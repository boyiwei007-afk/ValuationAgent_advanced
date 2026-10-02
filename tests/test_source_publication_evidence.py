from datetime import date

import pytest
from pydantic import ValidationError

from valuationagent.application.observation_extraction import (
    ExtractObservations, ObservationSelection, PUBLICATION_PENDING, prepare_reviews,
)
from test_multisource_extraction import attach, runtime_at
from test_observation_extraction import example, review, submit


def publication_example(runtime, *, metadata_date=None, claim="2026-04-20", public=False):
    args = example(runtime, public=public)
    blocks = runtime.service.store.research_blocks(runtime.session.session_id, args["file_id"])
    blocks[0]["text"] += "\nFirst published: April 20, 2026.\nApproved by the board: April 18, 2026."
    blocks[0]["location"]["published_at"] = metadata_date
    runtime.service.store.save_research_blocks(runtime.session.session_id, args["file_id"], blocks)
    args["anchors"]["publication"] = {"block_id": blocks[0]["block_id"], "start_line": 6}
    args["basis"].update(source_published_at=claim, source_publication_refs=["publication"])
    return args


def publication_checks(value):
    return {**{key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")},
            "publication": value}


def test_missing_metadata_can_use_anchored_llm_date_only_after_separate_review(tmp_path):
    runtime = runtime_at(tmp_path)
    args = publication_example(runtime)
    assert submit(runtime, args)["saved_count"] == 1
    fact = runtime.session.facts[0]
    assert fact.published_at == date(2026, 4, 20) and fact.status == "proposed"
    assert PUBLICATION_PENDING in fact.warnings
    packet = prepare_reviews(runtime, ObservationSelection(fact_ids=[fact.fact_id]))["packets"][0]
    assert packet["required_checks"][-1] == "publication"
    assert packet["publication"]["metadata_date"] is None
    with pytest.raises(ValueError, match="PUBLICATION_REVIEW_REQUIRED"):
        review(runtime)
    assert fact.status == "proposed"
    review(runtime, checks=publication_checks("supported"))
    assert fact.status == "confirmed" and not fact.warnings
    assert fact.verification["semantic_review"]["checks"]["publication"] == "supported"


@pytest.mark.parametrize("judgment", ["ambiguous", "contradicted"])
def test_report_period_or_board_date_is_not_automatically_accepted(tmp_path, judgment):
    runtime = runtime_at(tmp_path)
    args = publication_example(runtime, claim="2026-04-18")
    args["anchors"]["publication"]["start_line"] = 7
    submit(runtime, args)
    review(runtime, checks=publication_checks(judgment))
    fact = runtime.session.facts[0]
    assert fact.status == "proposed" and PUBLICATION_PENDING in fact.warnings
    assert fact.verification["semantic_review"]["status"] == "needs_evidence"


def test_llm_claim_cannot_overwrite_known_source_date_or_hide_future_disclosure(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, publication_example(runtime, metadata_date="2026-10-15"))
    review(runtime, checks=publication_checks("supported"))
    fact = runtime.session.facts[0]
    assert fact.published_at == date(2026, 10, 15)
    assert fact.status == "proposed"
    assert any(warning.startswith("SOURCE_PUBLICATION_CONFLICT") for warning in fact.warnings)
    assert "来源期间或披露日期晚于信息截止日" in fact.warnings


def test_public_source_date_does_not_upgrade_grade_or_bypass_corroboration(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, publication_example(runtime, public=True))
    review(runtime, checks=publication_checks("supported"))
    fact = runtime.session.facts[0]
    assert fact.status == "proposed"
    assert fact.verification["source_assessment"]["source_tier"] == "C"
    assert any(warning.startswith("PUBLIC_SOURCE_UNCORROBORATED") for warning in fact.warnings)


def test_publication_claim_requires_local_evidence_and_is_integrity_bound(tmp_path):
    runtime = runtime_at(tmp_path)
    args = publication_example(runtime)
    args["basis"]["source_publication_refs"] = []
    with pytest.raises(ValidationError, match="须同时提供"):
        ExtractObservations.model_validate(args)
    args["basis"]["source_publication_refs"] = ["publication"]
    other = attach(runtime, "other.txt", "Published April 20, 2026")[0]
    args["anchors"]["publication"]["block_id"] = other["block_id"]
    result = submit(runtime, args)
    assert result["saved_count"] == 0 and "ANCHOR_SCOPE" in result["rows"][0]["error"]
    args["anchors"]["publication"]["block_id"] = args["file_id"] + ":1"
    assert submit(runtime, args)["saved_count"] == 1
    runtime.session.facts[0].published_at = date(2026, 4, 19)
    with pytest.raises(ValueError, match="OBSERVATION_CHANGED"):
        review(runtime, checks=publication_checks("supported"))
