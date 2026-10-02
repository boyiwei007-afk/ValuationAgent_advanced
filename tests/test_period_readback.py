import json

import pytest

from valuationagent.application.observation_consistency import invalidate_unread_periods, period_readback_issue
from valuationagent.application.observation_extraction import ObservationSelection, prepare_reviews, ReviewObservations, review_observations
from valuationagent.application.research_valuation import ResearchValuationAssembler
from valuationagent.core.tools import canonical, ToolRegistry, ToolSpec
from valuationagent.llm.observation_review import focused_review_request
from observation_fixtures import fixture_observations
from test_multisource_extraction import runtime_at
from test_observation_extraction import review, submit


def shares(runtime):
    submit(runtime, fixture_observations(runtime, [{"metric": "common_shares", "raw_value": "1000股", "unit": "股", "period": "2025-12-31"}]))
    return runtime.session.facts[-1]


def test_review_reconstructs_instant_date_without_seeing_claimed_date(tmp_path):
    runtime = runtime_at(tmp_path)
    fact = shares(runtime)
    packet = prepare_reviews(runtime, ObservationSelection(fact_ids=[fact.fact_id]))
    messages = [{"role": "assistant", "tool_calls": [{"id": "review_packet", "function": {"name": "prepare_observation_review", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "review_packet", "content": canonical(packet)}]
    tools = ToolRegistry([ToolSpec("review_observations", "review", ReviewObservations, lambda _: None)]).schemas()
    focused, schemas = focused_review_request(messages, tools)
    shown = json.loads(focused[-1]["content"])
    assert shown["packets"][0]["requires_period_readback"]
    assert "period_end" not in shown["packets"][0]["interpretation"]["row"]
    assert "2025-12-31" in canonical(shown["original_context"])
    assert "source_period_end" in schemas[0]["function"]["parameters"]["$defs"]["ObservationReview"]["required"]
    assert packet["packets"][0]["interpretation"]["row"]["period_end"] == "2025-12-31"


@pytest.mark.parametrize("source_date,expected", [(None, "ambiguous"), ("2025-09-04", "contradicted")])
def test_supported_checkbox_cannot_override_missing_or_different_readback_date(tmp_path, source_date, expected):
    runtime = runtime_at(tmp_path)
    fact = shares(runtime)
    result = review(runtime, source_period_end=source_date)
    assert fact.status == "proposed" and fact.warnings
    assert fact.verification["semantic_review"]["checks"]["period"] == expected
    assert result["reviews"][0]["period_readback"]["source_period_end"] == source_date
    assert fact.period == "2025-12-31"
    assert not ResearchValuationAssembler._verified_dated_issuer_shares(fact)


def test_matching_date_readback_accepts_unit_suffix_and_is_recorded(tmp_path):
    runtime = runtime_at(tmp_path)
    fact = shares(runtime)
    review(runtime)
    assert fact.status == "confirmed" and not period_readback_issue(fact)
    assert ResearchValuationAssembler._verified_dated_issuer_shares(fact)
    assert fact.verification["semantic_review"]["source_period_end"] == "2025-12-31"


def test_omitted_readback_fails_before_applying_review(tmp_path):
    runtime = runtime_at(tmp_path)
    fact = shares(runtime)
    packet = prepare_reviews(runtime, ObservationSelection(fact_ids=[fact.fact_id]))["packets"][0]
    with pytest.raises(ValueError, match="PERIOD_READBACK_REQUIRED"):
        review_observations(runtime, ReviewObservations(reviews=[{"fact_id": fact.fact_id, "packet_id": packet["packet_id"],
            "rationale": "没有提交独立日期读取结果，即使勾选全部通过也不能确认时点。",
            "checks": {key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")}}]))
    assert fact.status == "proposed"


def test_previous_instant_review_is_downgraded_without_changing_source_or_frozen_value(tmp_path):
    runtime = runtime_at(tmp_path)
    fact = shares(runtime)
    review(runtime)
    proof = canonical(fact.verification["reading_proof"])
    del fact.verification["semantic_review"]["source_period_end"]
    assert not ResearchValuationAssembler._verified_dated_issuer_shares(fact)
    assert invalidate_unread_periods(runtime.session) == [fact.fact_id]
    assert fact.status == "proposed" and canonical(fact.verification["reading_proof"]) == proof
    assert not invalidate_unread_periods(runtime.session)
    review(runtime)
    assert fact.status == "confirmed" and not fact.warnings
