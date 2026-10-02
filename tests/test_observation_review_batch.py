import copy

import pytest

from valuationagent.application.observation_extraction import (
    ObservationSelection, ReviewObservations, prepare_reviews, review_observations,
)
from test_multisource_extraction import runtime_at
from test_observation_extraction import example, submit


def prepared_review(runtime):
    submit(runtime, example(runtime))
    fact = runtime.session.facts[0]
    packet = prepare_reviews(runtime, ObservationSelection(fact_ids=[fact.fact_id]))["packets"][0]
    return {"fact_id": fact.fact_id, "packet_id": packet["packet_id"],
            "rationale": "依据原始上下文核对主体、期间、金额单位和报表口径，各项均具有明确的原文支持。",
            "checks": {key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")}}


def test_invalid_later_review_does_not_partially_confirm_earlier_fact(tmp_path):
    runtime = runtime_at(tmp_path)
    valid = prepared_review(runtime)
    before = runtime.session.facts[0].model_dump(mode="json")
    invalid = {**valid, "fact_id": "fact_does_not_exist"}
    with pytest.raises(ValueError, match="OBSERVATION_NOT_FOUND"):
        review_observations(runtime, ReviewObservations(reviews=[valid, invalid]))
    assert runtime.session.facts[0].model_dump(mode="json") == before


def test_duplicate_review_cannot_overwrite_judgment_inside_same_batch(tmp_path):
    runtime = runtime_at(tmp_path)
    valid = prepared_review(runtime)
    contradicted = copy.deepcopy(valid)
    contradicted["checks"]["period"] = "contradicted"
    with pytest.raises(ValueError, match="REVIEW_DUPLICATE"):
        review_observations(runtime, ReviewObservations(reviews=[valid, contradicted]))
    assert runtime.session.facts[0].status == "proposed"


def test_rejected_observation_is_explicit_and_does_not_issue_other_packets(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    submit(runtime, example(runtime, name="prior.txt", year="2024"))
    runtime.session.facts[1].status = "rejected"
    selection = ObservationSelection(fact_ids=[fact.fact_id for fact in runtime.session.facts])
    with pytest.raises(ValueError, match="OBSERVATION_REJECTED") as failure:
        prepare_reviews(runtime, selection)
    assert runtime.session.facts[1].fact_id in str(failure.value)
    assert "inspect_context" in str(failure.value)
    assert "issued_packet_id" not in runtime.session.facts[0].verification["semantic_review"]
