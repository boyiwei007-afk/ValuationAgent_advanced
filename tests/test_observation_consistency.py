import copy

import pytest

from valuationagent.application.observation_consistency import PERIOD_CELL_CONFLICT, period_cell_conflicts
from valuationagent.application.research_valuation import ResearchValuationAssembler
from test_multisource_extraction import runtime_at
from test_observation_extraction import example, submit, review


def conflicting_rows(runtime):
    args = example(runtime)
    older = copy.deepcopy(args["rows"][0])
    older.update(period_start="2024-01-01", period_end="2024-12-31")
    args["rows"].append(older)
    submit(runtime, args)
    return args


def test_same_value_cell_cannot_be_confirmed_for_different_years_by_all_supported_review(tmp_path):
    runtime = runtime_at(tmp_path)
    conflicting_rows(runtime)
    review(runtime)
    assert len(period_cell_conflicts(runtime.session.facts)) == 2
    assert all(fact.status == "proposed" and PERIOD_CELL_CONFLICT in fact.warnings for fact in runtime.session.facts)
    with pytest.raises(ValueError, match="SOURCE_PERIOD_COLLISION"):
        ResearchValuationAssembler()._structured_financials(runtime.session)
    runtime.session.facts[1].status = "rejected"
    review(runtime, [runtime.session.facts[0].fact_id])
    assert runtime.session.facts[0].status == "confirmed"
    assert not period_cell_conflicts(runtime.session.facts)


def test_explicit_same_cell_period_correction_can_replace_wrong_interpretation(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    args["rows"][0].update(period_start="2024-01-01", period_end="2024-12-31")
    submit(runtime, args)
    previous = runtime.session.facts[0]
    args["rows"][0].update(period_start="2025-01-01", period_end="2025-12-31", replaces=[previous.fact_id])
    submit(runtime, args)
    assert not period_cell_conflicts(runtime.session.facts)
    review(runtime, [runtime.session.facts[-1].fact_id])
    assert previous.status == "rejected"


def test_equal_values_in_different_cells_remain_distinct_evidence(tmp_path):
    runtime = runtime_at(tmp_path)
    first = example(runtime, year="2025")
    second = example(runtime, year="2024", name="prior.txt")
    submit(runtime, first)
    submit(runtime, second)
    review(runtime)
    assert not period_cell_conflicts(runtime.session.facts)
    assert all(fact.status == "confirmed" for fact in runtime.session.facts)


def test_different_quote_windows_cannot_hide_the_same_numeric_cell(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    args["anchors"]["value"]["quote"] = ""
    submit(runtime, args)
    args["anchors"]["value"].update(start_line=1, end_line=2)
    args["rows"][0].update(period_start="2024-01-01", period_end="2024-12-31")
    submit(runtime, args)
    assert len(period_cell_conflicts(runtime.session.facts)) == 2


def test_resumed_turn_invalidates_previously_accepted_collisions_before_model_sees_them(tmp_path):
    from observation_fixtures import ObservationModel

    runtime = runtime_at(tmp_path)
    conflicting_rows(runtime)
    for fact in runtime.session.facts:
        fact.status, fact.warnings = "confirmed", []
    model = ObservationModel([])
    runtime.run(model)
    assert all(fact.status == "proposed" and PERIOD_CELL_CONFLICT in fact.warnings for fact in runtime.session.facts)
    assert "SOURCE_PERIOD_COLLISION" in str(model.calls[0])
    assert any(event.type == "evidence.consistency" for event in runtime.service.store.list_events(runtime.session.session_id))


def test_explicit_entity_contradiction_retires_interpretation_without_rewriting_company(tmp_path):
    from valuationagent.application.observation_consistency import retire_contradicted_entities

    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    fact = runtime.session.facts[0]
    assert not retire_contradicted_entities(runtime.session.facts)
    checks = {key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")}
    checks["entity"] = "ambiguous"
    review(runtime, checks=checks)
    assert not retire_contradicted_entities(runtime.session.facts)
    checks["entity"] = "contradicted"
    review(runtime, checks=checks)
    proof = copy.deepcopy(fact.verification["reading_proof"])
    ticker = runtime.session.draft.ticker
    assert fact.status == "rejected"
    assert not retire_contradicted_entities(runtime.session.facts)
    fact.status = "proposed"
    assert retire_contradicted_entities(runtime.session.facts) == [fact.fact_id]
    assert fact.status == "rejected"
    assert fact.verification["reading_proof"] == proof
    assert runtime.session.draft.ticker == ticker
    assert fact.verification["disposition"]["actor"] == "semantic_review_policy"
    assert not retire_contradicted_entities(runtime.session.facts)
