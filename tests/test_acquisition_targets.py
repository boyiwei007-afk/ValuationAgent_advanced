from datetime import date

from valuationagent.application.research_plan import research_plan
from valuationagent.application.research_valuation import ResearchValuationAssembler
from observation_fixtures import fixture_observations
from test_multisource_extraction import runtime_at
from test_observation_extraction import review, submit


def test_ready_method_appears_before_other_method_gaps_without_executing(tmp_path):
    from test_input_workspace import fixture, values
    from valuationagent.application.record_inputs import record_inputs

    _, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    runtime.session.draft.methods = ["dcf", "pe"]
    plan = research_plan(runtime.session, runtime.service.valuation_assembler)
    first = plan["next_work"][0]
    assert first["kind"] == "calculate" and first["methods"] == ["pe"]
    assert first["tool"] == "calculate_valuation" and "review" in first["instruction"]
    assert any(item.get("method") == "dcf" and item["status"] == "blocked" for item in plan["next_work"])
    assert runtime.session.valuation_run_id is None
    runtime.session.pending_action = None
    assert not any(item.get("kind") == "calculate" for item in research_plan(runtime.session, runtime.service.valuation_assembler)["next_work"])


def test_empty_workspace_exposes_current_year_and_share_window_without_claiming_availability(tmp_path):
    runtime = runtime_at(tmp_path)
    plan = research_plan(runtime.session, runtime.service.valuation_assembler)
    annual, shares = plan["acquisition_targets"]
    assert annual["kind"] == "annual_baseline" and annual["target_year"] == 2025
    assert annual["metrics"] == ["net_income_parent", "revenue"]
    assert "不证明年报已经发布" in annual["instruction"]
    assert shares["required_since"] == "2026-06-02"
    assert runtime.working_state()["research_plan"]["acquisition_targets"] == plan["acquisition_targets"]


def test_old_pending_shares_remain_visible_but_not_the_first_repair_target(tmp_path):
    runtime = runtime_at(tmp_path)
    args = fixture_observations(runtime, [{"metric": "common_shares", "raw_value": "1000", "unit": "股", "period": "2024"}])
    submit(runtime, args)
    old = runtime.session.facts[0]
    plan = research_plan(runtime.session, runtime.service.valuation_assembler)
    repair = next(item for item in plan["repair_candidates"] if item["fact_id"] == old.fact_id)
    assert not repair["current_acquisition_priority"]
    assert not any(item.get("fact_id") == old.fact_id for item in plan["next_work"])
    assert any(item["kind"] == "capital_structure" and "required_since" in item for item in plan["next_work"])
    assert old.status == "proposed" and old.warnings


def test_current_share_proof_closes_acquisition_target_not_annual_baseline(tmp_path):
    runtime = runtime_at(tmp_path)
    args = fixture_observations(runtime, [{"metric": "common_shares", "raw_value": "1000", "unit": "股", "period": "2026-06-30"}])
    submit(runtime, args)
    assert any(item["kind"] == "capital_structure" for item in research_plan(runtime.session, runtime.service.valuation_assembler)["acquisition_targets"])
    review(runtime)
    targets = research_plan(runtime.session, runtime.service.valuation_assembler)["acquisition_targets"]
    assert [item["kind"] for item in targets] == ["annual_baseline"]
    assert targets[0]["target_year"] == 2025


def test_recent_share_date_cannot_hide_pending_annual_financial_inputs(tmp_path):
    runtime = runtime_at(tmp_path)
    args = fixture_observations(runtime, [
        {"metric": "common_shares", "raw_value": "1000", "unit": "股", "period": "2026-06-30"},
        {"metric": "revenue", "raw_value": "120000", "period": "2025"},
        {"metric": "net_income_parent", "raw_value": "20000", "period": "2025"},
    ])
    submit(runtime, args)
    shares, revenue, income = runtime.session.facts
    review(runtime, [shares.fact_id, revenue.fact_id])
    assert income in ResearchValuationAssembler.pending_blockers(runtime.session)
    assert runtime.session.draft.valuation_date == date(2026, 9, 30)
