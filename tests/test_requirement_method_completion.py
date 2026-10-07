from decimal import Decimal

from test_input_workspace import fixture, values
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.research_plan import research_plan


def partial_valuation(tmp_path):
    app, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    runtime.session.draft.methods = ["pe", "ev_ebitda"]
    result = runtime.calculate(CalculateValuation())
    assert result["status"] in {"completed", "completed_with_warnings"}
    assert runtime.session.valuation_methods_override == ["pe"]
    return app, runtime


def test_partial_calculation_does_not_remove_requested_method_requirements(tmp_path):
    _, runtime = partial_valuation(tmp_path)
    before = runtime.session.model_dump(mode="json")
    plan = runtime.requirements()
    assert plan["methods"] == ["pe", "ev_ebitda"]
    assert "ebitda" in plan["required_metrics"]
    remaining = next(row for row in plan["method_readiness"] if row["method"] == "ev_ebitda")
    assert remaining["status"] == "blocked" and remaining["reason"]
    assert any(row.get("method") == "ev_ebitda" for row in plan["next_work"])
    assert plan["method_completion"]["completed_methods"] == ["pe"]
    assert "ev_ebitda" in plan["method_completion"]["remaining_methods"]
    assert not any("pe" in row.get("methods", []) for row in plan["next_work"])
    assert runtime.session.model_dump(mode="json") == before
    assert runtime.working_state()["research_plan"]["method_completion"] == plan["method_completion"]


def test_changed_inputs_do_not_reuse_old_method_completion(tmp_path):
    app, runtime = partial_valuation(tmp_path)
    original = app.state.store.get_run(runtime.session.valuation_run_id)
    earnings = next(row for row in runtime.session.input_dataset.records if row.metric == "net_income_parent")
    earnings.value = Decimal("1100000000")
    runtime.session.pending_action = "valuation"
    plan = runtime.requirements()
    assert plan["method_completion"]["completed_in_run"] == ["pe"]
    assert plan["method_completion"]["completed_methods"] == []
    assert plan["method_completion"]["current_inputs_match"] is False
    assert set(plan["method_completion"]["remaining_methods"]) == {"pe", "ev_ebitda"}
    assert any("pe" in row.get("methods", []) for row in plan["next_work"])
    assert app.state.store.get_run(original.run_id) == original


def test_snapshot_matches_agent_and_read_only_inspection_does_not_authorize_calculation(tmp_path):
    app, runtime = partial_valuation(tmp_path)
    workspace = app.state.store.workspace_for_research(runtime.session.session_id)
    plan = app.state.workspaces.snapshot(workspace.workspace_id)["research_plan"]
    assert plan["methods"] == ["pe", "ev_ebitda"]
    assert plan["method_completion"]["completed_methods"] == ["pe"]
    runtime.session.turn_control.effects = ["read"]
    runtime.session.turn_control.decision.actions = ["discuss"]
    runtime.session.pending_action = "valuation"
    before = runtime.session.model_dump(mode="json")
    assert not any(row.get("kind") == "calculate" for row in runtime.requirements()["next_work"])
    assert runtime.session.model_dump(mode="json") == before
    assert runtime.working_state()["research_plan"] == {
        "instruction": "这是最新持久状态，每次工具执行后更新；完整字段字典、年度覆盖和修复清单用inspect_requirements读取。"
    }


def test_without_calculation_record_readiness_never_claims_completion(tmp_path):
    _, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    runtime.session.draft.methods = ["pe", "ev_ebitda"]
    runtime.session.valuation_methods_override = ["pe"]
    plan = research_plan(runtime.session, runtime.service.valuation_assembler)
    assert plan["methods"] == ["pe", "ev_ebitda"]
    assert plan["method_completion"]["completed_methods"] == []
    assert plan["method_completion"]["all_requested_methods_completed"] is False


def test_declared_assumption_change_invalidates_completed_method(tmp_path):
    _, runtime = partial_valuation(tmp_path)
    multiple = next(row for row in runtime.session.input_dataset.records if row.metric == "pe_multiple")
    multiple.value = Decimal("25")
    plan = runtime.requirements()
    assert plan["method_completion"]["current_inputs_match"] is False
    assert not plan["method_completion"]["completed_methods"]


def test_document_observation_repairs_keep_excluded_requested_method_in_scope(tmp_path):
    from observation_fixtures import fixture_observations
    from test_multisource_extraction import runtime_at
    from test_observation_extraction import submit

    runtime = runtime_at(tmp_path)
    runtime.session.draft.methods = ["pe", "ev_ebitda"]
    runtime.session.valuation_methods_override = ["pe"]
    runtime.session.valuation_method_exclusions = {"ev_ebitda": "本次部分计算尚未包含企业价值倍数"}
    submitted = submit(runtime, fixture_observations(runtime, [
        {"metric": "ebitda", "raw_value": "250000", "period": "2025"},
    ]))
    assert submitted["saved_count"] == 1
    assert runtime.session.input_dataset is None
    observation = runtime.session.facts[0]
    assert observation.status == "proposed"
    before = runtime.session.model_dump(mode="json")
    plan = runtime.requirements()
    repair = next(item for item in plan["repair_candidates"] if item["fact_id"] == observation.fact_id)
    assert repair["blocks_selected_methods"] is True
    assert any(item.get("kind") == "repair" and item.get("fact_id") == observation.fact_id for item in plan["next_work"])
    assert plan["methods"] == ["pe", "ev_ebitda"]
    assert runtime.session.model_dump(mode="json") == before
