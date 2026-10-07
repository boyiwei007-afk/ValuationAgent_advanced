import json
from decimal import Decimal

import pytest

from test_user_peers import scenario
from valuationagent.application.agent_runtime import AgentResponse, CalculateValuation
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.application.workspace_artifacts import ReportWrite, write_report
from valuationagent.schemas.inputs import RecordInputs


def partial_scenario(tmp_path):
    app, runtime = scenario(tmp_path, missing="lease_liabilities")
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] == "completed_with_warnings"
    write_report(app.state.research, runtime.session, ReportWrite(format="md"))
    return app, runtime, outcome


def test_partial_calculation_returns_remaining_requested_methods(tmp_path):
    _, _, outcome = partial_scenario(tmp_path)
    completion = outcome["method_completion"]
    assert set(completion["completed_methods"]) == {"pe", "ps"}
    assert set(completion["remaining_methods"]) == {"ev_ebitda"}
    assert "lease_liabilities" in completion["remaining_methods"]["ev_ebitda"]
    assert completion["all_requested_methods_completed"] is False


def test_partial_result_cannot_be_called_complete_but_discussion_and_honest_deferral_work(tmp_path):
    _, runtime, _ = partial_scenario(tmp_path)
    with pytest.raises(ValueError, match="VALUATION_METHODS_INCOMPLETE.*ev_ebitda"):
        runtime.finish(AgentResponse(answer="所有方法已完成"))
    response = runtime.finish(AgentResponse(answer="仅部分方法完成，缺少可比租赁输入。", outcome="insufficient_data"))
    assert response["_terminal"]
    runtime.session.turn_control.decision.actions = ["discuss"]
    assert runtime.finish(AgentResponse(answer="说明方法区别"))["_terminal"]


def test_supplemented_inputs_reopen_excluded_method_without_altering_frozen_result(tmp_path):
    app, runtime, previous = partial_scenario(tmp_path)
    frozen = app.state.store.get_run(previous["run_id"]).model_dump(mode="json")
    rows = runtime.session.input_dataset.active_records()
    sources = [row for row in rows if row.role == "comparable" and row.metric == "interest_bearing_debt"]
    record_inputs(runtime, RecordInputs(user_basis={"period_quote": "2025年度", "as_of_quote": "2025年12月31日"},
        user_values=[{"metric": "lease_liabilities", "amount_text": "0万元", "unit": "万元",
            "role": "comparable", "entity": row.entity,
            "amount_occurrence": json.loads(row.source.locator)["amount_occurrence"],
            "period_end": "2025-12-31", "as_of": "2025-12-31"} for row in sources]))
    assert runtime.session.valuation_methods_override == []
    assert runtime.session.valuation_method_exclusions == {}
    result = runtime.calculate(CalculateValuation())
    assert result["method_completion"]["all_requested_methods_completed"]
    current = app.state.store.get_run(result["run_id"])
    assert {item.method: item.per_share_value for item in current.result.relative} == {
        "pe": Decimal("22.5"), "ps": Decimal(15), "ev_ebitda": Decimal(26)}
    assert replay_bundle(build_valuation_bundle(app.state.store, current))["passed"]
    assert app.state.store.get_run(previous["run_id"]).model_dump(mode="json") == frozen


def test_ingestion_without_current_value_action_does_not_request_a_new_calculation(tmp_path):
    from test_input_workspace import fixture, values

    _, runtime = fixture(tmp_path)
    runtime.session.pending_action = ""
    runtime.session.turn_control.decision.actions = ["ingest"]
    record_inputs(runtime, values())
    assert runtime.session.pending_action == ""
