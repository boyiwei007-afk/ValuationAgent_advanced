import pytest

from test_input_workspace import fixture, values
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.record_inputs import record_inputs


def test_requested_report_is_saved_with_calculation_without_another_model_call(tmp_path):
    app, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    result = runtime.calculate(CalculateValuation())
    delivery = result["report_delivery"]
    assert delivery["status"] == "saved"
    metadata, content = app.state.store.get_artifact(runtime.session.session_id, delivery["artifact_id"])
    assert metadata["valuation_run_id"] == result["run_id"]
    assert metadata["numeric_result_available"]
    assert "40.00" in content.decode("utf-8")
    assert delivery["download_url"].endswith(delivery["artifact_id"])


@pytest.mark.parametrize("blocked_by", ["action", "permission", "review"])
def test_auto_report_does_not_expand_user_actions_or_permissions(tmp_path, blocked_by):
    app, runtime = fixture(tmp_path)
    if blocked_by == "action":
        runtime.session.turn_control.decision.actions = ["discuss"]
    elif blocked_by == "permission":
        runtime.session.turn_control.effects = [effect for effect in runtime.session.turn_control.effects if effect != "artifacts"]
    else:
        workspace = app.state.store.workspace_for_research(runtime.session.session_id)
        workspace.run_policy = "review"
        app.state.store.save_workspace(workspace)
    record_inputs(runtime, values())
    result = runtime.calculate(CalculateValuation())
    assert "report_delivery" not in result
    assert not app.state.store.list_artifacts(runtime.session.session_id)


def test_value_action_delivers_audit_report_without_separate_report_action(tmp_path):
    app, runtime = fixture(tmp_path)
    runtime.session.turn_control.decision.actions = ["value"]
    record_inputs(runtime, values())
    result = runtime.calculate(CalculateValuation())
    metadata, content = app.state.store.get_artifact(runtime.session.session_id, result["report_delivery"]["artifact_id"])
    assert metadata["valuation_run_id"] == result["run_id"]
    assert metadata["numeric_result_available"] and "40.00" in content.decode("utf-8")


def test_value_without_report_action_still_requires_actual_file_delivery(tmp_path, monkeypatch):
    from valuationagent.application.agent_runtime import AgentResponse

    _, runtime = fixture(tmp_path)
    runtime.session.turn_control.decision.actions = ["value"]
    record_inputs(runtime, values())

    def fail(*args):
        raise OSError("synthetic storage failure")

    monkeypatch.setattr("valuationagent.application.agent_runtime.write_report", fail)
    runtime.calculate(CalculateValuation())
    with pytest.raises(ValueError, match="REPORT_NOT_DELIVERED"):
        runtime.finish(AgentResponse(answer="已完成"))


def test_artifact_opt_out_applies_to_sensitivity_and_interruption_not_just_valuation(tmp_path):
    from valuationagent.application.workspace_artifacts import save_interruption_report
    from valuationagent.application.workspace_sensitivity import SensitivityRequest, analyze_sensitivity

    app, runtime = fixture(tmp_path)
    runtime.session.turn_control.effects.remove("artifacts")
    record_inputs(runtime, values())
    result = runtime.calculate(CalculateValuation())
    assert "report_delivery" not in result
    outcome = analyze_sensitivity(runtime, SensitivityRequest(method="pe", parameter="multiple", values=[15, 25]))
    assert [row["per_share_value"] for row in outcome["scenarios"]] == ["30.0000", "50.0000"]
    assert outcome["artifact"] is None and outcome["artifact_status"] == "disabled_by_user"
    runtime.session.pending_action = "valuation"
    assert save_interruption_report(runtime.service, runtime.session, {"numeric_result_available": False}, "interruption") is None
    assert not app.state.store.list_artifacts(runtime.session.session_id)


def test_working_state_keeps_completion_without_repeating_full_valuation_payload(tmp_path):
    from test_user_peers import scenario

    _, runtime = scenario(tmp_path, missing="lease_liabilities")
    result = runtime.calculate(CalculateValuation())
    checkpoint = runtime.working_state()["valuation"]
    assert checkpoint["method_completion"] == result["method_completion"]
    assert checkpoint["run_id"] == result["run_id"]
    assert checkpoint["retrieve_with"] == "read_valuation"
    assert "result" not in checkpoint and "financial_display" not in checkpoint


def test_report_failure_preserves_successful_calculation_and_reports_separate_error(tmp_path, monkeypatch):
    app, runtime = fixture(tmp_path)
    record_inputs(runtime, values())

    def fail(*args):
        raise OSError("synthetic report storage failure")

    monkeypatch.setattr("valuationagent.application.agent_runtime.write_report", fail)
    result = runtime.calculate(CalculateValuation())
    assert result["status"] in {"completed", "completed_with_warnings"}
    assert result["report_delivery"]["status"] == "failed"
    assert "synthetic report storage failure" in result["report_delivery"]["error"]
    assert runtime.calculation_attempt["status"] == result["status"]
    assert app.state.store.get_run(result["run_id"]).result.relative[0].per_share_value == 40


def test_partial_report_keeps_prices_but_does_not_claim_all_methods_completed(tmp_path):
    from test_user_peers import scenario

    app, runtime = scenario(tmp_path, missing="lease_liabilities")
    result = runtime.calculate(CalculateValuation())
    metadata, content = app.state.store.get_artifact(runtime.session.session_id, result["report_delivery"]["artifact_id"])
    assert metadata["status"] == "partially_valued"
    assert metadata["numeric_result_available"]
    assert "部分方法已完成" in content.decode("utf-8")
    assert "EV/EBITDA" in content.decode("utf-8")
    document = app.state.store.research_report(runtime.session.session_id)
    assert document["method_completion"]["remaining_methods"].keys() == {"ev_ebitda"}
    assert not document["method_completion"]["all_requested_methods_completed"]


def test_report_cancellation_is_not_swallowed_as_a_storage_error(tmp_path, monkeypatch):
    from valuationagent.llm.client import LlmError

    app, runtime = fixture(tmp_path)
    record_inputs(runtime, values())

    def cancel(*args):
        raise LlmError("EXECUTION_CANCELLED: user cancelled")

    monkeypatch.setattr("valuationagent.application.agent_runtime.write_report", cancel)
    with pytest.raises(LlmError, match="EXECUTION_CANCELLED"):
        runtime.calculate(CalculateValuation())
    record = app.state.store.get_run(runtime.session.valuation_run_id)
    assert record.result.relative[0].per_share_value == 40
    assert not app.state.store.list_artifacts(runtime.session.session_id)
