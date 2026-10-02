import json
from decimal import Decimal

import pytest

from valuationagent.application.agent_runtime import WorkspaceAgentRuntime
from valuationagent.application.workspace_sensitivity import SensitivityRequest, analyze_sensitivity
from test_unified_workspace_agent import completed_workspace


def sensitivity_runtime(tmp_path):
    app, service, workspace, _ = completed_workspace(tmp_path)
    session = app.state.store.get_research(workspace.research_session_id)
    return WorkspaceAgentRuntime(service.research, session, service), service, workspace


def test_custom_sensitivity_is_immutable_reproducible_and_not_source_data(tmp_path):
    runtime, service, workspace = sensitivity_runtime(tmp_path)
    before = service.store.get_run(workspace.active_run_id).model_dump(mode="json")
    request = SensitivityRequest(method="pe", parameter="earnings_scale", values=["0.9", "1", "1.1"])
    output = analyze_sensitivity(runtime, request)
    base = Decimal(output["baseline_per_share"])
    assert abs(Decimal(output["scenarios"][0]["per_share_value"]) - base * Decimal("0.9")) < Decimal("0.02")
    assert Decimal(output["scenarios"][1]["per_share_value"]) == base
    assert Decimal(output["scenarios"][2]["per_share_value"]) > base
    assert service.store.get_run(workspace.active_run_id).model_dump(mode="json") == before
    assert service.get(workspace.workspace_id).active_run_id == workspace.active_run_id
    assert not runtime.session.facts
    metadata, payload = service.store.get_artifact(runtime.session.session_id, output["artifact"]["artifact_id"])
    saved = json.loads(payload)
    assert saved["frozen_inputs"]["financials"] == before["result"]["effective_financials"]
    assert saved["baseline_input_hash"] == before["input_hash"]
    assert metadata["kind"] == "sensitivity_analysis"
    assert saved["scenarios"] == analyze_sensitivity(runtime, request)["scenarios"]


def test_dcf_sensitivity_handles_invalid_g_without_inventing_price(tmp_path):
    runtime, _, _ = sensitivity_runtime(tmp_path)
    output = analyze_sensitivity(runtime, SensitivityRequest(method="dcf", parameter="terminal_growth", values=["0.01", "0.19"]))
    assert output["scenarios"][0]["status"] == "completed"
    assert output["scenarios"][1]["status"] == "invalid"
    assert output["scenarios"][1]["per_share_value"] is None


@pytest.mark.parametrize("parameters,error", [
    ({"method": "pe", "parameter": "wacc", "values": ["0.08"]}, "PARAMETER_SCOPE"),
    ({"method": "ps", "parameter": "multiple", "values": ["10"]}, "METHOD_UNAVAILABLE"),
    ({"method": "pe", "parameter": "earnings_scale", "values": ["-1"]}, "RANGE"),
])
def test_sensitivity_rejects_invalid_scope_and_bounds(tmp_path, parameters, error):
    runtime, _, _ = sensitivity_runtime(tmp_path)
    with pytest.raises(ValueError, match=error):
        analyze_sensitivity(runtime, SensitivityRequest(**parameters))
    assert not runtime.service.store.list_artifacts(runtime.session.session_id)


def test_sensitivity_requires_completed_baseline(tmp_path):
    from test_multisource_extraction import runtime_at
    runtime = runtime_at(tmp_path)
    with pytest.raises(ValueError, match="BASELINE_REQUIRED"):
        analyze_sensitivity(runtime, SensitivityRequest(method="pe", parameter="multiple", values=["10"]))
