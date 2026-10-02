"""Synthetic, scripted evidence-to-report integration, not a live company test."""
from datetime import date
from decimal import Decimal

import pytest

from valuationagent.api.main import create_app
from valuationagent.application.agent_runtime import CalculateValuation, WorkspaceAgentRuntime
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.application.workspace_artifacts import ArtifactRead, ReportWrite, read_artifact, write_report
from valuationagent.application.workspace_sensitivity import SensitivityRequest, analyze_sensitivity
from valuationagent.schemas.research import ResearchDraft
from observation_fixtures import fixture_observations
from test_observation_extraction import review, submit
from test_peer_inputs import peer_component


@pytest.mark.parametrize("method,baseline,parameter", [("pe", Decimal(20), "earnings_scale"), ("ps", Decimal(4), "revenue_scale")])
def test_components_reach_frozen_valuation_numeric_report_sensitivity_and_replay(tmp_path, method, baseline, parameter):
    service = create_app(tmp_path).state.workspaces
    workspace = service.create(run_policy="automatic", data_source_preference="upload", title="Synthetic handoff fixture")
    session = service.store.get_research(workspace.research_session_id)
    session.draft = ResearchDraft(company="Synthetic target", ticker="600123", industry="家电",
                                  valuation_date=date(2026, 9, 30), methods=[method], objective="Synthetic contract regression, not investment research")
    session.pending_action = "valuation"
    runtime = WorkspaceAgentRuntime(service.research, session, service)
    submit(runtime, fixture_observations(runtime, [
        {"metric": "net_income_parent", "raw_value": "100000000", "period": "2025"},
        {"metric": "revenue", "raw_value": "1000000000", "period": "2025"},
        {"metric": "common_shares", "raw_value": "100000000", "period": "2026-06-30", "unit": "股"},
    ]))
    review(runtime)
    for ticker, cap in [("600456", "180000"), ("600457", "200000"), ("600458", "220000")]:
        peer_component(runtime, "market_cap", cap, ticker=ticker)
        peer_component(runtime, "net_income_parent", "10000", ticker=ticker)
        peer_component(runtime, "revenue", "500000", ticker=ticker)
    service.store.save_research(session)
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] in {"completed", "completed_with_warnings"}, outcome
    record = service.store.get_run(outcome["run_id"])
    assert record.request.mode != "demo"
    assert record.request.assumptions.wacc is None
    event = next(event for event in service.store.list_events(session.session_id) if event.type == "valuation.plan_confirmed")
    assert event.payload["approved_by"] == "automatic_policy" and "非人工批准" in event.summary
    assert record.result.relative[0].per_share_value == baseline
    assert record.result.relative[0].sample_size == 3
    assert all(peer.calculation_methods["pe"] == "market_cap / net_income_parent" for peer in record.result.effective_peers)
    saved = write_report(service.research, runtime.session, ReportWrite(format="json"))
    assert saved["numeric_result_available"] is True and saved["kind"] == "result_report"
    assert read_artifact(service.store, session.session_id, ArtifactRead(artifact_id=saved["artifact_id"]))
    before = record.model_dump(mode="json")
    sensitivity = analyze_sensitivity(runtime, SensitivityRequest(method=method, parameter=parameter, values=["0.9", "1", "1.1"]))
    assert [Decimal(row["per_share_value"]) for row in sensitivity["scenarios"]] == [baseline * Decimal("0.9"), baseline, baseline * Decimal("1.1")]
    assert service.store.get_run(record.run_id).model_dump(mode="json") == before
    replay = replay_bundle(build_valuation_bundle(service.store, record))
    assert replay["passed"] and not replay["network_used"]
