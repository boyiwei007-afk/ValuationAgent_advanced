from __future__ import annotations

from fastapi.testclient import TestClient

from valuationagent.api.main import create_app


def demo_body():
    return {
        "request": {
            "company": {"name": "API 演示公司", "currency": "CNY"},
            "valuation_date": "2026-09-12",
            "data_source": "structured",
            "assumption_source": "automatic",
            "mode": "demo",
            "forecast_years": 5,
            "methods": ["dcf", "pe", "ev_ebitda"],
            "assumptions": {},
            "peers": [],
            "file_ids": [],
            "user_goal": "通过 Web API 完成演示估值",
        }
    }


def test_api_run_events_results_and_conversation(tmp_path):
    from test_unified_workspace_agent import ScriptedModel, completed_workspace
    from valuationagent.schemas.research import ResearchTurn

    app, service, workspace, _ = completed_workspace(
        tmp_path, ScriptedModel(("read_valuation", {}), ("finish_response", {"answer": "WACC 已列在计算假设中。"}), actions=("discuss",))
    )
    record = app.state.store.get_run(workspace.active_run_id)
    with TestClient(app) as client:
        assert client.post("/api/runs", json=demo_body()).status_code == 405
        assert client.get(f"/api/runs/{record.run_id}/results").status_code == 200
        reply = client.post(f"/api/workspaces/{workspace.workspace_id}/messages", json={"content": "请说明 WACC"})
        assert reply.status_code == 202
        snapshot = client.get(f"/api/workspaces/{workspace.workspace_id}").json()
        assert "WACC" in snapshot["messages"][-1]["content"]
        assert snapshot["workspace"]["active_run_id"] == record.run_id
        with client.stream("GET", f"/api/runs/{record.run_id}/events") as stream:
            assert stream.status_code == 200
            lines = list(stream.iter_lines())
        assert any("event: run.completed" in line for line in lines)


def test_capabilities_report_runtime_adapters(tmp_path):
    app = create_app(tmp_path / "api-runtime")
    with TestClient(app) as client:
        capabilities = {
            item["capability_id"]: item
            for item in client.get("/api/capabilities").json()
        }
    assert capabilities["structured_financial_input"]["available"] is True
    assert capabilities["durable_conversation_context"]["available"] is True
    assert capabilities["interactive_agent_recovery"]["available"] is True
    assert capabilities["agent_tool_extensions"]["available"] is True
    assert capabilities["ticker_data_provider"]["available"] is True
    assert capabilities["research_document_ingestion"]["available"] is True


def test_upload_records_hash_but_does_not_claim_to_parse(tmp_path):
    app = create_app(tmp_path / "api-runtime")
    with TestClient(app) as client:
        response = client.post(
            "/api/files",
            data={"role": "historical_financials"},
            files={
                "file": (
                    "financials.xlsx",
                    b"placeholder workbook bytes",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert response.status_code == 201
        assert response.json()["sha256"]
        capabilities = {
            item["capability_id"]: item
            for item in client.get("/api/capabilities").json()
        }
        assert capabilities["research_document_ingestion"]["available"] is True
