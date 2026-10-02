from __future__ import annotations

from unittest.mock import patch
from fastapi.testclient import TestClient

from valuationagent.api.main import create_app


def test_web_is_served_with_api_and_private_files_are_not_exposed(
    tmp_path, monkeypatch
):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text(
        "<html><title>ValuationAgent</title></html>", encoding="utf-8"
    )
    (tmp_path / "private.json").write_text("private", encoding="utf-8")
    monkeypatch.setenv("VALUATION_WEB_DIR", str(web))
    app = create_app(tmp_path / "runtime")
    with TestClient(app) as client:
        assert "ValuationAgent" in client.get("/").text
        assert client.get("/health").json()["service"] == "valuationagent"
        assert client.get("/api/runs").json() == []
        assert client.get("/private.json").status_code == 404
        assert client.get("/var/runs.sqlite3").status_code == 404
        assert client.get("/api/unknown").status_code == 404


def test_reconnecting_a_model_does_not_execute_or_revise_the_task(tmp_path):
    from test_unified_workspace_agent import ScriptedModel, completed_workspace
    app, service, workspace, _ = completed_workspace(tmp_path, ScriptedModel())
    record = app.state.store.get_run(workspace.active_run_id)
    url = f"/api/workspaces/{workspace.workspace_id}/model-session"
    with TestClient(app) as client:
        session = client.post("/api/model-sessions", json={"model": "mock", "api_key": "TEST_ONLY_NOT_A_REAL_KEY"}).json()
        with patch.object(app.state.runner, "execute", side_effect=AssertionError("must not execute")):
            response = client.post(url, json={"model_session_id": session["session_id"]})
        assert response.status_code == 204
        attached = app.state.research._clients[workspace.research_session_id]
        assert attached.config.model == "mock"
        assert app.state.store.get_run(record.run_id) == record
        assert client.post(url, json={}).status_code == 422
        assert client.post(url, json={"model_session_id": "missing"}).status_code == 404
        assert app.state.store.acquire(workspace.research_session_id, "test-owner")
        assert client.post(url, json={"model_session_id": session["session_id"]}).status_code == 409
        app.state.store.release(workspace.research_session_id, "test-owner")
        assert client.delete(f"/api/model-sessions/{session['session_id']}").status_code == 204
        assert attached.revoked.is_set()


def test_workspace_report_preserves_summary_and_escapes_untrusted_text(tmp_path):
    app = create_app(tmp_path)
    workspace = app.state.workspaces.create()
    session = app.state.store.get_research(workspace.research_session_id)
    session.summary = "资料摘要 <script>alert(1)</script>"
    session.gaps = ["需要核对 <行业口径> & 原文依据"]
    app.state.store.save_research(session)
    with TestClient(app) as client:
        response = client.get(f"/api/workspaces/{workspace.workspace_id}/export?format=html")
        assert response.status_code == 200
        assert "资料摘要 &lt;script&gt;" in response.text
        assert "&lt;行业口径&gt;" in response.text
        assert "保留" not in session.summary
        assert "<script>" not in response.text
        assert 'name="viewport"' in response.text
        assert client.get(f"/api/workspaces/{workspace.workspace_id}/export?format=pdf").status_code == 200
