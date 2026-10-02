from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from valuationagent.api.main import create_app
from valuationagent.schemas.models import CompanyInput, RunStatus, ValuationRequest
from valuationagent.schemas.workspace import AgentAction
from valuationagent.storage.sqlite import SQLiteRunStore


def delete(client, workspace, revision=None):
    return client.delete(f"/api/workspaces/{workspace.workspace_id}", params={"revision": workspace.revision if revision is None else revision})


def test_deletion_removes_owned_graph_but_preserves_other_tasks_and_shared_files(tmp_path):
    app = create_app(tmp_path)
    service, store = app.state.workspaces, app.state.store
    target, other = service.create(title="Delete me"), service.create(title="Keep me")
    session_id = target.research_session_id
    uploaded = store.save_upload("shared.txt", "evidence", "text/plain", b"source")
    for workspace in (target, other):
        store.save_research_blocks(workspace.research_session_id, uploaded["file_id"], [{"text": "source"}])
        store.save_artifact(workspace.research_session_id, {"kind": "research_note", "filename": "note.md"}, b"note")
    run = store.create_run("run_owned", ValuationRequest(company=CompanyInput(name="Issuer"), valuation_date=date(2026, 9, 30)))
    store.update_run(run.run_id, status=RunStatus.CANCELLED)
    target.active_run_id = run.run_id
    store.save_workspace(target)
    store.add_message(run.run_id, "assistant", "owned result", "result")
    store.append_event(run.run_id, type="test", summary="owned event")
    store.reserve_research_job(session_id, "request_1", {"content": "test"})
    store.update_research_job(session_id, "request_1", status="completed")
    store.save_research_report(session_id, {"source_revision": 0, "report_id": "report_owned"}, {})
    with TestClient(app) as client:
        assert delete(client, target).status_code == 204
        assert client.get(f"/api/workspaces/{target.workspace_id}").status_code == 404
        assert client.get(f"/api/workspaces/{other.workspace_id}").status_code == 200
        assert [row["workspace_id"] for row in client.get("/api/workspaces").json()] == [other.workspace_id]
        assert delete(client, target).status_code == 404
    with store._connect() as database:
        for table in ("research_sessions", "research_jobs", "research_documents", "research_reports", "workspace_artifacts"):
            assert not database.execute(f"SELECT 1 FROM {table} WHERE session_id=?", (session_id,)).fetchone()
        for identity in (session_id, run.run_id):
            for table in ("runs", "lineage", "run_sources", "events", "event_summaries", "messages", "leases", "checkpoints"):
                assert not database.execute(f"SELECT 1 FROM {table} WHERE run_id=?", (identity,)).fetchone()
        assert not database.execute("SELECT 1 FROM workspace_records WHERE workspace_id=?", (target.workspace_id,)).fetchone()
    assert store.list_artifacts(other.research_session_id)
    assert Path(store.get_file(uploaded["file_id"])["storage_path"]).read_bytes() == b"source"


@pytest.mark.parametrize("busy", ["queued", "running", "lease", "calculation"])
def test_busy_delete_is_atomic_and_returns_conflict(tmp_path, busy):
    app = create_app(tmp_path)
    service, store = app.state.workspaces, app.state.store
    workspace = service.create()
    if busy in {"queued", "running"}:
        store.reserve_research_job(workspace.research_session_id, "request_busy", {})
        store.update_research_job(workspace.research_session_id, "request_busy", status=busy)
    elif busy == "lease":
        store.acquire(workspace.research_session_id, "test-owner")
    else:
        run = store.create_run("run_busy", ValuationRequest(company=CompanyInput(name="Issuer"), valuation_date=date(2026, 9, 30)))
        workspace.active_run_id = run.run_id
        store.save_workspace(workspace)
    before = store.get_research(workspace.research_session_id).model_dump_json()
    with TestClient(app) as client:
        assert delete(client, workspace).status_code == 409
    assert store.get_research(workspace.research_session_id).model_dump_json() == before
    assert store.get_workspace(workspace.workspace_id)


def test_delete_requires_current_revision_and_cannot_recreate_deleted_state(tmp_path):
    app = create_app(tmp_path)
    service, store = app.state.workspaces, app.state.store
    workspace = service.create()
    previous = workspace.revision
    store.save_workspace(workspace)
    with TestClient(app) as client:
        assert client.delete(f"/api/workspaces/{workspace.workspace_id}").status_code == 422
        assert delete(client, workspace, previous).status_code == 409
        assert delete(client, workspace).status_code == 204
    with pytest.raises(KeyError):
        store.reserve_research_job(workspace.research_session_id, "late-request", {})
    with pytest.raises(KeyError):
        store.save_workspace_record(workspace.workspace_id, "action", AgentAction(action_id="late-action",
            workspace_id=workspace.workspace_id, actor="user", action_type="test", status="completed", summary="late update"))


def test_reserve_and_delete_race_has_only_one_winner_across_connections(tmp_path):
    app = create_app(tmp_path)
    workspace = app.state.workspaces.create()
    first, second = SQLiteRunStore(tmp_path), SQLiteRunStore(tmp_path)
    gate = Barrier(2)
    def reserve():
        gate.wait()
        try:
            return first.reserve_research_job(workspace.research_session_id, "racing-request", {})
        except KeyError:
            return False
    def remove():
        gate.wait()
        try:
            second.delete_workspace(workspace.workspace_id, expected_revision=workspace.revision)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as executor:
        reserved, removed = executor.submit(reserve), executor.submit(remove)
        assert reserved.result() != removed.result()
    assert bool(first.research_job(workspace.research_session_id)) == (workspace.workspace_id in {item.workspace_id for item in first.list_workspaces()})


def test_shared_calculation_prevents_destructive_cascade(tmp_path):
    app = create_app(tmp_path)
    target, other = app.state.workspaces.create(), app.state.workspaces.create()
    for workspace in (target, other):
        workspace.active_run_id = "run_shared"
        app.state.store.save_workspace(workspace)
    with TestClient(app) as client:
        assert delete(client, target).status_code == 409
    assert len(app.state.store.list_workspaces()) == 2
