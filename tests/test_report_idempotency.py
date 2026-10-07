import copy
import json

import pytest

from test_input_workspace import fixture, values
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.result_document import ensure_result_document
from valuationagent.application.workspace_artifacts import ReportWrite, write_report
from valuationagent.llm.client import LlmError


def completed(tmp_path):
    app, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["report_delivery"]["status"] == "saved"
    return app, runtime, outcome


@pytest.mark.parametrize("format_name", ["md", "html", "pdf"])
def test_report_reuses_same_frozen_content_after_bookkeeping_revision(tmp_path, format_name):
    app, runtime, outcome = completed(tmp_path)
    first = write_report(app.state.research, runtime.session, ReportWrite(format=format_name))
    first_metadata, first_content = app.state.store.get_artifact(runtime.session.session_id, first["artifact_id"])
    old_revision = runtime.session.revision
    app.state.store.save_research(runtime.session)
    assert runtime.session.revision > old_revision
    second = write_report(app.state.research, runtime.session, ReportWrite(format=format_name))
    assert second == first
    assert second["valuation_run_id"] == outcome["run_id"]
    assert second["source_revision"] == first_metadata["source_revision"] < runtime.session.revision
    assert app.state.store.get_artifact(runtime.session.session_id, second["artifact_id"]) == (first_metadata, first_content)
    reports = [row for row in app.state.store.list_artifacts(runtime.session.session_id) if row["filename"] == first["filename"]]
    assert len(reports) == 1


@pytest.mark.parametrize("change", ["narrative", "schema", "input"])
def test_changed_report_content_is_not_hidden_by_same_run_cache(tmp_path, monkeypatch, change):
    app, runtime, outcome = completed(tmp_path)
    first = outcome["report_delivery"]
    app.state.store.save_research(runtime.session)
    document = copy.deepcopy(ensure_result_document(app.state.research, runtime.session))
    if change == "narrative":
        document["research_summary"] = "更新后的研究说明，保留新增风险判断。"
    elif change == "schema":
        document["schema"] = "test-new-report-schema"
    else:
        document["input_records"][0]["value"] = "987654321"
    monkeypatch.setattr("valuationagent.application.workspace_artifacts.ensure_result_document", lambda *args: document)
    second = write_report(app.state.research, runtime.session, ReportWrite(format="md"))
    assert second["artifact_id"] != first["artifact_id"]
    assert second["valuation_run_id"] == first["valuation_run_id"]


def test_json_audit_changes_create_new_export_without_rewriting_previous(tmp_path):
    app, runtime, _ = completed(tmp_path)
    first = write_report(app.state.research, runtime.session, ReportWrite(format="json"))
    first_content = app.state.store.get_artifact(runtime.session.session_id, first["artifact_id"])[1]
    assert write_report(app.state.research, runtime.session, ReportWrite(format="json"))["artifact_id"] == first["artifact_id"]
    app.state.store.append_event(runtime.session.session_id, type="test.audit_added", stage="reporting", status="completed", summary="新增审计记录")
    second = write_report(app.state.research, runtime.session, ReportWrite(format="json"))
    assert second["artifact_id"] != first["artifact_id"]
    assert app.state.store.get_artifact(runtime.session.session_id, first["artifact_id"])[1] == first_content
    content = app.state.store.get_artifact(runtime.session.session_id, second["artifact_id"])[1]
    assert "test.audit_added" in {event["type"] for event in json.loads(content)["events"]}


@pytest.mark.parametrize("failure", ["corrupt", "missing"])
def test_matching_report_cannot_be_reused_if_storage_verification_fails(tmp_path, monkeypatch, failure):
    app, runtime, outcome = completed(tmp_path)
    app.state.store.save_research(runtime.session)
    expected = "ARTIFACT_INTEGRITY" if failure == "corrupt" else "ARTIFACT_NOT_FOUND"
    original = app.state.store.get_artifact

    def fail(session_id, artifact_id):
        if artifact_id == outcome["report_delivery"]["artifact_id"]:
            raise ValueError(expected)
        return original(session_id, artifact_id)

    monkeypatch.setattr(app.state.store, "get_artifact", fail)
    with pytest.raises(ValueError, match=expected):
        write_report(app.state.research, runtime.session, ReportWrite(format="md"))
    assert len(app.state.store.list_artifacts(runtime.session.session_id)) == 1


def test_cancel_after_document_build_does_not_deliver_cached_report(tmp_path, monkeypatch):
    app, runtime, _ = completed(tmp_path)
    checks = []

    def cancel():
        checks.append(True)
        if len(checks) > 1:
            raise LlmError("EXECUTION_CANCELLED")

    monkeypatch.setattr(app.state.research, "_check_execution", cancel)
    with pytest.raises(LlmError, match="EXECUTION_CANCELLED"):
        write_report(app.state.research, runtime.session, ReportWrite(format="md"))
    assert len(app.state.store.list_artifacts(runtime.session.session_id)) == 1


def test_report_formats_have_separate_verified_artifacts(tmp_path):
    app, runtime, _ = completed(tmp_path)
    reports = [write_report(app.state.research, runtime.session, ReportWrite(format=format_name))
        for format_name in ["md", "html", "pdf", "json"]]
    assert len({report["artifact_id"] for report in reports}) == 4
    for report in reports:
        assert app.state.store.get_artifact(runtime.session.session_id, report["artifact_id"])[0]["sha256"] == report["sha256"]
