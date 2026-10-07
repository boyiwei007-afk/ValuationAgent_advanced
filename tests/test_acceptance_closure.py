"""Acceptance boundaries: terminal reports, lost attachments and run updates.

Fixtures are synthetic; these tests do not prove real-company accuracy.
"""
import json

import pytest

from test_finance_team_model import request
from test_production_outcomes import empty_session
from valuationagent.application.research_export import build_research_export
from valuationagent.application.result_document import ensure_result_document
from valuationagent.application.runner import ValuationRunner
from valuationagent.finance.team_model import FinanceTeamModel
from valuationagent.schemas.research import ResearchTurn


@pytest.mark.parametrize("missing_first", [True, False])
def test_unknown_attachment_does_not_discard_valid_files_or_outcome(tmp_path, missing_first):
    from test_research_sessions import ScriptedModel

    service, session = empty_session(tmp_path)
    service.attach(session.session_id, ScriptedModel([
        ("finish_response", {"answer": "有效附件已读取，但尚未形成可计算输入。", "outcome": "insufficient_data"})]))
    meta = service.store.save_upload(
        "可读资料.txt", "historical_financials", "text/plain",
        "合成验收公司 合并报表\n单位：元\n项目 2025年 2024年\n营业收入 100 90".encode(),
    )
    ids = ["file_expired_fixture", meta["file_id"]]
    if not missing_first:
        ids.reverse()
    state = service.turn(session.session_id, ResearchTurn(content="开始自动化 DCF 估值", file_ids=ids))
    assert any(doc["file_id"] == meta["file_id"] and doc["parse_status"] == "parsed"
               for doc in state["session"]["documents"])
    assert state["result_document"]["status"] == "insufficient_data"
    assert not state["result_document"]["numeric_result_available"]
    # The loss must be disclosed instead of silently treating a missing file
    # as empty financial evidence. An event or explicit issue both qualify.
    assert "file_expired_fixture" in json.dumps(state, ensure_ascii=False)
    assert not state["execution"]["active"]


def test_replayed_request_does_not_regenerate_terminal_report(tmp_path):
    service, session = empty_session(tmp_path)
    turn = ResearchTurn(content="开始自动化 DCF 估值", request_id="request_closure_fixture")
    original = service.turn(session.session_id, turn)
    repeated = service.turn(session.session_id, turn)
    assert repeated["result_document"]["report_id"] == original["result_document"]["report_id"]
    assert len([event for event in service.store.list_events(session.session_id)
                if event.type == "report.generated"]) == 1
    assert len([message for message in repeated["messages"] if message["role"] == "user"]) == 1


def test_report_refreshes_after_async_calculation_without_new_chat_turn(tmp_path):
    service, session = empty_session(tmp_path)
    runner = ValuationRunner(service.store, FinanceTeamModel())
    record = runner.create_run(request())
    session.valuation_run_id = record.run_id
    service.store.save_research(session)
    pending = ensure_result_document(service, session)
    assert pending["status"] == "calculating" and not pending["numeric_result_available"]
    completed = runner.execute(record.run_id)
    assert completed.result is not None
    # No save_research/turn call occurs between these exports. The linked run
    # timestamp must invalidate the pre-calculation report.
    body, _ = build_research_export(service, session.session_id)
    final = json.loads(body)["result_document"]
    assert final["status"] == "valued" and final["numeric_result_available"]
    assert final["source_revision"] == pending["source_revision"]
    assert final["report_id"] != pending["report_id"]
    assert service.store.research_report(session.session_id, report_id=pending["report_id"]) == pending


def test_corrupted_cached_report_is_not_silently_served(tmp_path):
    service, session = empty_session(tmp_path)
    report = ensure_result_document(service, session)
    corrupt = {**report, "numeric_result_available": True, "conclusion": "tampered fixture"}
    with service.store._connect() as db:
        db.execute("UPDATE research_reports SET report_json=? WHERE session_id=? AND report_id=?",
                   (json.dumps(corrupt), session.session_id, report["report_id"]))
    with pytest.raises(ValueError, match="完整性校验失败"):
        build_research_export(service, session.session_id)


@pytest.mark.parametrize("storage_failure", [False, True])
def test_automatic_ready_task_cannot_end_with_unexecuted_next_steps(tmp_path, monkeypatch, storage_failure):
    from test_input_workspace import fixture, values
    from valuationagent.application.agent_runtime import AgentResponse, CalculateValuation
    from valuationagent.application.record_inputs import record_inputs
    from valuationagent.application.workspace_artifacts import ReportWrite, write_report

    _, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    with pytest.raises(ValueError, match="VALUATION_READY_NOT_EXECUTED"):
        runtime.finish(AgentResponse(answer="数据已经齐备，下一步我将计算。"))
    assert not runtime.session.valuation_run_id
    if storage_failure:
        def unavailable(*args):
            raise OSError("synthetic report storage failure")

        monkeypatch.setattr("valuationagent.application.agent_runtime.write_report", unavailable)
    result = runtime.calculate(CalculateValuation())
    if storage_failure:
        with pytest.raises(ValueError, match="REPORT_NOT_DELIVERED"):
            runtime.finish(AgentResponse(answer="估值已经完成。"))
        write_report(runtime.service, runtime.session, ReportWrite(format="md"))
    else:
        metadata, content = runtime.service.store.get_artifact(runtime.session.session_id, result["report_delivery"]["artifact_id"])
        assert metadata["numeric_result_available"] and metadata["valuation_run_id"] == result["run_id"]
        assert "40.00" in content.decode("utf-8")
    assert runtime.finish(AgentResponse(answer="计算及数值报告已经完成。"))["_terminal"]


def test_completion_guard_does_not_force_valuation_for_discussion_or_review(tmp_path):
    from test_input_workspace import fixture, values
    from valuationagent.application.agent_runtime import AgentResponse
    from valuationagent.application.record_inputs import record_inputs
    from valuationagent.application.turn_control import resolve_control
    from valuationagent.schemas.control import TurnDecision

    app, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    workspace = app.state.store.workspace_for_research(runtime.session.session_id)
    workspace.run_policy = "review"
    app.state.store.save_workspace(workspace)
    assert runtime.finish(AgentResponse(answer="输入准备完成，等待审阅批准。", outcome="needs_input"))["_terminal"]
    workspace.run_policy = "automatic"
    app.state.store.save_workspace(workspace)
    message = app.state.store.add_message(runtime.session.session_id, "user", "先只讨论，不要计算。", "agent")
    runtime.session.turn_control, runtime.session.execution_permissions = resolve_control(runtime.session, message,
        TurnDecision(summary="仅讨论", actions=["discuss"]))
    assert runtime.finish(AgentResponse(answer="只讨论方法，不运行计算。"))["_terminal"]
    assert not runtime.session.valuation_run_id


def test_file_extraction_cannot_claim_saved_data_with_only_prose(tmp_path):
    from test_input_workspace import fixture
    from test_multisource_extraction import attach
    from valuationagent.application.agent_runtime import AgentResponse
    from valuationagent.application.turn_control import resolve_control
    from valuationagent.schemas.control import TurnDecision

    app, runtime = fixture(tmp_path)
    attach(runtime, "report.txt", "营业收入100元")
    message = app.state.store.add_message(runtime.session.session_id, "user", "提取文件数据，不估值", "agent")
    runtime.session.turn_control, runtime.session.execution_permissions = resolve_control(runtime.session, message,
        TurnDecision(summary="提取数据", actions=["ingest"]))
    with pytest.raises(ValueError, match="INGESTION_NOT_SAVED"):
        runtime.finish(AgentResponse(answer="已提取收入100元。"))
    assert runtime.finish(AgentResponse(answer="文件没有提供年度，尚不能保存完整年度数据。", outcome="insufficient_data"))["_terminal"]
    runtime.session.turn_control.decision.actions = ["read"]
    assert runtime.finish(AgentResponse(answer="原文只写营业收入100元，没有年度。"))["_terminal"]
