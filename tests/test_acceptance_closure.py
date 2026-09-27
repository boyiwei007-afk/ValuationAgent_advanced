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
    service, session = empty_session(tmp_path)
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
