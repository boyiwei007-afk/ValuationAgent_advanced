from valuationagent.schemas.research import DocumentSummary, ResearchTurn
from test_evidence_recovery import configured_service
from test_research_sessions import ScriptedModel


def test_loaded_pdf_page_window_is_not_reparsed_or_marked_partial(tmp_path, monkeypatch):
    service, session, _ = configured_service(tmp_path)
    meta = service.store.save_upload("fixture.pdf", "historical_financials", "application/pdf", b"synthetic cache fixture")
    file_id = meta["file_id"]
    service.store.save_research_blocks(session.session_id, file_id, [
        {"file_id": file_id, "block_id": file_id + ":1", "text": "合成公司 已读取原文", "location": {"page": 2}},
    ])
    session.documents.append(DocumentSummary(file_id=file_id, name="fixture.pdf", role="historical_financials",
                                              block_count=1, sha256=meta["sha256"], parse_status="parsed"))
    service.store.save_research(session)

    def no_reparse(*args, **kwargs):
        raise AssertionError("already loaded PDF must not be extracted again")

    monkeypatch.setattr("valuationagent.application.research.parse_document", no_reparse)
    service._clients[session.session_id] = ScriptedModel([
        ("read_document", {"file_id": file_id, "start_page": 1, "limit": 8}),
        ("finish_response", {"answer": "缺失估值必要数据", "outcome": "insufficient_data"}),
    ])
    state = service.turn(session.session_id, ResearchTurn(content="开始自动化DCF估值"))
    doc = next(d for d in state["session"]["documents"] if d["file_id"] == file_id)
    assert not doc["warnings"] and doc["parse_status"] == "parsed"
    reads = [e for e in service.store.list_events(session.session_id) if e.type == "tool.completed" and e.tool == "read_document"]
    assert len(reads) == 1
    assert reads[0].payload["output"]["blocks"][0]["location"]["page"] == 2
