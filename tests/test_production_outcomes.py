"""Failure-path deliverables and independent financial-output gate."""
import json
from datetime import date
from decimal import Decimal as D
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from test_evidence_recovery import RepeatingRetrievalModel, configured_service
from test_finance_team_model import request
from valuationagent.api.main import create_app
from valuationagent.application.research import ResearchService
from valuationagent.application.research_export import build_research_export
from valuationagent.application.result_document import ensure_result_document
from valuationagent.application.runner import ValuationRunner
from valuationagent.finance.integrity import validate_peer_inputs
from valuationagent.finance.team_model import FinanceTeamModel
from valuationagent.schemas.models import PeerCompany
from valuationagent.schemas.research import ResearchDraft, ResearchTurn
from valuationagent.search.providers import MockSearchProvider
from valuationagent.storage.sqlite import SQLiteRunStore


def empty_session(tmp_path):
    service = ResearchService(SQLiteRunStore(tmp_path))
    session = service.create(data_source_preference="web")
    session.draft = ResearchDraft(company="零资料验收样本（合成）", industry="电子", methods=["dcf", "pe"], valuation_date=date(2026, 9, 26))
    service.store.save_research(session)
    return service, session


def test_zero_data_still_delivers_report_and_download_is_model_independent(tmp_path):
    service, session = empty_session(tmp_path)
    state = service.turn(session.session_id, ResearchTurn(content="开始自动化估值"))
    assert state["result_document"]["status"] == "insufficient_data"
    body, _ = build_research_export(service, session.session_id)
    report = json.loads(body)["result_document"]
    assert report["valuation_result"] is None
    assert not report["numeric_result_available"]
    assert not report["verified_facts"] and not report["searches"]
    assert len(report["methods"]) == 2 and report["gaps"]
    html, _ = build_research_export(service, session.session_id, "html")
    assert "未计算" in html and "方法适用性" in html and "尚无已记录的网络检索" in html
    pdf, media = build_research_export(service, session.session_id, "pdf")
    assert media == "application/pdf"
    text = "".join(p.extract_text() for p in PdfReader(BytesIO(pdf)).pages)
    assert "估值结果报告" in text and "未计算" in text
    assert not service.store.list_runs()
    fresh = ResearchService(SQLiteRunStore(tmp_path))
    again = ensure_result_document(fresh, fresh.store.get_research(session.session_id))
    assert again["report_id"] == report["report_id"]
    assert len([e for e in fresh.store.list_events(session.session_id) if e.type == "report.generated"]) == 1


def test_exhausted_target_survives_restart_without_closing_whole_task(tmp_path):
    model = RepeatingRetrievalModel()
    service, session, _ = configured_service(tmp_path, model)
    session.draft.ticker = ""
    session.pending_action = "valuation"
    service.search_provider = MockSearchProvider()
    service.store.save_research(session)
    first = service.turn(session.session_id, ResearchTurn(content="开始估值"))
    assert model.calls == 7 and len(first["session"]["search_history"]) == 3
    assert first["result_document"] and "question" not in first["session"]
    restarted = ResearchService(SQLiteRunStore(tmp_path), search_provider=MockSearchProvider())
    restarted.attach(session.session_id, model)
    again = restarted.turn(session.session_id, ResearchTurn(content="继续"))
    assert model.calls == 11
    assert len(again["session"]["search_history"]) == 3
    assert not again["session"]["outcome_status"]
    assert again["session"]["resume_context"]["reason"] == "AGENT_NO_PROGRESS"
    assert again["result_document"]["status"] == "insufficient_data"


def test_report_api_accepts_no_upload_no_model_and_escapes_untrusted_text(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        created = client.post("/api/workspaces", json={"data_source_preference": "web"}).json()
        workspace = created["workspace"]
        session = app.state.store.get_research(workspace["research_session_id"])
        session.draft.company = "<script>alert(1)</script>"
        app.state.store.save_research(session)
        url = f"/api/workspaces/{workspace['workspace_id']}"
        response = client.get(url + "/export?format=html")
        assert response.status_code == 200 and "<script>" not in response.text
        assert "&lt;script&gt;" in response.text
        pdf = client.get(url + "/export?format=pdf")
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
        assert "attachment" in pdf.headers["content-disposition"]
        assert client.get(url + "/export?format=exe").status_code == 422
        assert client.get(url).json()["research"]["result_document"]


def test_output_gate_blocks_corrupted_per_share_even_if_plugin_reports_success(tmp_path):
    class BrokenShareModel(FinanceTeamModel):
        def dcf(self, *args, **kwargs):
            result = super().dcf(*args, **kwargs)
            return result.model_copy(update={"per_share_value": result.per_share_value + D("3")})
    record = ValuationRunner(SQLiteRunStore(tmp_path), BrokenShareModel()).run(request())
    assert record.status == "waiting_review" and record.result is None
    assert "PER_SHARE" in record.review["message"]


def test_independent_gate_preserves_valid_output_and_reports_checks(tmp_path):
    record = ValuationRunner(SQLiteRunStore(tmp_path), FinanceTeamModel()).run(request())
    assert record.result, record.review
    assert {"DCF_PRESENT_VALUE", "EQUITY_BRIDGE", "PER_SHARE", "DCF_RANGE"} <= set(record.result.calculation_checks)


@pytest.mark.parametrize("changes,code", [({"as_of_date": date(2026, 9, 24)}, "PEER_PRICING_DATE"), ({"multiple_basis": "TTM"}, "PEER_DENOMINATOR_BASIS")])
def test_peer_date_and_denominator_mismatch_are_blocked(changes, code):
    req = request().model_copy(update={"methods": ["pe"]})
    peer = PeerCompany(ticker="TEST1", name="合成同业", pe="12", **changes)
    assert any(f.rule_id == code for f in validate_peer_inputs(req, [peer]))


def test_peer_fiscal_period_and_mixed_pricing_dates_are_blocked():
    req = request().model_copy(update={"methods": ["pe"]})
    peers = [
        PeerCompany(
            ticker="TEST1", name="同业一", pe="12", multiple_basis="FY",
            as_of_date=date(2026, 9, 20),
            financial_period_end=date(2024, 12, 31),
        ),
        PeerCompany(
            ticker="TEST2", name="同业二", pe="13", multiple_basis="FY",
            as_of_date=date(2026, 9, 21),
            financial_period_end=req.financials.period_end,
        ),
    ]
    rules = {finding.rule_id for finding in validate_peer_inputs(req, peers)}
    assert {"PEER_FISCAL_PERIOD", "PEER_MIXED_PRICING_DATES"} <= rules


def test_duplicate_peers_are_not_counted_as_independent_companies():
    req = request().model_copy(update={"methods": ["pe"]})
    peers = [PeerCompany(ticker=t, name="合成同业", pe="12") for t in ["600123", "600123.SH", "600125"]]
    assert any(f.rule_id == "PEER_DUPLICATE" for f in validate_peer_inputs(req, peers))


def test_unfinished_formal_run_also_has_pdf_and_explicitly_nonreplayable_json(tmp_path):
    app = create_app(tmp_path)
    record = ValuationRunner(app.state.store, FinanceTeamModel()).create_run(request())
    with TestClient(app) as client:
        response = client.get(f"/api/runs/{record.run_id}/export?format=pdf")
        assert response.status_code == 200
        text = "".join(page.extract_text() for page in PdfReader(BytesIO(response.content)).pages)
        assert "执行诊断" in text and "未计算" in text
        diagnostic = client.get(f"/api/runs/{record.run_id}/export?format=json").json()
        assert diagnostic["schema"] == "valuation-diagnostic-v1"
        assert not diagnostic["numeric_result_available"] and diagnostic["valuation_result"] is None
        assert diagnostic["input_hash"] == record.input_hash
        assert client.get(f"/api/runs/{record.run_id}/export?format=xlsx").status_code == 409


def test_model_can_deliver_missing_data_outcome_without_another_confirmation(tmp_path):
    from test_research_sessions import ScriptedModel
    answer = "缺项尚未取得，不输出价格。"
    model = ScriptedModel([("finish_response", {"answer": answer, "outcome": "insufficient_data"})])
    service, session, _ = configured_service(tmp_path, model)
    state = service.turn(session.session_id, ResearchTurn(content="开始估值"))
    assert state["messages"][-1]["content"] == answer
    assert state["session"]["outcome_status"] == "insufficient_data"
    assert state["result_document"]["status"] == "insufficient_data"
    assert not service.store.list_runs()


def test_repeated_unchanged_reads_stop_early_and_preserve_non_numeric_report(tmp_path):
    class EndlessInspections:
        calls = 0
        def chat(self, messages, **kwargs):
            self.calls += 1
            return {"tool_calls": [{"id": str(self.calls), "type": "function",
                "function": {"name": "inspect_context", "arguments": "{}"}}]}
    model = EndlessInspections()
    service, session, _ = configured_service(tmp_path, model)
    state = service.turn(session.session_id, ResearchTurn(content="检查资料"))
    assert model.calls == 4
    assert state["session"]["last_issue"]["code"] == "AGENT_NO_PROGRESS"
    assert state["session"]["resume_context"]
    assert any(event.type == "tool.failed" and event.tool == "inspect_context"
               for event in service.store.list_events(session.session_id))
    assert state["execution"]["status"] == "failed"
    assert not state["result_document"]["numeric_result_available"]
    assert not service.store.list_runs()


def test_unsupported_missing_shares_paths_end_in_report_not_fake_approval(tmp_path):
    from test_research_sessions import ScriptedModel
    # The live-model failure shape, using only synthetic names and numbers.
    invalid = {"answer": "普通股股数未通过来源校验。",
               "question": "普通股股数没有以股为单位的原文。请选择如何处理。",
               "options": ["剔除依赖股数的口径，按现有基期与十年预测提交整套估值方案",
                           "采信普通股股数为123,456,789股（面值1元/股）",
                           "改从其他正式公告检索普通股股份总数"]}
    model = ScriptedModel([("finish_response", invalid), ("finish_response", invalid)])
    service, session, _ = configured_service(tmp_path, model)
    state = service.turn(session.session_id, ResearchTurn(content="开始自动化估值"))
    assert "question" not in state["session"]
    assert state["session"]["valuation_run_id"] is None
    assert not state["result_document"]["numeric_result_available"]
    assert "采信普通股股数" not in state["messages"][-1]["content"]
    assert state["session"]["last_issue"]["code"] == "AGENT_NO_PROGRESS"
    assert build_research_export(service, session.session_id, "pdf")[0].startswith(b"%PDF")
