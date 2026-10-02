"""Synthetic controller integration: risk omissions cannot enter an EV model.

Core financial facts below are pre-reviewed test contracts, not live findings
or an automatic approval workflow.  Source rows exercise the real SQLite block
loader used by ResearchService, without a network or a model.
"""
from datetime import date

import pytest

from valuationagent.application.research import ResearchService
from valuationagent.application.research_valuation import ResearchValuationAssembler
from valuationagent.application.result_document import build_result_document, document_sections
from valuationagent.schemas.research import DocumentSummary, FactCandidate, ResearchDraft
from valuationagent.storage.sqlite import SQLiteRunStore


BASE = {
    "revenue": "1000", "ebit_margin": "0.2", "tax_rate": "0.25",
    "depreciation_amortization": "10", "capital_expenditure": "15",
    "change_operating_nwc": "5", "cash_and_non_operating_assets": "100",
    "interest_bearing_debt": "40", "common_shares": "100",
    "net_income_parent": "140", "ebitda": "210",
}


def configured(tmp_path, *, methods=("dcf",), risk_row="少数股东权益 10.00 9.00"):
    service = ResearchService(SQLiteRunStore(tmp_path))
    session = service.create()
    session.draft = ResearchDraft(
        company="合成风险回归公司", ticker="600123", industry="电子",
        valuation_date=date(2025, 6, 30), methods=list(methods),
    )
    session.data_source_preference = "upload"
    text = "合成风险回归公司600123\n2024年合并资产负债表\n单位：元\n" + risk_row
    meta = service.store.save_upload("合成公司2024年年度报告.txt", "historical_financials", "text/plain", text.encode())
    fid = meta["file_id"]
    blocks = [{"block_id": fid + ":1", "file_id": fid, "text": text, "location": {"page": 6}}]
    service.store.save_research_blocks(session.session_id, fid, blocks)
    session.documents = [DocumentSummary(
        file_id=fid, name=meta["original_name"], role="historical_financials",
        block_count=1, sha256=meta["sha256"], size_bytes=meta["size_bytes"],
    )]
    for year in range(2021, 2025):
        for metric, value in BASE.items():
            unit = "ratio" if metric in {"ebit_margin", "tax_rate"} else "股" if metric == "common_shares" else "元"
            session.facts.append(FactCandidate(
                fact_id=f"synthetic_{year}_{metric}", metric=metric, raw_value=value,
                normalized_value=value, unit=unit, period=str(year), scope="consolidated",
                block_id=fid + ":1", quote=f"合成测试已复核输入 {metric} {value}", status="confirmed",
            ))
    service.store.save_research(session)
    return service, session, blocks


def add_peers(session):
    for index, value in enumerate(("10", "12", "14"), 1):
        session.facts.append(FactCandidate(
            fact_id=f"synthetic_peer_{index}", metric="pe", raw_value=value,
            normalized_value=value, unit="ratio", period="2025-06-30", scope="unknown",
            role="comparable", peer_ticker=f"TEST{index}", peer_name=f"合成同业{index}",
            multiple_basis="FY", denominator_period_end=date(2024, 12, 31), block_id="peers:1", quote=f"合成 FY2024 倍数 {value}", status="confirmed",
        ))


def test_all_eleven_fields_do_not_override_omitted_risk_in_loaded_source(tmp_path):
    service, session, _ = configured(tmp_path)
    assert len({fact.metric for fact in session.facts}) == 11
    # The amount-only adapter cannot see omitted disclosures.  The service's
    # actual handoff has a source loader and must reject that incomplete view.
    assert ResearchValuationAssembler().build(session).financials.period_end == date(2024, 12, 31)
    with pytest.raises(ValueError, match="原文存在尚未提取核验.*少数股东权益"):
        service.valuation_assembler.build(session)
    assert session.valuation_run_id is None


@pytest.mark.parametrize("method", ["dcf", "ev_ebitda"])
def test_real_service_wiring_loads_source_blocks_from_sqlite(tmp_path, method):
    service, session, blocks = configured(tmp_path, methods=(method,))
    stored_session = service.store.get_research(session.session_id)
    inventory = service.valuation_assembler.source_risks(stored_session, date(2024, 12, 31))
    assert inventory["unresolved"][0]["block_id"] == blocks[0]["block_id"]
    assert inventory["unresolved"][0]["source_sha256"] == stored_session.documents[0].sha256
    assert inventory["scanned_block_count"] == 1
    with pytest.raises(ValueError, match="不能把漏提取当零"):
        service.valuation_assembler.build(stored_session)


def test_independent_pe_handoff_is_not_blocked_by_enterprise_bridge_omission(tmp_path):
    service, session, _ = configured(tmp_path, methods=("pe",))
    add_peers(session)
    assert service.valuation_assembler.source_risks(session, date(2024, 12, 31))["unresolved"]
    request = service.valuation_assembler.build(session)
    assert request.methods == ["pe"] and len(request.peers) == 3
    assert str(request.financials.net_income_parent) == "140"


def test_reviewed_pe_subset_does_not_reenter_rejected_enterprise_method(tmp_path):
    service, session, _ = configured(tmp_path, methods=("dcf", "pe"))
    add_peers(session)
    session.valuation_methods_override = ["pe"]
    session.valuation_method_exclusions = {"dcf": "合成测试：用户确认暂不使用存在桥接缺口的 DCF。"}
    request = service.valuation_assembler.build(session)
    assert request.methods == ["pe"] and request.requested_methods == ["dcf", "pe"]
    assert "dcf" in request.excluded_methods


def test_source_rows_are_reloaded_not_stale_cached_between_handoffs(tmp_path):
    service, session, blocks = configured(tmp_path, risk_row="营业收入 1000.00 900.00")
    assert service.valuation_assembler.build(session).financials is not None
    blocks[0]["text"] += "\n少数股东权益 10.00 9.00"
    service.store.save_research_blocks(session.session_id, session.documents[0].file_id, blocks)
    with pytest.raises(ValueError, match="原文存在尚未提取核验"):
        service.valuation_assembler.build(session)


def test_diagnostic_uses_candidate_year_without_promoting_financial_baseline(tmp_path):
    service, session, _ = configured(tmp_path)
    candidate = next(fact for fact in session.facts if fact.metric == "revenue" and fact.period == "2024")
    candidate.status = "proposed"
    session.facts = [candidate]
    document = build_result_document(service, session)
    inventory = document["source_risk_review"]
    assert inventory["period_end"] == "2024-12-31"
    assert "最新候选年度" in inventory["baseline_selection"]
    assert "不表示已确认财务基期" in inventory["baseline_selection"]
    assert document["counts"]["verified"] == 0 and document["counts"]["staged"] == 1
    assert not document["verified_facts"] and not document["numeric_result_available"]
    assert document["valuation_result"] is None and document["status"] == "insufficient_data"
    assert session.facts[0].status == "proposed"
    sections = dict(document_sections(document))
    rendered_review = str(sections["原文风险科目检查"])
    assert "不表示已确认财务基期" in rendered_review and "未加载" in rendered_review


def test_rejected_or_future_candidate_does_not_change_diagnostic_year(tmp_path):
    service, session, _ = configured(tmp_path)
    candidate = next(fact for fact in session.facts if fact.metric == "revenue" and fact.period == "2024")
    candidate.status = "proposed"
    session.facts = [candidate, candidate.model_copy(update={"fact_id": "future", "period": "2025"}),
                     candidate.model_copy(update={"fact_id": "rejected", "period": "2026", "status": "rejected"})]
    document = build_result_document(service, session)
    assert document["source_risk_review"]["period_end"] == "2024-12-31"
    assert not document["numeric_result_available"] and not document["verified_facts"]
