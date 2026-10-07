import json
from datetime import date
from decimal import Decimal

import pytest

from valuationagent.application.agent_runtime import WorkspaceAgentRuntime
from valuationagent.application.ledgers import sync_evidence_and_fact_ledgers
from valuationagent.application.financial_evidence import (
    EvidenceRequest, FinancialBatch, FactSelection, read_evidence, propose_batch,
    corroborate, reject_candidates, PUBLIC_REVIEW,
)
from valuationagent.application.research import CandidateInput, ProposeFacts, ResearchService, SearchSources
from valuationagent.core.documents import parse_document
from valuationagent.llm.client import LlmError
from valuationagent.market.research_data import FinancialHistoryRequest, fetch_history
from valuationagent.market.tushare import TushareDataProvider
from valuationagent.schemas.research import DocumentSummary
from valuationagent.schemas.workspace import EvidenceRecord, ValuationWorkspace
from valuationagent.storage.sqlite import SQLiteRunStore
from test_agent_recovery import EmptySearch


def runtime_at(tmp_path, company="样本科技股份有限公司", ticker="600123"):
    from control_fixtures import allow_tool_testing

    service = ResearchService(SQLiteRunStore(tmp_path), search_provider=EmptySearch())
    session = service.create(data_source_preference="web")
    session.draft.company = company
    session.draft.ticker = ticker
    session.draft.methods = ["pe", "ps"]
    session.draft.valuation_date = date(2026, 9, 30)
    allow_tool_testing(session, service.store)
    return WorkspaceAgentRuntime(service, session)


def attach(runtime, name, body, public=False, domain="one.example.test", published="2026-04-20"):
    meta = runtime.service.store.save_upload(name, "evidence", "text/plain", body.encode())
    blocks, warnings = parse_document(runtime.service.store.get_file(meta["file_id"]))
    for block in blocks:
        block["location"].update(source_url=f"https://{domain}/{name}", published_at=published,
                                 source_type="remote_web_document" if public else "remote_document")
    runtime.service.store.save_research_blocks(runtime.session.session_id, meta["file_id"], blocks)
    runtime.session.documents.append(DocumentSummary(file_id=meta["file_id"], name=name, role="evidence",
        block_count=len(blocks), sha256=meta["sha256"], provenance_type="public_web" if public else "official_filing",
        authority_tier="C" if public else "A", source_url=f"https://{domain}/{name}"))
    return blocks


@pytest.mark.parametrize("company,ticker", [("样本科技股份有限公司", "600123"), ("样本食品股份有限公司", "000321"), ("样本设备股份有限公司", "300456")])
@pytest.mark.parametrize("suffix", ["html", "csv"])
def test_batch_extracts_multiple_periods_without_company_specific_rules(tmp_path, company, ticker, suffix):
    runtime = runtime_at(tmp_path, company, ticker)
    if suffix == "html":
        text = f"<h1>{company} {ticker}</h1><h2>合并利润表</h2><p>单位：万元</p><table><tr><th>项目</th><th>2025年度</th><th>2024年度</th></tr>"
        text += "".join(f"<tr><td>其他项目{index}</td><td>1</td><td>2</td></tr>" for index in range(20))
        text += "<tr><td>营业收入</td><td>1,000</td><td>900</td></tr><tr><td>归属于母公司股东的净利润</td><td>100</td><td>80</td></tr></table>"
    else:
        text = f'{company} {ticker}\n合并利润表\n单位：万元\n项目,2025年度,2024年度\n营业收入,"1,000",900\n归属于母公司股东的净利润,100,80'
    blocks = attach(runtime, f"annual.{suffix}", text)
    source = next(block for block in blocks if block.get("location", {}).get("cells", [""])[0] == "营业收入")
    args = FinancialBatch.model_validate({"defaults": {"unit": "万元", "scope": "consolidated"}, "rows": [
        {"block_id": source["block_id"], "metric": "营业收入", "period": "2025", "value_column": 1},
        {"block_id": source["block_id"], "metric": "营业收入", "period": "2024", "value_column": 2},
        {"block_id": source["block_id"], "metric": "营业收入", "period": "2023", "value_column": 2},
    ]})
    packet = read_evidence(runtime, EvidenceRequest(file_ids=[source["file_id"]], query="营业收入"))
    assert packet["packets"][0]["blocks"]
    result = propose_batch(runtime, args)
    clean = [fact for fact in runtime.session.facts if fact.status == "confirmed"]
    assert [(fact.period, Decimal(fact.normalized_value)) for fact in clean] == [("2025", Decimal("10000000")), ("2024", Decimal("9000000"))], result
    assert runtime.session.facts[-1].warnings
    assert clean[0].verification["year_column"] == 2025


def test_batch_error_does_not_lose_other_rows_or_accept_blank_as_zero(tmp_path):
    runtime = runtime_at(tmp_path)
    blocks = attach(runtime, "blank.csv", "样本科技股份有限公司600123\n合并利润表\n单位：元\n项目,2025年度,2024年度\n营业收入,100,90\n营业成本,,50")
    income, cost = blocks[-2:]
    result = propose_batch(runtime, FinancialBatch.model_validate({"defaults": {"unit": "元", "period": "2025", "scope": "consolidated"}, "rows": [
        {"block_id": cost["block_id"], "metric": "营业成本", "value_column": 1},
        {"block_id": income["block_id"], "metric": "营业收入", "value_column": 1},
    ]}))
    assert result["rows"][0].get("error")
    assert len(runtime.session.facts) == 1 and runtime.session.facts[0].status == "confirmed"


def public_facts(runtime, values=("100", "100"), domains=("one.example.test", "two.other.test")):
    facts = []
    for index, (value, domain) in enumerate(zip(values, domains)):
        body = f"样本科技股份有限公司600123 来源{index}\n合并利润表\n单位：元\n项目 2025年度 2024年度\n营业收入 {value} 90"
        blocks = attach(runtime, f"source{index}.txt", body, public=True, domain=domain)
        result = runtime.facts(ProposeFacts(candidates=[CandidateInput(metric="营业收入", raw_value=value, unit="元", period="2025",
            scope="consolidated", block_id=blocks[0]["block_id"], quote=f"营业收入 {value} 90")]))
        facts.append(runtime.session.facts[-1])
        assert PUBLIC_REVIEW in facts[-1].warnings, result
    return facts


def test_corroborated_web_fields_keep_lower_grade_and_exported_limitations(tmp_path):
    runtime = runtime_at(tmp_path)
    facts = public_facts(runtime)
    workspace = ValuationWorkspace(workspace_id="workspace_evidence", research_session_id=runtime.session.session_id)
    runtime.service.store.create_workspace(workspace)
    sync_evidence_and_fact_ledgers(runtime.service.store, workspace, runtime.session)
    runtime.session.outcome_status = "insufficient_data"
    runtime.session.outcome_reason = "等待另一来源"
    assert all(fact.status == "proposed" for fact in facts)
    result = corroborate(runtime, FactSelection(fact_ids=[fact.fact_id for fact in facts], reason="两个下载原文的同主体同年度同口径金额一致，来源仍为第三方。"))
    assert not result["source_grade_upgraded"]
    assert all(fact.status == "confirmed" and not fact.warnings for fact in facts)
    assert all(fact.verification["source_assessment"]["source_tier"] == "C" for fact in facts)
    assert runtime.session.outcome_status == runtime.session.outcome_reason == ""
    sync_evidence_and_fact_ledgers(runtime.service.store, workspace, runtime.session)
    records = runtime.service.store.list_workspace_records(workspace.workspace_id, "evidence", EvidenceRecord)
    assert len(records) == 4
    assert sum(record.status == "candidate" for record in records) == 2
    assert sum(record.status == "verified" and "不证明上游独立" in str(record.limitations) for record in records) == 2
    reference = runtime.service.valuation_assembler._evidence(runtime.session, facts[0])
    assert "corroborated_draft" in reference.note and "不证明上游独立" in reference.note


@pytest.mark.parametrize("failure", ["amount", "period", "domain", "subdomain", "snapshot", "date", "mapping", "label"])
def test_corroboration_cannot_clear_unrelated_failures(tmp_path, failure):
    runtime = runtime_at(tmp_path)
    facts = public_facts(runtime)
    if failure == "amount":
        facts[-1].normalized_value = "999"
    elif failure == "period":
        facts[-1].period = "2024"
    elif failure == "domain":
        facts[-1].source_url = facts[0].source_url + "/copy"
    elif failure == "subdomain":
        facts[-1].source_url = "https://copy.one.example.test/report"
    elif failure == "snapshot":
        facts[-1].source_sha256 = facts[0].source_sha256
    elif failure == "date":
        facts[-1].published_at = date(2027, 1, 1)
    elif failure == "label":
        facts[-1].metric = "营业总收入"
    else:
        facts[-1].warnings.append("语义映射未解决")
    with pytest.raises(ValueError):
        corroborate(runtime, FactSelection(fact_ids=[fact.fact_id for fact in facts], reason="明确记录核对依据并检查所有相关维度"))
    assert all(fact.status == "proposed" for fact in facts)


def test_corroboration_cannot_cherry_pick_around_another_conflicting_candidate(tmp_path):
    runtime = runtime_at(tmp_path)
    facts = public_facts(runtime)
    conflict = facts[0].model_copy(deep=True, update={"fact_id": "fact_conflicting", "normalized_value": "101"})
    runtime.session.facts.append(conflict)
    with pytest.raises(ValueError, match="未解决的同口径冲突"):
        corroborate(runtime, FactSelection(fact_ids=[fact.fact_id for fact in facts], reason="两个下载原文的同主体同年度同口径金额一致，但仍需排查冲突。"))
    assert all(fact.status == "proposed" for fact in facts)


@pytest.mark.parametrize("cells", ["<th colspan='2'>2025年度 2024年度</th>", "<th><table><tr><td>2025年度</td></tr></table></th>"])
def test_ambiguous_html_layout_cannot_be_flattened_into_confirmed_fields(tmp_path, cells):
    runtime = runtime_at(tmp_path)
    blocks = attach(runtime, "merged.html", f"<h1>样本科技600123</h1><h2>合并利润表</h2><p>单位：元</p><table><tr><td>项目</td>{cells}</tr><tr><td>营业收入</td><td>100</td><td>90</td></tr></table>")
    row = blocks[-1]
    assert row["location"]["merged_cells"]
    result = propose_batch(runtime, FinancialBatch.model_validate({"defaults": {"unit": "元", "period": "2025", "scope": "consolidated"},
        "rows": [{"metric": "营业收入", "block_id": row["block_id"], "value_column": 1}]}))
    assert result["rows"][0].get("error")
    runtime.facts(ProposeFacts(candidates=[CandidateInput(metric="营业收入", raw_value="100", unit="元", period="2025", scope="consolidated", block_id=row["block_id"], quote=row["text"])]))
    assert not any(fact.status == "confirmed" for fact in runtime.session.facts)


def test_bulk_download_obeys_upload_only_even_when_search_lead_already_exists(tmp_path, monkeypatch):
    runtime = runtime_at(tmp_path)
    runtime.session.data_source_preference = "upload"
    runtime.session.documents.append(DocumentSummary(file_id="web_fixture", name="Cached lead", role="evidence", block_count=1))
    runtime.service.store.save_research_blocks(runtime.session.session_id, "web_fixture", [
        {"block_id": "web_fixture:1", "file_id": "web_fixture", "text": "lead", "location": {"source_type": "web_search", "url": "https://public.example.test/report"}}])
    monkeypatch.setattr(runtime.service, "_download_public_source", lambda url: pytest.fail("upload-only must not download"))
    result = read_evidence(runtime, EvidenceRequest(file_ids=["web_fixture"], download=True))
    assert "NETWORK_OUT_OF_SCOPE" in result["failures"][0]["error"]


def test_single_llm_tool_loop_selects_and_maps_bulk_evidence(tmp_path):
    from observation_fixtures import fixture_observations, ObservationModel, extraction_steps
    runtime = runtime_at(tmp_path)
    args = fixture_observations(runtime, [{"metric": "revenue", "raw_value": "100"},
                                         {"metric": "net_income_parent", "raw_value": "10"}])
    model = ObservationModel([
        ("read_financial_evidence", {"file_ids": [args["file_id"]], "query": "revenue net_income_parent"}),
        *extraction_steps(args)])
    runtime.run(model)
    assert len(model.calls) == 5
    assert len(runtime.session.facts) == 2
    assert all(fact.status == "confirmed" for fact in runtime.session.facts)
    assert runtime.session.valuation_run_id is None

def test_web_route_does_not_force_another_annual_pdf_search(tmp_path):
    runtime = runtime_at(tmp_path)
    class NoCatalogue:
        def search_reports(self, *args, **kwargs):
            pytest.fail("web route must not use catalogue")
        search_annual_reports = search_reports
    runtime.service.official_search_provider = NoCatalogue()
    result = runtime.search(SearchSources(query="样本科技2025年度报告财务表网页", purpose="financials", reason="换表格来源", source_route="web"))
    assert result["status"] == "no_results"
    assert runtime.session.search_history[-1]["source_route"] == "web"


class RecordsClient:
    def __init__(self):
        self.calls = []

    def query_snapshot(self, statement, **kwargs):
        from valuationagent.market.tushare import TushareSnapshot

        self.calls.append((statement, kwargs))
        baseline = {"ts_code": "600123.SH", "report_type": "1", "end_date": "20251231", "f_ann_date": "20260420", "revenue": "1000"}
        rows = [baseline, {**baseline, "end_date": "20241231", "f_ann_date": "20250420", "revenue": "900"},
                {**baseline, "f_ann_date": "20271010"}, {**baseline, "ts_code": "000999.SZ"},
                {**baseline, "report_type": "6"}, {**baseline, "f_ann_date": None}]
        if statement == "stock_basic":
            baseline = {"ts_code": "600123.SH", "name": "样本科技", "fullname": "样本科技股份有限公司", "industry": "半导体", "list_date": "20100101"}
            rows = [baseline]
        fields = list(baseline)
        raw = json.dumps({"code": 0, "data": {"fields": fields, "items": [[row.get(field) for field in fields] for row in rows]}}).encode()
        return TushareSnapshot(raw=raw, fields=fields, records=rows)


def test_structured_history_is_dated_scoped_cached_evidence_not_automatic_facts(tmp_path):
    runtime = runtime_at(tmp_path)
    client = RecordsClient()
    runtime.service._market_clients[runtime.session.session_id] = TushareDataProvider(client)
    request = FinancialHistoryRequest(years=[2024, 2025], statements=["income"])
    result = fetch_history(runtime, request)
    assert result["documents"][0]["accepted_records"] == 2
    assert result["documents"][0]["excluded_records"] == 4
    assert not runtime.session.facts
    document = next(document for document in runtime.session.documents if ":income:" in document.provider)
    assert document.authority_tier == "B" and document.sha256
    raw = runtime.service.store.get_file(document.file_id)
    assert raw["original_name"].endswith(".json")
    assert fetch_history(runtime, request)["documents"][0]["cached"]
    assert len(client.calls) == 2
    blocks = runtime.service.store.research_blocks(runtime.session.session_id, document.file_id)
    result = propose_batch(runtime, FinancialBatch.model_validate({"defaults": {"unit": "元", "scope": "consolidated"},
        "rows": [{"block_id": block["block_id"], "metric": "营业收入", "raw_value": value, "period": period, "start_line": 5, "end_line": 5}
                 for block, value, period in zip(blocks, ["1000", "900"], ["2025", "2024"])]}))
    assert len(runtime.session.facts) == 2 and all(fact.status == "confirmed" for fact in runtime.session.facts), result
    workspace = ValuationWorkspace(workspace_id="workspace_structured", research_session_id=runtime.session.session_id)
    runtime.service.store.create_workspace(workspace)
    sync_evidence_and_fact_ledgers(runtime.service.store, workspace, runtime.session)
    records = runtime.service.store.list_workspace_records(workspace.workspace_id, "evidence", EvidenceRecord)
    assert len(records) == 2 and all(record.source_type == "licensed_market_data" and record.authority_tier == "B" for record in records)


def test_structured_data_respects_upload_only_and_missing_connection(tmp_path):
    runtime = runtime_at(tmp_path)
    args = FinancialHistoryRequest(years=[2025])
    assert fetch_history(runtime, args)["status"] == "not_configured"
    runtime.session.data_source_preference = "upload"
    with pytest.raises(ValueError, match="NETWORK_OUT_OF_SCOPE"):
        fetch_history(runtime, args)


def test_wrong_period_annotation_can_be_repaired_from_same_source(tmp_path):
    runtime = runtime_at(tmp_path)
    blocks = attach(runtime, "annual.txt", "样本科技股份有限公司600123\n合并利润表\n单位：元\n项目 2025年度 2024年度\n营业收入 100 90")
    candidate = CandidateInput(metric="营业收入", raw_value="90", unit="元", scope="consolidated", period="2025",
        block_id=blocks[0]["block_id"], quote="营业收入 100 90")
    runtime.facts(ProposeFacts(candidates=[candidate]))
    previous = runtime.session.facts[0]
    assert previous.warnings
    runtime.facts(ProposeFacts(candidates=[candidate.model_copy(update={"period": "2024"})], replaces=[previous.fact_id]))
    assert previous.status == "rejected" and runtime.session.facts[-1].status == "confirmed"


def test_retracting_a_candidate_is_audited_and_does_not_fill_gap(tmp_path):
    runtime = runtime_at(tmp_path)
    facts = public_facts(runtime)
    reject_candidates(runtime, FactSelection(fact_ids=[facts[0].fact_id], reason="该网页无法证明数据的独立上游来源，撤回本次候选。"))
    assert facts[0].status == "rejected" and facts[0].verification["withdrawal"]
    assert not [fact for fact in facts if fact.status == "confirmed"]


def test_progress_checkpoint_continues_same_agent_within_bounded_windows(tmp_path, monkeypatch):
    runtime = runtime_at(tmp_path)
    turns = []
    def loop(llm, messages, registry, call, **kwargs):
        turns.append(messages)
        if len(turns) == 1:
            attach(runtime, "source.txt", "新增已下载的原文")
            raise LlmError("AGENT_STEP_LIMIT: fixture")
        assert "最后窗口" in messages[1]["content"]
        return {"answer": "进度已交付"}
    monkeypatch.setattr("valuationagent.application.agent_runtime.run_tool_loop", loop)
    assert runtime.run(object())["answer"] == "进度已交付"
    assert len(turns) == 2
    assert any(event.type == "agent.checkpoint" for event in runtime.service.store.list_events(runtime.session.session_id))


def test_no_progress_does_not_get_extra_windows(tmp_path, monkeypatch):
    runtime = runtime_at(tmp_path)
    calls = []
    def loop(*args, **kwargs):
        calls.append(1)
        raise LlmError("AGENT_STEP_LIMIT: fixture")
    monkeypatch.setattr("valuationagent.application.agent_runtime.run_tool_loop", loop)
    with pytest.raises(LlmError, match="AGENT_STEP_LIMIT"):
        runtime.run(object())
    assert len(calls) == 1
