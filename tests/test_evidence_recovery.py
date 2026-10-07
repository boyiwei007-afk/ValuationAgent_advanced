"""Company-independent regressions for real annual-report layouts and recovery.

Names and amounts below are synthetic, not financial data about real issuers.
"""
import json
from datetime import date

import pytest

from valuationagent.application.agent_runtime import WorkspaceAgentRuntime
from valuationagent.application.reporting import ValuationReportExporter
from valuationagent.application.research import CandidateInput, ProposeFacts, ResearchService
from valuationagent.application.research_valuation import (
    METRIC_ALIASES,
    mapped_financial_metric,
    normalize_financial_metric,
)
from valuationagent.application.runner import ValuationRunner
from valuationagent.core.evidence import bind_evidence, evidence_context, numeric_tokens
from valuationagent.finance.team_model import FinanceTeamModel
from valuationagent.schemas.models import required_financial_metrics
from valuationagent.schemas.research import (
    DocumentSummary,
    FactCandidate,
    ResearchDraft,
    ResearchTurn,
)
from valuationagent.search.providers import MockSearchProvider
from valuationagent.storage.sqlite import SQLiteRunStore


def draft(name="样本制造公司", ticker="600123"):
    return ResearchDraft(company=name, ticker=ticker, industry="电子", methods=["dcf"], valuation_date=date(2026, 9, 24))


def block(text, index=1, **location):
    return {"block_id": f"file_test:{index}", "file_id": "file_test", "text": text, "location": location}


def item(metric="营业收入", value="100", **changes):
    return CandidateInput.model_validate({"metric": metric, "raw_value": value, "unit": "万元", "period": "2025",
        "scope": "consolidated", "block_id": "file_test:1", "quote": f"{metric} {value}", **changes})


def bind(candidate, source, sources=None, company=None):
    sources = sources or [source]
    aliases = METRIC_ALIASES.get(normalize_financial_metric(candidate.metric), ())
    return bind_evidence(candidate, source, evidence_context(source, sources, candidate.context_block_ids),
                         company or draft(), aliases, identity_text="\n".join(b["text"] for b in sources[:2]))


@pytest.mark.parametrize("name,ticker,separator", [
    ("样本制造公司", "600123", "  "), ("样本消费公司", "000321", " | "),
    ("样本医药公司", "300123", "\t"), ("样本新能源公司", "688123", "      "),
])
def test_exact_source_metric_wins_over_similar_rows_for_any_company(name, ticker, separator):
    rows = [separator.join(row) for row in [
        ["项目", "附注", "2025年度", "2024年度"],
        ["一、营业总收入", "", "110", "95"],
        ["其中：营业收入", "44", "100", "90"],
        ["利息支出", "45", "30", "20"],
        ["其中：利息费用", "", "3", "2"],
    ]]
    source = block(f"{name} {ticker}\n合并利润表\n单位：万元\n" + "\n".join(rows))
    for metric, value, row in [("营业收入", "100", rows[2]), ("利息费用", "3", rows[4])]:
        warnings, checks = bind(item(metric, value, quote=row), source, company=draft(name, ticker))
        assert not warnings, warnings
        assert checks["year_column"] == 2025
        assert checks["source_row"] == row
    assert bind(item(quote=rows[2]), source, company=draft(name, ticker))[1]["note_column_excluded"]


def test_revenue_never_aliases_a_different_statement_concept():
    source = block("样本制造公司600123 合并报表\n单位：万元\n项目 2025年 2024年\n营业总收入 110 95\n营业收入 100 90")
    assert not bind(item("revenue", quote=source["text"]), source)[0]
    assert not bind(item("revenue", quote="营业收入 100 90"), source)[0]
    assert bind(item("revenue", value="110", quote="营业总收入 110 95"), source)[0]
    assert normalize_financial_metric("营业总收入") == "total_revenue"
    assert normalize_financial_metric("主营业务收入") == "main_business_revenue"


def test_chinese_note_reference_after_label_is_not_part_of_metric_name():
    source = block(
        "样本制造公司600123\n合并利润表\n单位：万元\n"
        "项目 附注 2025年 2024年\n"
        "营业外支出 七、75 15 16\n"
        "减：所得税费用 七、76 100 90"
    )
    warnings, checks = bind(item("所得税费用", "100", quote="减：所得税费用 七、76 100 90"), source)
    assert not any("字段名未与数值绑定" in warning for warning in warnings)
    assert checks["source_row"] == "减：所得税费用 七、76 100 90"


def test_wrapped_metric_can_cross_an_inserted_chinese_note_reference():
    source = block(
        "样本制造公司600123\n合并现金流量表\n单位：万元\n"
        "项目 附注 2025年 2024年\n"
        "购建固定资产、无形资产和其他 七、78（2） 100 90\n"
        "长期资产支付的现金"
    )
    candidate = item(
        "购建固定资产、无形资产和其他长期资产支付的现金",
        "100",
        quote="购建固定资产、无形资产和其他 七、78（2） 100 90\n长期资产支付的现金",
    )
    warnings, checks = bind(candidate, source)
    assert not any("字段名未与数值绑定" in warning for warning in warnings)
    assert "长期资产支付的现金" in checks["source_row"]


def test_distinct_revenue_lines_do_not_create_an_artificial_confirmation_conflict(tmp_path):
    service, session, _ = configured_service(tmp_path)
    session.draft.methods = ["ps"]
    session.facts = [FactCandidate(**item(metric, str(value), unit=unit).model_dump(),
                                  fact_id=f"f_{index}", status="confirmed", normalized_value=str(value))
                     for index, (metric, value, unit) in enumerate([
                         ("营业收入", 100, "元"), ("营业总收入", 110, "元"),
                         ("主营业务收入", 90, "元"), ("普通股股数", 10, "股")])]
    snapshot = service.valuation_assembler._structured_financials(session)[0]
    assert snapshot.revenue == 100
    assert snapshot.statement_items["total_revenue"] == 110
    assert snapshot.statement_items["main_business_revenue"] == 90


@pytest.mark.parametrize("headers,values,year,value", [
    ("2025年度 2024年度", "100 90", "2025", "100"),
    ("2024年 2025年", "90 100", "2025", "100"),
    ("2025年 2024年 本年比上年增减(%) 2023年", "100 90 11.11 80", "2023", "80"),
    ("2025年 2024年 本年比上年增减(%) 2023年", "100 90 11.11% 80", "2023", "80"),
])
def test_explicit_year_columns_and_comparison_columns(headers, values, year, value):
    source = block(f"样本制造公司600123 合并报表\n单位：万元\n项目 附注 {headers}\n营业收入 7 {values}")
    candidate = item(value=value, period=year, quote=f"营业收入 7 {values}")
    assert not bind(candidate, source)[0]
    wrong = candidate.model_copy(update={"period": "2024"})
    assert any("年度列冲突" in w for w in bind(wrong, source)[0])


def test_missing_column_does_not_shift_a_value_into_a_different_year():
    source = block("样本制造公司600123 合并报表\n单位：万元\n项目 2025年 2024年\n营业收入     90")
    assert bind(item(value="90", quote="营业收入     90"), source)[0]


def test_verified_blank_debt_cell_can_be_zero_but_ordinary_missing_data_cannot(tmp_path):
    row1 = f"{'应付账款':<16}{'20':>4}{'100':>14}{'90':>14}"
    row2 = f"{'合同负债':<16}{'21':>4}{'200':>14}{'180':>14}"
    blank_row = "短期借款"
    text = (
        "样本制造公司600123\n2025年12月31日\n合并资产负债表\n单位：万元\n"
        "项目 附注 2025年 2024年\n" + "\n".join((row1, row2, blank_row))
    )
    source = block(text)
    candidate = item("短期借款", "0", quote=blank_row)
    warnings, checks = bind(candidate, source)
    assert not warnings, warnings
    assert checks["source_blank_as_zero"] is True
    assert checks["year_column"] == 2025

    service, session, _ = configured_service(tmp_path)
    service.store.save_research_blocks(session.session_id, "file_test", [source])
    accepted, rejected = service._validate_candidates(session, [candidate])
    assert not rejected
    assert accepted[0].normalized_value == "0"

    narrative = block("样本制造公司600123 说明：短期借款资料未取得")
    unsafe = item("短期借款", "0", quote=narrative["text"])
    assert bind(unsafe, narrative)[0]


def test_contextual_semantic_mapping_separates_finance_subsidiary_interest(tmp_path):
    source = block(
        "样本制造公司600123\n合并利润表\n单位：万元\n"
        "项目 附注 2025年 2024年\n"
        "利息收入 40 500 450\n"
        "利息支出 41 300 260\n"
        "财务费用 46 -20 -18\n"
        "其中：利息费用 46 3 2"
    )
    service, session, _ = configured_service(tmp_path)
    service.store.save_research_blocks(session.session_id, "file_test", [source])
    finance_sub = item(
        "利息支出", "300", quote="利息支出 41 300 260",
        standard_metric="financial_subsidiary_interest_expense",
        semantic_role="financial_subsidiary",
        ebit_treatment="exclude", fcff_treatment="exclude",
        equity_bridge_treatment="exclude", mapping_confidence=0.96,
        mapping_rationale="该行与利息收入并列，附注说明属于财务子公司吸收存款业务的经营成本。",
    )
    financing = item(
        "利息费用", "3", quote="其中：利息费用 46 3 2",
        standard_metric="interest_expense", semantic_role="financing",
        ebit_treatment="include", fcff_treatment="include",
        equity_bridge_treatment="exclude", mapping_confidence=0.93,
        mapping_rationale="该行位于财务费用明细，属于租赁负债等融资利息，用于程序推导EBIT。",
    )
    accepted, rejected = service._validate_candidates(session, [finance_sub, financing])
    assert not rejected
    assert all(not fact.warnings for fact in accepted), [fact.warnings for fact in accepted]
    assert [mapped_financial_metric(fact) for fact in accepted] == [
        "financial_subsidiary_interest_expense", "interest_expense",
    ]
    assert accepted[0].verification["semantic_mapping"]["resolution_method"] == "llm_context_mapping"
    assert accepted[0].verification["source_context"]["adjacent_rows"]

    unsafe = financing.model_copy(update={
        "metric": "利息支出",
        "quote": "利息支出 41 300 260",
        "raw_value": "300",
        "semantic_role": "financial_subsidiary",
        "ebit_treatment": "exclude",
    })
    warned, rejected = service._validate_candidates(session, [unsafe])
    assert not rejected
    assert any("融资利息准入冲突" in warning for warning in warned[0].warnings)

    legacy, rejected = service._validate_candidates(
        session, [item("利息费用", "3", quote="其中：利息费用 46 3 2")]
    )
    assert not rejected
    assert any("LLM语义映射" in warning for warning in legacy[0].warnings)


def test_unique_verbatim_context_block_repairs_a_page_boundary_block_id(tmp_path):
    first = block(
        "样本制造公司600123\n2025年12月31日\n合并资产负债表\n单位：万元\n"
        "项目 附注 2025年 2024年\n"
        f"{'应付账款':<16}{'20':>4}{'100':>14}{'90':>14}\n"
        f"{'合同负债':<16}{'21':>4}{'200':>14}{'180':>14}\n"
        "短期借款",
        index=1,
    )
    continuation = block("样本制造公司600123\n负债表下页\n应付账款 100 90", index=2)
    service, session, _ = configured_service(tmp_path)
    service.store.save_research_blocks(session.session_id, "file_test", [first, continuation])
    candidate = item(
        "短期借款", "0", quote="短期借款", block_id="file_test:2",
        context_block_ids=["file_test:1"], standard_metric="short_term_borrowings",
        semantic_role="financing", ebit_treatment="exclude",
        fcff_treatment="exclude", equity_bridge_treatment="include",
        mapping_confidence=0.9,
        mapping_rationale="合并资产负债表目标年度短期借款金额格为空，前置完整行证明年度列对齐。",
    )
    accepted, rejected = service._validate_candidates(session, [candidate])
    assert not rejected
    assert not accepted[0].warnings
    assert accepted[0].block_id == "file_test:1"
    context = accepted[0].verification["source_context"]
    assert context["block_relocated_from_context"] is True
    assert context["supplied_block_id"] == "file_test:2"


def test_issuer_share_total_in_strict_capital_note_is_a_count_not_balance_sheet_capital():
    source = block(
        "样本制造公司600123 2025年年度报告\n"
        "53、 股本\n单位：元 币种：人民币\n"
        "本次变动增减（+、-）\n期初余额 发行新股 送股 公积金转股 其他 小计 期末余额\n"
        "股份总数 5,560,600,544 5,560,600,544\n"
        "于2025年12月31日，本公司注册资本包括普通股，每股面值人民币1元。所有普通股同股同权。"
    )
    candidate = item(
        "股份总数", "5,560,600,544", unit="股", period="2025-12-31",
        scope="issuer", quote="股份总数 5,560,600,544 5,560,600,544",
        standard_metric="common_shares", semantic_role="equity",
        ebit_treatment="exclude", fcff_treatment="exclude",
        equity_bridge_treatment="include", mapping_confidence=0.98,
        mapping_rationale="股本附注明确标注股份总数期末列，并说明普通股类别及每股1元面值。",
    )
    warnings, checks = bind(candidate, source)
    assert not warnings, warnings
    assert checks["binding"] == "issuer_share_capital_note"
    assert checks["period_end"] == "2025-12-31"
    assert checks["unit"] == "股"

    unsafe = candidate.model_copy(update={"metric": "实收资本（或股本）", "quote": source["text"]})
    assert bind(unsafe, source)[0]


def test_report_year_alone_does_not_identify_multiple_amount_columns():
    source = block("样本制造公司600123 2025年合并报表\n单位：万元\n项目 期初余额 期末余额\n营业收入 100 90")
    assert any("不能默认首列" in w for w in bind(item(), source)[0])


def test_metric_name_cannot_match_a_different_longer_concept():
    source = block("样本制造公司600123 合并报表\n单位：万元\n项目 2025年 2024年\n其他业务营业收入 100 90")
    assert bind(item(), source)[0]


def test_amounts_resembling_years_are_not_used_as_headers():
    source = block("样本制造公司600123 合并报表\n单位：万元\n项目 2024年 2025年\n营业成本 2025 2024\n营业收入 100 90")
    assert any("年度列冲突" in w for w in bind(item(quote="营业收入 100 90"), source)[0])


def test_long_worksheet_retains_its_own_headers():
    sources = [block("样本制造公司600123 合并报表", 1, sheet="合并"),
               block("单位：万元", 2, sheet="合并"),
               block("项目 2025年 2024年", 3, sheet="合并")]
    sources += [block(f"其他科目{i} 10 9", i, sheet="合并") for i in range(4, 30)]
    sources.append(block("营业收入 100 90", 30, sheet="合并"))
    assert not bind(item(block_id=sources[-1]["block_id"], quote=sources[-1]["text"]), sources[-1], sources)[0]


def test_adjacent_negative_pdf_amounts_keep_their_sign():
    assert list(map(str, numeric_tokens("-815.24-1470.21"))) == ["-815.24", "-1470.21"]


def test_concatenated_pdf_money_columns_are_split_without_guessing_decimals():
    text = "29,444,936,771.4130,303,850,168.56"
    assert list(map(str, numeric_tokens(text))) == ["29444936771.41", "30303850168.56"]
    assert list(map(str, numeric_tokens("比率 0.123456"))) == ["0.123456"]


def test_concatenated_note_columns_bind_and_infer_unique_source_year():
    target = "减：所得税费用                       58      29,444,936,771.4130,303,850,168.56"
    source = block(
        "样本制造公司600123\n合并利润表\n单位：元\n"
        "项目                    附注           2025年度              2024年度\n"
        "加：营业外收入                       56          74,947,039.70    70,936,575.97\n"
        "减：营业外支出                       57         128,635,598.86   120,937,834.74\n"
        + target
    )
    candidate = item(
        "所得税费用",
        "29,444,936,771.41",
        unit="元",
        period="unknown",
        quote=target,
    )
    warnings, checks = bind(candidate, source)

    assert not warnings, warnings
    assert checks["year_column"] == 2025
    assert checks["period"] == "2025"
    assert checks["period_resolution"] == "unique_source_column"


def test_wrapped_statement_label_and_concatenated_columns_bind_to_the_requested_year():
    row = "购建固定资产、无形资产和其3,127,594,916.414,678,712,053.56\n他长期资产支付的现金"
    source = block(
        "样本制造公司600123\n合并现金流量表\n单位：元\n项目 2025年 2024年\n" + row
    )
    candidate = item(
        "购建固定资产、无形资产和其他长期资产支付的现金",
        "3,127,594,916.41",
        unit="元",
        quote=row,
    )
    warnings, checks = bind(candidate, source)
    assert not warnings, warnings
    assert checks["year_column"] == 2025
    assert checks["source_row"].startswith("购建固定资产、无形资产和其")
    assert checks["source_row"].endswith("他长期资产支付的现金")


def test_cross_page_header_follows_physical_pages_not_append_order():
    title = block("样本制造公司600123\n合并利润表\n单位：万元\n项目 附注 2025年 2024年", 20, page=8)
    row = block("营业收入 7 100 90", 2, page=9)
    later = block("母公司利润表\n单位：元\n项目 2025年 2024年", 3, page=10)
    warnings, checks = bind(item(block_id=row["block_id"], quote=row["text"]), row, [row, later, title])
    assert not warnings and title["block_id"] in checks["context_block_ids"]


def test_statement_header_can_span_three_consecutive_loaded_pages():
    header = block("样本制造公司600123\n合并资产负债表\n单位：万元\n项目 附注 2025年 2024年", 1, page=8)
    continuation = block("应付账款 7 10 9", 2, page=9)
    row = block("长期借款 8 100 90", 3, page=10)
    warnings, _ = bind(item("长期借款", block_id=row["block_id"], quote=row["text"]), row, [header, continuation, row])
    assert not warnings
    assert bind(item("长期借款", block_id=row["block_id"]), row, [header, row])[0]


def test_parent_statement_cannot_borrow_consolidated_scope_or_unit():
    first = block("样本制造公司600123\n合并利润表\n单位：万元\n项目 2025年 2024年", 1, page=8)
    second = block("母公司利润表\n项目 2025年 2024年\n营业收入 100 90", 2, page=9)
    warnings, _ = bind(item(block_id=second["block_id"], quote="营业收入 100 90"), second, [first, second])
    assert any("口径冲突" in w for w in warnings)
    assert any("单位缺少" in w for w in warnings)


def test_distant_or_future_or_other_sheet_header_is_not_authority():
    for location, wrong_location in [({"page": 9}, {"page": 6}), ({"page": 9}, {"page": 10}),
                                     ({"sheet": "经营"}, {"sheet": "母公司"})]:
        header = block("样本制造公司600123 合并报表\n单位：万元\n项目 2025年 2024年", 1, **wrong_location)
        row = block("营业收入 100 90", 2, **location)
        warnings, _ = bind(item(block_id=row["block_id"], context_block_ids=[header["block_id"]]), row, [header, row])
        assert any("单位缺少" in w for w in warnings)


@pytest.mark.parametrize("period,amount", [("2025", "-100"), ("2024", "-90")])
def test_notes_section_scope_and_relative_columns_with_wrapped_labels(period, amount):
    sources = [
        block("样本制造公司600123 2025年年度报告\n七、合并财务报表项目注释", 1, page=3),
        block("61、现金流量表补充资料\n单位：万元", 2, page=4),
        block("样本制造公司2025年年度报告\n补充资料 本期金额 上期金额\n"
              "经营性应收项目的减少（增加以“－”号    -100    -90\n填列）", 3, page=5),
    ]
    metric = "经营性应收项目的减少（增加以“－”号填列）"
    assert normalize_financial_metric(metric) == "operating_receivables_decrease"
    candidate = item(metric, amount, period=period, block_id=sources[-1]["block_id"], quote=sources[-1]["text"])
    warnings, checks = bind(candidate, sources[-1], sources)
    assert not warnings, warnings
    assert checks["year_column"] == int(period)
    assert checks["scope"] == "consolidated"
    # Removing an intervening page prevents unjustified section inheritance.
    assert any("口径缺少" in w for w in bind(candidate, sources[-1], [sources[0], sources[-1]])[0])


def configured_service(tmp_path, llm=None):
    service = ResearchService(SQLiteRunStore(tmp_path))
    session = service.create(llm=llm)
    session.draft = draft()
    session.data_source_preference = "web"
    session.documents.append(DocumentSummary(file_id="file_test", name="synthetic-annual.txt", role="historical_financials", block_count=1))
    source = block("样本制造公司600123 合并报表\n单位：万元\n项目 2025年 2024年\n营业收入 100 90")
    service.store.save_research_blocks(session.session_id, "file_test", [source])
    service.store.save_research(session)
    return service, session, source


def test_verified_evidence_reopens_closed_outcome_and_persists_inferred_period(tmp_path):
    service, session, _ = configured_service(tmp_path)
    target = "减：所得税费用                       58      29,444,936,771.4130,303,850,168.56"
    source = block(
        "样本制造公司600123\n合并利润表\n单位：元\n"
        "项目                    附注           2025年度              2024年度\n"
        "加：营业外收入                       56          74,947,039.70    70,936,575.97\n"
        "减：营业外支出                       57         128,635,598.86   120,937,834.74\n"
        + target
    )
    service.store.save_research_blocks(session.session_id, "file_test", [source])
    session.outcome_status = "insufficient_data"
    session.outcome_reason = "previous bounded attempt"
    candidate = item(
        "所得税费用",
        "29,444,936,771.41",
        unit="元",
        period="unknown",
        quote=target,
    )

    result = WorkspaceAgentRuntime(service, session).facts(ProposeFacts(candidates=[candidate]))

    assert result["candidates"] and not result.get("_terminal")
    fact = next(fact for fact in session.facts if fact.metric == "所得税费用")
    assert fact.period == "2025"
    assert not fact.warnings
    assert session.outcome_status == ""
    assert session.outcome_reason == ""


class RepeatingRetrievalModel:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        from control_fixtures import control_reply

        if reply := control_reply(kwargs):
            return reply
        self.calls += 1
        return {"tool_calls": [{
            "id": str(self.calls),
            "type": "function",
            "function": {
                "name": "search_sources",
                "arguments": json.dumps({
                    "query": f"样本公司 缺失财务字段 正式披露 尝试{self.calls}",
                    "reason": "补齐DCF基期字段",
                    "purpose": "financials",
                }, ensure_ascii=False),
            },
        }]}


def test_repeating_one_search_target_stops_with_resumable_checkpoint(tmp_path):
    model = RepeatingRetrievalModel()
    service, session, _ = configured_service(tmp_path, model)
    session.draft.ticker = ""  # Keep the synthetic test off the real official catalogue.
    session.pending_action = "valuation"
    service.search_provider = MockSearchProvider()
    service.store.save_research(session)

    state = service.turn(session.session_id, ResearchTurn(content="继续自动估值"))

    assert model.calls == 7
    assert len(state["session"]["search_history"]) == 3
    assert "question" not in state["session"]
    assert not state["session"]["outcome_status"]
    assert state["session"]["resume_context"]["reason"] == "AGENT_NO_PROGRESS"
    assert not service.store.list_runs()
    assert state["result_document"]["status"] == "insufficient_data"


class RepairModel:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if len(self.calls) == 3:
            name, args = "finish_response", {"answer": "已依据原文修正单位。"}
        else:
            candidate = item(quote="营业收入 100 90", unit="unknown" if len(self.calls) == 1 else "万元")
            name, args = "propose_facts", {"candidates": [candidate.model_dump()]}
            if len(self.calls) == 2:
                result = json.loads(messages[-1]["content"])
                assert result["candidates"][0]["warnings"]
                args["replaces"] = [result["candidates"][0]["fact_id"]]
        return {"tool_calls": [{"id": str(len(self.calls)), "type": "function",
                               "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_llm_repairs_unknown_unit_with_verified_evidence_in_same_turn(tmp_path):
    from copy import deepcopy
    from valuationagent.application.observation_extraction import ExtractObservations, extract_observations
    from observation_fixtures import fixture_observations, run_turn, extraction_steps
    service, session, _ = configured_service(tmp_path)
    runtime = WorkspaceAgentRuntime(service, session)
    correct = fixture_observations(runtime)
    uncertain = deepcopy(correct)
    uncertain["rows"][0]["uncertainties"] = ["单位的适用范围待补读"]
    previous = extract_observations(runtime, ExtractObservations.model_validate(uncertain))["rows"][0]["fact_id"]
    correct["rows"][0]["replaces"] = [previous]
    state, model = run_turn(runtime, extraction_steps(correct))
    assert len(model.calls) == 4
    assert [fact["status"] for fact in state["session"]["facts"]] == ["rejected", "confirmed"]
    assert not state["session"]["facts"][-1]["warnings"]
    assert state["execution"]["status"] == "completed"
    assert not service.store.list_runs()

def test_repeat_warned_candidates_are_deduplicated_and_never_present_impossible_confirmation(tmp_path):
    service, session, _ = configured_service(tmp_path)
    for _ in range(5):
        result = WorkspaceAgentRuntime(service, session).facts(ProposeFacts(candidates=[item(quote="营业收入 100 90", unit="unknown")]))
        assert any(fact.warnings for fact in session.facts)
        assert not result.get("_terminal") and "question" not in session.model_dump()
    assert len(session.facts) == 1 and session.facts[0].status == "proposed"


def test_financial_claims_in_model_prose_never_create_official_values(tmp_path):
    from test_research_sessions import ScriptedModel
    model = ScriptedModel([("finish_response", {"answer": "基期已补全，假设每股估值是999元。"})])
    service, session, _ = configured_service(tmp_path, model)
    state = service.turn(session.session_id, ResearchTurn(content="开始估值"))
    assert not state["session"]["facts"]
    assert state["session"]["valuation_run_id"] is None
    assert not state["result_document"]["numeric_result_available"]
    assert not service.store.list_runs()


def test_repeated_bad_evidence_remains_unverified_and_never_enters_calculation(tmp_path):
    from observation_fixtures import fixture_observations, run_turn
    service, session, _ = configured_service(tmp_path)
    runtime = WorkspaceAgentRuntime(service, session)
    args = fixture_observations(runtime)
    args["rows"][0]["uncertainties"] = ["尚未核清计量单位的适用范围"]
    state, _ = run_turn(runtime, [("extract_observations", args)] * 3 + [
        ("finish_response", {"answer": "没有可用单位证据，不输出估值。", "outcome": "insufficient_data"})])
    assert len(state["session"]["facts"]) == 1
    assert state["session"]["facts"][0]["status"] == "proposed"
    assert state["session"]["outcome_status"] == "insufficient_data"
    assert not state["result_document"]["numeric_result_available"]
    assert not service.store.list_runs()

def test_currency_share_capital_is_not_accepted_as_a_share_count(tmp_path):
    service, session, _ = configured_service(tmp_path)
    source = block("样本制造公司600123 2025年合并报表\n单位：万元\n总股本（万元）100")
    service.store.save_research_blocks(session.session_id, "file_test", [source])
    result = WorkspaceAgentRuntime(service, session).facts(ProposeFacts(candidates=[item("总股本", quote="总股本（万元）100")]))
    assert any(fact.warnings for fact in session.facts)
    assert any("维度" in w for w in session.facts[0].warnings)


def test_method_specific_pending_checks_preserve_conflicts_and_ignore_research_only_facts(tmp_path):
    service, session, _ = configured_service(tmp_path)
    session.draft.methods = ["pe"]
    def fact(metric, value="100", **changes):
        return FactCandidate(**{**item(metric, value).model_dump(), "fact_id": metric,
                               "normalized_value": value, **changes})
    session.facts = [fact("net_income_parent", status="confirmed"),
                     fact("财务费用", warnings=["待查"]), fact("revenue", warnings=["待查"])]
    assert not service.valuation_assembler.pending_blockers(session)
    session.facts.append(fact("net_income_parent", "90", fact_id="conflict"))
    assert [f.fact_id for f in service.valuation_assembler.pending_blockers(session)] == ["conflict"]
    session.facts[-1].normalized_value = "100"
    assert not service.valuation_assembler.pending_blockers(session)
    session.facts.append(fact("common_shares", unit="unknown"))
    assert service.valuation_assembler.pending_blockers(session)


def test_four_year_confirmed_dcf_starts_despite_unrelated_pending_note_and_exports_report(tmp_path):
    from test_finance_team_model import history
    service, session, _ = configured_service(tmp_path)
    fields = sorted(required_financial_metrics(["dcf"]))
    sources = []
    for index, row in enumerate(history()[-4:], 1):
        year = str(row.period_end.year)
        units = {key: "ratio" if key in {"ebit_margin", "tax_rate"} else "股" if key == "common_shares" else "元" for key in fields}
        lines = {key: f"{key}（{units[key]}） {getattr(row, key)}" for key in fields}
        source = block(f"样本制造公司600123 {year}年合并报表\n单位：元\n" + "\n".join(lines.values()), index)
        sources.append(source)
        service.store.save_research_blocks(session.session_id, "file_test", sources)
        candidates = [item(key, str(getattr(row, key)), period=year, unit=units[key],
                           block_id=source["block_id"], quote=lines[key]) for key in fields]
        WorkspaceAgentRuntime(service, session).facts(ProposeFacts(candidates=candidates))
        assert not any(f.warnings for f in session.facts), [(f.metric, f.warnings) for f in session.facts if f.warnings]
    session.facts.append(FactCandidate(**item("财务费用", "9").model_dump(), fact_id="unused", warnings=["缺少原文"] ))
    service.store.save_research(session)
    request = service.valuation_assembler.build(service.store.get_research(session.session_id))
    record = ValuationRunner(service.store, FinanceTeamModel()).run(request)
    assert record.result and record.result.dcf and record.result.sensitivity
    assert ValuationReportExporter().xlsx(record).startswith(b"PK")
