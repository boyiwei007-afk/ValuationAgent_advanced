"""Regression cases from real-statement audit; no live data or model required.

The four disclosed D&A rows in the first test are transcribed from page 116
of Kweichow Moutai's 2024 annual report, not a live investment valuation:
https://static.cninfo.com.cn/finalpage/2025-04-03/1222993920.PDF
All other financial amounts below are explicitly synthetic.
"""
from datetime import date
from decimal import Decimal as D

import pytest

from valuationagent.application.research_valuation import (
    ResearchValuationAssembler,
    normalize_financial_metric,
)
from valuationagent.schemas.research import FactCandidate
from valuationagent.schemas.research import ResearchDraft, ResearchSession
from valuationagent.finance.integrity import (
    EQUITY_BRIDGE_REVIEW_LABELS,
    equity_bridge_review_findings,
    validate_equity_bridge_inputs,
    verify_calculations,
)
from valuationagent.finance.team_model import FinanceTeamModel
from valuationagent.schemas.models import CompanyInput, FinancialSnapshot, ValuationRequest


def fact(metric, value, *, quote=None):
    row = quote or f"{metric} {value}"
    return FactCandidate(
        fact_id="test_" + metric, metric=metric, raw_value=value,
        normalized_value=value, unit="元", period="2024年度",
        scope="consolidated", block_id="test:statement", quote=row,
        status="confirmed", verification={"source_row": row},
    )


def component_values(rows):
    return {
        normalize_financial_metric(metric): (D(value), [fact(metric, value)])
        for metric, value in rows.items()
    }


def synthetic_components():
    return component_values({
        "固定资产折旧": "100", "无形资产摊销": "20",
        "长期待摊费用摊销": "5", "使用权资产摊销": "10",
        "租赁负债": "50",
    })


def test_real_report_separately_disclosed_lease_charge_is_included_once():
    rows = {
        "固定资产折旧、油气资产折耗、生产性生物资产折旧": "1721165327.14",
        "使用权资产摊销": "94492678.29",
        "无形资产摊销": "249170059.35",
        "长期待摊费用摊销": "20191550.34",
    }
    values, methods = component_values(rows), {}
    assembler = ResearchValuationAssembler()
    assembler._derive_period(values, methods)
    assembler._reconcile_period(date(2024, 12, 31), values, methods)
    assert values["depreciation_amortization"][0] == D("2085019615.12")
    assert "depreciation_right_of_use" in methods["depreciation_amortization"]
    assert len(values["depreciation_amortization"][1]) == 4


def test_synthetic_missing_lease_charge_is_not_assumed_zero():
    values = synthetic_components()
    values.pop("depreciation_right_of_use")
    with pytest.raises(ValueError, match="不能把缺失项当作零"):
        ResearchValuationAssembler()._derive_period(values, {})
    assert "depreciation_amortization" not in values


def test_synthetic_lease_gap_does_not_block_methods_without_da():
    values = synthetic_components()
    values.pop("depreciation_right_of_use")
    assembler = ResearchValuationAssembler()
    assembler._derive_period(values, {}, require_da=False)
    assembler._reconcile_period(date(2024, 12, 31), values, {}, require_da=False)
    assert "depreciation_amortization" not in values


def test_synthetic_complete_direct_total_does_not_need_partial_component_sum():
    values = synthetic_components()
    values.pop("depreciation_right_of_use")
    values.update(component_values({"折旧摊销": "135"}))
    methods = {"depreciation_amortization": "direct_confirmed_fact"}
    assembler = ResearchValuationAssembler()
    assembler._derive_period(values, methods)
    assembler._reconcile_period(date(2024, 12, 31), values, methods)
    assert values["depreciation_amortization"][0] == D("135")
    assert methods["depreciation_amortization"] == "direct_confirmed_fact"


def test_synthetic_direct_total_is_not_incremented_by_lease_charge_again():
    values = synthetic_components()
    values.update(component_values({"折旧摊销": "135"}))
    methods = {"depreciation_amortization": "direct_confirmed_fact"}
    assembler = ResearchValuationAssembler()
    assembler._derive_period(values, methods)
    assembler._reconcile_period(date(2024, 12, 31), values, methods)
    assert values["depreciation_amortization"][0] == D("135")
    assert "depreciation_right_of_use" in methods["reconciliation.depreciation_amortization"]


def test_synthetic_direct_total_must_reconcile_with_complete_independent_rows():
    values = synthetic_components()
    values.update(component_values({"折旧摊销": "125"}))
    with pytest.raises(ValueError, match="勾稽"):
        ResearchValuationAssembler()._reconcile_period(
            date(2024, 12, 31), values,
            {"depreciation_amortization": "direct_confirmed_fact"},
        )


@pytest.mark.parametrize("row", [
    "固定资产折旧（含使用权资产折旧）110",
    "depreciation_fixed_assets including right_of_use depreciation 110",
    "折旧总额110",
])
def test_synthetic_inclusive_or_ambiguous_fixed_charge_cannot_double_count(row):
    values = synthetic_components()
    values["depreciation_fixed_assets"] = (
        D("110"), [fact("depreciation_fixed_assets", "110", quote=row)],
    )
    with pytest.raises(ValueError, match="不能重复相加"):
        ResearchValuationAssembler()._derive_period(values, {})


def test_synthetic_same_fact_cannot_supply_two_independent_components():
    values = synthetic_components()
    values["depreciation_right_of_use"] = values["depreciation_fixed_assets"]
    with pytest.raises(ValueError, match="不能重复相加"):
        ResearchValuationAssembler()._derive_period(values, {})


def test_synthetic_explicit_zero_lease_charge_is_valid_not_missing():
    values = synthetic_components()
    values["depreciation_right_of_use"] = (
        D(0), [fact("使用权资产摊销", "0")],
    )
    ResearchValuationAssembler()._derive_period(values, {})
    assert values["depreciation_amortization"][0] == D("125")


@pytest.mark.parametrize("metric,expected", [
    ("普通股股数（总股本）", "common_shares"),
    ("总股本(普通股股数)", "common_shares"),
    ("common_shares（总股本）", "common_shares"),
    ("普通股股份总数", "common_shares"),
    ("普通股股份总额", "common_shares"),
    ("股份总数", "common_shares"),
    ("股本", None),
    ("营业收入（营业总收入）", None),
    ("营业总收入（营业收入）", None),
    ("普通股股数（实收资本）", None),
    ("固定资产折旧（含使用权资产折旧）", None),
])
def test_parenthesized_synonyms_must_have_the_same_exact_basis(metric, expected):
    assert normalize_financial_metric(metric) == expected


def structured_request(method, metric, amount="10"):
    snapshot = FinancialSnapshot(
        period_end=date(2024, 12, 31), revenue="1000", ebit_margin="0.2",
        tax_rate="0.25", depreciation_amortization="10", capital_expenditure="15",
        change_operating_nwc="5", cash_and_non_operating_assets="100",
        interest_bearing_debt="40", common_shares="100", ebitda="210",
        net_income_parent="140", statement_items={metric: D(amount)},
    )
    return ValuationRequest(
        company=CompanyInput(name="合成桥接门禁样本", industry="电子"),
        valuation_date=date(2025, 6, 30), methods=[method], financials=snapshot,
    )


@pytest.mark.parametrize("metric", list(EQUITY_BRIDGE_REVIEW_LABELS))
@pytest.mark.parametrize("method", ["dcf", "ev_ebitda"])
def test_explicit_complex_bridge_inputs_block_only_enterprise_value_methods(metric, method):
    request = structured_request(method, metric)
    findings = FinanceTeamModel().validate(request, request.financials)
    gate = next(f for f in findings if f.rule_id == "EQUITY_BRIDGE_COMPLEX_SCOPE")
    assert gate.severity == "blocking"
    assert EQUITY_BRIDGE_REVIEW_LABELS[metric] in gate.message
    assert "不能把调整默认为零" in gate.message
    assert request.financials.statement_items[metric] == D("10")
    with pytest.raises(ValueError, match="桥接需专项复核"):
        verify_calculations(request, request.financials, None, [], None, [])


@pytest.mark.parametrize("method", ["pe", "ps"])
def test_bridge_review_does_not_block_independent_equity_multiples(method):
    request = structured_request(method, "minority_interest")
    assert not validate_equity_bridge_inputs(request, request.financials)
    assert not any(f.rule_id == "EQUITY_BRIDGE_COMPLEX_SCOPE"
                   for f in FinanceTeamModel().validate(request, request.financials))


def test_explicit_zero_risk_fact_does_not_trigger_positive_exposure_gate():
    request = structured_request("dcf", "minority_interest", "0")
    assert not validate_equity_bridge_inputs(request, request.financials)


@pytest.mark.parametrize("metric", list(EQUITY_BRIDGE_REVIEW_LABELS))
@pytest.mark.parametrize("method", ["dcf", "ev_ebitda"])
def test_nonzero_negative_bridge_amount_cannot_bypass_review(metric, method):
    # A deficit is not an approved zero market-value adjustment.  Negative
    # restricted funds / deposits likewise need review of sign and basis.
    request = structured_request(method, metric, "-10")
    findings = validate_equity_bridge_inputs(request, request.financials)
    assert findings and findings[0].severity == "blocking"
    assert EQUITY_BRIDGE_REVIEW_LABELS[metric] + "=-10" in findings[0].message
    with pytest.raises(ValueError, match="桥接需专项复核"):
        verify_calculations(request, request.financials, None, [], None, [])


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-Infinity"])
def test_independent_bridge_gate_rejects_nonfinite_values_from_plugins(amount):
    # Normal schema validation rejects these already.  An independent output
    # guard also protects against model_construct / third-party plugin bypass.
    findings = equity_bridge_review_findings(["dcf"], {"minority_interest": D(amount)})
    assert findings and findings[0].severity == "blocking"


def test_zero_amount_is_explicitly_different_from_unmeasured_exposure():
    # Document the function's coverage boundary.  The caller must separately
    # ensure that source review is complete; these two returns do NOT establish
    # equivalent evidence coverage.
    assert not equity_bridge_review_findings(["dcf"], {"minority_interest": D(0)})
    assert not equity_bridge_review_findings(["dcf"], {})


def test_research_bridge_review_retains_raw_evidence_and_allows_pe_subset():
    session = ResearchSession(session_id="test_finance_scope", draft=ResearchDraft(methods=["dcf"]))
    session.facts = [fact("少数股东权益", "10"), fact("归母净利润", "140"),
                     fact("common_shares", "100")]
    session.facts[-1].unit = "股"
    assembler = ResearchValuationAssembler()
    assert "少数股东权益=10" in assembler.structured_readiness_error(session)
    assert session.facts[0].metric == "少数股东权益"
    session.draft.methods = ["pe"]
    snapshot = assembler._structured_financials(session)[0]
    assert snapshot.net_income_parent == D("140")
    assert snapshot.statement_items["minority_interest"] == D("10")


def test_only_share_count_accepts_issuer_scope_alongside_consolidated_income():
    session = ResearchSession(session_id="test_share_scope", draft=ResearchDraft(methods=["pe"]))
    shares = fact("普通股股数（总股本）", "100")
    shares.scope, shares.unit = "issuer", "股"
    income = fact("归母净利润", "140")
    issuer_cash = fact("货币资金", "999")
    issuer_cash.scope = "issuer"
    session.facts = [income, shares, issuer_cash]
    snapshot = ResearchValuationAssembler()._structured_financials(session)[0]
    assert snapshot.common_shares == D("100")
    assert snapshot.net_income_parent == D("140")
    assert snapshot.cash_and_non_operating_assets is None
    assert "cash_and_non_operating_assets" not in snapshot.statement_items
    assert shares.metric == "普通股股数（总股本）"


def test_generic_total_shares_requires_issuer_scope_not_a_statement_amount():
    session = ResearchSession(session_id="test_total_shares", draft=ResearchDraft(methods=["pe"]))
    shares = fact("股份总数", "100")
    shares.unit = "股"
    session.facts = [fact("归母净利润", "140"), shares]
    assembler = ResearchValuationAssembler()
    assert "普通股股数" in assembler.structured_readiness_error(session)
    shares.scope = "issuer"
    assert assembler._structured_financials(session)[0].common_shares == D("100")


def test_real_midea_year_end_shares_cannot_override_later_disclosed_issuer_total():
    # Two different issuer totals appear in the same 2024 annual report:
    # page 136 year end and page 3 as of the later annual-report disclosure.
    session = ResearchSession(session_id="test_midea_later_shares", draft=ResearchDraft(
        company="美的集团", ticker="000333.SZ", valuation_date=date(2025, 6, 30), methods=["pe"],
    ))
    shares = fact("股份总数", "7655955883")
    shares.fact_id = "midea:share"
    shares.block_id = "midea:136"
    shares.scope, shares.unit = "issuer", "股"
    session.facts = [shares]
    blocks = [
        {"file_id": "midea", "block_id": "midea:136", "text": "三、股份总数 7,655,955,883"},
        {"file_id": "midea", "block_id": "midea:3", "text":
         "以截至本报告披露之日公司总股本 7,660,355,772股扣除回购专户股份后的股本为分红基数"},
    ]
    assembler = ResearchValuationAssembler(block_loader=lambda _: blocks)
    issue = assembler.later_issuer_shares_issue(session)
    assert "7660355772股不同" in issue
    assert "分红基数不能替代" in issue
    with pytest.raises(ValueError, match="不能将年末数直接用作较晚估值日"):
        assembler._structured_financials(session)


def test_later_issuer_shares_guard_does_not_borrow_other_source_or_dividend_base():
    session = ResearchSession(session_id="test_share_source_scope", draft=ResearchDraft(
        valuation_date=date(2025, 6, 30), methods=["pe"],
    ))
    shares = fact("股份总数", "100")
    shares.scope, shares.unit, shares.block_id = "issuer", "股", "issuer_report:136"
    session.facts = [shares]
    assembler = ResearchValuationAssembler(block_loader=lambda _: [
        {"file_id": "issuer_report", "text": "截至本报告披露之日以总股本100股扣除回购后，分红基数为90股"},
        {"file_id": "other_company", "text": "截至本报告披露之日公司总股本200股"},
    ])
    assert assembler.later_issuer_shares_issue(session) == ""


def test_verified_report_disclosure_shares_replace_latest_per_share_denominator_only():
    session = ResearchSession(session_id="test_midea_share_timeline", draft=ResearchDraft(
        company="美的集团", ticker="000333.SZ", valuation_date=date(2025, 6, 30), methods=["pe"],
    ))
    annual = fact("股份总数", "7655955883")
    annual.scope, annual.unit, annual.block_id = "issuer", "股", "midea:136"
    newer = fact("总股本", "7660355772")
    newer.fact_id, newer.scope, newer.unit = "midea:current", "issuer", "股"
    newer.block_id, newer.period = "midea:3", "2025-03-29"
    newer.verification = {"binding": "issuer_report_disclosure_shares", "scope": "issuer",
                          "period_end": "2025-03-29"}
    session.facts = [fact("归母净利润", "140"), annual, newer]
    assembler = ResearchValuationAssembler(block_loader=lambda _: [
        {"file_id": "midea", "block_id": "midea:3", "text":
         "截至本报告披露之日公司总股本7,660,355,772股扣除已回购股份，"
         "可参与分红股份数为7,631,903,546股"},
    ])
    assert not assembler.later_issuer_shares_issue(session)
    snapshots = assembler._structured_financials(session)
    assert len(snapshots) == 1
    assert snapshots[0].period_end == date(2024, 12, 31)
    assert snapshots[0].common_shares == D("7660355772")
    assert snapshots[0].common_shares_as_of == date(2025, 3, 29)
    assert snapshots[0].calculation_methods["common_shares"] == "issuer_shares_as_of[2025-03-29]"
    assert snapshots[0].evidence["common_shares"][0].evidence_id == newer.fact_id

    # Uploading the same official PDF and later fetching its public copy gives
    # two file IDs, but their verified content hashes identify one disclosure.
    annual.source_sha256 = newer.source_sha256 = "a" * 64
    newer.block_id = "official_copy:3"
    assert not assembler.later_issuer_shares_issue(session)
    assert assembler._structured_financials(session)[0].common_shares == D("7660355772")


@pytest.mark.parametrize("share_date", [date(2024, 1, 1), date(2025, 7, 1)])
def test_structured_request_rejects_share_date_outside_period_to_valuation_window(share_date):
    request = structured_request("pe", "minority_interest")
    request.financials.common_shares_as_of = share_date
    with pytest.raises(ValueError, match="issuer share-count date"):
        ValuationRequest.model_validate(request.model_dump(mode="json"))
