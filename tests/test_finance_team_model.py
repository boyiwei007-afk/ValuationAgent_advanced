from datetime import date
from decimal import Decimal as D

import pytest

from valuationagent.application.runner import ValuationRunner
from valuationagent.finance.industry import (
    FinancialIndustryUnsupported,
    IndustryParameterRegistry,
)
from valuationagent.finance.revenue import FinanceTeamRevenueModel
from valuationagent.finance.team_model import FinanceTeamModel
from valuationagent.finance.production import normalized_terminal_cash_flow
from valuationagent.schemas.models import (
    AssumptionInputs,
    CompanyInput,
    EvidenceRef,
    FinancialSnapshot,
    PeerCompany,
    RevisionInput,
    ValuationRequest,
)
from valuationagent.storage.sqlite import SQLiteRunStore


def history():
    revenues = [20, 22, 24, 27, 30, 33, 35, 32, 40, 44]
    margins = [0.18, 0.19, 0.20, 0.19, 0.21, 0.22, 0.20, 0.17, 0.22, 0.24]
    rows = []
    for offset, (revenue, margin) in enumerate(zip(revenues, margins)):
        amount = D(revenue) * D("100000000")
        rows.append(
            FinancialSnapshot(
                period_end=date(2016 + offset, 12, 31),
                revenue=amount,
                ebit_margin=D(str(margin)),
                tax_rate=D("0.08"),
                depreciation_amortization=amount * D("0.037"),
                capital_expenditure=amount * D("0.0529"),
                change_operating_nwc=amount * D("0.02"),
                cash_and_non_operating_assets=D("1225000000"),
                interest_bearing_debt=D("47000000"),
                common_shares=D("420949000"),
                net_income_parent=amount * D("0.20"),
                ebitda=amount * D("0.27"),
                tax_policy="高新技术企业15%",
                statement_items={
                    "market_cap": D("75000000000"),
                    "operating_nwc": amount * D("0.234"),
                    "total_equity": D("20000000000"),
                },
            )
        )
    return rows


def request(industry="电子"):
    rows = history()
    return ValuationRequest(
        company=CompanyInput(ticker="603893.SH", name="测试公司", industry=industry),
        valuation_date=date(2026, 9, 23),
        forecast_years=10,
        discount_policy="year_end",
        methods=["dcf"],
        financials=rows[-1],
        historical_financials=rows[:-1],
    )


def test_zero_terminal_growth_is_not_replaced_by_default_and_negative_growth_releases_capital():
    req = request()
    req.assumptions = AssumptionInputs(
        revenue_growth=[D("0")] * 10,
        ebit_margin=[D("0.2")] * 10,
        terminal_growth=D("0"),
    )
    resolved = FinanceTeamModel().resolve_assumptions(req, req.financials)
    assert resolved.terminal_growth == D("0")
    terminal = normalized_terminal_cash_flow(D("100"), D("-0.02"), D("0.10"))
    assert terminal.nopat == D("98")
    assert terminal.reinvestment_rate == D("-0.2")
    assert terminal.reinvestment == D("-19.6")
    assert terminal.fcff == D("117.6")


def test_wacc_counts_capitalized_lease_debt_once():
    req = request()
    req.assumptions = AssumptionInputs(market_cap=D("150"))
    base = req.financials.model_copy(update={
        "interest_bearing_debt": D("100"),
        "lease_liabilities": D("50"),
        "interest_bearing_debt_includes_leases": False,
    })
    _, components, _ = FinanceTeamModel._wacc(req, base, D("1"), D("0.25"))
    assert components["debt_weight"] == D("0.5")
    included = base.model_copy(update={"interest_bearing_debt_includes_leases": True})
    _, included_components, _ = FinanceTeamModel._wacc(req, included, D("1"), D("0.25"))
    assert included_components["debt_weight"] == D("0.4")


def test_loss_forecast_does_not_assume_immediate_cash_tax_refund():
    req = request()
    req.assumptions = AssumptionInputs(
        revenue_growth=[D("0")] * 10,
        ebit_margin=[D("-0.1")] * 10,
        terminal_growth=D("0"),
    )
    model = FinanceTeamModel()
    resolved = model.resolve_assumptions(req, req.financials)
    forecast = model.forecast(req, req.financials, resolved)
    assert forecast[0].ebit < 0
    assert forecast[0].nopat == forecast[0].ebit


def test_registry_is_versioned_and_financial_industry_is_out_of_scope():
    registry = IndustryParameterRegistry()
    row = registry.resolve("申万电子/半导体")
    assert row.domestic_growth == D("0.0639")
    assert row.decay_years == 8
    assert row.rnd_wage_factor == D("1.14")
    assert len(registry.source_sha256) == 64
    with pytest.raises(FinancialIndustryUnsupported):
        registry.resolve("银行")

    liquor = registry.resolve("食品饮料 / 白酒")
    assert liquor.industry_id == "consumer_staples_fallback"
    assert liquor.quality == "C"
    assert liquor.metadata_completeness == "fallback"
    assert "缺少食品饮料/白酒专属行" in liquor.provenance_note


def test_consumer_staples_fallback_is_explicitly_warned():
    model = FinanceTeamModel()
    req = request("食品饮料 / 白酒")

    findings = model.validate(req, req.financials)

    assert any(item.rule_id == "INDUSTRY_PARAMETER_FALLBACK" for item in findings)


def test_evidence_coverage_counts_fields_required_by_selected_methods():
    rows = [
        row.model_copy(update={
            "evidence": {
                "revenue": [EvidenceRef(
                    evidence_id=f"revenue-{row.period_end.year}",
                    source="audited annual report",
                )]
            }
        })
        for row in history()
    ]
    req = request()
    req.historical_financials = rows[:-1]
    req.financials = rows[-1]

    quality = FinanceTeamModel().assess_quality(req, req.financials, [], [])

    # DCF requires nine fields; unused NI and EBITDA must not dilute coverage.
    assert quality.evidence_coverage == D("0.1111")


def test_revenue_model_uses_ten_years_and_terminal_constraint():
    registry = IndustryParameterRegistry()
    projection = FinanceTeamRevenueModel().project(
        history(), registry.resolve("医药生物"), D("0.03"), 10
    )
    assert len(projection.annual_growth_history) == 9
    assert projection.effective_decay_years == 9
    assert projection.scenarios["base"][-1] == D("0.03")
    assert any("T_eff=9" in item for item in projection.decisions)


def test_formal_model_builds_traceable_fcff_dcf_and_sensitivity():
    model = FinanceTeamModel()
    req = request()
    financials = req.financials
    findings = model.validate(req, financials)
    assert not [item for item in findings if item.severity == "blocking"]
    assumptions = model.resolve_assumptions(req, financials)
    assert assumptions.industry_parameters["industry_id"] == "electronics"
    assert set(assumptions.revenue_growth_scenarios) == {
        "pessimistic",
        "base",
        "optimistic",
    }
    forecast = model.forecast(req, financials, assumptions)
    assert len(forecast) == 10
    assert forecast[-1].revenue_growth == D("0.0300")
    assert forecast[-1].calculation_methods["depreciation_amortization"] == (
        "revenue_ratio_fallback"
    )
    result = model.dcf(req, financials, assumptions, forecast)
    assert result.present_value_explicit > 0
    assert result.terminal_value > 0
    assert "minority_interest" in result.bridge_unmeasured_items
    assert result.range_low <= result.per_share_value <= result.range_high
    assert set(result.scenario_values) == {"pessimistic", "base", "optimistic"}
    assert len(model.sensitivity(req, financials, assumptions)) == 9


def test_relative_methods_remain_independent_and_ps_is_supported():
    model = FinanceTeamModel()
    req = request()
    req.methods = ["dcf", "ps"]
    peers = [
        PeerCompany(ticker="A", name="A", ps="3"),
        PeerCompany(ticker="B", name="B", ps="4"),
        PeerCompany(ticker="C", name="C", ps="5"),
    ]
    relative = model.relative(req, req.financials, peers)
    assert relative[0].method == "ps"
    dcf = model.dcf(
        req,
        req.financials,
        model.resolve_assumptions(req, req.financials),
        model.forecast(
            req,
            req.financials,
            model.resolve_assumptions(req, req.financials),
        ),
    )
    compared = model.reconcile(dcf, relative)
    assert compared.combined_range is None
    assert compared.overlap_range == (
        max(compared.dcf_range[0], compared.relative_range[0]),
        min(compared.dcf_range[1], compared.relative_range[1]),
    )
    assert compared.method_comparison["policy"] == (
        "dcf_primary_relative_cross_check_no_mechanical_average"
    )
    assert relative[0].sample_quality == "limited"
    assert relative[0].statistic == "P25/P50/P75"


def test_formal_validation_rejects_wrong_horizon_and_financial_sector():
    model = FinanceTeamModel()
    bad = request("银行")
    bad.forecast_years = 5
    rules = {item.rule_id for item in model.validate(bad, bad.financials)}
    assert "FINANCIAL_INDUSTRY_OUT_OF_SCOPE" in rules


def test_manual_forecast_discloses_short_history_and_wacc_quality_range():
    model = FinanceTeamModel()
    req = request()
    req.historical_financials = []
    req.assumptions.revenue_growth = [D("0.08")] * 9 + [D("0.03")]
    req.assumptions.ebit_margin = [D("0.20")] * 10

    findings = model.validate(req, req.financials)
    rules = {item.rule_id for item in findings}
    assert "HISTORY_SHORT_WITH_MANUAL_FORECAST" in rules
    assert "REVENUE_HISTORY_SHORT" not in rules

    assumptions = model.resolve_assumptions(req, req.financials)
    forecast = model.forecast(req, req.financials, assumptions)
    dcf = model.dcf(req, req.financials, assumptions, forecast)

    assert len(set(dcf.scenario_values.values())) == 1
    assert dcf.range_low < dcf.per_share_value < dcf.range_high
    assert any("经营情景区间退化为单点" in item for item in dcf.scenario_warnings)
    assert any("WACC输入未完全核验" in item for item in dcf.scenario_warnings)


def test_manual_three_scenario_paths_produce_a_real_valuation_range():
    model = FinanceTeamModel()
    req = request()
    req.historical_financials = []
    req.assumption_source = "manual"
    req.assumptions = AssumptionInputs(
        revenue_growth_scenarios={
            "pessimistic": [D("0.03")] * 10,
            "base": [D("0.06")] * 9 + [D("0.03")],
            "optimistic": [D("0.09")] * 9 + [D("0.03")],
        },
        ebit_margin_scenarios={
            "pessimistic": [D("0.18")] * 10,
            "base": [D("0.22")] * 10,
            "optimistic": [D("0.26")] * 10,
        },
        wacc=D("0.09"),
        terminal_growth=D("0.03"),
    )

    findings = model.validate(req, req.financials)
    assert not any(item.rule_id == "REVENUE_HISTORY_MINIMUM" for item in findings)
    assumptions = model.resolve_assumptions(req, req.financials)
    dcf = model.dcf(
        req, req.financials, assumptions,
        model.forecast(req, req.financials, assumptions),
    )

    assert assumptions.revenue_growth == assumptions.revenue_growth_scenarios["base"]
    assert assumptions.ebit_margin == assumptions.ebit_margin_scenarios["base"]
    assert dcf.range_low < dcf.per_share_value < dcf.range_high
    assert not any("区间已退化为单点" in item for item in dcf.scenario_warnings)


def test_manual_scenario_inputs_require_all_three_named_paths():
    with pytest.raises(ValueError, match="pessimistic, base and optimistic"):
        AssumptionInputs(revenue_growth_scenarios={"base": [D("0.05")]})


def test_wacc_only_revision_preserves_manual_scenario_paths(tmp_path):
    req = request()
    req.assumption_source = "manual"
    req.assumptions = AssumptionInputs(
        revenue_growth_scenarios={
            "pessimistic": [D("0.03")] * 10,
            "base": [D("0.06")] * 9 + [D("0.03")],
            "optimistic": [D("0.09")] * 9 + [D("0.03")],
        },
        ebit_margin_scenarios={
            "pessimistic": [D("0.18")] * 10,
            "base": [D("0.22")] * 10,
            "optimistic": [D("0.26")] * 10,
        },
        wacc=D("0.095"), terminal_growth=D("0.03"),
    )
    runner = ValuationRunner(SQLiteRunStore(tmp_path / "scenario-revision"), FinanceTeamModel())
    parent = runner.run(req)

    child = runner.revise(
        parent.run_id,
        RevisionInput(changes={"assumptions": {"wacc": "0.09"}}, reason="WACC复核"),
    )

    assert child.request.assumptions.revenue_growth_scenarios == req.assumptions.revenue_growth_scenarios
    assert child.request.assumptions.ebit_margin_scenarios == req.assumptions.ebit_margin_scenarios


def test_formal_model_runs_through_audited_agent_workflow(tmp_path):
    store = SQLiteRunStore(tmp_path / "formal-model")
    record = ValuationRunner(store, FinanceTeamModel()).run(request())

    assert record.result is not None
    assert record.result.model_version == "1.4.0-finance-team-production-20260927"
    assert record.result.data_quality.confidence in {"low", "medium", "high"}
    assert record.result.assumptions.industry_parameters["industry_id"] == "electronics"
    assert len(record.result.forecast) == 10
    events = store.list_events(record.run_id)
    assert any(event.type == "industry.parameters.resolved" for event in events)
    assert any(
        event.tool == "lookup_industry_parameters" and event.type == "tool.completed"
        for event in events
    )


def _history_with_operating_drivers():
    rows = history()
    enriched = []
    for row in rows:
        revenue = row.revenue
        items = dict(row.statement_items)
        items.update({
            "fixed_assets_net": revenue * D("0.30"),
            "intangible_assets": revenue * D("0.05"),
            "long_term_deferred_expenses": revenue * D("0.02"),
            "depreciation_fixed_assets": revenue * D("0.025"),
            "amortization_intangibles": revenue * D("0.007"),
            "amortization_long_term_deferred": revenue * D("0.003"),
            "accounts_receivable": revenue * D("0.12"),
            "inventory": revenue * D("0.15"),
            "accounts_payable": revenue * D("0.10"),
            "operating_cost": revenue * D("0.62"),
            "taxes_and_surcharges": revenue * D("0.01"),
            "selling_expense": revenue * D("0.03"),
            "administrative_expense": revenue * D("0.05"),
            "research_expense": revenue * D("0.08"),
            "impairment_loss": revenue * D("0.01"),
            "other_income": revenue * D("0.02"),
            "operating_nwc": revenue * D("0.17"),
            "quarterly_average_market_cap": D("72000000000"),
            "annual_average_market_cap": D("71500000000"),
            "market_cap_period_low": D("60000000000"),
            "market_cap_period_high": D("82000000000"),
        })
        if row.period_end.year >= 2019:
            items.update({
                "right_of_use_assets": revenue * D("0.03"),
                "depreciation_right_of_use": revenue * D("0.002"),
            })
        enriched.append(row.model_copy(update={"statement_items": items}))
    return enriched


def test_document_formulas_drive_da_capex_nwc_exit_check_and_catalogue():
    model = FinanceTeamModel()
    rows = _history_with_operating_drivers()
    req = request()
    req.financials = rows[-1]
    req.historical_financials = rows[:-1]

    findings = model.validate(req, req.financials)
    assert "DA_REVENUE_RATIO_FALLBACK" not in {item.rule_id for item in findings}
    assumptions = model.resolve_assumptions(req, req.financials)
    assert assumptions.calculation_methods["capex"] == "alpha_da_plus_kappa_delta_revenue"
    assert assumptions.calculation_methods["change_operating_nwc"] == "dso_dio_dpo"
    assert assumptions.calculation_methods["ebit"] == "operating_component_build"
    assert assumptions.wacc_components["equity_weight"] > D("0.99")

    forecast = model.forecast(req, req.financials, assumptions)
    assert forecast[0].calculation_methods["depreciation_amortization"] == "asset_rollforward"
    assert forecast[0].calculation_methods["change_operating_nwc"] == "dso_dio_dpo"
    assert forecast[0].calculation_methods["ebit"] == "operating_component_build"
    assert set(forecast[0].operating_items) >= {
        "operating_cost", "selling_expense", "research_expense", "other_income"
    }
    assert forecast[-1].operating_items["depreciation_right_of_use"] > 0
    assert (
        forecast[-1].capital_expenditure
        == forecast[-1].operating_items["cash_asset_depreciation"]
    )
    assert forecast[-1].capital_expenditure < forecast[-1].depreciation_amortization

    dcf = model.dcf(req, req.financials, assumptions, forecast)
    assert dcf.exit_multiple_cross_check is not None
    assert dcf.exit_multiple_per_share is not None
    assert dcf.terminal_method_gap is not None
    assert "operating_cash_requirement" in dcf.bridge

    studies = model.sensitivity_studies(req, req.financials, assumptions)
    assert {f"S{number}" for number in range(1, 21)} <= {
        item.study_id.split("-")[0] for item in studies
    }
    assert next(item for item in studies if item.study_id == "S3").status == "completed"
    assert next(item for item in studies if item.study_id == "S5").status == "completed"
    assert next(item for item in studies if item.study_id == "S10").status == "completed"
    assert next(item for item in studies if item.study_id == "S11").status == "completed"
    assert next(item for item in studies if item.study_id == "S19").status == "completed"
    assert next(item for item in studies if item.study_id == "S8").status == "not_available"


def test_missing_cash_flow_drivers_degrade_but_still_produce_auditable_dcf():
    model = FinanceTeamModel()
    req = request()
    req.historical_financials = [
        row.model_copy(update={
            "depreciation_amortization": None,
            "capital_expenditure": None,
            "change_operating_nwc": None,
            "statement_items": {
                key: value
                for key, value in row.statement_items.items()
                if key != "operating_nwc"
            },
        })
        for row in req.historical_financials
    ]
    req.financials = req.financials.model_copy(update={
        "depreciation_amortization": None,
        "capital_expenditure": None,
        "change_operating_nwc": None,
        "statement_items": {
            key: value
            for key, value in req.financials.statement_items.items()
            if key != "operating_nwc"
        },
    })

    findings = model.validate(req, req.financials)
    assert not [item for item in findings if item.severity == "blocking"]
    rules = {item.rule_id for item in findings}
    assert "HISTORY_DRIVER_DEGRADED_DEPRECIATION_AMORTIZATION" in rules
    assert "HISTORY_DRIVER_DEGRADED_CAPITAL_EXPENDITURE" in rules
    assert "HISTORY_DRIVER_DEGRADED_CHANGE_OPERATING_NWC" in rules

    assumptions = model.resolve_assumptions(req, req.financials)
    forecast = model.forecast(req, req.financials, assumptions)
    dcf = model.dcf(req, req.financials, assumptions, forecast)
    quality = model.assess_quality(req, req.financials, [], [])

    assert assumptions.calculation_methods["capex"] == "revenue_ratio_fallback"
    assert assumptions.calculation_methods["change_operating_nwc"] == "policy_delta_revenue_fallback"
    assert all(row.depreciation_amortization > 0 for row in forecast)
    assert all(row.capital_expenditure > 0 for row in forecast)
    assert dcf.per_share_value.is_finite()
    assert quality.result_grade == "C"
    assert quality.degraded_fields == [
        "capital_expenditure",
        "change_operating_nwc",
        "depreciation_amortization",
        "operating_cash_requirement",
        "stable_roic",
    ]


def test_structurally_missing_first_delta_nwc_does_not_force_policy_grade():
    model = FinanceTeamModel()
    req = request()
    req.historical_financials[0] = req.historical_financials[0].model_copy(
        update={"change_operating_nwc": None}
    )

    quality = model.assess_quality(req, req.financials, [], [])

    assert "change_operating_nwc" not in quality.degraded_fields


def test_terminal_value_requires_growth_reinvestment_and_reconciles():
    model = FinanceTeamModel()
    req = request()
    assumptions = model.resolve_assumptions(req, req.financials)
    forecast = model.forecast(req, req.financials, assumptions)
    dcf = model.dcf(req, req.financials, assumptions, forecast)

    expected_rate = assumptions.terminal_growth / dcf.stable_roic
    expected_nopat = forecast[-1].nopat * (D(1) + assumptions.terminal_growth)
    expected_reinvestment = expected_nopat * expected_rate
    expected_fcff = expected_nopat - expected_reinvestment

    assert abs(dcf.terminal_reinvestment_rate - expected_rate) < D("0.0001")
    assert abs(dcf.terminal_nopat - expected_nopat) < D("1")
    assert abs(dcf.terminal_reinvestment - expected_reinvestment) < D("1")
    assert abs(dcf.terminal_fcff - expected_fcff) < D("1")
    assert abs(
        dcf.terminal_value
        - dcf.terminal_fcff / (assumptions.wacc - assumptions.terminal_growth)
    ) < D("1")


def test_complete_equity_bridge_uses_leases_claims_associates_and_diluted_shares():
    model = FinanceTeamModel()
    req = request()
    req.financials = req.financials.model_copy(update={
        "cash_and_non_operating_assets": D("1000"),
        "interest_bearing_debt": D("400"),
        "lease_liabilities": D("50"),
        "interest_bearing_debt_includes_leases": False,
        "minority_interest": D("40"),
        "preferred_equity": D("30"),
        "associates_and_non_operating_investments": D("70"),
        "unfunded_pension": D("20"),
        "non_operating_provisions": D("10"),
        "common_shares": D("100"),
        "diluted_shares": D("110"),
    })
    assumptions = model.resolve_assumptions(req, req.financials)
    forecast = model.forecast(req, req.financials, assumptions)
    dcf = model.dcf(req, req.financials, assumptions, forecast)

    expected_equity = dcf.enterprise_value + D("1000") + D("70") - D("450") - D("100")
    assert abs(dcf.equity_value - expected_equity) < D("0.01")
    assert abs(dcf.per_share_value - expected_equity / D("110")) < D("0.0001")
    assert dcf.bridge["total_debt_including_incremental_leases"] == D("-450.0000")
    assert dcf.bridge["diluted_or_common_shares"] == D("110.0000")
    assert dcf.bridge_unmeasured_items == []
    assert any("账面值作为市场价值代理" in item for item in dcf.scenario_warnings)


def test_user_stable_roic_must_exceed_terminal_growth():
    req = request()
    req.assumptions = AssumptionInputs(
        terminal_growth=D("0.04"), stable_roic=D("0.03")
    )
    findings = FinanceTeamModel().validate(req, req.financials)
    assert "STABLE_ROIC_NOT_ABOVE_GROWTH" in {
        item.rule_id for item in findings if item.severity == "blocking"
    }

    req.assumptions = AssumptionInputs(stable_roic=D("0.02"))
    findings = FinanceTeamModel().validate(req, req.financials)
    assert "STABLE_ROIC_NOT_ABOVE_GROWTH" in {
        item.rule_id for item in findings if item.severity == "blocking"
    }


def test_manual_wacc_must_exceed_effective_terminal_growth():
    req = request()
    req.assumptions = AssumptionInputs(wacc=D("0.02"))
    findings = FinanceTeamModel().validate(req, req.financials)
    assert "WACC_NOT_ABOVE_TERMINAL_GROWTH" in {
        item.rule_id for item in findings if item.severity == "blocking"
    }


def test_sensitivity_marks_growth_above_stable_roic_invalid_instead_of_crashing():
    req = request()
    req.assumptions = AssumptionInputs(
        wacc=D("0.09"),
        terminal_growth=D("0.03"),
        stable_roic=D("0.035"),
    )
    model = FinanceTeamModel()
    assumptions = model.resolve_assumptions(req, req.financials)

    cells = model.sensitivity(req, req.financials, assumptions)

    assert [cell for cell in cells if cell.terminal_growth == D("0.0400")]
    assert all(
        not cell.valid
        for cell in cells
        if cell.terminal_growth == D("0.0400")
    )


def test_strict_bridge_policy_rejects_unverified_book_value_proxy():
    req = request()
    req.assumptions.equity_bridge_policy = "require_market_values"
    req.financials = req.financials.model_copy(update={
        "minority_interest": D("100"),
    })
    model = FinanceTeamModel()
    assumptions = model.resolve_assumptions(req, req.financials)
    forecast = model.forecast(req, req.financials, assumptions)

    with pytest.raises(ValueError, match="minority_interest_market_value"):
        model.dcf(req, req.financials, assumptions, forecast)


def test_bridge_rejects_conflicting_top_level_and_statement_item_values():
    req = request()
    req.financials = req.financials.model_copy(update={
        "minority_interest": D("100"),
        "statement_items": {
            **req.financials.statement_items,
            "minority_interest": D("90"),
        },
    })
    model = FinanceTeamModel()
    assumptions = model.resolve_assumptions(req, req.financials)
    forecast = model.forecast(req, req.financials, assumptions)

    with pytest.raises(ValueError, match="不一致"):
        model.dcf(req, req.financials, assumptions, forecast)


def test_reference_compatibility_accepts_diluted_share_denominator():
    req = request(industry=None)
    original_shares = req.financials.common_shares
    req.financials = req.financials.model_copy(update={
        "common_shares": None,
        "diluted_shares": original_shares * D("1.1"),
    })
    model = FinanceTeamModel()
    findings = model.validate(req, req.financials)
    assert not [item for item in findings if item.severity == "blocking"]
    assumptions = model.resolve_assumptions(req, req.financials)
    forecast = model.forecast(req, req.financials, assumptions)
    result = model.dcf(req, req.financials, assumptions, forecast)
    assert result.per_share_value.is_finite()


def test_comparability_controls_block_review_years_and_exclude_marked_years():
    model = FinanceTeamModel()
    req = request()
    req.historical_financials[2] = req.historical_financials[2].model_copy(update={
        "comparability_status": "review_required",
        "comparability_note": "收入确认政策发生变化，尚未完成重述。",
    })
    findings = model.validate(req, req.financials)
    assert "HISTORY_COMPARABILITY_REVIEW" in {
        item.rule_id for item in findings if item.severity == "blocking"
    }

    req.historical_financials[2] = req.historical_financials[2].model_copy(update={
        "comparability_status": "excluded",
        "comparability_note": "已确认不可比并从统计窗口排除。",
    })
    assert len(model._history(req, req.financials)) == 9
    findings = model.validate(req, req.financials)
    assert "HISTORY_COMPARABILITY_REVIEW" not in {
        item.rule_id for item in findings
    }


@pytest.mark.parametrize(
    ("revenues", "expected_cycle"),
    [
        ([10, 11, 12, 13, 14, 15, 16, 18, 21, 25], "up"),
        ([10, 12, 14, 16, 18, 20, 22, 20, 19, 18], "down"),
    ],
)
def test_revenue_scenarios_preserve_cycle_identity_without_sorting(
    revenues, expected_cycle
):
    rows = history()
    rows = [
        row.model_copy(update={"revenue": D(value) * D("100000000")})
        for row, value in zip(rows, revenues)
    ]
    projection = FinanceTeamRevenueModel().project(
        rows, IndustryParameterRegistry().resolve("电子"), D("0.03"), 10
    )
    assert projection.cycle == expected_cycle
    assert projection.anchors["pessimistic"] <= projection.anchors["base"]
    assert projection.anchors["base"] <= projection.anchors["optimistic"]
    assert projection.scenarios["pessimistic"][0] <= projection.scenarios["base"][0]
    assert projection.scenarios["base"][0] <= projection.scenarios["optimistic"][0]
    expected_base = projection.center + projection.rho * (
        projection.cagr - projection.center
    )
    assert projection.scenarios["base"][0] == expected_base
    assert any("不通过排序改写情景身份" in item for item in projection.decisions)


def test_workforce_expense_model_and_its_sensitivities_activate_only_with_full_inputs():
    rows = _history_with_operating_drivers()
    latest = rows[-1]
    items = dict(latest.statement_items)
    for prefix, headcount in (("research", 500), ("administrative", 120)):
        items.update({
            f"{prefix}_headcount": D(headcount),
            f"{prefix}_average_cost": D("500000"),
            f"{prefix}_non_labor_ratio": D("0.4"),
            f"{prefix}_headcount_growth_center": D("0.04"),
            f"{prefix}_lifecycle_factor": D("1"),
            f"{prefix}_wage_premium": D("0.01"),
        })
    items["workforce_ramp_years"] = D(5)
    rows[-1] = latest.model_copy(update={"statement_items": items})
    req = request()
    req.financials = rows[-1]
    req.historical_financials = rows[:-1]
    model = FinanceTeamModel()
    assumptions = model.resolve_assumptions(req, req.financials)

    assert assumptions.calculation_methods["research_expense"] == "workforce_cost_model_l1_l3"
    forecast = model.forecast(req, req.financials, assumptions)
    assert forecast[0].operating_items["research_expense"] > 0
    studies = {item.study_id: item for item in model.sensitivity_studies(
        req, req.financials, assumptions
    )}
    assert studies["S8"].status == "completed"
    assert studies["S13"].status == "completed"
    assert studies["S14"].status == "completed"
