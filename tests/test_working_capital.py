from datetime import date
from decimal import Decimal

import pytest

from valuationagent.application.input_balances import derive_balance_changes, verify_balance_changes
from valuationagent.finance.team_model import FinanceTeamModel
from valuationagent.finance.working_capital import TURNOVER_METHOD, working_capital_drivers
from valuationagent.schemas.models import AssumptionInputs, CompanyInput, FinancialSnapshot, ValuationRequest


def request_with_balances():
    rows = [FinancialSnapshot(period_end=date(year, 12, 31), revenue=1000, ebit_margin=Decimal("0.2"),
        tax_rate=Decimal("0.25"), depreciation_amortization=30, capital_expenditure=40,
        change_operating_nwc=20, cash_and_non_operating_assets=50, interest_bearing_debt=0, common_shares=10,
        statement_items={"operating_nwc": Decimal(-100), "accounts_receivable": Decimal(120),
            "inventory": Decimal(150), "accounts_payable": Decimal(120), "operating_cost": Decimal(600)})
        for year in range(2022, 2026)]
    return ValuationRequest(company=CompanyInput(name="合成营运资本情景", industry="电子"), valuation_date=date(2026, 10, 4),
        forecast_years=10, methods=["dcf"], analysis_basis="user_scenario", financials=rows[-1], historical_financials=rows[:-1],
        assumptions=AssumptionInputs(revenue_growth=[Decimal("0.1")] * 10, wacc=Decimal("0.1"), terminal_growth=Decimal("0.02")))


def test_full_negative_operating_balance_is_not_replaced_by_three_positive_components():
    request = request_with_balances()
    model = FinanceTeamModel()
    assumptions = model.resolve_assumptions(request, request.financials)
    assert assumptions.calculation_methods["change_operating_nwc"] == "operating_nwc_revenue_ratio"
    forecast = model.forecast(request, request.financials, assumptions)
    assert forecast[0].operating_items["operating_nwc"] == -110
    assert forecast[0].change_operating_nwc == -10
    previous = Decimal(-100)
    for row in forecast:
        assert abs(row.change_operating_nwc - (row.operating_items["operating_nwc"] - previous)) <= Decimal("0.0001")
        previous = row.operating_items["operating_nwc"]


def test_manual_ending_balance_days_preserve_other_operating_liabilities():
    request = request_with_balances()
    request.assumptions.dso_days = Decimal("43.8")
    request.assumptions.dio_days = Decimal("91.25")
    request.assumptions.dpo_days = Decimal("73")
    request.assumptions.operating_cost_ratio = Decimal("0.6")
    model = FinanceTeamModel()
    assumptions = model.resolve_assumptions(request, request.financials)
    assert assumptions.calculation_methods["change_operating_nwc"] == TURNOVER_METHOD
    assert assumptions.operating_drivers["other_operating_nwc_ratio"] == Decimal("-0.25")
    forecast = model.forecast(request, request.financials, assumptions)
    assert forecast[0].operating_items["other_operating_nwc"] == -275
    assert forecast[0].operating_items["operating_nwc"] == -110
    assert forecast[0].change_operating_nwc == -10
    request.financials.statement_items.pop("inventory")
    with pytest.raises(ValueError, match="NWC_SCOPE_BASELINE"):
        model.resolve_assumptions(request, request.financials)
    assert "NWC_SCOPE_BASELINE" in {finding.rule_id for finding in model.validate(request, request.financials)}


def test_three_component_forecast_declares_partial_scope_and_anchors_actual_opening():
    request = request_with_balances()
    for row in [*request.historical_financials, request.financials]:
        row.statement_items.pop("operating_nwc")
    request.financials.statement_items["accounts_receivable"] = Decimal(180)
    model = FinanceTeamModel()
    assumptions = model.resolve_assumptions(request, request.financials)
    assert assumptions.operating_drivers["nwc_scope_partial"] == 1
    first = model.forecast(request, request.financials, assumptions)[0]
    assert first.operating_items["operating_nwc"] == 165
    assert first.change_operating_nwc == -45
    request.analysis_basis = "research"
    quality = model.assess_quality(request, request.financials, [], [], assumptions)
    assert "change_operating_nwc" in quality.degraded_fields


def test_flow_to_revenue_fallback_multiplies_revenue_not_zero_revenue_growth():
    request = request_with_balances()
    request.assumptions.revenue_growth = [Decimal(0)] * 10
    for row in [*request.historical_financials, request.financials]:
        row.statement_items.clear()
    model = FinanceTeamModel()
    assumptions = model.resolve_assumptions(request, request.financials)
    assert assumptions.operating_drivers["nwc_change_revenue_ratio"] == Decimal("0.02")
    forecast = model.forecast(request, request.financials, assumptions)
    assert all(row.change_operating_nwc == 20 for row in forecast)


def test_manual_days_are_not_silently_discarded_and_marginal_factor_not_clipped():
    request = request_with_balances()
    request.assumptions.dso_days = Decimal(30)
    with pytest.raises(ValueError, match="NWC_DRIVER_PARTIAL"):
        working_capital_drivers(request, [*request.historical_financials, request.financials])
    request.assumptions.dso_days = None
    for row in [*request.historical_financials, request.financials]:
        row.statement_items.clear()
    request.financials.revenue = Decimal(1001)
    request.assumptions.revenue_growth = [Decimal("0.1")] * 10
    model = FinanceTeamModel()
    assumptions = model.resolve_assumptions(request, request.financials)
    assert assumptions.operating_drivers["nwc_delta_revenue_factor"] == 20
    assert model.forecast(request, request.financials, assumptions)[0].change_operating_nwc == 2002


@pytest.mark.parametrize("field,value", [("period_end", date(2020, 12, 31)), ("period_end", date(2024, 6, 30)),
    ("currency", "USD"), ("statement_scope", "parent"), ("comparability_status", "review_required")])
def test_balance_changes_do_not_cross_gaps_scopes_or_unreviewed_restatements(field, value):
    request = request_with_balances()
    previous, current = request.historical_financials[-1], request.financials
    previous = previous.model_copy(update={field: value})
    current.change_operating_nwc = None
    assert derive_balance_changes([previous, current])[-1].change_operating_nwc is None


def test_balance_change_preserves_first_year_and_explicit_cashflow_adjustment():
    request = request_with_balances()
    rows = [*request.historical_financials, request.financials]
    rows[0].change_operating_nwc = None
    rows[1].change_operating_nwc = None
    rows[1].statement_items["operating_nwc"] = Decimal(-120)
    derived = derive_balance_changes(rows)
    assert derived[0].change_operating_nwc is None
    assert derived[1].change_operating_nwc == -20
    assert derived[2].change_operating_nwc == 20
    assert rows[1].change_operating_nwc is None
    request.historical_financials, request.financials = derived[:-1], derived[-1]
    assert verify_balance_changes(request)
    request.historical_financials[1].change_operating_nwc += 1
    with pytest.raises(ValueError, match="INPUT_BALANCE_REPLAY"):
        verify_balance_changes(request)
