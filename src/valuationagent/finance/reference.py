from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from datetime import date

from valuationagent.schemas.models import (
    AssumptionSet,
    DcfResult,
    FinancialSnapshot,
    ForecastYear,
    MultipleResult,
    PeerCompany,
    ReconciliationResult,
    SensitivityCell,
    ValidationFinding,
    ValuationMethod,
    ValuationRequest,
)


D = Decimal
FOUR_PLACES = D("0.0001")


def _q(value: Decimal) -> Decimal:
    return value.quantize(FOUR_PLACES, rounding=ROUND_HALF_UP)


def _percentile(values: list[Decimal], q: Decimal) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    if len(ordered) == 1:
        return ordered[0]
    position = D(len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - D(lower)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class ReferenceFinancialModel:
    """Transparent integration model. A finance reviewer should replace or approve it."""

    plugin_id = "reference_fcff_relative"
    version = "0.3.0-reference"
    supports_incremental_inputs = True

    @staticmethod
    def demo_financials() -> FinancialSnapshot:
        # Compatibility helper; workflow data comes from the injected DataProvider.
        from valuationagent.core.data import demo_financials

        return demo_financials()

    @staticmethod
    def demo_peers() -> list[PeerCompany]:
        from valuationagent.core.data import demo_peers

        return demo_peers()

    def validate(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        *,
        allow_dcf_driver_degradation: bool = False,
    ) -> list[ValidationFinding]:
        from valuationagent.schemas.models import required_financial_metrics
        findings: list[ValidationFinding] = []
        required = required_financial_metrics(request.methods)
        if allow_dcf_driver_degradation and "dcf" in request.methods:
            # The finance-team implementation has deterministic, disclosed
            # fallbacks for these three cash-flow drivers.  The reference model
            # remains strict because it does not implement those fallbacks.
            required -= {
                "depreciation_amortization",
                "capital_expenditure",
                "change_operating_nwc",
            }
        missing = sorted(
            key
            for key in required
            if getattr(financials, key) is None
            and not (
                key == "common_shares" and financials.diluted_shares is not None
            )
        )
        if missing:
            return [ValidationFinding(rule_id="METHOD_INPUTS_MISSING", severity="blocking",
                                      message="所选估值方法缺少已确认字段：" + "、".join(missing))]
        if financials.published_at and financials.published_at > request.valuation_date:
            findings.append(
                ValidationFinding(
                    rule_id="PUBLICATION_DATE",
                    severity="blocking",
                    message="资料公告日晚于估值基准日，请提供当时已公开的数据。",
                )
            )
        if (
            financials.currency != request.company.currency
            or financials.statement_scope != "consolidated"
        ):
            findings.append(
                ValidationFinding(
                    rule_id="FINANCIAL_SCOPE",
                    severity="blocking",
                    message="参考模型要求币种一致的合并报表，请先复核币种和口径。",
                )
            )
        if "dcf" in request.methods and (
            financials.period_end.month != 12
            or financials.period_end.day != 31
            or request.valuation_date.year > financials.period_end.year + 1
        ):
            findings.append(
                ValidationFinding(
                    rule_id="FORECAST_BASE_PERIOD",
                    severity="blocking",
                    message="参考模型需要最近年末基期；更早或非年末数据需要金融团队提供期间衔接模型。",
                )
            )
        items = financials.statement_items
        for rule, fields in [
            ("BALANCE_SHEET", ("total_assets", "total_liabilities", "total_equity")),
            ("CASH_FLOW", ("cash_end", "cash_begin", "net_cash_change")),
        ]:
            if all(k in items for k in fields):
                actual, expected = items[fields[0]], items[fields[1]] + items[fields[2]]
                if abs(actual - expected) > D("0.01"):
                    findings.append(
                        ValidationFinding(
                            rule_id=rule,
                            severity="blocking",
                            message=f"{rule} 勾稽不平衡，请更正字段或单位。",
                            actual=actual,
                            expected=expected,
                        )
                    )
        if financials.period_end and financials.period_end > request.valuation_date:
            findings.append(
                ValidationFinding(
                    rule_id="POINT_IN_TIME_001",
                    severity="blocking",
                    message="财务数据期末日晚于估值基准日，可能存在未来信息泄漏。",
                    actual=str(financials.period_end),
                    expected=f"<= {request.valuation_date}",
                    recommended_action="更换估值日或使用当时已公开的数据。",
                )
            )
        if (
            "ev_ebitda" in request.methods
            and financials.ebitda is not None
            and financials.ebitda <= 0
        ):
            findings.append(
                ValidationFinding(
                    rule_id="MULTIPLE_APPLICABILITY_001",
                    severity="warning",
                    message="EBITDA 非正，EV/EBITDA 相对估值将不适用。",
                )
            )
        if (
            "pe" in request.methods
            and financials.net_income_parent is not None
            and financials.net_income_parent <= 0
        ):
            findings.append(
                ValidationFinding(
                    rule_id="MULTIPLE_APPLICABILITY_002",
                    severity="warning",
                    message="归母净利润非正，P/E 相对估值将不适用。",
                )
            )
        if "dcf" not in request.methods:
            return findings
        reinvestment_inputs = (
            financials.capital_expenditure,
            financials.depreciation_amortization,
            financials.change_operating_nwc,
        )
        reinvestment = (
            financials.capital_expenditure
            - financials.depreciation_amortization
            + financials.change_operating_nwc
            if all(value is not None for value in reinvestment_inputs)
            else None
        )
        if reinvestment is not None and reinvestment < 0:
            findings.append(
                ValidationFinding(
                    rule_id="REINVESTMENT_001",
                    severity="warning",
                    message="基期净再投资为负，需要核对资本开支、折旧和营运资本口径。",
                    actual=_q(reinvestment),
                )
            )
        if financials.cash_and_non_operating_assets > financials.revenue:
            findings.append(
                ValidationFinding(
                    rule_id="CASH_CLASSIFICATION_001",
                    severity="warning",
                    message="现金及非经营资产超过年收入，需要复核受限资金及非经营资产分类。",
                )
            )
        return findings

    def resolve_assumptions(
        self, request: ValuationRequest, financials: FinancialSnapshot
    ) -> AssumptionSet:
        if "dcf" not in request.methods:
            return AssumptionSet(revenue_growth=[], ebit_margin=[], wacc=D(0), terminal_growth=D(0),
                                 source="not_applicable_relative_only",
                                 rationale={"scope": "仅进行相对估值，不构建现金流预测；WACC和永续增长率不适用。"})
        years = request.forecast_years
        provided_growth = request.assumptions.revenue_growth
        if provided_growth:
            growth = self._fit_series(provided_growth, years)
            growth_source = "user_input"
        else:
            start = D("0.08")
            end = D("0.04")
            growth = [
                start - (start - end) * D(i) / D(max(years - 1, 1))
                for i in range(years)
            ]
            growth_source = "reference_policy_default"

        provided_margin = request.assumptions.ebit_margin
        margins = (
            self._fit_series(provided_margin, years)
            if provided_margin
            else [financials.ebit_margin] * years
        )
        wacc = request.assumptions.wacc or D("0.095")
        terminal_growth = (
            request.assumptions.terminal_growth
            if request.assumptions.terminal_growth is not None
            else D("0.03")
        )
        if "dcf" in request.methods and terminal_growth >= wacc:
            raise ValueError("terminal growth must be lower than WACC")
        if any(item <= D("-0.95") or item >= D("1") for item in growth):
            raise ValueError("revenue growth assumption is outside the supported range")
        if any(item <= D("-1") or item >= D("1") for item in margins):
            raise ValueError("EBIT margin assumption is outside the supported range")
        source = (
            "user_override_with_reference_defaults"
            if any(
                value is not None
                for value in [
                    provided_growth,
                    provided_margin,
                    request.assumptions.wacc,
                    request.assumptions.terminal_growth,
                ]
            )
            else "reference_policy_default"
        )
        return AssumptionSet(
            revenue_growth=[_q(item) for item in growth],
            ebit_margin=[_q(item) for item in margins],
            wacc=_q(wacc),
            terminal_growth=_q(terminal_growth),
            source=source,
            rationale={
                "revenue_growth": f"{growth_source}; transparent convergence baseline for integration testing",
                "ebit_margin": "user input or base-period margin held constant",
                "wacc": "user input or 9.5% integration-test default",
                "terminal_growth": "user input or 3.0% integration-test default",
            },
        )

    @staticmethod
    def _fit_series(values: list[Decimal], years: int) -> list[Decimal]:
        fitted = list(values[:years])
        while len(fitted) < years:
            fitted.append(fitted[-1])
        return fitted

    def forecast(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        assumptions: AssumptionSet,
    ) -> list[ForecastYear]:
        revenue = financials.revenue
        da_ratio = financials.depreciation_amortization / financials.revenue
        capex_ratio = financials.capital_expenditure / financials.revenue
        nwc_ratio = financials.change_operating_nwc / financials.revenue
        rows: list[ForecastYear] = []
        for index in range(request.forecast_years):
            year = financials.period_end.year + index + 1
            year_start, year_end = date(year, 1, 1), date(year + 1, 1, 1)
            remaining_start = max(year_start, request.valuation_date)
            remaining = max(0, (year_end - remaining_start).days)
            fraction = D(remaining) / D((year_end - year_start).days)
            discount_period = (
                D((remaining_start - request.valuation_date).days) + D(remaining) / 2
            ) / D("365.25")
            if request.discount_policy == "year_end":
                fraction = D(1)
                discount_period = D(index + 1)
            growth = assumptions.revenue_growth[index]
            margin = assumptions.ebit_margin[index]
            revenue = revenue * (D(1) + growth)
            ebit = revenue * margin
            nopat = ebit * (D(1) - financials.tax_rate)
            da = revenue * da_ratio
            capex = revenue * capex_ratio
            nwc = revenue * nwc_ratio
            fcff = nopat + da - capex - nwc
            rows.append(
                ForecastYear(
                    year=year,
                    discount_period=discount_period,
                    cash_flow_fraction=fraction,
                    revenue=_q(revenue),
                    revenue_growth=_q(growth),
                    ebit_margin=_q(margin),
                    ebit=_q(ebit),
                    nopat=_q(nopat),
                    depreciation_amortization=_q(da),
                    capital_expenditure=_q(capex),
                    change_operating_nwc=_q(nwc),
                    fcff=_q(fcff),
                )
            )
        return rows

    def _dcf_from_forecast(
        self,
        financials: FinancialSnapshot,
        forecast: list[ForecastYear],
        wacc: Decimal,
        terminal_growth: Decimal,
        discount_policy: str = "annual_midyear_remaining",
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        if wacc <= 0 or terminal_growth >= wacc:
            raise ValueError("terminal growth must be lower than WACC")
        present_value = D(0)
        for row in forecast:
            present_value += (
                row.fcff
                * row.cash_flow_fraction
                / ((D(1) + wacc) ** row.discount_period)
            )
        terminal_fcff = forecast[-1].fcff * (D(1) + terminal_growth)
        terminal_value = terminal_fcff / (wacc - terminal_growth)
        terminal_period = (
            forecast[-1].discount_period + forecast[-1].cash_flow_fraction / 2
        )
        if discount_policy == "year_end":
            terminal_period = forecast[-1].discount_period
        discounted_terminal = terminal_value / ((D(1) + wacc) ** terminal_period)
        enterprise_value = present_value + discounted_terminal
        equity_value = (
            enterprise_value
            + financials.cash_and_non_operating_assets
            - financials.interest_bearing_debt
        )
        shares = financials.diluted_shares or financials.common_shares
        per_share = equity_value / shares
        terminal_share = (
            discounted_terminal / enterprise_value if enterprise_value else D(0)
        )
        return enterprise_value, equity_value, per_share, terminal_share

    def dcf(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        assumptions: AssumptionSet,
        forecast: list[ForecastYear],
    ) -> DcfResult:
        enterprise, equity, per_share, terminal_share = self._dcf_from_forecast(
            financials, forecast, assumptions.wacc, assumptions.terminal_growth, request.discount_policy
        )
        pessimistic = assumptions.model_copy(
            update={
                "revenue_growth": [
                    max(D("-0.5"), item - D("0.02"))
                    for item in assumptions.revenue_growth
                ],
                "ebit_margin": [item - D("0.01") for item in assumptions.ebit_margin],
                "wacc": assumptions.wacc + D("0.01"),
                "terminal_growth": assumptions.terminal_growth - D("0.005"),
            }
        )
        optimistic = assumptions.model_copy(
            update={
                "revenue_growth": [
                    item + D("0.02") for item in assumptions.revenue_growth
                ],
                "ebit_margin": [item + D("0.01") for item in assumptions.ebit_margin],
                "wacc": assumptions.wacc - D("0.01"),
                "terminal_growth": assumptions.terminal_growth + D("0.005"),
            }
        )
        values = [per_share]
        scenario_warnings = []
        for label, scenario in [("悲观", pessimistic), ("乐观", optimistic)]:
            valid = (
                D(0) < scenario.wacc < D("0.5")
                and D("-0.1") <= scenario.terminal_growth < min(scenario.wacc, D("0.2"))
                and all(D("-0.95") < x < D(1) for x in scenario.revenue_growth)
                and all(D(-1) < x < D(1) for x in scenario.ebit_margin)
            )
            if not valid:
                scenario_warnings.append(
                    f"{label}情景超出参数有效范围，已排除；区间仅覆盖有效情景，需复核。"
                )
                continue
            rows = self.forecast(request, financials, scenario)
            values.append(
                self._dcf_from_forecast(
                    financials, rows, scenario.wacc, scenario.terminal_growth, request.discount_policy
                )[2]
            )
        return DcfResult(
            enterprise_value=_q(enterprise),
            equity_value=_q(equity),
            per_share_value=_q(per_share),
            range_low=_q(min(values)),
            range_high=_q(max(values)),
            scenario_warnings=scenario_warnings,
            terminal_value_share=_q(terminal_share),
            bridge={
                "operating_enterprise_value": _q(enterprise),
                "cash_and_non_operating_assets": _q(
                    financials.cash_and_non_operating_assets
                ),
                "interest_bearing_debt": _q(-financials.interest_bearing_debt),
                "common_equity_value": _q(equity),
            },
        )

    def relative(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        peers: list[PeerCompany],
    ) -> list[MultipleResult]:
        results: list[MultipleResult] = []
        shares = financials.diluted_shares or financials.common_shares
        if ValuationMethod.PE in request.methods:
            pe_values = [peer.pe for peer in peers if peer.pe is not None]
            if financials.net_income_parent <= 0 or not pe_values:
                reason = (
                    "归母净利润非正"
                    if financials.net_income_parent <= 0
                    else "没有有效 P/E 同业样本"
                )
                results.append(
                    MultipleResult(method="pe", status="not_applicable", reason=reason)
                )
            else:
                values = [
                    multiple * financials.net_income_parent / shares
                    for multiple in pe_values
                ]
                results.append(
                    MultipleResult(
                        method="pe",
                        status="success",
                        per_share_value=_q(_percentile(values, D("0.5"))),
                        range_low=_q(_percentile(values, D("0.25"))),
                        range_high=_q(_percentile(values, D("0.75"))),
                        sample_size=len(values),
                    )
                )
        if ValuationMethod.EV_EBITDA in request.methods:
            multiples = [peer.ev_ebitda for peer in peers if peer.ev_ebitda is not None]
            if financials.ebitda <= 0 or not multiples:
                reason = (
                    "EBITDA 非正"
                    if financials.ebitda <= 0
                    else "没有有效 EV/EBITDA 同业样本"
                )
                results.append(
                    MultipleResult(
                        method="ev_ebitda", status="not_applicable", reason=reason
                    )
                )
            else:
                values = [
                    (
                        multiple * financials.ebitda
                        + financials.cash_and_non_operating_assets
                        - financials.interest_bearing_debt
                    )
                    / shares
                    for multiple in multiples
                ]
                results.append(
                    MultipleResult(
                        method="ev_ebitda",
                        status="success",
                        per_share_value=_q(_percentile(values, D("0.5"))),
                        range_low=_q(_percentile(values, D("0.25"))),
                        range_high=_q(_percentile(values, D("0.75"))),
                        sample_size=len(values),
                    )
                )
        return results

    def sensitivity(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        assumptions: AssumptionSet,
    ) -> list[SensitivityCell]:
        forecast = self.forecast(request, financials, assumptions)
        cells: list[SensitivityCell] = []
        for wacc_delta in [D("-0.02"), D("-0.01"), D(0), D("0.01"), D("0.02")]:
            wacc = assumptions.wacc + wacc_delta
            for growth_delta in [D("-0.01"), D("-0.005"), D(0), D("0.005"), D("0.01")]:
                growth = assumptions.terminal_growth + growth_delta
                if wacc <= 0 or growth >= wacc:
                    cells.append(
                        SensitivityCell(
                            wacc=_q(wacc),
                            terminal_growth=_q(growth),
                            per_share_value=None,
                            valid=False,
                        )
                    )
                    continue
                _, _, value, _ = self._dcf_from_forecast(
                    financials, forecast, wacc, growth, request.discount_policy
                )
                cells.append(
                    SensitivityCell(
                        wacc=_q(wacc),
                        terminal_growth=_q(growth),
                        per_share_value=_q(value),
                        valid=True,
                    )
                )
        return cells

    def reconcile(
        self, dcf: DcfResult | None, relative: list[MultipleResult]
    ) -> ReconciliationResult:
        dcf_range = (dcf.range_low, dcf.range_high) if dcf else None
        successful = [item for item in relative if item.status == "success"]
        relative_range = None
        if successful:
            relative_range = (
                min(
                    item.range_low for item in successful if item.range_low is not None
                ),
                max(
                    item.range_high
                    for item in successful
                    if item.range_high is not None
                ),
            )
        overlap = None
        if dcf_range and relative_range:
            low = max(dcf_range[0], relative_range[0])
            high = min(dcf_range[1], relative_range[1])
            if low <= high:
                overlap = (_q(low), _q(high))
                conclusion = "两类方法存在重叠区间；重叠仅表示结果一致程度，不构成单独的推荐区间。"
            else:
                conclusion = "两类估值区间没有重叠，需要复核增长、利润率、资本成本和可比公司口径。"
        elif dcf_range:
            conclusion = "当前仅有可用的 DCF 结果；相对估值未形成有效区间。"
        elif relative_range:
            conclusion = "当前仅有可用的相对估值结果；DCF 未形成有效区间。"
        else:
            conclusion = "当前没有可用的估值区间。"
        return ReconciliationResult(
            dcf_range=dcf_range,
            relative_range=relative_range,
            overlap_range=overlap,
            conclusion=conclusion,
        )
