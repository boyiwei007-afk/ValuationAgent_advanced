from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from itertools import pairwise
from statistics import median

from valuationagent.finance.industry import (
    FinancialIndustryUnsupported,
    IndustryParameterRegistry,
    IndustryResolutionError,
)
from valuationagent.finance.reference import ReferenceFinancialModel
from valuationagent.finance.integrity import validate_equity_bridge_inputs
from valuationagent.finance.production import (
    effective_capitalized_debt,
    normalized_terminal_cash_flow,
    resolve_equity_bridge,
)
from valuationagent.finance.revenue import FinanceTeamRevenueModel
from valuationagent.schemas.models import (
    AssumptionSet,
    DataQualityAssessment,
    DcfResult,
    FinancialSnapshot,
    ForecastYear,
    MultipleResult,
    PeerCompany,
    ReconciliationResult,
    SensitivityCell,
    SensitivityStudy,
    ValidationFinding,
    ValuationMethod,
    ValuationRequest,
)

D = Decimal
FOUR = D("0.0001")
ZERO = D(0)


def _q(value: Decimal) -> Decimal:
    return value.quantize(FOUR, rounding=ROUND_HALF_UP)


def _mean(values: Iterable[Decimal]) -> Decimal:
    rows = list(values)
    if not rows:
        raise ValueError("平均值至少需要一个数据点。")
    return sum(rows, D(0)) / D(len(rows))


def _percentile(values: list[Decimal], q: Decimal) -> Decimal:
    rows = sorted(values)
    if not rows:
        raise ValueError("分位数至少需要一个数据点。")
    if len(rows) == 1:
        return rows[0]
    position = D(len(rows) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(rows) - 1)
    fraction = position - D(lower)
    return rows[lower] + (rows[upper] - rows[lower]) * fraction


def _fit(values: list[Decimal], years: int) -> list[Decimal]:
    rows = list(values[:years])
    while rows and len(rows) < years:
        rows.append(rows[-1])
    if not rows:
        raise ValueError("预测序列不能为空。")
    return rows


class FinanceTeamModel:
    """Deterministic implementation of the supplied non-financial A-share model."""

    plugin_id = "finance_team_nonfinancial_fcff_relative"
    version = "1.4.0-finance-team-production-20260927"
    supports_incremental_inputs = True

    def __init__(self, registry: IndustryParameterRegistry | None = None):
        self.registry = registry or IndustryParameterRegistry()
        self.revenue_model = FinanceTeamRevenueModel()
        self.reference = ReferenceFinancialModel()

    @staticmethod
    def _use_reference_compatibility(request: ValuationRequest) -> bool:
        """Route demos and historical requests without an industry transparently."""

        return request.mode == "demo" or not (request.company.industry or "").strip()

    def model_version_for(self, request: ValuationRequest) -> str:
        """Expose the calculation engine that actually produced the result."""

        return (
            self.reference.version
            if self._use_reference_compatibility(request)
            else self.version
        )

    def resolve_industry_parameters(self, request: ValuationRequest) -> dict:
        if self._use_reference_compatibility(request):
            return {
                "mode": (
                    "synthetic_demo"
                    if request.mode == "demo"
                    else "legacy_reference_compatibility"
                ),
                "financial_industry_supported": False,
            }
        return self.registry.resolve(request.company.industry).as_dict()

    @staticmethod
    def _history(
        request: ValuationRequest, financials: FinancialSnapshot
    ) -> list[FinancialSnapshot]:
        rows = [*request.historical_financials, financials]
        by_year: dict[int, FinancialSnapshot] = {}
        for row in sorted(rows, key=lambda item: item.period_end):
            if row.period_end <= request.valuation_date and (
                row.published_at is None or row.published_at <= request.valuation_date
            ) and row.comparability_status != "excluded":
                by_year[row.period_end.year] = row
        return list(by_year.values())[-10:]

    def validate(
        self, request: ValuationRequest, financials: FinancialSnapshot
    ) -> list[ValidationFinding]:
        bridge_findings = validate_equity_bridge_inputs(request, financials)
        blocking_bridge = [item for item in bridge_findings if item.severity == "blocking"]
        if blocking_bridge:
            return [
                *self.reference.validate(
                    request,
                    financials,
                    allow_dcf_driver_degradation=True,
                ),
                *bridge_findings,
            ]
        if self._use_reference_compatibility(request):
            findings = self.reference.validate(request, financials)
            if request.mode != "demo":
                findings.append(
                    ValidationFinding(
                        rule_id="LEGACY_REFERENCE_COMPATIBILITY",
                        severity="warning",
                        message=(
                            "未提供行业，当前任务使用参考模型兼容路径；补充非金融行业后，"
                            "系统才会启用金融小组正式模型。"
                        ),
                        recommended_action="补充公司所属行业并重新运行正式估值。",
                    )
                )
            return findings
        findings = self.reference.validate(
            request,
            financials,
            allow_dcf_driver_degradation=True,
        )
        findings.extend(bridge_findings)
        if "dcf" not in request.methods or any(f.severity == "blocking" for f in findings):
            return findings
        try:
            industry = self.registry.resolve(request.company.industry)
        except FinancialIndustryUnsupported as exc:
            findings.append(
                ValidationFinding(
                    rule_id="FINANCIAL_INDUSTRY_OUT_OF_SCOPE",
                    severity="blocking",
                    message=str(exc),
                )
            )
            return findings
        except IndustryResolutionError as exc:
            findings.append(
                ValidationFinding(
                    rule_id="INDUSTRY_CONFIRMATION_REQUIRED",
                    severity="blocking",
                    message=str(exc),
                )
            )
            return findings
        raw_history = [*request.historical_financials, financials]
        historical_core = {"revenue", "ebit_margin"}
        degradable_dcf_fields = {
            "depreciation_amortization",
            "capital_expenditure",
            "change_operating_nwc",
        }
        degraded_years: dict[str, list[int]] = {
            key: [] for key in degradable_dcf_fields
        }
        for row in raw_history:
            if row.comparability_status == "excluded":
                continue
            missing = sorted(key for key in historical_core if getattr(row, key) is None)
            if missing:
                findings.append(ValidationFinding(
                    rule_id="HISTORY_CORE_INPUTS_MISSING",
                    severity="blocking",
                    message=f"{row.period_end} 历史收入/利润率输入不完整：" + "、".join(missing),
                ))
            for key in degradable_dcf_fields:
                if getattr(row, key) is None:
                    degraded_years[key].append(row.period_end.year)
        for key, years in degraded_years.items():
            if years:
                findings.append(ValidationFinding(
                    rule_id="HISTORY_DRIVER_DEGRADED_" + key.upper(),
                    severity="warning",
                    message=(
                        f"{key}在历史年度" + "、".join(map(str, years))
                        + "缺失；模型将按可得历史比例或显式低置信默认值降级，不再把缺失当作零。"
                    ),
                ))
        if any(f.severity == "blocking" for f in findings):
            return findings
        active_history = [
            row for row in raw_history
            if row.period_end <= request.valuation_date
            and (row.published_at is None or row.published_at <= request.valuation_date)
            and row.comparability_status != "excluded"
        ]
        duplicate_years = sorted({
            row.period_end.year
            for row in active_history
            if sum(item.period_end.year == row.period_end.year for item in active_history) > 1
        })
        if duplicate_years:
            findings.append(
                ValidationFinding(
                    rule_id="HISTORY_DUPLICATE_YEAR",
                    severity="blocking",
                    message=(
                        "历史数据包含重复年度："
                        + "、".join(str(year) for year in duplicate_years)
                        + "。请明确重述后版本，并将旧版本标记为excluded。"
                    ),
                )
            )
        review_years = [
            row.period_end.year
            for row in active_history
            if row.comparability_status == "review_required"
        ]
        if review_years:
            findings.append(
                ValidationFinding(
                    rule_id="HISTORY_COMPARABILITY_REVIEW",
                    severity="blocking",
                    message=(
                        "以下年度存在尚未确认的可比性问题："
                        + "、".join(str(year) for year in review_years)
                        + "。请确认调整口径或排除年度后再估值。"
                    ),
                )
            )
        adjusted_years = [
            row.period_end.year
            for row in active_history
            if row.comparability_status == "adjusted"
        ]
        if adjusted_years:
            findings.append(
                ValidationFinding(
                    rule_id="HISTORY_ADJUSTED_YEARS",
                    severity="warning",
                    message=(
                        "历史数据包含调整后年度："
                        + "、".join(str(year) for year in adjusted_years)
                        + "；报告将保留调整说明。"
                    ),
                )
            )
        excluded_years = [
            row.period_end.year
            for row in raw_history
            if row.comparability_status == "excluded"
        ]
        if excluded_years:
            findings.append(
                ValidationFinding(
                    rule_id="HISTORY_EXCLUDED_YEARS",
                    severity="warning",
                    message=(
                        "以下年度因不可比而未进入统计窗口："
                        + "、".join(str(year) for year in excluded_years)
                        + "。"
                    ),
                )
            )
        currencies = {row.currency for row in active_history}
        scopes = {row.statement_scope for row in active_history}
        if len(currencies) > 1 or currencies != {request.company.currency}:
            findings.append(
                ValidationFinding(
                    rule_id="HISTORY_CURRENCY_MISMATCH",
                    severity="blocking",
                    message="历史财务币种不一致，不能在未换算且未说明的情况下合并预测。",
                )
            )
        if len(scopes) > 1 or scopes != {"consolidated"}:
            findings.append(
                ValidationFinding(
                    rule_id="HISTORY_SCOPE_MISMATCH",
                    severity="blocking",
                    message="历史财务合并范围不一致；正式模型只接受口径一致的合并报表。",
                )
            )
        if "dcf" in request.methods and request.forecast_years != 10:
            findings.append(
                ValidationFinding(
                    rule_id="FINANCE_TEAM_FORECAST_HORIZON",
                    severity="blocking",
                    message="金融小组正式模型固定使用10年显性预测期，请将预测期设为10年。",
                    actual=str(request.forecast_years),
                    expected="10",
                )
            )
        if "dcf" in request.methods and request.discount_policy != "year_end":
            findings.append(
                ValidationFinding(
                    rule_id="FINANCE_TEAM_DISCOUNT_POLICY",
                    severity="blocking",
                    message="金融小组公式按年度期末折现，请将discount_policy设为year_end。",
                    actual=request.discount_policy,
                    expected="year_end",
                )
            )
        history = self._history(request, financials)
        manual_growth = (
            request.assumptions.revenue_growth
            or (
                request.assumptions.revenue_growth_scenarios.get("base")
                if request.assumptions.revenue_growth_scenarios else None
            )
        )
        if not manual_growth:
            if len(history) < 4:
                findings.append(
                    ValidationFinding(
                        rule_id="REVENUE_HISTORY_MINIMUM",
                        severity="blocking",
                        message="自动收入预测至少需要4个连续年度收入；目标为近10年。也可以提供手工收入增长路径。",
                        actual=str(len(history)),
                        expected="4-10 annual observations",
                    )
                )
            elif len(history) < 10:
                findings.append(
                    ValidationFinding(
                        rule_id="REVENUE_HISTORY_SHORT",
                        severity="warning",
                        message=f"仅取得{len(history)}个年度数据，自动收入模型按全部可得年份计算，CAGR和分位判断样本敏感。",
                    )
                )
        elif len(history) < 10:
            findings.append(
                ValidationFinding(
                    rule_id="HISTORY_SHORT_WITH_MANUAL_FORECAST",
                    severity="warning",
                    message=(
                        f"仅取得{len(history)}个历史年度；收入路径使用用户指定值，未计算CAGR或位置判断。"
                        "历史样本仍不足以验证利润率、折旧、资本开支和营运资本假设。"
                    ),
                )
            )
        years = [row.period_end.year for row in history]
        if len(years) >= 2 and years != list(range(years[0], years[-1] + 1)):
            findings.append(
                ValidationFinding(
                    rule_id="REVENUE_HISTORY_GAP",
                    severity="blocking",
                    message="历史财务年份不连续，不允许自动插值或填零。",
                )
            )
        if industry.quality in {"B", "C"}:
            findings.append(
                ValidationFinding(
                    rule_id="INDUSTRY_PARAMETER_QUALITY",
                    severity="warning",
                    message=f"行业参数行的数据等级为{industry.quality}，报告将保留来源与待复核提示。",
                )
            )
        if industry.metadata_completeness != "complete":
            findings.append(
                ValidationFinding(
                    rule_id="INDUSTRY_PARAMETER_METADATA_PARTIAL",
                    severity="warning",
                    message=(
                        "行业参数库元数据尚不完整，缺少："
                        + "、".join(industry.required_metadata_missing)
                        + "。本次结果保留参数版本与原始文件哈希，但参数置信区间仍待金融小组补充。"
                    ),
                )
            )
        if industry.provenance_note:
            findings.append(
                ValidationFinding(
                    rule_id="INDUSTRY_PARAMETER_FALLBACK",
                    severity="warning",
                    message=industry.provenance_note,
                )
            )
        special_routes = {
            "real_estate": "房地产存在预售、合同负债和项目周期，需人工复核营运资本与收入确认。",
            "energy": "资源行业收入受产量与商品价格驱动，统计收入路径仅作为基础情景。",
            "pharma": "医药公司可能受集采、管线和专利到期影响，需人工复核经营驱动。",
        }
        if industry.industry_id in special_routes:
            findings.append(
                ValidationFinding(
                    rule_id="SPECIAL_INDUSTRY_MODEL_REVIEW",
                    severity="warning",
                    message=special_routes[industry.industry_id],
                )
            )
        if request.assumptions.wacc is None and not (
            request.assumptions.quarterly_average_market_cap
            or financials.statement_items.get("quarterly_average_market_cap")
            or request.assumptions.market_cap
            or financials.statement_items.get("market_cap")
            or financials.statement_items.get("total_equity")
        ):
            findings.append(
                ValidationFinding(
                    rule_id="WACC_EQUITY_WEIGHT_FALLBACK",
                    severity="warning",
                    message="缺少市值或权益数据，自动WACC暂按100%股权权重计算；建议补充估值日前可得市值。",
                )
            )
        if request.assumptions.wacc is None:
            market_defaults = [
                ("risk_free_rate", request.assumptions.risk_free_rate, "无风险利率Rf"),
                ("equity_risk_premium", request.assumptions.equity_risk_premium, "股权风险溢价ERP"),
                ("beta", request.assumptions.beta, "Beta"),
                ("debt_cost", request.assumptions.debt_cost, "税前债务成本Kd"),
            ]
            missing = [label for _, value, label in market_defaults if value is None]
            if missing:
                findings.append(
                    ValidationFinding(
                        rule_id="WACC_MARKET_INPUT_DEFAULTS",
                        severity="warning",
                        message=(
                            "WACC缺少" + "、".join(missing)
                            + "的基准日证据，当前使用模型默认值/行业Beta；结果会保留具体数值。"
                        ),
                        recommended_action="联网取得基准日中国10年期国债收益率、ERP、Beta与债务成本并附来源。",
                    )
                )
        market_as_of = request.assumptions.market_inputs_as_of
        market_source = request.assumptions.market_inputs_source.strip()
        if market_as_of is None:
            findings.append(
                ValidationFinding(
                    rule_id="WACC_MARKET_DATE_UNVERIFIED",
                    severity="warning",
                    message=(
                        "WACC市场参数未记录基准日；数值结果将降为低置信并扩大WACC区间，"
                        "不得把模型默认值表述为估值日事实。"
                    ),
                    recommended_action="记录Rf、ERP、Beta、Kd和市值快照的估值日或最近交易日。",
                )
            )
        else:
            age = (request.valuation_date - market_as_of).days
            if age > request.assumptions.market_inputs_stale_after_days:
                findings.append(
                    ValidationFinding(
                        rule_id="WACC_MARKET_DATA_STALE",
                        severity="warning",
                        message=(
                            f"WACC市场参数距估值日{age}天，超过"
                            f"{request.assumptions.market_inputs_stale_after_days}天新鲜度阈值；"
                            "结果将降低置信等级并扩大区间。"
                        ),
                    )
                )
        if not market_source:
            findings.append(
                ValidationFinding(
                    rule_id="WACC_MARKET_SOURCE_UNVERIFIED",
                    severity="warning",
                    message="WACC市场参数未记录可核验来源；报告将明确标记为未核验输入。",
                )
            )
        if (
            request.assumptions.stable_roic is not None
            and request.assumptions.stable_roic
            <= (
                request.assumptions.terminal_growth
                if request.assumptions.terminal_growth is not None
                else D("0.03")
            )
        ):
            findings.append(
                ValidationFinding(
                    rule_id="STABLE_ROIC_NOT_ABOVE_GROWTH",
                    severity="blocking",
                    message="用户指定的稳定期ROIC必须高于永续增长率，否则g/ROIC再投资率不成立。",
                )
            )
        if (
            request.assumptions.wacc is not None
            and request.assumptions.wacc
            <= (
                request.assumptions.terminal_growth
                if request.assumptions.terminal_growth is not None
                else D("0.03")
            )
        ):
            findings.append(
                ValidationFinding(
                    rule_id="WACC_NOT_ABOVE_TERMINAL_GROWTH",
                    severity="blocking",
                    message="WACC必须高于永续增长率，否则Gordon终值不成立。",
                )
            )
        required_assets = {
            "fixed_assets_net",
            "intangible_assets",
            "long_term_deferred_expenses",
            "depreciation_fixed_assets",
            "amortization_intangibles",
            "amortization_long_term_deferred",
        }
        if financials.period_end.year >= 2019:
            required_assets |= {
                "right_of_use_assets",
                "depreciation_right_of_use",
            }
        if not required_assets <= set(financials.statement_items):
            findings.append(
                ValidationFinding(
                    rule_id="DA_REVENUE_RATIO_FALLBACK",
                    severity="warning",
                    message="资产和折旧摊销明细不完整，D&A将按收入比例法降级，并在结果中记录。",
                )
            )
        days = (
            request.assumptions.dso_days,
            request.assumptions.dio_days,
            request.assumptions.dpo_days,
        )
        if any(value is not None for value in days) and not all(
            value is not None for value in days
        ):
            findings.append(
                ValidationFinding(
                    rule_id="NWC_DRIVER_PARTIAL",
                    severity="blocking",
                    message="DSO、DIO、DPO必须成组提供；不能用部分周转天数与其他口径混算。",
                    recommended_action="补齐三个周转天数，或全部留空让模型从历史报表计算。",
                )
            )
        if all(value is not None for value in days) and (
            request.assumptions.operating_cost_ratio is None
            and sum(
                row.statement_items.get("operating_cost") is not None
                for row in history[-5:]
            )
            < 4
        ):
            findings.append(
                ValidationFinding(
                    rule_id="NWC_COST_RATIO_MISSING",
                    severity="blocking",
                    message="已指定DSO/DIO/DPO，但缺少营业成本率且历史完整样本不足4年，无法计算营运资本。",
                    recommended_action="补充营业成本率，或提供至少4年的营业成本与收入。",
                )
            )
        items = financials.statement_items
        direct_keys = {
            "accounts_receivable",
            "inventory",
            "accounts_payable",
        }
        if direct_keys <= set(items) and items.get("operating_nwc") is not None:
            direct_ratio = (
                items["accounts_receivable"]
                + items["inventory"]
                - items["accounts_payable"]
            ) / financials.revenue
            reported_ratio = items["operating_nwc"] / financials.revenue
            gap = abs(direct_ratio - reported_ratio)
            if gap > D("0.05"):
                findings.append(
                    ValidationFinding(
                        rule_id="NWC_DEFINITION_GAP",
                        severity="warning",
                        message="应收+存货-应付与报表经营营运资本口径相差超过5个百分点。",
                        actual=f"{gap:.2%}",
                        expected="<=5.00%",
                        recommended_action="确认其他经营流动资产/负债是否应纳入口径。",
                    )
                )
        return findings

    @staticmethod
    def _margin_scenarios(
        history: list[FinancialSnapshot], years: int
    ) -> dict[str, list[Decimal]]:
        margins = [row.ebit_margin for row in history]
        current = margins[-1]
        targets = {
            "pessimistic": _percentile(margins, D("0.25")),
            "base": D(str(median(margins))),
            "optimistic": _percentile(margins, D("0.75")),
        }
        result = {}
        for name, target in targets.items():
            result[name] = [
                current + (target - current) * D(i) / D(max(years - 1, 1))
                for i in range(years)
            ]
        return result

    @staticmethod
    def _terminal_tax_rate(
        request: ValuationRequest, financials: FinancialSnapshot
    ) -> Decimal:
        if request.assumptions.terminal_tax_rate is not None:
            return request.assumptions.terminal_tax_rate
        policy = (financials.tax_policy or "").lower()
        if "重点集成电路" in policy or "重点软件" in policy:
            return D("0.10")
        if any(word in policy for word in ("高新", "技术先进型")):
            return D("0.15")
        return D("0.25")

    @staticmethod
    def _tax_path(
        request: ValuationRequest,
        financials: FinancialSnapshot,
        terminal: Decimal,
    ) -> list[Decimal]:
        years = request.forecast_years
        current = financials.tax_rate
        if current == terminal:
            return [terminal] * years
        base_year = financials.period_end.year
        expiry = financials.tax_policy_expiry_year
        if expiry is None:
            hold = years // 2
        else:
            hold = max(0, min(years, expiry - base_year))
        configured = request.assumptions.tax_transition_years
        transition = (
            min(configured, max(0, years - hold))
            if configured is not None
            else max(1, min(3, years - hold))
        )
        path = []
        for index in range(years):
            step = index + 1
            if step <= hold:
                value = current
            elif transition and step <= hold + transition:
                value = current + (terminal - current) * D(step - hold) / D(transition)
            else:
                value = terminal
            path.append(value)
        path[-1] = terminal
        return path

    @staticmethod
    def _wacc(
        request: ValuationRequest,
        financials: FinancialSnapshot,
        beta_default: Decimal,
        marginal_tax_rate: Decimal,
    ) -> tuple[Decimal, dict[str, Decimal], str]:
        supplied = request.assumptions
        market_age = (
            D((request.valuation_date - supplied.market_inputs_as_of).days)
            if supplied.market_inputs_as_of is not None
            else D(-1)
        )
        market_verified = D(
            int(
                supplied.market_inputs_as_of is not None
                and bool(supplied.market_inputs_source.strip())
            )
        )
        market_stale = D(
            int(
                market_age >= 0
                and market_age > D(supplied.market_inputs_stale_after_days)
            )
        )
        if supplied.wacc is not None:
            return supplied.wacc, {
                "wacc": supplied.wacc,
                "manual_wacc": D(1),
                "market_inputs_age_days": market_age,
                "market_inputs_verified": market_verified,
                "market_inputs_stale": market_stale,
                "wacc_default_count": D(0),
            }, "用户指定WACC；市场参数日期与来源仍单独披露"
        rf = supplied.risk_free_rate if supplied.risk_free_rate is not None else (
            D("0.0168") if request.valuation_date >= date(2026, 1, 1) else D("0.0185")
        )
        erp = (
            supplied.equity_risk_premium
            if supplied.equity_risk_premium is not None
            else D("0.06")
        )
        beta = supplied.beta if supplied.beta is not None else beta_default
        kd = supplied.debt_cost if supplied.debt_cost is not None else D("0.045")
        default_count = sum(
            value is None
            for value in (
                supplied.risk_free_rate,
                supplied.equity_risk_premium,
                supplied.beta,
                supplied.debt_cost,
            )
        )
        market_cap = (
            supplied.quarterly_average_market_cap
            or financials.statement_items.get("quarterly_average_market_cap")
            or supplied.market_cap
            or financials.statement_items.get("market_cap")
            or financials.statement_items.get("total_equity")
        )
        # Capitalized leases are debt for both the WACC weights and the
        # enterprise-to-equity bridge.  Resolve them with the same policy to
        # avoid discounting lease cash flows with an inconsistent capital mix.
        debt = effective_capitalized_debt(financials)[0]
        if market_cap is None:
            equity_weight, debt_weight = D(1), D(0)
            weight_note = "缺少市值/权益数据，暂按100%股权权重"
        else:
            total = market_cap + debt
            equity_weight = market_cap / total
            debt_weight = debt / total
            if (
                supplied.quarterly_average_market_cap is not None
                or financials.statement_items.get("quarterly_average_market_cap") is not None
            ):
                weight_note = "按估值日前四个季度市值均值与有息负债加权"
            else:
                weight_note = "四季度平均市值不可得，按估值日前可得市值/权益与有息负债加权"
        ke = rf + beta * erp
        wacc = equity_weight * ke + debt_weight * kd * (D(1) - marginal_tax_rate)
        return wacc, {
            "risk_free_rate": rf,
            "equity_risk_premium": erp,
            "beta": beta,
            "cost_of_equity": ke,
            "debt_cost": kd,
            "marginal_tax_rate": marginal_tax_rate,
            "equity_weight": equity_weight,
            "debt_weight": debt_weight,
            "wacc": wacc,
            "manual_wacc": D(0),
            "market_inputs_age_days": market_age,
            "market_inputs_verified": market_verified,
            "market_inputs_stale": market_stale,
            "wacc_default_count": D(default_count),
        }, weight_note

    def resolve_assumptions(
        self, request: ValuationRequest, financials: FinancialSnapshot
    ) -> AssumptionSet:
        if "dcf" not in request.methods or self._use_reference_compatibility(request):
            return self.reference.resolve_assumptions(request, financials)
        industry = self.registry.resolve(request.company.industry)
        history = self._history(request, financials)
        terminal_growth = (
            request.assumptions.terminal_growth
            if request.assumptions.terminal_growth is not None
            else D("0.03")
        )
        decisions: list[str] = [
            "全球行业增速仅作有限期合理性校验，不直接作为永续增长率。",
            "金融行业被排除，参数库不会为金融公司返回通用FCFF参数。",
        ]
        revenue_drivers: dict[str, Decimal] = {}
        if request.assumptions.revenue_growth_scenarios:
            revenue_scenarios = {
                name: _fit(path, request.forecast_years)
                for name, path in request.assumptions.revenue_growth_scenarios.items()
            }
            growth_rationale = "用户分别指定悲观、基准和乐观收入增长路径"
        elif request.assumptions.revenue_growth:
            revenue_scenarios = {
                "base": _fit(request.assumptions.revenue_growth, request.forecast_years)
            }
            revenue_scenarios["pessimistic"] = list(revenue_scenarios["base"])
            revenue_scenarios["optimistic"] = list(revenue_scenarios["base"])
            growth_rationale = "用户指定收入增长路径；三情景暂共享该路径"
        else:
            projection = self.revenue_model.project(
                history, industry, terminal_growth, request.forecast_years
            )
            revenue_scenarios = projection.scenarios
            decisions.extend(projection.decisions)
            revenue_drivers = {
                "revenue_center": projection.center,
                "revenue_cagr": projection.cagr,
                "revenue_a3": projection.a3,
                "revenue_rho": projection.rho,
                "revenue_pessimistic_anchor": projection.anchors["pessimistic"],
                "revenue_optimistic_anchor": projection.anchors["optimistic"],
                "revenue_decay_years": D(projection.effective_decay_years),
            }
            growth_rationale = (
                f"十年历史模型：CAGR={projection.cagr:.4%}，A3={projection.a3:.4%}，"
                f"A5={projection.a5:.4%}，位置={projection.position}"
            )
        if request.assumptions.ebit_margin_scenarios:
            margin_scenarios = {
                name: _fit(path, request.forecast_years)
                for name, path in request.assumptions.ebit_margin_scenarios.items()
            }
            margin_rationale = "用户分别指定悲观、基准和乐观EBIT利润率路径"
        elif request.assumptions.ebit_margin:
            base_margin = _fit(request.assumptions.ebit_margin, request.forecast_years)
            margin_scenarios = {name: list(base_margin) for name in revenue_scenarios}
            margin_rationale = "用户指定EBIT利润率路径"
        else:
            margin_scenarios = self._margin_scenarios(history, request.forecast_years)
            margin_rationale = "最新实际利润率向历史P25/中位数/P75收敛；详细科目不足时的可追溯降级"
        terminal_tax = self._terminal_tax_rate(request, financials)
        tax_path = self._tax_path(request, financials, terminal_tax)
        wacc, components, weight_note = self._wacc(
            request, financials, industry.beta_default, terminal_tax
        )
        if terminal_growth >= wacc:
            raise ValueError("永续增长率必须低于WACC。")
        decisions.append(weight_note)
        if request.assumptions.stable_roic is not None:
            if request.assumptions.stable_roic <= terminal_growth:
                raise ValueError("稳定期ROIC必须高于永续增长率。")
            stable_roic = request.assumptions.stable_roic
            stable_roic_note = "稳定期ROIC使用用户已确认输入"
            stable_roic_policy_fallback = False
        else:
            invested_capital = financials.statement_items.get("invested_capital")
            if invested_capital is not None and invested_capital > 0:
                current_ebit = financials.revenue * financials.ebit_margin
                current_nopat = current_ebit - max(current_ebit, ZERO) * terminal_tax
                observed_roic = current_nopat / invested_capital
                stable_roic = min(D("0.30"), max(D("0.06"), observed_roic))
                stable_roic_note = "稳定期ROIC由最新NOPAT/投入资本推导并限制在6%–30%审慎区间"
                stable_roic_policy_fallback = stable_roic != observed_roic
            else:
                lifecycle_roic = {
                    "成长期": D("0.15"),
                    "成熟期": D("0.10"),
                    "衰退期": D("0.08"),
                }
                stable_roic = lifecycle_roic.get(industry.lifecycle, D("0.10"))
                stable_roic_note = "缺少投入资本，稳定期ROIC使用按生命周期披露的低置信政策值"
                stable_roic_policy_fallback = True
        stable_roic = max(stable_roic, terminal_growth + D("0.01"))
        decisions.append(
            stable_roic_note
            + f"；稳定期再投资率=g/ROIC={terminal_growth / stable_roic:.2%}。"
        )
        decisions.append(
            "预测EBIT为负时不预设当期可收取所得税退款；税亏结转及以后年度抵扣未单独建模。"
        )
        capex_alpha, capex_kappa, capex_method, capex_note = self._capex_drivers(
            request, history
        )
        decisions.append(capex_note)
        nwc_drivers, nwc_method, nwc_note = self._nwc_drivers(request, history)
        decisions.append(nwc_note)
        ebit_drivers, ebit_methods, ebit_note = self._ebit_drivers(
            history, industry.window_years
        )
        decisions.append(ebit_note)
        exit_multiple, exit_low, exit_high, exit_note = self._exit_multiple(
            request, industry
        )
        decisions.append(exit_note)
        operating_drivers = {
            "capex_alpha": _q(capex_alpha),
            "operating_cash_ratio": _q(
                request.assumptions.operating_cash_ratio or D(0)
            ),
            "stable_roic": _q(stable_roic),
            "stable_roic_policy_fallback": D(int(stable_roic_policy_fallback)),
            "operating_cash_policy_fallback": D(
                int(
                    request.assumptions.operating_cash_ratio is None
                    and financials.cash_and_non_operating_assets > ZERO
                )
            ),
        }
        if operating_drivers["operating_cash_policy_fallback"]:
            decisions.append(
                "经营必需现金比例未建立；暂按0计算可分配现金，可能高估普通股价值，"
                "结果按低置信等级披露。"
            )
        da_ratios = [
            row.depreciation_amortization / row.revenue
            for row in history[-5:]
            if row.depreciation_amortization is not None
        ]
        capex_ratios = [
            row.capital_expenditure / row.revenue
            for row in history[-5:]
            if row.capital_expenditure is not None
        ]
        da_ratio = D(str(median(da_ratios))) if da_ratios else D("0.03")
        capex_ratio = (
            D(str(median(capex_ratios)))
            if capex_ratios
            else max(da_ratio, D("0.05"))
        )
        operating_drivers["da_revenue_ratio"] = _q(da_ratio)
        operating_drivers["capex_revenue_ratio"] = _q(capex_ratio)
        operating_drivers["da_policy_fallback"] = D(
            int(not da_ratios and self._asset_rates(history) is None)
        )
        operating_drivers["capex_policy_fallback"] = D(
            int(
                not capex_ratios
                and capex_method == "revenue_ratio_fallback"
            )
        )
        if not da_ratios:
            decisions.append("D&A历史值不可得，采用收入3%低置信比例并进入扩大敏感性区间。")
        if not capex_ratios:
            decisions.append("CapEx历史值不可得，采用收入比例政策值并明确标记为降级。")
        wacc_range_delta = (
            D("0.02")
            if components.get("market_inputs_verified", D(0)) == 0
            or components.get("wacc_default_count", D(0)) > 0
            else D("0.015")
            if components.get("market_inputs_stale", D(0)) > 0
            else D("0.01")
        )
        operating_drivers["wacc_range_delta"] = wacc_range_delta
        operating_drivers.update(
            {key: _q(value) for key, value in revenue_drivers.items()}
        )
        if capex_kappa is not None:
            operating_drivers["capex_kappa"] = _q(capex_kappa)
        if exit_multiple is not None:
            operating_drivers["exit_multiple"] = _q(exit_multiple)
        if exit_low is not None:
            operating_drivers["exit_multiple_low"] = _q(exit_low)
        if exit_high is not None:
            operating_drivers["exit_multiple_high"] = _q(exit_high)
        operating_drivers.update({key: _q(value) for key, value in nwc_drivers.items()})
        operating_drivers.update({key: _q(value) for key, value in ebit_drivers.items()})
        return AssumptionSet(
            revenue_growth=[_q(x) for x in revenue_scenarios["base"]],
            ebit_margin=[_q(x) for x in margin_scenarios["base"]],
            wacc=_q(wacc),
            terminal_growth=_q(terminal_growth),
            source="finance_team_model_with_user_overrides",
            rationale={
                "revenue_growth": growth_rationale,
                "ebit_margin": margin_rationale,
                "wacc": "CAPM与资本结构；" + weight_note,
                "terminal_growth": "长期稳态参数，默认3%；与有限期全球行业增速分开",
                "tax": f"实际税率{financials.tax_rate:.2%}向可持续税率{terminal_tax:.2%}过渡",
                "stable_roic": stable_roic_note,
            },
            revenue_growth_scenarios={
                key: [_q(x) for x in values] for key, values in revenue_scenarios.items()
            },
            ebit_margin_scenarios={
                key: [_q(x) for x in values] for key, values in margin_scenarios.items()
            },
            tax_rate_path=[_q(x) for x in tax_path],
            wacc_components={key: _q(value) for key, value in components.items()},
            industry_parameters=industry.as_dict(),
            model_decisions=decisions,
            operating_drivers=operating_drivers,
            calculation_methods={
                "depreciation_amortization": (
                    "policy_3_percent_revenue"
                    if operating_drivers["da_policy_fallback"]
                    else "asset_rollforward_or_historical_revenue_ratio"
                ),
                "capex": capex_method,
                "change_operating_nwc": nwc_method,
                "terminal_value": "gordon_growth_normalized_reinvestment",
                **ebit_methods,
            },
        )

    @staticmethod
    def _asset_rates(history: list[FinancialSnapshot]) -> dict[str, Decimal] | None:
        pairs = {
            "fixed": ("fixed_assets_net", "depreciation_fixed_assets"),
            "intangible": ("intangible_assets", "amortization_intangibles"),
            "deferred": (
                "long_term_deferred_expenses",
                "amortization_long_term_deferred",
            ),
            "right_of_use": (
                "right_of_use_assets",
                "depreciation_right_of_use",
            ),
        }
        rates: dict[str, Decimal] = {}
        for name, (balance_key, charge_key) in pairs.items():
            values = []
            for previous, row in pairwise(history[-6:]):
                if name == "right_of_use" and row.period_end.year < 2019:
                    continue
                opening = previous.statement_items.get(balance_key)
                closing = row.statement_items.get(balance_key)
                charge = row.statement_items.get(charge_key)
                if (
                    opening is not None
                    and closing is not None
                    and opening + closing > 0
                    and charge is not None
                ):
                    average_balance = (opening + closing) / D(2)
                    values.append(charge / average_balance)
            if len(values) < 3:
                return None
            rates[name] = D(str(median(values)))
        return rates

    @staticmethod
    def _capex_drivers(
        request: ValuationRequest,
        history: list[FinancialSnapshot],
    ) -> tuple[Decimal, Decimal | None, str, str]:
        alpha = (
            request.assumptions.capex_alpha
            if request.assumptions.capex_alpha is not None
            else D(1)
        )
        if request.assumptions.capex_kappa is not None:
            return (
                alpha,
                request.assumptions.capex_kappa,
                "alpha_da_plus_kappa_delta_revenue",
                "资本开支采用方法A：用户指定α与κ；预测末期α收敛至1、κ收敛至0。",
            )
        kappas: list[Decimal] = []
        recent = history[-6:]
        for previous, current in pairwise(recent):
            if (
                previous.revenue is None
                or current.revenue is None
                or current.capital_expenditure is None
                or current.depreciation_amortization is None
            ):
                continue
            delta_revenue = current.revenue - previous.revenue
            materiality = abs(previous.revenue) * D("0.005")
            if abs(delta_revenue) <= materiality:
                continue
            kappas.append(
                (current.capital_expenditure - current.depreciation_amortization)
                / delta_revenue
            )
        if len(kappas) >= 3:
            raw_median = D(str(median(kappas)))
            deviations = [abs(value - raw_median) for value in kappas]
            mad = D(str(median(deviations)))
            robust = (
                [value for value in kappas if abs(value - raw_median) <= D(3) * mad]
                if mad > 0
                else kappas
            )
            kappa = max(D("-2"), min(D("2"), D(str(median(robust)))))
            return (
                alpha,
                kappa,
                "alpha_da_plus_kappa_delta_revenue",
                f"资本开支采用方法A：α={alpha}，κ取最近{len(robust)}个稳健历史样本中位数{kappa:.4f}（MAD去极值并限制在[-2,2]）；预测末期α→1、κ→0。",
            )
        return (
            alpha,
            None,
            "revenue_ratio_fallback",
            "有效κ历史样本不足3年，资本开支降级为收入比例法，并在预测末期令CapEx收敛至D&A。",
        )

    @staticmethod
    def _nwc_drivers(
        request: ValuationRequest,
        history: list[FinancialSnapshot],
    ) -> tuple[dict[str, Decimal], str, str]:
        supplied = request.assumptions
        cost_ratios = [
            row.statement_items["operating_cost"] / row.revenue
            for row in history[-5:]
            if row.statement_items.get("operating_cost") is not None
        ]
        manual_days = (supplied.dso_days, supplied.dio_days, supplied.dpo_days)
        if all(value is not None for value in manual_days):
            cost_ratio = supplied.operating_cost_ratio
            if cost_ratio is None and len(cost_ratios) >= 4:
                cost_ratio = D(str(median(cost_ratios)))
            if cost_ratio is not None:
                return (
                    {
                        "dso_days": supplied.dso_days,
                        "dio_days": supplied.dio_days,
                        "dpo_days": supplied.dpo_days,
                        "operating_cost_ratio": cost_ratio,
                    },
                    "dso_dio_dpo",
                    "营运资本采用用户指定DSO/DIO/DPO；营业成本率取用户值或最近4–5年中位数。",
                )

        day_rows: list[tuple[Decimal, Decimal, Decimal, Decimal]] = []
        for row in history[-5:]:
            items = row.statement_items
            cost = items.get("operating_cost")
            receivable = items.get("accounts_receivable")
            inventory = items.get("inventory")
            payable = items.get("accounts_payable")
            if (
                cost is None
                or cost <= 0
                or receivable is None
                or inventory is None
                or payable is None
            ):
                continue
            day_rows.append(
                (
                    receivable / row.revenue * D(365),
                    inventory / cost * D(365),
                    payable / cost * D(365),
                    cost / row.revenue,
                )
            )
        if len(day_rows) >= 4:
            return (
                {
                    "dso_days": D(str(median(row[0] for row in day_rows))),
                    "dio_days": D(str(median(row[1] for row in day_rows))),
                    "dpo_days": D(str(median(row[2] for row in day_rows))),
                    "operating_cost_ratio": D(str(median(row[3] for row in day_rows))),
                },
                "dso_dio_dpo",
                f"营运资本采用最近{len(day_rows)}个完整年度DSO/DIO/DPO及营业成本率中位数。",
            )

        nwc_ratios = [
            row.statement_items["operating_nwc"] / row.revenue
            for row in history[-5:]
            if row.statement_items.get("operating_nwc") is not None
        ]
        if len(nwc_ratios) >= 3:
            ratio = D(str(median(nwc_ratios)))
            return (
                {"operating_nwc_ratio": ratio},
                "operating_nwc_revenue_ratio",
                f"周转科目不足4年，按最近{len(nwc_ratios)}年经营营运资本/收入中位数{ratio:.2%}降级；应对该比例做±1个百分点敏感性。",
            )
        latest = history[-1]
        if latest.statement_items.get("operating_nwc") is not None:
            ratio = latest.statement_items["operating_nwc"] / latest.revenue
            return (
                {"operating_nwc_ratio": ratio},
                "latest_operating_nwc_revenue_ratio",
                "周转科目与历史等价比例均不足，暂用最近一期经营营运资本/收入；该低样本降级必须复核。",
            )
        latest_change_rows = [
            row for row in history if row.change_operating_nwc is not None
        ]
        if (
            len(history) >= 2
            and history[-1].revenue != history[-2].revenue
            and history[-1].change_operating_nwc is not None
        ):
            factor = history[-1].change_operating_nwc / (
                history[-1].revenue - history[-2].revenue
            )
            return (
                {"nwc_delta_revenue_factor": factor},
                "implied_from_latest_delta",
                "经营营运资本余额不可得，暂按最近一期ΔNWC/Δ收入降级；不得将其误称为周转天数模型。",
            )
        if latest_change_rows:
            ratio = D(str(median(
                row.change_operating_nwc / row.revenue
                for row in latest_change_rows[-5:]
            )))
            return (
                {"nwc_delta_revenue_factor": ratio},
                "historical_delta_nwc_revenue_fallback",
                "经营营运资本余额不可得，按可得历史ΔNWC/收入中位数低质量降级并要求人工复核。",
            )
        ratio = D("0.10")
        return (
            {"nwc_delta_revenue_factor": ratio},
            "policy_delta_revenue_fallback",
            "营运资本余额及变动均不可得，按Δ收入的10%政策值低置信降级；不得把缺失值填零。",
        )

    @staticmethod
    def _ebit_drivers(
        history: list[FinancialSnapshot], window_years: int
    ) -> tuple[dict[str, Decimal], dict[str, str], str]:
        component_keys = (
            "operating_cost",
            "taxes_and_surcharges",
            "selling_expense",
            "administrative_expense",
            "research_expense",
            "impairment_loss",
            "other_income",
        )
        window = history[-max(3, min(window_years, 8)):]
        ratios: dict[str, list[Decimal]] = {key: [] for key in component_keys}
        for row in window:
            for key in component_keys:
                value = row.statement_items.get(key)
                if value is not None:
                    ratios[key].append(value / row.revenue)
        latest = history[-1]
        if (
            any(len(ratios[key]) < 3 for key in component_keys)
            or any(latest.statement_items.get(key) is None for key in component_keys)
        ):
            return (
                {},
                {"ebit": "margin_path_fallback"},
                "经营利润科目不足3个年度，EBIT降级为最新利润率向历史分位数收敛；报告不把该路径误称为逐科目模型。",
            )

        drivers: dict[str, Decimal] = {}
        methods: dict[str, str] = {"ebit": "operating_component_build"}
        for key in component_keys:
            drivers[f"{key}_start_ratio"] = (
                latest.statement_items[key] / latest.revenue
            )
            drivers[f"{key}_target_ratio"] = _mean(ratios[key])

        workforce_specs = {
            "research": "research_expense",
            "administrative": "administrative_expense",
        }
        for prefix, expense_key in workforce_specs.items():
            required = {
                f"{prefix}_headcount",
                f"{prefix}_average_cost",
                f"{prefix}_non_labor_ratio",
                f"{prefix}_headcount_growth_center",
                f"{prefix}_lifecycle_factor",
                f"{prefix}_wage_premium",
                "workforce_ramp_years",
            }
            if required <= set(latest.statement_items):
                for key in required:
                    drivers[key] = latest.statement_items[key]
                methods[expense_key] = "workforce_cost_model_l1_l3"
            else:
                methods[expense_key] = "historical_expense_ratio_fallback"

        workforce_count = sum(
            method == "workforce_cost_model_l1_l3"
            for key, method in methods.items()
            if key in {"research_expense", "administrative_expense"}
        )
        note = (
            f"EBIT采用严格经营口径逐科目搭建；A类比率使用最近{len(window)}年算术平均作为稳态。"
            f"研发/管理费用中有{workforce_count}项具备用工成本法完整输入，其余按历史费用率显式降级。"
        )
        return drivers, methods, note

    @staticmethod
    def _exit_multiple(
        request, industry
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None, str]:
        if request.assumptions.exit_multiple is not None:
            value = request.assumptions.exit_multiple
            return value, value, value, f"退出倍数交叉校验使用用户指定{value}x；未擅自扩展区间，Gordon法仍为主估值。"
        if industry.industry_id == "utilities":
            return D(10), D(8), D(12), "公用事业退出倍数交叉校验采用文档区间8–12x、中点10x；Gordon法仍为主估值。"
        if industry.category == "高新技术类" and industry.lifecycle == "成长期":
            return D(20), D(15), D(25), "成长科技退出倍数交叉校验采用文档区间15–25x、中点20x；Gordon法仍为主估值。"
        if industry.category == "制造业" and industry.lifecycle == "成熟期":
            return D(8), D(6), D(10), "成熟制造业退出倍数交叉校验采用文档区间6–10x、中点8x；Gordon法仍为主估值。"
        return None, None, None, "文档没有给出该行业的通用退出倍数区间，本次只报告Gordon隐含倍数，不自动编造交叉校验倍数。"

    def _forecast_scenario(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        assumptions: AssumptionSet,
        scenario: str,
    ) -> list[ForecastYear]:
        history = self._history(request, financials)
        growth_path = assumptions.revenue_growth_scenarios.get(
            scenario, assumptions.revenue_growth
        )
        margin_path = assumptions.ebit_margin_scenarios.get(
            scenario, assumptions.ebit_margin
        )
        revenue = financials.revenue
        drivers = assumptions.operating_drivers
        current_da_ratio = (
            financials.depreciation_amortization / financials.revenue
            if financials.depreciation_amortization is not None
            else drivers.get("da_revenue_ratio", D("0.03"))
        )
        history_da = [
            row.depreciation_amortization / row.revenue
            for row in history[-5:]
            if row.depreciation_amortization is not None
        ]
        terminal_da_ratio = (
            D(str(median(history_da))) if history_da else current_da_ratio
        )
        current_capex_ratio = (
            financials.capital_expenditure / financials.revenue
            if financials.capital_expenditure is not None
            else drivers.get("capex_revenue_ratio", max(current_da_ratio, D("0.05")))
        )
        rates = self._asset_rates(history)
        asset_keys = {
            "fixed": "fixed_assets_net",
            "intangible": "intangible_assets",
            "deferred": "long_term_deferred_expenses",
            "right_of_use": "right_of_use_assets",
        }
        asset_balances = {
            name: financials.statement_items.get(key, D(0))
            for name, key in asset_keys.items()
        }
        asset_total = sum(asset_balances.values(), D(0))
        cash_asset_total = sum(
            value for name, value in asset_balances.items()
            if name != "right_of_use"
        )
        cash_asset_shares = (
            {
                key: value / cash_asset_total
                for key, value in asset_balances.items()
                if key != "right_of_use"
            }
            if cash_asset_total > 0
            else {}
        )
        capex_method = assumptions.calculation_methods.get(
            "capex", "revenue_ratio_fallback"
        )
        capex_alpha = drivers.get("capex_alpha", D(1))
        capex_kappa = drivers.get("capex_kappa")
        da_rate_multiplier = drivers.get("da_rate_multiplier", D(1))
        nwc_method = assumptions.calculation_methods.get(
            "change_operating_nwc", "delta_revenue_ratio_fallback"
        )
        if nwc_method == "dso_dio_dpo":
            dso = drivers["dso_days"]
            dio = drivers["dio_days"]
            dpo = drivers["dpo_days"]
            cost_ratio = drivers["operating_cost_ratio"]
            previous_nwc = financials.statement_items.get("operating_nwc")
            if previous_nwc is None:
                previous_nwc = financials.revenue * (
                    dso + dio * cost_ratio - dpo * cost_ratio
                ) / D(365)
        elif "operating_nwc_ratio" in drivers:
            nwc_ratio = drivers["operating_nwc_ratio"]
            previous_nwc = financials.statement_items.get(
                "operating_nwc", financials.revenue * nwc_ratio
            )
        else:
            nwc_factor = max(
                D("-2"), min(D("2"), drivers.get("nwc_delta_revenue_factor", D(0)))
            )
            previous_nwc = None
        ebit_method = assumptions.calculation_methods.get(
            "ebit", "margin_path_fallback"
        )
        workforce_state: dict[str, dict[str, Decimal]] = {}
        for prefix in ("research", "administrative"):
            if assumptions.calculation_methods.get(f"{prefix}_expense") == "workforce_cost_model_l1_l3":
                workforce_state[prefix] = {
                    "headcount": drivers[f"{prefix}_headcount"],
                    "average_cost": drivers[f"{prefix}_average_cost"],
                }
        rows: list[ForecastYear] = []
        previous_revenue = revenue
        for index in range(request.forecast_years):
            progress = D(index) / D(max(request.forecast_years - 1, 1))
            revenue = previous_revenue * (D(1) + growth_path[index])
            component_items: dict[str, Decimal] = {}
            if ebit_method == "operating_component_build":
                transition_keys = {"operating_cost", "selling_expense"}
                for key in (
                    "operating_cost",
                    "taxes_and_surcharges",
                    "selling_expense",
                    "administrative_expense",
                    "research_expense",
                    "impairment_loss",
                    "other_income",
                ):
                    prefix = (
                        "research" if key == "research_expense"
                        else "administrative" if key == "administrative_expense"
                        else None
                    )
                    if prefix in workforce_state:
                        state = workforce_state[prefix]
                        ramp_years = max(D(1), drivers["workforce_ramp_years"])
                        workforce_progress = min(
                            D(1), D(index) / max(ramp_years - D(1), D(1))
                        )
                        headcount_growth = (
                            drivers[f"{prefix}_headcount_growth_center"]
                            * drivers[f"{prefix}_lifecycle_factor"]
                            * (D(1) - workforce_progress)
                        )
                        wage_growth = (
                            (assumptions.terminal_growth + drivers[f"{prefix}_wage_premium"])
                            * (D(1) - workforce_progress)
                            + assumptions.terminal_growth * workforce_progress
                        )
                        state["headcount"] *= D(1) + headcount_growth
                        state["average_cost"] *= D(1) + wage_growth
                        component_items[key] = (
                            state["headcount"]
                            * state["average_cost"]
                            * (D(1) + drivers[f"{prefix}_non_labor_ratio"])
                        )
                    elif key == "other_income" and drivers.get(
                        "other_income_fixed_amount_mode", D(0)
                    ) == D(1):
                        component_items[key] = drivers["other_income_fixed_base"] * (
                            (D(1) + assumptions.terminal_growth) ** (index + 1)
                        )
                    else:
                        start_ratio = drivers[f"{key}_start_ratio"]
                        target_ratio = drivers[f"{key}_target_ratio"]
                        ratio = (
                            start_ratio + (target_ratio - start_ratio) * progress
                            if key in transition_keys
                            else target_ratio
                        )
                        component_items[key] = revenue * ratio
                ebit = (
                    revenue
                    - component_items["operating_cost"]
                    - component_items["taxes_and_surcharges"]
                    - component_items["selling_expense"]
                    - component_items["administrative_expense"]
                    - component_items["research_expense"]
                    - component_items["impairment_loss"]
                    + component_items["other_income"]
                )
                ebit_margin = ebit / revenue
            else:
                ebit_margin = margin_path[index]
                ebit = revenue * ebit_margin
            tax_rate = (
                assumptions.tax_rate_path[index]
                if assumptions.tax_rate_path
                else financials.tax_rate
            )
            # Losses do not create an immediate cash tax refund by default.
            # Tax-loss carryforwards require their own dated evidence/model.
            nopat = ebit - max(ebit, ZERO) * tax_rate
            if rates and asset_total > 0:
                da_components = {
                    name: asset_balances[name] * rates[name] * da_rate_multiplier
                    for name in rates
                }
                da = sum(da_components.values(), D(0))
                right_of_use_da = da_components.get("right_of_use", D(0))
                da_method = "asset_rollforward"
            else:
                da_ratio = current_da_ratio + (
                    terminal_da_ratio - current_da_ratio
                ) * progress
                da = revenue * da_ratio * da_rate_multiplier
                right_of_use_ratio = (
                    financials.statement_items.get("depreciation_right_of_use", D(0))
                    / financials.revenue
                )
                right_of_use_da = min(da, revenue * right_of_use_ratio)
                da_method = "revenue_ratio_fallback"
            cash_asset_da = max(D(0), da - right_of_use_da)

            if capex_method == "alpha_da_plus_kappa_delta_revenue" and capex_kappa is not None:
                remaining = request.forecast_years - index
                convergence = (
                    D(5 - remaining) / D(4) if remaining <= 4 else D(0)
                )
                effective_alpha = capex_alpha + (D(1) - capex_alpha) * convergence
                effective_kappa = capex_kappa * (D(1) - convergence)
                capex = max(
                    D(0),
                    effective_alpha * cash_asset_da
                    + (revenue - previous_revenue) * effective_kappa,
                )
                capex_formula = "alpha_da_plus_kappa_delta_revenue_terminal_convergence"
            else:
                terminal_cash_da_ratio = max(
                    D(0), terminal_da_ratio - (right_of_use_da / revenue)
                )
                capex_ratio = current_capex_ratio + (
                    terminal_cash_da_ratio - current_capex_ratio
                ) * progress
                capex = revenue * capex_ratio
                effective_alpha = D(1)
                effective_kappa = D(0)
                capex_formula = "revenue_ratio_converging_to_da"

            if rates and asset_total > 0:
                for name, balance in asset_balances.items():
                    cash_capex = capex * cash_asset_shares.get(name, D(0))
                    asset_balances[name] = max(
                        D(0),
                        balance
                        + cash_capex
                        - balance * rates[name] * da_rate_multiplier,
                    )

            operating_nwc = None
            if nwc_method == "dso_dio_dpo":
                operating_nwc = revenue * (
                    dso + dio * cost_ratio - dpo * cost_ratio
                ) / D(365)
                change_nwc = operating_nwc - previous_nwc
                previous_nwc = operating_nwc
            elif "operating_nwc_ratio" in drivers:
                operating_nwc = revenue * nwc_ratio
                change_nwc = operating_nwc - previous_nwc
                previous_nwc = operating_nwc
            else:
                change_nwc = (revenue - previous_revenue) * nwc_factor
            fcff = nopat + da - change_nwc - capex
            operating_items = {key: _q(value) for key, value in component_items.items()}
            if not component_items:
                for key in (
                    "operating_cost",
                    "taxes_and_surcharges",
                    "selling_expense",
                    "administrative_expense",
                    "research_expense",
                    "impairment_loss",
                    "other_income",
                ):
                    if key in financials.statement_items:
                        operating_items[key] = _q(
                            revenue * financials.statement_items[key] / financials.revenue
                        )
            if operating_nwc is not None:
                operating_items["operating_nwc"] = _q(operating_nwc)
            operating_items["capex_alpha"] = _q(effective_alpha)
            operating_items["capex_kappa"] = _q(effective_kappa)
            operating_items["depreciation_right_of_use"] = _q(right_of_use_da)
            operating_items["cash_asset_depreciation"] = _q(cash_asset_da)
            rows.append(
                ForecastYear(
                    year=financials.period_end.year + index + 1,
                    discount_period=D(index + 1),
                    cash_flow_fraction=D(1),
                    revenue=_q(revenue),
                    revenue_growth=_q(growth_path[index]),
                    ebit_margin=_q(ebit_margin),
                    ebit=_q(ebit),
                    tax_rate=_q(tax_rate),
                    nopat=_q(nopat),
                    depreciation_amortization=_q(da),
                    capital_expenditure=_q(capex),
                    change_operating_nwc=_q(change_nwc),
                    fcff=_q(fcff),
                    operating_items=operating_items,
                    calculation_methods={
                        "revenue": "finance_team_two_stage",
                        "ebit": ebit_method,
                        "tax": "policy_transition",
                        "depreciation_amortization": da_method,
                        "capex": capex_formula,
                        "change_operating_nwc": nwc_method,
                    },
                )
            )
            previous_revenue = revenue
        return rows

    def forecast(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        assumptions: AssumptionSet,
    ) -> list[ForecastYear]:
        if self._use_reference_compatibility(request):
            return self.reference.forecast(request, financials, assumptions)
        return self._forecast_scenario(request, financials, assumptions, "base")

    @staticmethod
    def _dcf_values(
        financials: FinancialSnapshot,
        forecast: list[ForecastYear],
        wacc: Decimal,
        terminal_growth: Decimal,
        operating_cash: Decimal = ZERO,
        stable_roic: Decimal | None = None,
        bridge_policy: str = "use_disclosed_book_values",
        discount_policy: str = "year_end",
    ) -> tuple:
        if wacc <= ZERO or terminal_growth >= wacc:
            raise ValueError("WACC必须为正，且永续增长率必须低于WACC。")
        pv_explicit = sum(
            (
                row.fcff
                * row.cash_flow_fraction
                / ((D(1) + wacc) ** row.discount_period)
                for row in forecast
            ),
            D(0),
        )
        if stable_roic is not None:
            terminal_cash_flow = normalized_terminal_cash_flow(
                forecast[-1].nopat,
                terminal_growth,
                stable_roic,
            )
            terminal_nopat = terminal_cash_flow.nopat
            terminal_reinvestment_rate = terminal_cash_flow.reinvestment_rate
            terminal_reinvestment = terminal_cash_flow.reinvestment
            terminal_fcff = terminal_cash_flow.fcff
        else:
            terminal_nopat = forecast[-1].nopat * (D(1) + terminal_growth)
            terminal_reinvestment_rate = D(0)
            terminal_reinvestment = D(0)
            terminal_fcff = forecast[-1].fcff * (D(1) + terminal_growth)
        terminal_value = terminal_fcff / (wacc - terminal_growth)
        terminal_period = forecast[-1].discount_period
        if discount_policy == "annual_midyear_remaining":
            terminal_period += forecast[-1].cash_flow_fraction / D(2)
        pv_terminal = terminal_value / ((D(1) + wacc) ** terminal_period)
        enterprise = pv_explicit + pv_terminal
        bridge = resolve_equity_bridge(
            financials,
            operating_cash,
            policy=bridge_policy,
        )
        equity = bridge.equity_value(enterprise)
        per_share = equity / bridge.share_count
        terminal_share = pv_terminal / enterprise if enterprise else D(0)
        terminal_ebitda = (
            forecast[-1].ebit + forecast[-1].depreciation_amortization
        )
        implied_multiple = (
            terminal_value / terminal_ebitda if terminal_ebitda > 0 else None
        )
        return (
            enterprise,
            equity,
            per_share,
            terminal_share,
            pv_explicit,
            pv_terminal,
            implied_multiple,
            terminal_value,
            terminal_nopat,
            terminal_reinvestment_rate,
            terminal_reinvestment,
            terminal_fcff,
            bridge,
        )

    def dcf(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        assumptions: AssumptionSet,
        forecast: list[ForecastYear],
    ) -> DcfResult:
        if self._use_reference_compatibility(request):
            return self.reference.dcf(request, financials, assumptions, forecast)
        scenario_values: dict[str, Decimal] = {}
        base_values = None
        scenario_warnings: list[str] = []
        operating_cash = (
            financials.revenue
            * assumptions.operating_drivers.get("operating_cash_ratio", D(0))
        )
        for scenario in ("pessimistic", "base", "optimistic"):
            rows = forecast if scenario == "base" else self._forecast_scenario(
                request, financials, assumptions, scenario
            )
            if any(row.fcff <= D("-100000000000000") for row in rows):
                scenario_warnings.append(f"{scenario}情景FCFF异常，请复核经营假设。")
            values = self._dcf_values(
                financials,
                rows,
                assumptions.wacc,
                assumptions.terminal_growth,
                operating_cash,
                assumptions.operating_drivers.get("stable_roic"),
                request.assumptions.equity_bridge_policy,
                request.discount_policy,
            )
            scenario_values[scenario] = values[2]
            if scenario == "base":
                base_values = values
        if len(set(scenario_values.values())) == 1:
            scenario_warnings.append(
                "悲观、中性、乐观情景使用了相同经营预测路径，经营情景区间退化为单点；"
                "最终区间仍保留WACC数据质量压力测试。请补充不同经营情景后再把经营区间"
                "解释为情景分布。"
            )
        assert base_values is not None
        (
            enterprise,
            equity,
            per_share,
            terminal_share,
            pv_explicit,
            pv_terminal,
            implied,
            terminal_value,
            terminal_nopat,
            terminal_reinvestment_rate,
            terminal_reinvestment,
            terminal_fcff,
            bridge_inputs,
        ) = base_values
        scenario_warnings.extend(bridge_inputs.warnings)
        range_values = list(scenario_values.values())
        lease_value = financials.lease_liabilities
        if lease_value is None:
            lease_value = financials.statement_items.get("lease_liabilities")
        debt_method = financials.calculation_methods.get("interest_bearing_debt", "")
        if (
            lease_value is not None
            and lease_value > ZERO
            and financials.interest_bearing_debt_includes_leases is None
            and "lease_liabilities" not in debt_method
        ):
            range_values.append(per_share + lease_value / bridge_inputs.share_count)
            scenario_warnings.append(
                "有息负债是否已包含租赁负债尚未核实；区间上界额外纳入"
                "租赁已包含时的每股差额，基准情景按单列计债。"
            )
        wacc_range_delta = assumptions.operating_drivers.get(
            "wacc_range_delta", D("0.01")
        )
        for stressed_wacc in (
            assumptions.wacc - wacc_range_delta,
            assumptions.wacc + wacc_range_delta,
        ):
            if stressed_wacc > max(ZERO, assumptions.terminal_growth):
                range_values.append(
                    self._dcf_values(
                        financials,
                        forecast,
                        stressed_wacc,
                        assumptions.terminal_growth,
                        operating_cash,
                        assumptions.operating_drivers.get("stable_roic"),
                        request.assumptions.equity_bridge_policy,
                        request.discount_policy,
                    )[2]
                )
        if wacc_range_delta > D("0.01"):
            scenario_warnings.append(
                f"WACC输入未完全核验或已过期，估值区间额外纳入±{wacc_range_delta:.2%}压力测试。"
            )
        exit_multiple = assumptions.operating_drivers.get("exit_multiple")
        exit_cross_check = None
        exit_per_share = None
        terminal_method_gap = None
        if exit_multiple is not None:
            terminal_ebitda = forecast[-1].ebit + forecast[-1].depreciation_amortization
            exit_terminal_value = terminal_ebitda * exit_multiple
            exit_terminal_period = forecast[-1].discount_period
            if request.discount_policy == "annual_midyear_remaining":
                exit_terminal_period += forecast[-1].cash_flow_fraction / D(2)
            exit_pv_terminal = exit_terminal_value / (
                (D(1) + assumptions.wacc) ** exit_terminal_period
            )
            exit_cross_check = pv_explicit + exit_pv_terminal
            exit_equity = bridge_inputs.equity_value(exit_cross_check)
            exit_per_share = exit_equity / bridge_inputs.share_count
            if per_share != 0:
                terminal_method_gap = (exit_per_share - per_share) / abs(per_share)
                if abs(terminal_method_gap) > D("0.20"):
                    scenario_warnings.append(
                        f"Gordon法与退出倍数交叉校验的每股价值相差{abs(terminal_method_gap):.2%}，"
                        "应复核永续增长率、WACC及终值EBITDA口径。"
                    )
        bridge_rows = bridge_inputs.entries(enterprise)
        bridge_rows.update({
            "cash_and_non_operating_assets": financials.cash_and_non_operating_assets,
            "operating_cash_requirement": -operating_cash,
            "surplus_cash": bridge_inputs.distributable_cash,
            "interest_bearing_debt": -financials.interest_bearing_debt,
        })
        return DcfResult(
            enterprise_value=_q(enterprise),
            equity_value=_q(equity),
            per_share_value=_q(per_share),
            range_low=_q(min(range_values)),
            range_high=_q(max(range_values)),
            terminal_value_share=_q(terminal_share),
            bridge={key: _q(value) for key, value in bridge_rows.items()},
            scenario_warnings=scenario_warnings,
            scenario_values={key: _q(value) for key, value in scenario_values.items()},
            present_value_explicit=_q(pv_explicit),
            present_value_terminal=_q(pv_terminal),
            terminal_value=_q(terminal_value),
            implied_exit_multiple=_q(implied) if implied is not None else None,
            exit_multiple_cross_check=(
                _q(exit_cross_check) if exit_cross_check is not None else None
            ),
            exit_multiple_per_share=(
                _q(exit_per_share) if exit_per_share is not None else None
            ),
            terminal_method_gap=(
                _q(terminal_method_gap) if terminal_method_gap is not None else None
            ),
            terminal_nopat=_q(terminal_nopat),
            terminal_reinvestment_rate=_q(terminal_reinvestment_rate),
            terminal_reinvestment=_q(terminal_reinvestment),
            terminal_fcff=_q(terminal_fcff),
            stable_roic=_q(assumptions.operating_drivers["stable_roic"]),
            bridge_unmeasured_items=list(bridge_inputs.unmeasured_items),
        )

    def relative(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        peers: list[PeerCompany],
    ) -> list[MultipleResult]:
        if self._use_reference_compatibility(request):
            return self.reference.relative(request, financials, peers)

        def calculate(
            method: ValuationMethod,
            metric: str,
            target_value: Decimal,
            *,
            enterprise_multiple: bool = False,
        ) -> MultipleResult:
            rows = [
                (peer, getattr(peer, metric))
                for peer in peers
                if getattr(peer, metric) is not None and getattr(peer, metric) > 0
            ]
            original_size = len(rows)
            if target_value <= 0:
                return MultipleResult(
                    method=method.value,
                    status="not_applicable",
                    original_sample_size=original_size,
                    sample_quality="insufficient",
                    peer_tickers=[peer.ticker for peer, _ in rows],
                    reason=f"目标公司的{metric.upper()}对应财务指标非正。",
                )
            if original_size < 3:
                return MultipleResult(
                    method=method.value,
                    status="not_applicable",
                    original_sample_size=original_size,
                    sample_size=original_size,
                    sample_quality="insufficient",
                    peer_tickers=[peer.ticker for peer, _ in rows],
                    reason=f"有效{metric.upper()}可比公司仅{original_size}家，正式区间至少需要3家。",
                )

            retained = rows
            if original_size >= 5:
                multiples = [value for _, value in rows]
                q1 = _percentile(multiples, D("0.25"))
                q3 = _percentile(multiples, D("0.75"))
                iqr = q3 - q1
                low_fence = max(D(0), q1 - D("1.5") * iqr)
                high_fence = q3 + D("1.5") * iqr
                candidate = [
                    (peer, value)
                    for peer, value in rows
                    if low_fence <= value <= high_fence
                ]
                if len(candidate) >= 3:
                    retained = candidate

            values = []
            bridge_inputs = (
                resolve_equity_bridge(
                    financials,
                    policy=request.assumptions.equity_bridge_policy,
                )
                if enterprise_multiple
                else None
            )
            share_count = (
                bridge_inputs.share_count
                if bridge_inputs is not None
                else financials.diluted_shares or financials.common_shares
            )
            for _, multiple in retained:
                equity_value = multiple * target_value
                if enterprise_multiple:
                    equity_value = bridge_inputs.equity_value(equity_value)
                values.append(equity_value / share_count)
            sample_size = len(values)
            outlier_count = original_size - sample_size
            quality = "adequate" if sample_size >= 5 else "limited"
            note_parts = []
            if quality == "limited":
                note_parts.append("有效样本不足5家，区间稳定性有限")
            if outlier_count:
                note_parts.append(f"按1.5×IQR规则剔除{outlier_count}个异常倍数")
            if bridge_inputs is not None:
                note_parts.extend(bridge_inputs.warnings)
            return MultipleResult(
                method=method.value,
                status="success",
                per_share_value=_q(_percentile(values, D("0.5"))),
                range_low=_q(_percentile(values, D("0.25"))),
                range_high=_q(_percentile(values, D("0.75"))),
                sample_size=sample_size,
                original_sample_size=original_size,
                outlier_count=outlier_count,
                sample_quality=quality,
                peer_tickers=[peer.ticker for peer, _ in retained],
                reason="；".join(note_parts) or None,
            )

        results: list[MultipleResult] = []
        if ValuationMethod.PE in request.methods:
            results.append(
                calculate(
                    ValuationMethod.PE,
                    "pe",
                    financials.net_income_parent,
                )
            )
        if ValuationMethod.PS in request.methods:
            results.append(
                calculate(ValuationMethod.PS, "ps", financials.revenue)
            )
        if ValuationMethod.EV_EBITDA in request.methods:
            results.append(
                calculate(
                    ValuationMethod.EV_EBITDA,
                    "ev_ebitda",
                    financials.ebitda,
                    enterprise_multiple=True,
                )
            )
        return results

    def sensitivity(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        assumptions: AssumptionSet,
    ) -> list[SensitivityCell]:
        if self._use_reference_compatibility(request):
            return self.reference.sensitivity(request, financials, assumptions)
        forecast = self._forecast_scenario(request, financials, assumptions, "base")
        cells: list[SensitivityCell] = []
        wacc_step = assumptions.operating_drivers.get("wacc_range_delta", D("0.01"))
        operating_cash = financials.revenue * assumptions.operating_drivers.get(
            "operating_cash_ratio", D(0)
        )
        for wacc_delta in (-wacc_step, D(0), wacc_step):
            wacc = assumptions.wacc + wacc_delta
            for growth_delta in (D("-0.01"), D(0), D("0.01")):
                growth = assumptions.terminal_growth + growth_delta
                stable_roic = assumptions.operating_drivers.get("stable_roic")
                if (
                    wacc <= 0
                    or growth >= wacc
                    or (stable_roic is not None and growth >= stable_roic)
                ):
                    cells.append(
                        SensitivityCell(
                            wacc=_q(wacc), terminal_growth=_q(growth), per_share_value=None, valid=False
                        )
                    )
                    continue
                value = self._dcf_values(
                    financials,
                    forecast,
                    wacc,
                    growth,
                    operating_cash,
                    assumptions.operating_drivers.get("stable_roic"),
                    request.assumptions.equity_bridge_policy,
                    request.discount_policy,
                )[2]
                cells.append(
                    SensitivityCell(
                        wacc=_q(wacc), terminal_growth=_q(growth), per_share_value=_q(value), valid=True
                    )
                )
        return cells

    def sensitivity_studies(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        assumptions: AssumptionSet,
    ) -> list[SensitivityStudy]:
        """Run the document's sensitivity catalogue without manufacturing inputs.

        Studies supported by the current FCFF data contract are recalculated.
        Items that depend on the detailed expense/labour models or unavailable
        market histories are returned explicitly as ``not_available`` so the
        report distinguishes an untested factor from a low-impact factor.
        """

        if self._use_reference_compatibility(request):
            return []

        def per_share(variant: AssumptionSet) -> Decimal | None:
            wacc = variant.wacc
            growth = variant.terminal_growth
            if wacc <= 0 or growth >= wacc:
                return None
            try:
                rows = self._forecast_scenario(request, financials, variant, "base")
                operating_cash = financials.revenue * variant.operating_drivers.get(
                    "operating_cash_ratio", D(0)
                )
                return self._dcf_values(
                    financials,
                    rows,
                    wacc,
                    growth,
                    operating_cash,
                    variant.operating_drivers.get("stable_roic"),
                    request.assumptions.equity_bridge_policy,
                    request.discount_policy,
                )[2]
            except ValueError:
                return None

        baseline = per_share(assumptions)

        def completed(
            study_id: str,
            parameter: str,
            baseline_input: str,
            low_input: str,
            high_input: str,
            low_value: Decimal | None,
            high_value: Decimal | None,
            rationale: str,
        ) -> SensitivityStudy:
            if baseline is None or low_value is None or high_value is None:
                return SensitivityStudy(
                    study_id=study_id,
                    parameter=parameter,
                    baseline_input=baseline_input,
                    low_input=low_input,
                    high_input=high_input,
                    classification="not_available",
                    status="not_available",
                    rationale=rationale + "；扰动后出现g≥WACC或其他不可计算组合。",
                )
            impact = max(
                abs(low_value - baseline), abs(high_value - baseline)
            ) / max(abs(baseline), D("0.0001"))
            classification = (
                "high" if impact >= D("0.05") else "medium" if impact >= D("0.01") else "low"
            )
            return SensitivityStudy(
                study_id=study_id,
                parameter=parameter,
                baseline_input=baseline_input,
                low_input=low_input,
                high_input=high_input,
                baseline_per_share=_q(baseline),
                low_per_share=_q(low_value),
                high_per_share=_q(high_value),
                max_relative_change=_q(impact),
                classification=classification,
                status="completed",
                rationale=rationale,
            )

        def unavailable(study_id: str, parameter: str, rationale: str) -> SensitivityStudy:
            return SensitivityStudy(
                study_id=study_id,
                parameter=parameter,
                baseline_input="当前模型口径",
                classification="not_available",
                status="not_available",
                rationale=rationale,
            )

        studies: list[SensitivityStudy] = []
        wacc_step = assumptions.operating_drivers.get("wacc_range_delta", D("0.01"))
        low_wacc = assumptions.model_copy(
            update={"wacc": assumptions.wacc - wacc_step}
        )
        high_wacc = assumptions.model_copy(
            update={"wacc": assumptions.wacc + wacc_step}
        )
        studies.append(completed(
            "S1", "WACC", f"{assumptions.wacc:.2%}",
            f"{low_wacc.wacc:.2%}", f"{high_wacc.wacc:.2%}",
            per_share(low_wacc), per_share(high_wacc),
            f"WACC按本次数据质量上下扰动{wacc_step:.2%}。"
        ))

        low_g = assumptions.model_copy(update={"terminal_growth": D("0.02")})
        high_g = assumptions.model_copy(update={"terminal_growth": D("0.04")})
        studies.append(completed(
            "S2", "永续增长率g", f"{assumptions.terminal_growth:.2%}", "2.00%", "4.00%",
            per_share(low_g), per_share(high_g), "按文档固定测试2%、3%、4%档位。"
        ))

        stable_roic = assumptions.operating_drivers["stable_roic"]
        low_roic = max(
            assumptions.terminal_growth + D("0.005"), stable_roic * D("0.8")
        )
        high_roic = min(D(1), stable_roic * D("1.2"))
        low_roic_drivers = dict(
            assumptions.operating_drivers, stable_roic=low_roic
        )
        high_roic_drivers = dict(
            assumptions.operating_drivers, stable_roic=high_roic
        )
        studies.append(completed(
            "S2-roic",
            "稳定期ROIC",
            f"{stable_roic:.2%}",
            f"{low_roic:.2%}",
            f"{high_roic:.2%}",
            per_share(assumptions.model_copy(update={"operating_drivers": low_roic_drivers})),
            per_share(assumptions.model_copy(update={"operating_drivers": high_roic_drivers})),
            "验证永续增长所需再投资率g/ROIC，防止正增长与零净再投资并存。",
        ))

        grid_values = [cell.per_share_value for cell in self.sensitivity(
            request, financials, assumptions
        ) if cell.valid and cell.per_share_value is not None]
        studies.append(completed(
            "S3", "WACC×g双因素矩阵",
            f"{assumptions.wacc:.2%} × {assumptions.terminal_growth:.2%}",
            "矩阵最低值", "矩阵最高值",
            min(grid_values) if grid_values else None,
            max(grid_values) if grid_values else None,
            "九格矩阵保留每个有效组合；本行汇总其最低和最高每股价值。",
        ))

        base_growth = list(assumptions.revenue_growth_scenarios.get(
            "base", assumptions.revenue_growth
        ))

        def growth_variant(path: list[Decimal]) -> AssumptionSet:
            scenarios = dict(assumptions.revenue_growth_scenarios)
            scenarios["base"] = path
            return assumptions.model_copy(update={
                "revenue_growth": path,
                "revenue_growth_scenarios": scenarios,
            })

        fade_denominator = D(max(len(base_growth) - 1, 1))
        low_start = [
            value - D("0.01") * (D(1) - D(index) / fade_denominator)
            for index, value in enumerate(base_growth)
        ]
        high_start = [
            value + D("0.01") * (D(1) - D(index) / fade_denominator)
            for index, value in enumerate(base_growth)
        ]
        low_start[-1] = high_start[-1] = assumptions.terminal_growth
        studies.append(completed(
            "S4-start", "收入预测起点", f"{base_growth[0]:.2%}",
            f"{low_start[0]:.2%}", f"{high_start[0]:.2%}",
            per_share(growth_variant(low_start)), per_share(growth_variant(high_start)),
            "首年增速上下扰动1个百分点，影响线性衰减并在末年回到同一永续增速。",
        ))

        drivers = assumptions.operating_drivers
        if {"revenue_center", "revenue_cagr", "revenue_decay_years"} <= set(drivers):
            center = drivers["revenue_center"]
            cagr = drivers["revenue_cagr"]
            industry_growth = D(str(assumptions.industry_parameters["domestic_growth"]))
            decay_years = int(drivers["revenue_decay_years"])

            def rho_path(rho: Decimal, anchor: Decimal) -> list[Decimal]:
                start = center + rho * (anchor - center)
                return self.revenue_model._path(
                    start, industry_growth, assumptions.terminal_growth,
                    decay_years, request.forecast_years,
                )

            studies.append(completed(
                "S4-rho", "收入情景收缩系数ρ", f"{drivers['revenue_rho']}", "0.4", "1.0",
                per_share(growth_variant(rho_path(D("0.4"), cagr))),
                per_share(growth_variant(rho_path(D("1.0"), cagr))),
                "保持中性锚=CAGR，仅改变ρ，验证情景机制本身。",
            ))
            a3 = drivers.get("revenue_a3", cagr)
            studies.append(completed(
                "S4-anchor", "中性收入锚", "CAGR", "CAGR", "A3",
                per_share(growth_variant(rho_path(drivers["revenue_rho"], cagr))),
                per_share(growth_variant(rho_path(drivers["revenue_rho"], a3))),
                "比较中性锚使用长期CAGR与近期A3，避免只展示机械情景区间。",
            ))
        else:
            studies.append(unavailable(
                "S4-rho", "收入情景收缩系数ρ与锚",
                "本次收入路径由用户直接指定，没有CAGR/A3/center元数据，不能伪造ρ与锚敏感性。",
            ))

        if assumptions.calculation_methods.get("ebit") == "operating_component_build":
            cost_target = drivers["operating_cost_target_ratio"]
            low_cost = dict(
                drivers,
                operating_cost_target_ratio=max(D(0), cost_target - D("0.02")),
            )
            high_cost = dict(
                drivers,
                operating_cost_target_ratio=cost_target + D("0.02"),
            )
            if "operating_cost_ratio" in drivers:
                low_cost["operating_cost_ratio"] = max(
                    D(0), drivers["operating_cost_ratio"] - D("0.02")
                )
                high_cost["operating_cost_ratio"] = (
                    drivers["operating_cost_ratio"] + D("0.02")
                )
            studies.append(completed(
                "S5", "稳态营业成本率CR", f"{cost_target:.2%}",
                f"{max(D(0), cost_target-D('0.02')):.2%}",
                f"{cost_target+D('0.02'):.2%}",
                per_share(assumptions.model_copy(update={"operating_drivers": low_cost})),
                per_share(assumptions.model_copy(update={"operating_drivers": high_cost})),
                "稳态营业成本率上下扰动2个百分点，并同步调整周转天数公式中的营业成本率。",
            ))
        else:
            studies.append(unavailable(
                "S5", "稳态营业成本率CR", "本次EBIT因经营科目不完整使用利润率路径，无法单独扰动营业成本率。"
            ))

        low_tax = assumptions.model_copy(update={
            "tax_rate_path": [D("0.15")] * request.forecast_years
        })
        high_tax = assumptions.model_copy(update={
            "tax_rate_path": [D("0.25")] * request.forecast_years
        })
        studies.append(completed(
            "S6", "所得税率", "政策过渡路径", "15%", "25%",
            per_share(low_tax), per_share(high_tax),
            "比较高新税率与一般企业税率；正式基准仍采用政策到期与过渡路径。",
        ))

        if "operating_nwc_ratio" in drivers:
            ratio = drivers["operating_nwc_ratio"]
            low_drivers = dict(drivers, operating_nwc_ratio=ratio - D("0.01"))
            high_drivers = dict(drivers, operating_nwc_ratio=ratio + D("0.01"))
            studies.append(completed(
                "S7", "经营营运资本/收入", f"{ratio:.2%}",
                f"{ratio-D('0.01'):.2%}", f"{ratio+D('0.01'):.2%}",
                per_share(assumptions.model_copy(update={"operating_drivers": low_drivers})),
                per_share(assumptions.model_copy(update={"operating_drivers": high_drivers})),
                "等价比例口径上下扰动1个百分点。",
            ))
        elif {"dso_days", "dio_days", "dpo_days"} <= set(drivers):
            low_drivers = dict(drivers)
            high_drivers = dict(drivers)
            for key in ("dso_days", "dio_days"):
                low_drivers[key] = drivers[key] * D("0.9")
                high_drivers[key] = drivers[key] * D("1.1")
            low_drivers["dpo_days"] = drivers["dpo_days"] * D("1.1")
            high_drivers["dpo_days"] = drivers["dpo_days"] * D("0.9")
            studies.append(completed(
                "S7", "DSO/DIO/DPO", "历史或用户基准", "现金转换周期改善10%", "现金转换周期恶化10%",
                per_share(assumptions.model_copy(update={"operating_drivers": low_drivers})),
                per_share(assumptions.model_copy(update={"operating_drivers": high_drivers})),
                "应收与存货周转天数、应付周转天数按相反方向扰动10%。",
            ))
        else:
            studies.append(unavailable(
                "S7", "NWC率/周转天数", "本次仅有ΔNWC/Δ收入低质量降级，无法形成文档要求的周转敏感性。"
            ))

        workforce_prefixes = [
            prefix for prefix in ("research", "administrative")
            if assumptions.calculation_methods.get(f"{prefix}_expense")
            == "workforce_cost_model_l1_l3"
        ]
        if workforce_prefixes:
            low_people = dict(drivers)
            high_people = dict(drivers)
            for prefix in workforce_prefixes:
                low_people[f"{prefix}_wage_premium"] = D(0)
                high_people[f"{prefix}_wage_premium"] = D("0.02")
                low_people[f"{prefix}_headcount_growth_center"] = (
                    drivers[f"{prefix}_headcount_growth_center"] - D("0.01")
                )
                high_people[f"{prefix}_headcount_growth_center"] = (
                    drivers[f"{prefix}_headcount_growth_center"] + D("0.01")
                )
            studies.append(completed(
                "S8", "研发/管理用工参数", "Δw与人数中枢基准",
                "Δw=0%，人数中枢-1pp", "Δw=2%，人数中枢+1pp",
                per_share(assumptions.model_copy(update={"operating_drivers": low_people})),
                per_share(assumptions.model_copy(update={"operating_drivers": high_people})),
                "按文档同时覆盖调薪溢价0%/2%和职能人数历史中枢±1个百分点。",
            ))
        else:
            studies.append(unavailable(
                "S8", "研发与管理费用自由参数", "缺少职能人数、人均成本、非人力结构比或ramp等完整用工模型输入。"
            ))

        dcf_result = self.dcf(
            request,
            financials,
            assumptions,
            self._forecast_scenario(request, financials, assumptions, "base"),
        )
        if dcf_result.exit_multiple_per_share is not None and baseline is not None:
            exit_forecast = self._forecast_scenario(
                request, financials, assumptions, "base"
            )
            exit_operating_cash = financials.revenue * drivers.get(
                "operating_cash_ratio", D(0)
            )
            pv_explicit = self._dcf_values(
                financials,
                exit_forecast,
                assumptions.wacc,
                assumptions.terminal_growth,
                exit_operating_cash,
                assumptions.operating_drivers.get("stable_roic"),
                request.assumptions.equity_bridge_policy,
                request.discount_policy,
            )[4]
            exit_bridge = resolve_equity_bridge(
                financials,
                exit_operating_cash,
                policy=request.assumptions.equity_bridge_policy,
            )

            def exit_price(multiple: Decimal) -> Decimal:
                terminal_ebitda = (
                    exit_forecast[-1].ebit
                    + exit_forecast[-1].depreciation_amortization
                )
                exit_terminal_period = exit_forecast[-1].discount_period
                if request.discount_policy == "annual_midyear_remaining":
                    exit_terminal_period += exit_forecast[-1].cash_flow_fraction / D(2)
                pv_terminal = terminal_ebitda * multiple / (
                    (D(1) + assumptions.wacc) ** exit_terminal_period
                )
                equity = exit_bridge.equity_value(pv_explicit + pv_terminal)
                return equity / exit_bridge.share_count

            low_multiple = drivers.get(
                "exit_multiple_low", drivers["exit_multiple"]
            )
            high_multiple = drivers.get(
                "exit_multiple_high", drivers["exit_multiple"]
            )
            studies.append(completed(
                "S9", "终值方法", "Gordon",
                f"退出倍数{low_multiple}x", f"退出倍数{high_multiple}x",
                exit_price(low_multiple), exit_price(high_multiple),
                "展示行业退出倍数区间相对Gordon基准的差异；退出倍数只用于交叉校验。",
            ))
        else:
            studies.append(unavailable(
                "S9", "终值方法", "行业没有文档支持的退出倍数区间，且用户未指定倍数。"
            ))

        if len(base_growth) >= 3:
            start_growth = base_growth[0]
            terminal_growth = assumptions.terminal_growth
            periods = len(base_growth)
            exponential_q = D("0.7")
            exponential_denominator = D(1) - exponential_q ** D(periods - 1)
            exponential_path = []
            s_curve_path = []
            for index in range(periods):
                x = D(index) / D(periods - 1)
                exponential_weight = (
                    exponential_q ** D(index) - exponential_q ** D(periods - 1)
                ) / exponential_denominator
                smoothstep = D(3) * x * x - D(2) * x * x * x
                s_curve_weight = D(1) - smoothstep
                exponential_path.append(
                    terminal_growth
                    + (start_growth - terminal_growth) * exponential_weight
                )
                s_curve_path.append(
                    terminal_growth
                    + (start_growth - terminal_growth) * s_curve_weight
                )
            exponential_path[-1] = terminal_growth
            s_curve_path[-1] = terminal_growth
            shape_values = [
                ("指数衰减", per_share(growth_variant(exponential_path))),
                ("S型衰减", per_share(growth_variant(s_curve_path))),
            ]
            calculable_shapes = [item for item in shape_values if item[1] is not None]
            if len(calculable_shapes) == 2:
                low_shape = min(calculable_shapes, key=lambda item: item[1])
                high_shape = max(calculable_shapes, key=lambda item: item[1])
                studies.append(completed(
                    "S10", "收入衰减形状", "线性/既定基准路径",
                    low_shape[0], high_shape[0], low_shape[1], high_shape[1],
                    "固定首年增速、末年永续增速与预测期，仅比较线性、指数和S型的收敛形状。",
                ))
        else:
            studies.append(unavailable(
                "S10", "收入衰减形状", "预测期少于3年，无法稳定比较线性、指数与S型衰减。"
            ))

        historical_rows = self._history(request, financials)
        historical_growth = [
            current.revenue / previous.revenue - D(1)
            for previous, current in pairwise(historical_rows)
            if previous.revenue > 0
        ]
        window_candidates: list[tuple[str, Decimal | None]] = []
        industry_growth = D(str(
            assumptions.industry_parameters.get(
                "domestic_growth", assumptions.terminal_growth
            )
        ))
        decay_years = int(drivers.get("revenue_decay_years", D(request.forecast_years)))
        for years in (3, 5, 10):
            observation_count = years - 1
            if len(historical_growth) < observation_count:
                continue
            observations = historical_growth[-observation_count:]
            for statistic, start in (
                ("均值", _mean(observations)),
                ("中位数", D(str(median(observations)))),
            ):
                path = self.revenue_model._path(
                    start,
                    industry_growth,
                    assumptions.terminal_growth,
                    decay_years,
                    request.forecast_years,
                )
                window_candidates.append(
                    (f"近{years}年{statistic}", per_share(growth_variant(path)))
                )
        valid_windows = [item for item in window_candidates if item[1] is not None]
        if len(valid_windows) >= 2:
            low_window = min(valid_windows, key=lambda item: item[1])
            high_window = max(valid_windows, key=lambda item: item[1])
            studies.append(completed(
                "S11", "均值窗口与统计量", "当前十年收入模型",
                low_window[0], high_window[0], low_window[1], high_window[1],
                "在可用历史范围内比较近3/5/10年窗口的均值与中位数，并对每条路径全模型重算。",
            ))
        else:
            studies.append(unavailable(
                "S11", "均值窗口与统计量", "可比历史增速不足，无法形成至少两组可复算窗口。"
            ))

        low_da = dict(drivers, da_rate_multiplier=D("0.8"))
        high_da = dict(drivers, da_rate_multiplier=D("1.2"))
        studies.append(completed(
            "S15", "D&A折旧摊销率", "100%", "80%", "120%",
            per_share(assumptions.model_copy(update={"operating_drivers": low_da})),
            per_share(assumptions.model_copy(update={"operating_drivers": high_da})),
            "按文档对D&A参数η上下扰动20%。",
        ))

        if "capex_kappa" in drivers:
            low_capex = dict(drivers, capex_alpha=D(1), capex_kappa=D(0))
            high_capex = dict(drivers, capex_alpha=D("1.1"))
            studies.append(completed(
                "S16", "CapEx α/κ", f"α={drivers['capex_alpha']}, κ={drivers['capex_kappa']}",
                "α=1, κ=0", f"α=1.1, κ={drivers['capex_kappa']}",
                per_share(assumptions.model_copy(update={"operating_drivers": low_capex})),
                per_share(assumptions.model_copy(update={"operating_drivers": high_capex})),
                "覆盖文档给出的α=1/1.1与κ=0/历史中位数边界。",
            ))
        else:
            studies.append(unavailable(
                "S16", "CapEx α/κ", "κ有效历史样本不足，基准已降级到收入比例法。"
            ))

        terminal_tax = assumptions.tax_rate_path[-1] if assumptions.tax_rate_path else financials.tax_rate
        immediate_tax = assumptions.model_copy(update={
            "tax_rate_path": [terminal_tax] * request.forecast_years
        })
        five_year_tax = assumptions.model_copy(update={
            "tax_rate_path": [
                financials.tax_rate + (terminal_tax - financials.tax_rate)
                * D(min(index + 1, 5)) / D(5)
                for index in range(request.forecast_years)
            ]
        })
        studies.append(completed(
            "S17", "税率过渡期", "基准政策路径", "0年", "5年",
            per_share(immediate_tax), per_share(five_year_tax),
            "比较立即切换与五年线性过渡。",
        ))

        if assumptions.calculation_methods.get("ebit") == "operating_component_build":
            low_a = dict(drivers)
            high_a = dict(drivers)
            for key in ("selling_expense_target_ratio", "taxes_and_surcharges_target_ratio"):
                low_a[key] = max(D(0), drivers[key] - D("0.005"))
                high_a[key] = drivers[key] + D("0.005")
            studies.append(completed(
                "S12", "A类费用固定比率", "历史窗口均值",
                "销售费用率与税金率各-0.5pp", "销售费用率与税金率各+0.5pp",
                per_share(assumptions.model_copy(update={"operating_drivers": low_a})),
                per_share(assumptions.model_copy(update={"operating_drivers": high_a})),
                "按文档较宽档位同时扰动销售费用率与税金及附加率。",
            ))

        if workforce_prefixes:
            low_structure = dict(drivers)
            high_structure = dict(drivers)
            for prefix in workforce_prefixes:
                low_structure[f"{prefix}_non_labor_ratio"] = (
                    drivers[f"{prefix}_non_labor_ratio"] * D("0.8")
                )
                high_structure[f"{prefix}_non_labor_ratio"] = (
                    drivers[f"{prefix}_non_labor_ratio"] * D("1.2")
                )
            studies.append(completed(
                "S13", "B类费用非人力结构比", "基期结构",
                "非人力结构比-20%", "非人力结构比+20%",
                per_share(assumptions.model_copy(update={"operating_drivers": low_structure})),
                per_share(assumptions.model_copy(update={"operating_drivers": high_structure})),
                "按文档对研发/管理非人力结构比做相对±20%扰动。",
            ))
            short_ramp = dict(drivers, workforce_ramp_years=D(3))
            long_ramp = dict(drivers, workforce_ramp_years=D(8))
            studies.append(completed(
                "S14", "用工参数ramp长度", f"{drivers['workforce_ramp_years']}年", "3年", "8年",
                per_share(assumptions.model_copy(update={"operating_drivers": short_ramp})),
                per_share(assumptions.model_copy(update={"operating_drivers": long_ramp})),
                "比较3年与8年过渡长度。",
            ))

        if assumptions.calculation_methods.get("ebit") == "operating_component_build":
            fixed_other = dict(
                drivers,
                other_income_fixed_amount_mode=D(1),
                other_income_fixed_base=financials.statement_items.get("other_income", D(0)),
            )
            studies.append(completed(
                "S18", "其他收益归属/增长口径", "按收入比例",
                "按收入比例", "固定额按g增长",
                baseline,
                per_share(assumptions.model_copy(update={"operating_drivers": fixed_other})),
                "比较其他收益随收入同比例与按基期固定额随g增长两种经营归属路径。",
            ))

        market_cap_low = (
            request.assumptions.market_cap_period_low
            or financials.statement_items.get("market_cap_period_low")
        )
        market_cap_high = (
            request.assumptions.market_cap_period_high
            or financials.statement_items.get("market_cap_period_high")
        )
        market_cap_base = (
            request.assumptions.annual_average_market_cap
            or financials.statement_items.get("annual_average_market_cap")
            or request.assumptions.quarterly_average_market_cap
            or financials.statement_items.get("quarterly_average_market_cap")
            or request.assumptions.market_cap
            or financials.statement_items.get("market_cap")
        )
        wacc_components = assumptions.wacc_components
        required_wacc_components = {
            "cost_of_equity", "debt_cost", "marginal_tax_rate"
        }
        if (
            market_cap_low is not None
            and market_cap_high is not None
            and market_cap_base is not None
            and required_wacc_components <= set(wacc_components)
        ):
            debt = financials.interest_bearing_debt

            def wacc_for_market_cap(market_cap: Decimal) -> Decimal:
                total_capital = market_cap + debt
                equity_weight = market_cap / total_capital
                debt_weight = debt / total_capital
                return (
                    equity_weight * wacc_components["cost_of_equity"]
                    + debt_weight * wacc_components["debt_cost"]
                    * (D(1) - wacc_components["marginal_tax_rate"])
                )

            low_market_wacc = wacc_for_market_cap(market_cap_low)
            high_market_wacc = wacc_for_market_cap(market_cap_high)
            variants = [
                ("区间最低市值", per_share(assumptions.model_copy(
                    update={"wacc": low_market_wacc}
                ))),
                ("区间最高市值", per_share(assumptions.model_copy(
                    update={"wacc": high_market_wacc}
                ))),
            ]
            valid_variants = [item for item in variants if item[1] is not None]
            if len(valid_variants) == 2:
                low_value = min(valid_variants, key=lambda item: item[1])
                high_value = max(valid_variants, key=lambda item: item[1])
                studies.append(completed(
                    "S19", "股权市值E取值时点", f"期间均值市值 {market_cap_base:,.0f}",
                    low_value[0], high_value[0], low_value[1], high_value[1],
                    "保持Ke、Kd、税率与有息负债不变，使用估值日前可得市值区间重算资本结构和WACC。",
                ))
        else:
            studies.append(unavailable(
                "S19", "股权市值取值时点",
                "缺少估值日前年度市值均值、期间高低端点或可拆分的WACC组成，不能伪造市值时点敏感性。",
            ))

        zero_cash = dict(drivers, operating_cash_ratio=D(0))
        two_cash = dict(drivers, operating_cash_ratio=D("0.02"))
        studies.append(completed(
            "S20", "经营必需现金", f"{drivers.get('operating_cash_ratio', D(0)):.2%}", "0%收入", "2%收入",
            per_share(assumptions.model_copy(update={"operating_drivers": zero_cash})),
            per_share(assumptions.model_copy(update={"operating_drivers": two_cash})),
            "从现金桥接中扣除0%或2%收入作为经营必需现金。",
        ))

        missing = {
            "S12": ("A类费用固定比率", "本次EBIT未进入逐科目模型，无法单独扰动销售费用率与税金率。"),
            "S13": ("B类费用结构参数", "缺少完整用工成本法输入，无法扰动非人力结构比。"),
            "S14": ("ramp长度", "缺少完整用工成本法输入或经确认的基准ramp。"),
            "S18": ("非经营项目归属", "输入未完成其他收益等项目的逐科目经营归属。"),
        }
        existing_ids = {item.study_id for item in studies}
        for study_id, (parameter, rationale) in missing.items():
            if study_id not in existing_ids:
                studies.append(unavailable(study_id, parameter, rationale))
        return sorted(studies, key=lambda item: (
            int(item.study_id.split("-")[0][1:]), item.study_id
        ))

    def assess_quality(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        peers: list[PeerCompany],
        relative: list[MultipleResult],
        assumptions: AssumptionSet | None = None,
    ) -> DataQualityAssessment:
        """Summarize data and model support without changing valuation numbers."""

        history = self._history(request, financials)
        raw_history = [*request.historical_financials, financials]
        adjusted_years = sorted(
            row.period_end.year
            for row in raw_history
            if row.comparability_status == "adjusted"
        )
        excluded_years = sorted(
            row.period_end.year
            for row in raw_history
            if row.comparability_status == "excluded"
        )
        core_evidence = {
            "revenue",
            "ebit_margin",
            "tax_rate",
            "depreciation_amortization",
            "capital_expenditure",
            "change_operating_nwc",
            "cash_and_non_operating_assets",
            "interest_bearing_debt",
            "common_shares",
            "net_income_parent",
            "ebitda",
        }
        from valuationagent.schemas.models import required_financial_metrics
        core_evidence = required_financial_metrics(request.methods)
        has_dcf = "dcf" in request.methods
        denominator = max(len(history) * len(core_evidence), 1)
        evidence_count = 0
        for row in history:
            available_evidence = set(row.evidence)
            for key in core_evidence:
                if key == "common_shares":
                    evidence_count += int(
                        bool({"common_shares", "diluted_shares"} & available_evidence)
                    )
                else:
                    evidence_count += int(key in available_evidence)
        evidence_coverage = D(evidence_count) / D(denominator)
        notes: list[str] = []
        if has_dcf and assumptions is None:
            assumptions = self.resolve_assumptions(request, financials)
        degraded_fields = []
        if has_dcf and assumptions is not None:
            if assumptions.operating_drivers.get("da_policy_fallback", D(0)):
                degraded_fields.append("depreciation_amortization")
            if assumptions.operating_drivers.get("capex_policy_fallback", D(0)):
                degraded_fields.append("capital_expenditure")
            if (
                assumptions.calculation_methods.get("change_operating_nwc")
                == "policy_delta_revenue_fallback"
            ):
                degraded_fields.append("change_operating_nwc")
            if assumptions.operating_drivers.get("stable_roic_policy_fallback", D(0)):
                degraded_fields.append("stable_roic")
            if assumptions.operating_drivers.get("operating_cash_policy_fallback", D(0)):
                degraded_fields.append("operating_cash_requirement")
        degraded_fields = sorted(degraded_fields)
        if degraded_fields:
            notes.append(
                "估值关键输入已降级：" + "、".join(degraded_fields)
                + "；具体代理值或政策值见假设与计算口径。"
            )
        if has_dcf and len(history) < 10:
            notes.append(f"仅有{len(history)}个可用历史年度，长期统计量样本有限。")
        if adjusted_years:
            notes.append("调整后年度已进入模型，需结合comparability_note复核。")
        if excluded_years:
            notes.append("不可比年度已排除，统计窗口相应缩短。")
        if evidence_coverage < D("0.8"):
            notes.append("核心历史字段的逐项证据覆盖不足80%。")

        industry_quality = "unknown"
        metadata_completeness = "unknown"
        if has_dcf and not self._use_reference_compatibility(request):
            industry = self.registry.resolve(request.company.industry)
            industry_quality = industry.quality
            metadata_completeness = industry.metadata_completeness
            if metadata_completeness != "complete":
                notes.append("行业参数保留版本与文件哈希，但样本期、样本量和置信区间尚未补全。")
            if industry.provenance_note:
                notes.append(industry.provenance_note)

        requested_relative = any(method != ValuationMethod.DCF for method in request.methods)
        if requested_relative and peers:
            notes.append(f"相对估值候选池共{len(peers)}家公司，质量结论按各倍数有效样本计算。")
        peer_provenance_missing = requested_relative and any(not p.evidence or p.as_of_date is None or p.multiple_basis == "unknown" for p in peers)
        if peer_provenance_missing:
            notes.append("部分可比公司缺少定价日、分母口径或原文证据，须人工核验，不能视为已核验同业。")
        bridge_book_proxy = any(
            item.rule_id == "EQUITY_BRIDGE_BOOK_VALUE_PROXY"
            for item in validate_equity_bridge_inputs(request, financials)
        )
        bridge_unmeasured = False
        bridge_lease_ambiguous = False
        if {"dcf", "ev_ebitda"} & {str(method) for method in request.methods}:
            bridge_unmeasured = bool(
                resolve_equity_bridge(
                    financials,
                    policy=request.assumptions.equity_bridge_policy,
                ).unmeasured_items
            )
            lease_value = financials.lease_liabilities
            if lease_value is None:
                lease_value = financials.statement_items.get("lease_liabilities")
            bridge_lease_ambiguous = bool(
                lease_value is not None
                and lease_value > ZERO
                and financials.interest_bearing_debt_includes_leases is None
                and "lease_liabilities" not in financials.calculation_methods.get(
                    "interest_bearing_debt", ""
                )
            )
        if bridge_book_proxy:
            notes.append("股权价值桥接使用已披露账面值代理市场价值，结果等级最高为B。")
        if bridge_unmeasured:
            notes.append("部分股权桥接项未建立，未将缺失值表述为确认0；结果按C级披露。")
        if bridge_lease_ambiguous:
            notes.append("有息债务是否包含租赁负债尚未核实，基准值和估值区间均披露该不确定性。")
        successful_quality = [
            item.sample_quality for item in relative if item.status == "success"
        ]
        if not requested_relative:
            peer_quality = "not_requested"
        elif not successful_quality:
            peer_quality = "insufficient"
            notes.append("相对估值没有形成满足最小样本要求的有效方法。")
        elif "limited" in successful_quality:
            peer_quality = "limited"
            notes.append("至少一种相对估值方法仅有3至4家有效可比公司。")
        else:
            peer_quality = "adequate"

        if not has_dcf:
            market_input_quality = "not_applicable"
        else:
            market_as_of = request.assumptions.market_inputs_as_of
            market_source = request.assumptions.market_inputs_source.strip()
            market_components_complete = (
                request.assumptions.wacc is not None
                or all(
                    value is not None
                    for value in (
                        request.assumptions.risk_free_rate,
                        request.assumptions.equity_risk_premium,
                        request.assumptions.beta,
                        request.assumptions.debt_cost,
                    )
                )
            )
            if market_as_of is None or not market_source or not market_components_complete:
                market_input_quality = "unverified"
                notes.append("WACC市场参数缺少完整分项、基准日或来源，估值区间已扩大。")
            elif (
                request.valuation_date - market_as_of
            ).days > request.assumptions.market_inputs_stale_after_days:
                market_input_quality = "verified_stale"
                notes.append("WACC市场参数有来源但已超过新鲜度阈值，估值区间已扩大。")
            else:
                market_input_quality = "verified_current"

        low = (
            (has_dcf and len(history) < 6)
            or evidence_coverage < D("0.4")
            or industry_quality == "C"
            or peer_quality == "insufficient"
            or peer_provenance_missing
            or bool(degraded_fields)
            or (has_dcf and market_input_quality == "unverified")
            or bridge_unmeasured
            or bridge_lease_ambiguous
        )
        medium = (
            (has_dcf and len(history) < 10)
            or evidence_coverage < D("0.8")
            or industry_quality == "B"
            or (has_dcf and metadata_completeness != "complete")
            or peer_quality == "limited"
            or bool(adjusted_years or excluded_years)
            or (has_dcf and market_input_quality == "verified_stale")
            or bridge_book_proxy
        )
        confidence = "low" if low else "medium" if medium else "high"
        result_grade = "C" if confidence == "low" else "B" if confidence == "medium" else "A"
        return DataQualityAssessment(
            historical_years=len(raw_history),
            comparable_years=len(history),
            adjusted_years=adjusted_years,
            excluded_years=excluded_years,
            evidence_coverage=_q(evidence_coverage),
            industry_parameter_quality=industry_quality,
            industry_metadata_completeness=metadata_completeness,
            peer_sample_quality=peer_quality,
            confidence=confidence,
            result_grade=result_grade,
            market_input_quality=market_input_quality,
            degraded_fields=degraded_fields,
            notes=notes,
        )

    def reconcile(
        self, dcf: DcfResult | None, relative: list[MultipleResult]
    ) -> ReconciliationResult:
        dcf_range = (dcf.range_low, dcf.range_high) if dcf else None
        successful = [item for item in relative if item.status == "success"]
        relative_range = None
        if successful:
            relative_range = (
                min(item.range_low for item in successful if item.range_low is not None),
                max(item.range_high for item in successful if item.range_high is not None),
            )
        comparison = {
            "policy": "dcf_primary_relative_cross_check_no_mechanical_average"
        }
        overlap_range = None
        if dcf_range and relative_range:
            dcf_mid = _mean(list(dcf_range))
            relative_mid = _mean(list(relative_range))
            gap = abs(dcf_mid - relative_mid) / max(abs(dcf_mid), D("0.0001"))
            comparison["midpoint_gap"] = f"{gap:.2%}"
            overlap_low = max(dcf_range[0], relative_range[0])
            overlap_high = min(dcf_range[1], relative_range[1])
            if overlap_low <= overlap_high:
                overlap_range = (overlap_low, overlap_high)
                comparison["status"] = "overlap"
                conclusion = (
                    "DCF为主估值、相对估值为市场交叉验证；两类区间存在重叠，"
                    "重叠部分可作为高置信参考区间，但不替代两套独立结果。"
                )
            elif gap <= D("0.20"):
                comparison["status"] = "adjacent_no_overlap"
                conclusion = (
                    "DCF与相对估值区间相邻但未重叠；保留两套独立结果，"
                    "不进行机械平均，并解释增长、利润率与同业定价差异。"
                )
            else:
                comparison["status"] = "conflict_review_required"
                conclusion = (
                    "DCF与相对估值差异较大，已触发复核；应检查收入增长、利润率、"
                    "WACC、终值占比、净债务和可比公司口径。"
                )
        elif dcf_range:
            comparison["status"] = "dcf_only"
            conclusion = "仅DCF形成有效区间；相对估值独立保留为不可用。"
        elif relative_range:
            comparison["status"] = "relative_only"
            conclusion = "仅相对估值形成有效区间；DCF独立保留为不可用。"
        else:
            comparison["status"] = "no_valid_method"
            conclusion = "当前没有有效估值区间。"
        return ReconciliationResult(
            dcf_range=dcf_range,
            relative_range=relative_range,
            overlap_range=overlap_range,
            conclusion=conclusion,
            method_comparison=comparison,
        )
