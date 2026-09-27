"""Independent arithmetic checks at the boundary between calculation and report."""
from decimal import Decimal

from valuationagent.schemas.models import ValidationFinding

D = Decimal

# This version has no reviewed adjustments contract for a consolidated
# financial subsidiary or non-controlling equity.  Preserve facts and decline
# the affected enterprise-value method, rather than silently treating unknown
# market-value adjustments as zero or subtracting accounting book values.
EQUITY_BRIDGE_REVIEW_LABELS = {
    "minority_interest": "少数股东权益",
    "restricted_cash": "受限货币资金/法定存款准备金",
    "financial_institution_deposits": "吸收存款及同业存放",
    "interbank_lending": "拆出资金",
    "restricted_interbank_deposits": "不能随时支取的同业存款/受限拆出资金",
}


def equity_bridge_review_findings(methods, statement_items):
    if not {"dcf", "ev_ebitda"} & set(methods):
        return []
    # An accounting deficit in non-controlling equity is not evidence that its
    # market value is zero.  Nor may a negative restricted-cash/deposit balance
    # silently bypass review.  This gate checks disclosed values only: absence
    # of a key is not proof that a source document contains no such exposure.
    present = []
    for key, label in EQUITY_BRIDGE_REVIEW_LABELS.items():
        value = statement_items.get(key)
        if value is None:
            continue
        amount = D(str(value))
        if not amount.is_finite() or amount != 0:
            present.append(f"{label}={value}")
    if not present:
        return []
    return [ValidationFinding(
        rule_id="EQUITY_BRIDGE_COMPLEX_SCOPE", severity="blocking",
        message=(
            "企业价值到普通股权益的桥接需专项复核：" + "；".join(present)
            + "。当前版本尚未实现经确认的少数股东权益/受限资金/金融子公司调整契约；"
            "不能把调整默认为零，也不能直接将账面值当作应扣市值。"
            "本次DCF、EV/EBITDA暂不生成价格；可保留证据并评估PE/PS是否独立具备条件。"
        ),
    )]


def validate_equity_bridge_inputs(request, financials):
    return equity_bridge_review_findings(request.methods, financials.statement_items)


def validate_peer_inputs(request, peers):
    if all(method == "dcf" for method in request.methods):
        return []
    findings, seen = [], set()
    for peer in peers:
        # Ignore a row that has no multiple used by this request.
        if not any(getattr(peer, method, None) is not None for method in request.methods if method != "dcf"):
            continue
        ticker = peer.ticker.strip().upper().split(".")[0]
        if not ticker or ticker in seen:
            findings.append(ValidationFinding(rule_id="PEER_DUPLICATE", severity="blocking",
                message="可比公司代码为空或重复，不能重复计入样本数。请核对：" + peer.ticker))
        seen.add(ticker)
        if peer.as_of_date and peer.as_of_date != request.valuation_date:
            findings.append(ValidationFinding(rule_id="PEER_PRICING_DATE", severity="blocking",
                message=f"可比公司 {peer.ticker} 定价日 {peer.as_of_date} 与估值日不一致，请统一日期。"))
        if peer.multiple_basis in {"TTM", "forward"}:
            findings.append(ValidationFinding(rule_id="PEER_DENOMINATOR_BASIS", severity="blocking",
                message=f"可比公司 {peer.ticker} 使用 {peer.multiple_basis} 倍数，不能乘以年度 FY 财务数据。"))
    return findings


def verify_calculations(request, financials, assumptions, forecast, dcf, relative, peers=()):
    scope_findings = validate_equity_bridge_inputs(request, financials)
    if scope_findings:
        raise ValueError(scope_findings[0].message)
    checks = []

    def check(code, actual, expected, tolerance=D("0.001")):
        if not actual.is_finite() or not expected.is_finite() or abs(actual - expected) > tolerance:
            raise ValueError(f"计算复核未通过 [{code}]：实际 {actual}，复算 {expected}；停止生成数值报告。")
        checks.append(code)

    for row in forecast:
        check(f"FCFF_{row.year}", row.fcff,
              row.nopat + row.depreciation_amortization - row.capital_expenditure - row.change_operating_nwc)
    if dcf is not None:
        if not forecast or assumptions.wacc <= assumptions.terminal_growth:
            raise ValueError("计算复核未通过：DCF预测为空或WACC不高于永续增长率。")
        explicit = sum((r.fcff * r.cash_flow_fraction / (1 + assumptions.wacc) ** r.discount_period for r in forecast), D(0))
        terminal_period = forecast[-1].discount_period
        if request.discount_policy == "annual_midyear_remaining":
            terminal_period += forecast[-1].cash_flow_fraction / 2
        terminal = forecast[-1].fcff * (1 + assumptions.terminal_growth) / (assumptions.wacc - assumptions.terminal_growth)
        enterprise = explicit + terminal / (1 + assumptions.wacc) ** terminal_period
        check("DCF_PRESENT_VALUE", dcf.enterprise_value, enterprise, D("0.01"))
        operating_cash = financials.revenue * assumptions.operating_drivers.get("operating_cash_ratio", D(0))
        surplus = max(D(0), financials.cash_and_non_operating_assets - operating_cash)
        check("EQUITY_BRIDGE", dcf.equity_value, dcf.enterprise_value + surplus - financials.interest_bearing_debt)
        check("PER_SHARE", dcf.per_share_value, dcf.equity_value / financials.common_shares,
              D("0.00011") + D("0.0001") / financials.common_shares)
        if not dcf.range_low <= dcf.per_share_value <= dcf.range_high:
            raise ValueError("计算复核未通过：DCF区间顺序错误或基准值落在区间之外。")
        checks.append("DCF_RANGE")
    for result in relative:
        if result.status != "success":
            continue
        if any(v is None or not v.is_finite() for v in (result.range_low, result.per_share_value, result.range_high)) or not result.range_low <= result.per_share_value <= result.range_high:
            raise ValueError(f"计算复核未通过：{result.method} 缺少有限结果或区间顺序错误。")
        checks.append(result.method.upper() + "_RANGE")
        selected = set(result.peer_tickers)
        rows = [peer for peer in peers if (not selected or peer.ticker in selected) and getattr(peer, result.method) is not None]
        if selected and {peer.ticker for peer in rows} != selected:
            raise ValueError("计算复核未通过：结果引用了不在有效输入中的可比公司。")
        if not rows:
            raise ValueError("计算复核未通过：相对估值结果没有可复算的同业样本。")
        metric = {"pe": "net_income_parent", "ps": "revenue", "ev_ebitda": "ebitda"}[result.method]
        values = []
        for peer in rows:
            equity = getattr(financials, metric) * getattr(peer, result.method)
            if result.method == "ev_ebitda":
                equity += financials.cash_and_non_operating_assets - financials.interest_bearing_debt
            values.append(equity / financials.common_shares)
        values.sort()
        for label, percentile, actual in (("P25", D(".25"), result.range_low), ("P50", D(".5"), result.per_share_value), ("P75", D(".75"), result.range_high)):
            position = (len(values) - 1) * percentile
            lower = int(position)
            expected = values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)
            check(result.method.upper() + "_" + label, actual, expected)
    return checks
