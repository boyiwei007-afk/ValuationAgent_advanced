"""Independent arithmetic checks at the boundary between calculation and report."""
from decimal import Decimal

from valuationagent.finance.production import (
    normalized_terminal_cash_flow,
    resolve_equity_bridge,
)
from valuationagent.schemas.models import ValidationFinding

D = Decimal

# Common non-financial bridge items have a disclosed book-value proxy policy.
# Restricted-cash and financial-institution balances remain outside this
# generic contract and block enterprise-value methods until specialist review.
EQUITY_BRIDGE_REVIEW_LABELS = {
    "minority_interest": "少数股东权益",
    "preferred_equity": "优先股权益",
    "unfunded_pension": "未弥补养老金缺口",
    "non_operating_provisions": "非经营性预计负债",
    "associates_and_non_operating_investments": "联营及非经营性投资",
    "restricted_cash": "受限货币资金/法定存款准备金",
    "financial_institution_deposits": "吸收存款及同业存放",
    "interbank_lending": "拆出资金",
    "restricted_interbank_deposits": "不能随时支取的同业存款/受限拆出资金",
}

UNSUPPORTED_COMPLEX_BRIDGE_KEYS = {
    "restricted_cash",
    "financial_institution_deposits",
    "interbank_lending",
    "restricted_interbank_deposits",
}

SUPPORTED_BOOK_PROXY_KEYS = {
    "minority_interest",
    "preferred_equity",
    "unfunded_pension",
    "non_operating_provisions",
    "associates_and_non_operating_investments",
}


def equity_bridge_review_findings(methods, statement_items):
    if not {"dcf", "ev_ebitda"} & set(methods):
        return []
    # An accounting deficit in non-controlling equity is not evidence that its
    # market value is zero.  Nor may a negative restricted-cash/deposit balance
    # silently bypass review.  This gate checks disclosed values only: absence
    # of a key is not proof that a source document contains no such exposure.
    invalid = []
    for key, label in EQUITY_BRIDGE_REVIEW_LABELS.items():
        value = statement_items.get(key)
        if value is None:
            continue
        try:
            amount = D(str(value))
        except Exception:
            invalid.append(f"{label}={value}")
            continue
        if not amount.is_finite() or amount < 0:
            invalid.append(f"{label}={value}")
    if invalid:
        return [ValidationFinding(
            rule_id="EQUITY_BRIDGE_INVALID_AMOUNT",
            severity="blocking",
            message=(
                "企业价值到普通股权益的桥接需专项复核：" + "；".join(invalid)
                + "。非有限值或负数不能直接作为桥接调整，也不能把调整默认为零。"
            ),
        )]

    present = []
    for key in UNSUPPORTED_COMPLEX_BRIDGE_KEYS:
        label = EQUITY_BRIDGE_REVIEW_LABELS[key]
        value = statement_items.get(key)
        if value is None:
            continue
        try:
            amount = D(str(value))
        except Exception:
            # Invalid values were already returned by the first pass.
            continue
        if not amount.is_finite() or amount != 0:
            present.append(f"{label}={value}")
    findings = []
    if present:
        findings.append(ValidationFinding(
            rule_id="EQUITY_BRIDGE_COMPLEX_SCOPE", severity="blocking",
            message=(
                "企业价值到普通股权益的桥接需专项复核：" + "；".join(present)
                + "。这些金融/受限资金科目不能按通用非金融企业桥接公式处理。"
                "不能把调整默认为零。"
                "本次DCF、EV/EBITDA暂不生成价格；可保留证据并评估PE/PS是否独立具备条件。"
            ),
        ))
    proxies = []
    for key in SUPPORTED_BOOK_PROXY_KEYS:
        value = statement_items.get(key)
        try:
            amount = D(str(value)) if value is not None else D(0)
        except Exception:
            continue
        if amount != 0 and statement_items.get(f"{key}_market_value") is None:
            proxies.append(EQUITY_BRIDGE_REVIEW_LABELS[key])
    if proxies:
        findings.append(ValidationFinding(
            rule_id="EQUITY_BRIDGE_BOOK_VALUE_PROXY",
            severity="warning",
            message=(
                "以下桥接项缺少市场价值，当前按已披露账面值代理："
                + "、".join(sorted(proxies))
                + "；报告将降低置信等级并保留该口径。"
            ),
        ))
    return findings


def validate_equity_bridge_inputs(request, financials):
    statement_items = dict(financials.statement_items)
    for key in EQUITY_BRIDGE_REVIEW_LABELS:
        value = getattr(financials, key, None)
        if value is not None:
            statement_items.setdefault(key, value)
    return equity_bridge_review_findings(request.methods, statement_items)


def validate_peer_inputs(request, peers):
    if all(method == "dcf" for method in request.methods):
        return []
    findings, seen = [], set()
    pricing_dates = set()
    target_period = request.financials.period_end if request.financials else None
    for peer in peers:
        # Ignore a row that has no multiple used by this request.
        if not any(getattr(peer, method, None) is not None for method in request.methods if method != "dcf"):
            continue
        ticker = peer.ticker.strip().upper().split(".")[0]
        if not ticker or ticker in seen:
            findings.append(ValidationFinding(rule_id="PEER_DUPLICATE", severity="blocking",
                message="可比公司代码为空或重复，不能重复计入样本数。请核对：" + peer.ticker))
        seen.add(ticker)
        if peer.as_of_date:
            pricing_dates.add(peer.as_of_date)
        if peer.as_of_date and (
            peer.as_of_date > request.valuation_date
            or (request.valuation_date - peer.as_of_date).days > 7
        ):
            findings.append(ValidationFinding(rule_id="PEER_PRICING_DATE", severity="blocking",
                message=f"可比公司 {peer.ticker} 定价日 {peer.as_of_date} 不在估值日前七天内，请统一可得交易日。"))
        if peer.financial_period_end and target_period and peer.financial_period_end != target_period:
            findings.append(ValidationFinding(rule_id="PEER_FISCAL_PERIOD", severity="blocking",
                message=f"可比公司 {peer.ticker} 的FY期间 {peer.financial_period_end} 与目标公司 {target_period} 不一致。"))
        if peer.multiple_basis in {"TTM", "forward"}:
            findings.append(ValidationFinding(rule_id="PEER_DENOMINATOR_BASIS", severity="blocking",
                message=f"可比公司 {peer.ticker} 使用 {peer.multiple_basis} 倍数，不能乘以年度 FY 财务数据。"))
    if len(pricing_dates) > 1:
        findings.append(ValidationFinding(rule_id="PEER_MIXED_PRICING_DATES", severity="blocking",
            message="可比公司倍数使用了不同交易日，须统一到同一可得交易日。"))
    return findings


def verify_calculations(request, financials, assumptions, forecast, dcf, relative, peers=()):
    scope_findings = validate_equity_bridge_inputs(request, financials)
    blocking_scope = [item for item in scope_findings if item.severity == "blocking"]
    if blocking_scope:
        raise ValueError(blocking_scope[0].message)
    checks = []

    def check(code, actual, expected, tolerance=D("0.001")):
        if not actual.is_finite() or not expected.is_finite() or abs(actual - expected) > tolerance:
            raise ValueError(f"计算复核未通过 [{code}]：实际 {actual}，复算 {expected}；停止生成数值报告。")
        checks.append(code)

    for row in forecast:
        check(f"FCFF_{row.year}", row.fcff,
              row.nopat + row.depreciation_amortization - row.capital_expenditure - row.change_operating_nwc)
    if dcf is not None:
        if (
            not forecast
            or assumptions.wacc <= 0
            or assumptions.wacc <= assumptions.terminal_growth
        ):
            raise ValueError("计算复核未通过：DCF预测为空、WACC非正或WACC不高于永续增长率。")
        explicit = sum((r.fcff * r.cash_flow_fraction / (1 + assumptions.wacc) ** r.discount_period for r in forecast), D(0))
        terminal_period = forecast[-1].discount_period
        if request.discount_policy == "annual_midyear_remaining":
            terminal_period += forecast[-1].cash_flow_fraction / 2
        if assumptions.calculation_methods.get("terminal_value") == "gordon_growth_normalized_reinvestment":
            terminal_cash_flow = normalized_terminal_cash_flow(
                forecast[-1].nopat,
                assumptions.terminal_growth,
                assumptions.operating_drivers["stable_roic"],
            )
            terminal_fcff = terminal_cash_flow.fcff
        else:
            terminal_fcff = forecast[-1].fcff * (1 + assumptions.terminal_growth)
        terminal = terminal_fcff / (assumptions.wacc - assumptions.terminal_growth)
        enterprise = explicit + terminal / (1 + assumptions.wacc) ** terminal_period
        check("DCF_PRESENT_VALUE", dcf.enterprise_value, enterprise, D("0.01"))
        operating_cash = financials.revenue * assumptions.operating_drivers.get("operating_cash_ratio", D(0))
        bridge = resolve_equity_bridge(
            financials,
            operating_cash,
            policy=request.assumptions.equity_bridge_policy,
        )
        check("EQUITY_BRIDGE", dcf.equity_value, bridge.equity_value(dcf.enterprise_value))
        check("PER_SHARE", dcf.per_share_value, dcf.equity_value / bridge.share_count,
              D("0.00011") + D("0.0001") / bridge.share_count)
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
                bridge = resolve_equity_bridge(
                    financials,
                    policy=request.assumptions.equity_bridge_policy,
                )
                equity = bridge.equity_value(equity)
                shares = bridge.share_count
            else:
                shares = financials.diluted_shares or financials.common_shares
            values.append(equity / shares)
        values.sort()
        for label, percentile, actual in (("P25", D(".25"), result.range_low), ("P50", D(".5"), result.per_share_value), ("P75", D(".75"), result.range_high)):
            position = (len(values) - 1) * percentile
            lower = int(position)
            expected = values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)
            check(result.method.upper() + "_" + label, actual, expected)
    return checks
