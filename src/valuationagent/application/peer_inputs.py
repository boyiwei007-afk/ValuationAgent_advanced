"""Assemble dated comparable multiples from admitted, source-backed inputs."""
from datetime import date
from decimal import Decimal

from valuationagent.schemas.models import PeerCompany
from valuationagent.application.observation_consistency import PERIOD_CELL_CONFLICT, period_cell_conflicts, period_readback_issue


MULTIPLES = {"pe", "ps", "ev_ebitda"}
COMPONENTS = {"market_cap", "revenue", "net_income_parent"}
FORMULAS = {"pe": "market_cap / net_income_parent", "ps": "market_cap / revenue"}


def assemble_peers(session, evidence, baseline_end=None):
    collisions = period_cell_conflicts([fact for fact in session.facts if fact.role == "comparable"])
    if collisions:
        raise ValueError(PERIOD_CELL_CONFLICT + "：" + ", ".join(list(collisions)[:6]))
    cutoff = session.information_cutoff_date or session.draft.valuation_date
    pricing_date = session.draft.peer_pricing_date or session.draft.valuation_date
    if pricing_date and (not session.draft.valuation_date or not 0 <= (session.draft.valuation_date - pricing_date).days <= 7
                         or cutoff and pricing_date > cutoff):
        raise ValueError("可比统一行情日期须在估值日前七天内且不晚于信息截止日")
    if pricing_date != session.draft.valuation_date and len(session.draft.peer_pricing_rationale.strip()) < 12:
        raise ValueError("必须记录不同行情日期的选择依据和陈旧性风险")
    if baseline_end is None:
        target_years = [date(int(fact.period), 12, 31) for fact in session.facts
                        if fact.role == "historical" and fact.scope == "consolidated"
                        and fact.status == "confirmed" and not fact.warnings and len(fact.period) == 4
                        and fact.period.isdecimal() and (fact.standard_metric or fact.metric) in {"revenue", "net_income_parent", "ebitda"}]
        baseline_end = max(target_years, default=None)
    groups = {}
    for fact in session.facts:
        if fact.role != "comparable" or fact.status != "confirmed":
            continue
        if issue := period_readback_issue(fact):
            raise ValueError(issue)
        if session.draft.valuation_date is None:
            raise ValueError("可比估值需要先明确估值日")
        if fact.warnings or not fact.peer_ticker or not fact.peer_name or fact.normalized_value is None:
            raise ValueError("可比公司存在未解决的来源、数值或主体警告")
        if fact.peer_ticker == session.draft.ticker:
            raise ValueError("目标公司不能作为自身相对估值的可比样本")
        metric = fact.standard_metric or fact.metric
        if metric not in MULTIPLES | COMPONENTS:
            raise ValueError("可比公司字段仅支持倍数或总市值、年度合并收入/归母净利润")
        if fact.published_at and cutoff and fact.published_at > cutoff:
            raise ValueError("可比资料的披露日晚于信息截止日")
        period = date.fromisoformat(fact.period + "-12-31" if len(fact.period) == 4 else fact.period)
        if cutoff and period > cutoff:
            raise ValueError("可比数据期间晚于信息截止日")
        if metric in MULTIPLES | {"market_cap"}:
            if len(fact.period) != 10 or period != pricing_date:
                raise ValueError(f"可比市值/倍数须采用明确选择的统一行情日{pricing_date}；不同日期不能混用。缺少估值日行情时可通过update_task明确peer_pricing_date及理由，不静默改写原文日期。")
        if metric in MULTIPLES:
            if fact.multiple_basis != "FY" or fact.unit != "ratio" or not fact.denominator_period_end:
                raise ValueError("直接可比倍数须明确FY分母截止日和倍数单位，不能混用TTM或预测倍数")
            denominator = fact.denominator_period_end
            if (denominator.month, denominator.day) != (12, 31) or denominator >= period:
                raise ValueError("可比FY分母须为定价日前完整日历年度")
            if baseline_end and denominator != baseline_end:
                raise ValueError("可比FY分母年度与目标公司的估值基期不一致")
        else:
            proof = fact.verification.get("reading_proof", {})
            basis, row = proof.get("basis", {}), proof.get("row", {})
            if basis.get("currency") != "CNY" or fact.unit not in {"元", "千元", "万元", "百万元", "亿元"}:
                raise ValueError("可比推导输入须经原文核验为人民币金额，不隐式换汇")
            if metric == "market_cap":
                if fact.scope != "issuer" or row.get("period_kind") != "instant":
                    raise ValueError("可比市值须为发行人总市值及独立时点")
            elif fact.scope != "consolidated" or row.get("period_kind") != "annual":
                raise ValueError("可比分母须为完整年度合并收入或归母净利润")
            elif period >= pricing_date:
                raise ValueError("可比年度财务须为定价日前的完整年度，不能使用未来财务")
        group = groups.setdefault(fact.peer_ticker, {"name": fact.peer_name, "facts": {}})
        key = (metric, period, fact.denominator_period_end if metric in MULTIPLES else None)
        previous = group["facts"].setdefault(key, [])
        if previous and Decimal(previous[0].normalized_value) != Decimal(fact.normalized_value):
            raise ValueError("同一可比公司同期间字段冲突，须更正而非选择或平均")
        previous.append(fact)

    peers = []
    for ticker, group in groups.items():
        row = {"ticker": ticker, "name": group["name"], "as_of_date": pricing_date,
               "multiple_basis": "FY", "evidence": {}, "calculation_methods": {},
               "rationale": "通过当前字段准入的同定价日、年度口径可比样本；不等于人工批准，适用性须结合业务和资本结构复核"
                            + (f"；估值日{session.draft.valuation_date}，行情日{pricing_date}；{session.draft.peer_pricing_rationale}" if pricing_date != session.draft.valuation_date else "")}
        facts = group["facts"]
        annuals = {period for (metric, period, _) in facts if metric in {"revenue", "net_income_parent"}}
        direct_years = {denominator for (metric, _, denominator) in facts if metric in MULTIPLES}
        if len(direct_years) > 1:
            raise ValueError("同一可比公司直接倍数的财务分母年度不一致")
        financial_end = baseline_end or next(iter(direct_years), None)
        if financial_end is None and len(annuals) == 1:
            financial_end = next(iter(annuals))
        row["financial_period_end"] = financial_end
        for (metric, _, _), items in facts.items():
            if metric in MULTIPLES:
                value = Decimal(items[0].normalized_value)
                if value <= 0 or not value.is_finite():
                    raise ValueError("直接可比倍数须为正的有限数值；亏损公司不能强行形成PE")
                row[metric] = value
                row["evidence"][metric] = [evidence(session, item) for item in items]
                row["calculation_methods"][metric] = "direct_disclosed_FY_multiple"
        market = facts.get(("market_cap", pricing_date, None), [])
        if market and Decimal(market[0].normalized_value) > 0:
            row["market_cap"] = Decimal(market[0].normalized_value)
            row["evidence"]["market_cap"] = [evidence(session, item) for item in market]
            for metric, denominator_metric in (("pe", "net_income_parent"), ("ps", "revenue")):
                denominator = facts.get((denominator_metric, financial_end, None), [])
                if not denominator or Decimal(denominator[0].normalized_value) <= 0:
                    continue
                value = row["market_cap"] / Decimal(denominator[0].normalized_value)
                if metric in row and abs(row[metric] - value) > max(Decimal("0.01"), abs(value) * Decimal("0.005")):
                    raise ValueError("直接披露的可比倍数与市值/年度财务推导不一致，须核对分母口径，不能静默覆盖")
                row[metric] = value
                row["calculation_methods"][metric] = FORMULAS[metric]
                row["evidence"][denominator_metric] = [evidence(session, item) for item in denominator]
                row["evidence"][metric] = [evidence(session, item, FORMULAS[metric]) for item in [*market, *denominator]]
        if any(metric in row for metric in MULTIPLES):
            peers.append(PeerCompany.model_validate(row))
    return peers
