from decimal import Decimal
from statistics import median


CORE_BALANCES = ("accounts_receivable", "inventory", "accounts_payable")
TURNOVER_METHOD = "ending_balance_dso_dio_dpo"


def core_balance(snapshot):
    items = snapshot.statement_items
    if not all(key in items for key in CORE_BALANCES):
        return None
    return items["accounts_receivable"] + items["inventory"] - items["accounts_payable"]


def turnover_scope(latest):
    total = latest.statement_items.get("operating_nwc")
    core = core_balance(latest)
    if total is not None:
        if core is None:
            raise ValueError("NWC_SCOPE_BASELINE: 已有完整营运资本余额，但缺少基期应收、存货或应付，无法分离其他经营项目；请补齐三项或取消周转天数改用完整余额比例，不能静默丢弃其他项目。")
        return {"other_operating_nwc_ratio": (total - core) / latest.revenue, "nwc_scope_partial": Decimal(0)}
    return {"nwc_scope_partial": Decimal(1)}


def working_capital_drivers(request, history):
    supplied, latest = request.assumptions, history[-1]
    recent = history[-5:]
    cost_ratios = [row.statement_items["operating_cost"] / row.revenue for row in recent
        if row.statement_items.get("operating_cost") is not None]
    manual_days = (supplied.dso_days, supplied.dio_days, supplied.dpo_days)
    if any(value is not None for value in manual_days):
        if not all(value is not None for value in manual_days):
            raise ValueError("NWC_DRIVER_PARTIAL: DSO、DIO、DPO须成组提供，不忽略用户部分输入。")
        cost_ratio = supplied.operating_cost_ratio
        if cost_ratio is None and len(cost_ratios) >= 4:
            cost_ratio = median(cost_ratios)
        if cost_ratio is None:
            raise ValueError("NWC_COST_RATIO_MISSING: 用户周转天数还需营业成本率或至少4年成本率历史。")
        scope = turnover_scope(latest)
        note = ("其他经营项目余额按基期完整NWC与三项余额之差/收入单独保留，不假定为零。"
            if not scope["nwc_scope_partial"] else "仅覆盖三项经营余额，未证明其他项目为零，按范围不完整降级。")
        return ({"dso_days": supplied.dso_days, "dio_days": supplied.dio_days, "dpo_days": supplied.dpo_days,
            "operating_cost_ratio": cost_ratio, **scope}, TURNOVER_METHOD,
            "采用用户指定的期末余额等价周转天数，不是期初期末平均余额法；" + note)
    nwc_ratios = [row.statement_items["operating_nwc"] / row.revenue for row in recent
        if "operating_nwc" in row.statement_items]
    if "operating_nwc" in latest.statement_items:
        if len(nwc_ratios) >= 3:
            ratio, method = median(nwc_ratios), "operating_nwc_revenue_ratio"
            note = f"使用最近{len(nwc_ratios)}年已选经营营运资本期末余额/收入中位数，不以三项模板覆盖完整经营范围。"
        else:
            ratio, method = latest.statement_items["operating_nwc"] / latest.revenue, "latest_operating_nwc_revenue_ratio"
            note = "历史完整余额不足3年，使用基期经营营运资本/收入低样本情景，须人工复核。"
        return ({"operating_nwc_ratio": ratio, "nwc_scope_partial": Decimal(0)}, method,
            note + "允许负营运资本；已选范围及跨年可比性仍需解释，比例预测不是披露事实。")
    day_rows = []
    for row in recent:
        items = row.statement_items
        cost = items.get("operating_cost")
        if cost is None or cost <= 0 or core_balance(row) is None:
            continue
        day_rows.append((items["accounts_receivable"] / row.revenue * 365,
            items["inventory"] / cost * 365, items["accounts_payable"] / cost * 365, cost / row.revenue))
    if len(day_rows) >= 4:
        return ({"dso_days": median(row[0] for row in day_rows), "dio_days": median(row[1] for row in day_rows),
            "dpo_days": median(row[2] for row in day_rows), "operating_cost_ratio": median(row[3] for row in day_rows),
            "nwc_scope_partial": Decimal(1)}, TURNOVER_METHOD,
            "按历史期末应收/收入、存货/成本、应付/成本校准等价天数，不是平均余额周转天数；仅覆盖三项，其他经营项目未知，按范围不完整降级。")
    if (len(history) >= 2 and latest.revenue != history[-2].revenue
            and latest.change_operating_nwc is not None):
        factor = latest.change_operating_nwc / (latest.revenue - history[-2].revenue)
        return ({"nwc_delta_revenue_factor": factor, "nwc_scope_partial": Decimal(1)}, "implied_from_latest_delta",
            "经营余额不可得，使用最近一期ΔNWC/Δ收入作为低置信边际投入假设；收入变动很小时可能不稳定，不暗中截断已计算比例。")
    changes = [row.change_operating_nwc / row.revenue for row in recent if row.change_operating_nwc is not None]
    if changes:
        return ({"nwc_change_revenue_ratio": median(changes), "nwc_scope_partial": Decimal(1)},
            "historical_delta_nwc_revenue_fallback",
            "经营余额不可得，按历史ΔNWC/收入中位数乘预测当年收入，不乘收入增量；这是流量比例降级，并非余额比例或周转模型。")
    return ({"nwc_delta_revenue_factor": Decimal("0.10"), "nwc_scope_partial": Decimal(1)},
        "policy_delta_revenue_fallback", "经营余额及变动均不可得，按Δ收入的10%政策值低置信降级，不将缺失当零。")
