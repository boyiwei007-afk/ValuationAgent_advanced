from decimal import Decimal


def input_source_label(source):
    if source["kind"] == "user":
        return "用户给数或假设，未经外部核验"
    if source.get("provider_binding"):
        return "供应商字段契约校验，非发行人原件或独立审计"
    return "来源解释已复核，非独立审计"


def money(value, currency):
    if value is None:
        return "未计算"
    amount = Decimal(str(value))
    if currency == "CNY":
        return f"{amount / Decimal(100000000):,.2f}亿元" if abs(amount) >= 100000000 else f"{amount:,.2f}元"
    return f"{amount:,.2f} {currency}"


def financial_display(result):
    rows = []
    if result.dcf:
        rows.append({"method": "DCF", "per_share": money(result.dcf.per_share_value, result.currency) + "/股",
                     "equity_value": money(result.dcf.equity_value, result.currency)})
    for row in result.relative:
        if row.status == "success":
            rows.append({"method": row.method.upper(), "per_share": money(row.per_share_value, result.currency) + "/股",
                         "equity_value": money(row.equity_value, result.currency), "selected_multiple": str(row.selected_multiple) if row.selected_multiple is not None else None})
    return {"results": rows, "instruction": "金额显示由确定性代码换算。引用时逐字使用此处数值及单位，不自行移小数点；股权估值不是实际交易市值。"}
