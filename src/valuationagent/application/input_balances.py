from itertools import pairwise


BALANCE_CHANGE_VERSION = "annual-operating-balances-20261004-v1"
BALANCE_CHANGE_FORMULA = "operating_nwc - previous_annual_operating_nwc"


def consecutive_balances(previous, current):
    start, end = previous.period_end, current.period_end
    return bool(start and end and end.year == start.year + 1
        and (start.month, start.day) == (end.month, end.day)
        and previous.currency == current.currency
        and previous.statement_scope == current.statement_scope == "consolidated"
        and previous.comparability_status == current.comparability_status == "comparable"
        and all("operating_nwc" in row.statement_items for row in (previous, current)))


def derive_balance_changes(snapshots):
    snapshots = [row.model_copy(deep=True) for row in snapshots]
    for previous, current in pairwise(snapshots):
        if current.change_operating_nwc is not None or not consecutive_balances(previous, current):
            continue
        current.change_operating_nwc = current.statement_items["operating_nwc"] - previous.statement_items["operating_nwc"]
        references = {ref.evidence_id: ref for row in (previous, current) for ref in row.evidence.get("operating_nwc", [])}
        current.evidence["change_operating_nwc"] = list(references.values())
        current.calculation_methods.update({"balance_change_policy": BALANCE_CHANGE_VERSION,
            "change_operating_nwc": BALANCE_CHANGE_FORMULA, "previous_balance_date": previous.period_end.isoformat()})
    return snapshots


def verify_balance_changes(request):
    snapshots = [row for row in [*request.historical_financials, request.financials] if row is not None]
    previous = None
    for current in snapshots:
        metadata = current.calculation_methods
        if "balance_change_policy" in metadata:
            if (metadata["balance_change_policy"] != BALANCE_CHANGE_VERSION
                    or metadata.get("change_operating_nwc") != BALANCE_CHANGE_FORMULA
                    or previous is None or not consecutive_balances(previous, current)
                    or metadata.get("previous_balance_date") != previous.period_end.isoformat()
                    or current.change_operating_nwc != current.statement_items["operating_nwc"] - previous.statement_items["operating_nwc"]):
                raise ValueError("INPUT_BALANCE_REPLAY: 连续年度余额、期间、版本或变动额与冻结推导不一致。")
        previous = current
    return True


def balance_change_rows(record):
    if record is None:
        return []
    return [{"period_end": str(row.period_end), "previous_period_end": row.calculation_methods["previous_balance_date"],
        "policy": row.calculation_methods["balance_change_policy"], "metric": "change_operating_nwc",
        "value": str(row.change_operating_nwc), "formula": BALANCE_CHANGE_FORMULA,
        "evidence_ids": [ref.evidence_id for ref in row.evidence.get("change_operating_nwc", [])],
        "limitation": "相同经营范围的连续年度期末余额差；不等于已剔除并购、汇率和重分类影响的现金流量表调整额。"}
        for row in [*record.request.historical_financials, record.request.financials]
        if row is not None and "balance_change_policy" in row.calculation_methods]
