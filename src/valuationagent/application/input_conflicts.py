import json
from collections import defaultdict

from valuationagent.core.tools import canonical


def conflict_detail(metric, rows):
    candidates = [row for row in rows if row.metric == metric]
    details = []
    for row in candidates[:6]:
        try:
            locator = json.loads(row.source.locator or "{}")
        except (ValueError, TypeError):
            locator = {}
        details.append({"input_id": row.input_id, "value": str(row.value),
            "original_amount": row.original_amount, "unit": row.unit,
            "period_end": str(row.period_end) if row.period_end else None,
            "as_of": str(row.as_of) if row.as_of else None,
            "source_id": row.source.source_id,
            "location": {key: locator[key] for key in ("amount_ref", "unit_ref", "period_ref", "as_of_ref",
                "amount_start", "amount_end", "page", "sheet", "cell", "block_id") if key in locator}})
    return {"metric": metric, "candidates": details, "candidates_omitted": max(0, len(candidates) - 6)}


def input_conflicts(rows, limit=8):
    groups = defaultdict(list)
    for row in rows:
        groups[(row.role, row.entity, row.metric, row.period_end, row.as_of, row.scope, row.currency)].append(row)
    conflicts = [conflict_detail(key[2], candidates) for key, candidates in groups.items()
        if len({row.value for row in candidates}) > 1]
    return {"conflicts": conflicts[:limit], "conflicts_omitted": max(0, len(conflicts) - limit)}


def conflict_error(metric, rows):
    return ValueError("INPUT_CONFLICT: 当前方法使用的输入有冲突；按原消息/来源核对后用replaces更正真实input_id，"
        "不自动挑选、不重录无冲突字段。候选：" + canonical(conflict_detail(metric, rows)))
