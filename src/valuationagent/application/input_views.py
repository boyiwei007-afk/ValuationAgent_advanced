from collections import Counter
from datetime import date
from typing import Literal

from pydantic import Field

from valuationagent.market.tushare import normalize_a_share_ticker
from valuationagent.application.input_conflicts import input_conflicts
from valuationagent.schemas.models import ApiModel


class InspectInputs(ApiModel):
    section: Literal["records", "calculations", "comparables", "acquisition_issues", "acquisition_coverage", "acquisition_sources"] = "records"
    input_ids: list[str] = Field(default_factory=list, max_length=24)
    metrics: list[str] = Field(default_factory=list, max_length=24)
    ticker: str = Field(default="", max_length=124)
    period_end: date | None = None
    as_of: date | None = None
    include_superseded: bool = False
    include_evidence: bool = Field(default=False, description="默认仅返回数值、口径、来源标识及去重后的限制；需要原始引文、定位及哈希时开启，不改变记录或准入。")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=6, ge=1, description="每页最多12条，较大请求自动限为12；calculations最多2条。next_action提供保留筛选的续读参数。")


def ordered_records(records):
    return sorted(records, key=lambda row: (row.role == "historical", row.as_of or row.period_end or date.max,
        row.entity_ticker, row.metric, row.input_id), reverse=True)


def input_overview(session, limit=12):
    dataset = session.input_dataset
    if dataset is None:
        return None
    records = ordered_records(dataset.active_records())
    periods = sorted({row.period_end.isoformat() for row in records if row.period_end}, reverse=True)
    return {"entity": dataset.entity, "analysis_basis": dataset.analysis_basis, **input_conflicts(records),
        "baseline_selection": dataset.baseline_selection,
        "available_target_annual_periods": [period.isoformat() for period in dataset.available_target_annual_periods],
        "active_count": len(records), "stored_count": len(dataset.records),
        "metrics": sorted({row.metric for row in records}), "periods": periods[:12], "periods_omitted": max(0, len(periods) - 12),
        "source_counts": dict(Counter(row.source.kind for row in records)),
        "comparables": [{"ticker": choice.ticker, "name": choice.name, "enabled": choice.enabled,
            "rationale": choice.rationale[:180], "rationale_truncated": len(choice.rationale) > 180} for choice in dataset.comparables.values()],
        "active_records": [{**row.model_dump(mode="json", exclude={"source"}),
            "source": {"kind": row.source.kind, "source_id": row.source.source_id, "authority_tier": row.source.authority_tier}}
            for row in records[:limit]],
        "records_omitted": max(0, len(records) - limit),
        "calculation_count": len(dataset.active_calculations()),
        "calculations": [{"calculation_id": row.calculation_id, "metric": row.metric, "period_end": str(row.period_end or "未指定"),
            "entity_ticker": row.entity_ticker, "term_count": len(row.terms)} for row in dataset.active_calculations()[:3]],
        "calculations_omitted": max(0, len(dataset.active_calculations()) - 3),
        "retrieve_with": "inspect_inputs",
        "instruction": "这是有界输入目录，不是全部记录。历史年度/可比/更正ID用inspect_inputs按指标、代码、日期分页；已有输入不重复取数。记录存在不等于所有方法均可计算。"}


def normalized_unit(item):
    if item["metric"] == "market_price":
        return "元/股"
    if item["unit"] in {"元", "千元", "万元", "百万元", "亿元"}:
        return "元"
    if item["unit"] in {"股", "万股", "亿股"}:
        return "股"
    if item["unit"] in {"%", "ratio"}:
        return "ratio"
    return item["unit"]


def record_page(items, include_evidence):
    notes, selected = [], []
    for item in items:
        row = {**item, "value_unit": normalized_unit(item)}
        if not include_evidence:
            source = item["source"]
            indexes = []
            for note in source["limitations"]:
                if note not in notes:
                    notes.append(note)
                indexes.append(notes.index(note))
            row["source"] = {key: source[key] for key in ("kind", "source_id", "file_id", "authority_tier", "published_at") if source[key]}
            row["source"]["note_indexes"] = indexes
        selected.append(row)
    groups = {}
    for row in selected:
        key = (row["entity"], row["entity_ticker"], row["role"], row["metric"], row["period_end"], row["as_of"], row["scope"], row["currency"], row["value"])
        groups.setdefault(key, []).append(row["input_id"])
    duplicates = [identifiers for identifiers in groups.values() if len(identifiers) > 1]
    return selected, notes, duplicates


def inspect_inputs(session, args):
    dataset = session.input_dataset
    ticker = args.ticker if args.ticker.startswith("user:") else normalize_a_share_ticker(args.ticker) if args.ticker else ""
    active_ids = {row.input_id for row in dataset.active_records()} if dataset else set()
    if args.section != "records" and (args.input_ids or args.metrics or args.period_end or args.as_of or args.include_superseded):
        raise ValueError("INPUT_QUERY_SCOPE: 指标/记录ID/日期/旧输入筛选仅用于records；其他目录可按ticker及offset/limit查看。")
    query_feedback = {}
    if args.section == "records":
        records = ordered_records(dataset.records if args.include_superseded else dataset.active_records()) if dataset else []
        if ticker:
            records = [row for row in records if (row.entity_ticker or (normalize_a_share_ticker(session.draft.ticker)
                if row.role == "historical" and session.draft.ticker else "")) == ticker]
        records = [row for row in records if not args.input_ids or row.input_id in args.input_ids]
        before_dates = records
        records = [row for row in records if (not args.period_end or row.period_end == args.period_end)
            and (not args.as_of or row.as_of == args.as_of)]
        available_metrics = sorted({row.metric for row in records})
        unmatched_metrics = sorted(set(args.metrics) - set(available_metrics))
        records = [row for row in records if not args.metrics or row.metric in args.metrics]
        query_feedback["unmatched_input_ids"] = sorted(set(args.input_ids) - {row.input_id for row in records})
        query_feedback["unmatched_metrics"] = unmatched_metrics
        if unmatched_metrics:
            query_feedback["available_metrics"] = available_metrics
            excluded = [row for row in before_dates if row.metric in unmatched_metrics]
            query_feedback["date_filtered_metrics"] = [{"metric": metric,
                "period_ends": sorted({row.period_end.isoformat() for row in excluded if row.metric == metric and row.period_end}),
                "as_of_dates": sorted({row.as_of.isoformat() for row in excluded if row.metric == metric and row.as_of}),
                "instruction": "已有记录被当前日期筛选排除；period_end是财报期间，as_of是独立时点，不能相互冒充。"}
                for metric in unmatched_metrics if any(row.metric == metric for row in excluded)]
        items = [{**row.model_dump(mode="json", exclude={"source": {"provider_binding"}}), "active": row.input_id in active_ids}
            for row in records]
    elif args.section == "calculations":
        items = [row.model_dump(mode="json") for row in dataset.active_calculations()] if dataset else []
        if ticker:
            items = [item for item in items if (item["entity_ticker"] or normalize_a_share_ticker(session.draft.ticker)) == ticker]
    elif args.section == "comparables":
        items = [choice.model_dump(mode="json") for choice in dataset.comparables.values()] if dataset else []
        if ticker:
            items = [item for item in items if item["ticker"] == ticker]
    else:
        key = args.section.removeprefix("acquisition_")
        items = session.input_acquisition.get(key, [])
        if ticker:
            items = [item for item in items if not item.get("ticker") or item["ticker"] == ticker]
    limit = min(args.limit, 2 if args.section == "calculations" else 12)
    selected = items[args.offset:args.offset + limit]
    end = args.offset + len(selected)
    next_offset = end if end < len(items) else None
    arguments = args.model_dump(mode="json", exclude_defaults=True)
    notes, duplicates = [], []
    evidence_action = None
    if args.section == "records":
        selected, notes, duplicates = record_page(selected, args.include_evidence)
        if selected and not args.include_evidence:
            evidence_action = {"tool": "inspect_inputs", "arguments": {
                "input_ids": [row["input_id"] for row in selected], "include_superseded": args.include_superseded,
                "include_evidence": True, "limit": limit}}
    return {"section": args.section, "items": selected, "total": len(items), **query_feedback,
        "offset": args.offset, "page_limit": limit, "next_offset": next_offset,
        "next_action": {"tool": "inspect_inputs", "arguments": {**arguments, "offset": next_offset, "limit": limit}} if next_offset is not None else None,
        "evidence_action": evidence_action, "source_notes": notes, "same_value_input_groups": duplicates,
        "active_count": len(active_ids),
        "analysis_basis": dataset.analysis_basis if dataset else None,
        "acquisition_status": session.input_acquisition.get("status"),
        "instruction": "这是已保存输入，不联网、不计算、不改变准入。value/value_unit为归一化数值/单位，original_amount/unit保留原始数量/单位，不重复乘倍率。source.note_indexes引用本页source_notes；include_evidence可取完整证据。same_value_input_groups为同科目同值的不同记录，不能相加，也不等于独立佐证。下一页用next_action，不重复相同请求；已有原始科目的财务解释用record_inputs(calculations)引用真实input_id，不重抄或重下载。"}
