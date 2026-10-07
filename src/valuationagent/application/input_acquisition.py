from collections import defaultdict
from datetime import date
from typing import Literal

from pydantic import Field, field_validator

from valuationagent.application.file_workspace import source_bytes, source_document
from valuationagent.application.input_calculations import CALCULATION_KINDS
from valuationagent.application.input_derivations import DERIVATIONS, RAW_INPUT_FIELDS, derivation_dependencies, derive_snapshot_inputs
from valuationagent.application.input_sources import dataset_basis
from valuationagent.application.input_workspace import FINANCIAL_FIELDS
from valuationagent.application.provider_inputs import provider_inventory
from valuationagent.market.research_data import FinancialHistoryRequest, fetch_history
from valuationagent.market.tushare import normalize_a_share_ticker
from valuationagent.market.tushare_contracts import CONTRACTS
from valuationagent.schemas.inputs import BaselineInstruction, ComparableSelection, InputDataset
from valuationagent.schemas.models import ApiModel, required_financial_metrics


PROVIDER_METRICS = {contract[0] for contracts in CONTRACTS.values() for contract in contracts.values()}
ACQUISITION_METRICS = FINANCIAL_FIELDS | RAW_INPUT_FIELDS | DERIVATIONS.keys() | CALCULATION_KINDS.keys() | PROVIDER_METRICS
STATEMENT_METRICS = {
    "income": {"ebit", "ebitda", "ebit_margin", "tax_rate"},
    "balancesheet": {"cash_and_non_operating_assets", "interest_bearing_debt", "operating_nwc",
        "lease_liabilities", "minority_interest", "preferred_equity", "associates_and_non_operating_investments",
        "unfunded_pension", "non_operating_provisions"},
    "cashflow": {"capital_expenditure", "depreciation_amortization", "depreciation_fixed_assets",
        "amortization_intangible_assets", "amortization_long_term_deferred_expenses", "depreciation_right_of_use",
        "change_operating_nwc", "inventory_decrease", "operating_receivables_decrease", "operating_payables_increase"},
    "statistics": {"common_shares", "diluted_shares", "market_price", "market_cap"},
}


def validate_acquisition_metrics(metrics):
    unknown = sorted(set(metrics) - ACQUISITION_METRICS)
    if unknown:
        raise ValueError("INPUT_ACQUISITION_METRIC: 未支持的请求指标：" + ", ".join(unknown)
            + "。本批尚未取数或修改输入；请使用标准字段或已登记raw.*科目，不自动映射API字段别名。可用字段："
            + ", ".join(sorted(ACQUISITION_METRICS)))
    return metrics


class AcquireFinancialInputs(ApiModel):
    years: list[int] = Field(min_length=1, max_length=10, description="历史覆盖年度，不是估值基期指令。默认同时选入响应中目标公司最新可得完整年度及同年度可比；用户明确指定历史基期时用baseline保留原话。")
    baseline: BaselineInstruction | None = None
    comparables: list[ComparableSelection] = Field(default_factory=list, max_length=10,
        description="可选：本次选择/剔除的可比公司及业务理由。程序一并取得目标与所有已启用可比的数据，不需逐家公司调用。")
    metrics: list[str] = Field(default_factory=list, max_length=24,
        description="省略按当前方法取数；显式填写则按指标目录取对应报表。只接受标准字段或已登记raw.*，不填operate_profit/money_cap等API原名或自造别名。raw.*仅保存原始科目，派生或解释字段仍需LLM判断与声明计算，不因可请求就自动入模。",
        json_schema_extra={"items": {"type": "string", "enum": sorted(ACQUISITION_METRICS)}})
    pricing_date: date | None = Field(default=None,
        description="已有明确统一行情日可填写。省略则沿用任务行情日；仍未指定时从实际记录选择估值日前七天内的最近共同可得日并披露，不冒充估值日行情。")
    extra_statements: list[Literal["income", "balancesheet", "cashflow", "statistics"]] = Field(default_factory=list, max_length=4,
        description="需要附加原始接口报表供语义分析时指定；来源被保存，不代表所有科目自动入模。")

    @field_validator("metrics")
    @classmethod
    def supported_metrics(cls, values):
        return validate_acquisition_metrics(values)


def input_key(row, target):
    return (row.entity_ticker or target, row.metric, row.period_end, row.as_of, row.scope, row.currency)


def acquisition_sources(session):
    sources = []
    for document in session.documents:
        parts = document.provider.split(":")
        if document.provenance_type == "structured_provider" and len(parts) == 4:
            sources.append({"file_id": document.file_id, "provider": parts[0], "ticker": parts[1], "statement": parts[2]})
    return {"sources": sources[-60:], "omitted": max(0, len(sources) - 60),
        "instruction": "这些原始快照已经取得，缺少显示不等于缺少数据。候选用list_input_candidates，其他科目用read_file；相同参数不必重新取数。"}


def acquire_financial_inputs(runtime, args):
    validate_acquisition_metrics(args.metrics)
    session = runtime.session
    target = normalize_a_share_ticker(session.draft.ticker)
    valuation_date = session.draft.valuation_date
    cutoff = min(day for day in (valuation_date, session.information_cutoff_date) if day is not None) if valuation_date else None
    if cutoff is None or any(year < 1990 or year >= cutoff.year for year in args.years):
        raise ValueError("INPUT_ACQUISITION_DATE: 先明确估值日；请求年度必须为信息截止日前已结束的完整年度。")
    pricing_date = args.pricing_date or session.draft.peer_pricing_date
    if args.pricing_date and session.draft.peer_pricing_date and args.pricing_date != session.draft.peer_pricing_date:
        raise ValueError("INPUT_ACQUISITION_DATE: 与已明确的任务行情日不同；先说明变更理由并update_task，不隐式替换。")
    if pricing_date and (pricing_date > cutoff or not 0 <= (valuation_date - pricing_date).days <= 7):
        raise ValueError("INPUT_ACQUISITION_DATE: 行情日必须不晚于截止日且在估值日前七天内。")
    dataset = session.input_dataset.model_copy(deep=True) if session.input_dataset else InputDataset(
        entity=session.draft.company or session.draft.ticker, analysis_basis="research")
    if dataset.entity not in {session.draft.company, session.draft.ticker}:
        raise ValueError("INPUT_SCOPE: 已有输入与当前主体不同，不跨主体合并。")
    from valuationagent.application.input_baseline import apply_baseline, selected_period

    if args.baseline:
        apply_baseline(runtime, dataset, args.baseline)
    requested_baseline = selected_period(dataset)
    query_years = sorted(set(args.years) | ({requested_baseline.year} if requested_baseline else set()))
    seen = set()
    for choice in args.comparables:
        try:
            ticker = normalize_a_share_ticker(choice.ticker)
        except ValueError:
            raise ValueError(f"INPUT_PEER_MARKET_UNSUPPORTED: 可比 {choice.name} 的代码 {choice.ticker!r} 不属于当前A股接口；仅支持六位数字或.SH/.SZ/.BJ。请从本批移除该样本或另选范围内公司，不要只改写同一港股代码的后缀。整批未取数，已有输入保留。") from None
        if ticker == target or ticker in seen:
            raise ValueError("INPUT_PEER_SELECTION: 同批可比不能重复或使用目标自身。")
        seen.add(ticker)
        dataset.comparables[ticker] = choice.model_copy(update={"ticker": ticker})
    tickers = [target, *[ticker for ticker, choice in dataset.comparables.items() if choice.enabled]]
    if len(tickers) > 11:
        raise ValueError("INPUT_ACQUISITION_SIZE: 一次最多目标及10家可比，先明确剔除不需要的样本。")
    metrics = set(args.metrics) or required_financial_metrics(session.draft.methods or ["pe", "ps"])
    if len(tickers) > 1 and (not args.metrics or metrics & {"revenue", "net_income_parent", "ebitda", "market_cap", "common_shares"}):
        metrics.update({"market_cap", "common_shares"})
    acquisition_metrics = derivation_dependencies(metrics)
    interpret_raw = bool(acquisition_metrics & (
        ACQUISITION_METRICS - PROVIDER_METRICS - DERIVATIONS.keys() - STATEMENT_METRICS["statistics"]))
    statements = set(args.extra_statements)
    statements.update(statement for statement, contracts in CONTRACTS.items()
        if any(contract[0] in acquisition_metrics and not (statement == "balancesheet" and contract[0] == "common_shares")
            for contract in contracts.values()))
    statements.update(statement for statement, fields in STATEMENT_METRICS.items() if acquisition_metrics & fields)
    if metrics - STATEMENT_METRICS["statistics"]:
        statements.add("income")
    if not args.metrics and set(session.draft.methods) & {"dcf", "ev_ebitda"}:
        statements.update({"balancesheet", "cashflow"})
    candidates, files, issues, available_periods = [], [], [], defaultdict(set)
    for ticker in tickers:
        runtime.service._check_execution()
        response = fetch_history(runtime, FinancialHistoryRequest(years=query_years, statements=sorted(statements), ticker=ticker))
        if response.get("issuer_identity"):
            from valuationagent.application.issuer_identity import require_identity

            require_identity(session, ticker, session.draft.company if ticker == target else dataset.comparables[ticker].name)
        if not response.get("documents"):
            issues.append({"ticker": ticker, "code": response["status"], "detail": response.get("instruction", "未取得接口原始响应")})
        issues.extend({"ticker": ticker, **failure} for failure in response.get("failures", []))
        for item in response.get("documents", []):
            document = source_document(session, item["file_id"])
            _, raw = source_bytes(runtime.service.store, session, document.file_id)
            rows, excluded = provider_inventory(session, document, raw)
            available_periods[ticker].update(row.period_end.isoformat() for _, row in rows
                if row.period_kind == "annual" and row.period_end and row.metric in {"revenue", "net_income_parent"})
            files.append({"ticker": ticker, "file_id": document.file_id, "statement": document.provider.split(":")[2],
                "cached": item.get("cached", False), "excluded_by_reason": excluded})
            candidates.extend((candidate_id, row) for candidate_id, row in rows if (
                row.metric in acquisition_metrics or row.metric.startswith("raw.")
                and (interpret_raw or document.provider.split(":")[2] in args.extra_statements))
                and (row.role == "historical" and row.metric != "market_cap" or row.role == "comparable" and row.metric != "common_shares"))
    dataset.available_target_annual_periods = sorted(set(dataset.available_target_annual_periods)
        | {date.fromisoformat(period) for period in available_periods[target]})
    baseline = selected_period(dataset) or max(dataset.available_target_annual_periods, default=None)
    selected_years = set(args.years) | ({baseline.year} if baseline else set())
    candidates = [(candidate_id, row) for candidate_id, row in candidates if row.period_end is None or row.period_end.year in selected_years]
    candidates = [(candidate_id, row) for candidate_id, row in candidates if not row.metric.startswith("raw.")
        or row.entity_ticker == target or row.period_end == baseline]
    common = set()
    if metrics & {"common_shares", "market_cap"}:
        date_sets = []
        for ticker in tickers:
            metric = "common_shares" if ticker == target else "market_cap"
            dates = {row.as_of for _, row in candidates if row.entity_ticker == ticker and row.metric == metric
                and row.as_of and 0 <= (valuation_date - row.as_of).days <= 7}
            date_sets.append(dates)
            if not dates:
                issues.append({"ticker": ticker, "metric": metric, "code": "INPUT_MARKET_DATE_MISSING"})
        common = set.intersection(*date_sets) if date_sets else set()
        if pricing_date is None:
            pricing_date = max(common) if common else None
        elif pricing_date not in common:
            issues.append({"code": "INPUT_PRICING_DATE_UNAVAILABLE", "requested_date": pricing_date.isoformat(),
                "observed_common_dates": sorted(day.isoformat() for day in common),
                "detail": "请求的行情日缺少全体样本原始记录，不把它保存为已核验共同日。若用户未要求固定该日，可按列出的真实共同日重新调用本工具；已有快照会复用，不重新下载。用户明确固定日期则披露缺口，不擅自改日期。"})
        if not common:
            issues.append({"code": "INPUT_NO_COMMON_MARKET_DATE", "detail": "没有最近七天内的共同可得行情日；不混用各公司不同日期。"})
    grouped = defaultdict(list)
    for candidate_id, row in candidates:
        if row.as_of and row.as_of != pricing_date:
            continue
        grouped[input_key(row, target)].append((candidate_id, row))
    existing = dataset.active_records()
    all_ids = {row.input_id for row in dataset.records}
    saved = []
    for key, entries in grouped.items():
        related = [row for row in existing if input_key(row, target) == key or row.source.kind == "user"
            and row.role == "historical" and entries[0][1].role == "historical" and row.metric == key[1]
            and row.period_end is None and row.as_of is None]
        values = {row.value for _, row in entries} | {row.value for row in related}
        if any(row.currency and row.currency != dataset.currency for _, row in entries):
            issues.append({"ticker": key[0], "metric": key[1], "code": "INPUT_CURRENCY", "detail": "币种不同，不自动换汇或重标单位。"})
            continue
        if len(values) > 1:
            issues.append({"ticker": key[0], "metric": key[1], "period_end": str(key[2] or ""), "as_of": str(key[3] or ""),
                "code": "INPUT_CONFLICT", "candidate_ids": [candidate_id for candidate_id, _ in entries][:12],
                "existing_input_ids": [row.input_id for row in related], "detail": "不自动择一、平均或覆盖；请复核来源后显式更正。"})
        for _, row in entries:
            if row.input_id not in all_ids:
                dataset.records.append(row)
                all_ids.add(row.input_id)
                saved.append(row.input_id)
    accepted = dataset.active_records()
    coverage = []
    for ticker in tickers:
        wanted = metrics - ({"market_cap"} if ticker == target else {"common_shares"})
        for year in sorted(selected_years, reverse=True):
            year_values = defaultdict(set)
            for row in accepted:
                if (row.entity_ticker or target) == ticker and row.period_end == date(year, 12, 31):
                    year_values[row.metric].add(row.value)
            unique = {metric: next(iter(values)) for metric, values in year_values.items() if len(values) == 1}
            derived = {}
            if ticker == target:
                for metric in sorted(wanted & DERIVATIONS.keys()):
                    try:
                        computed, _, formulas = derive_snapshot_inputs(unique, {}, {metric})
                    except ValueError as exc:
                        issues.append({"ticker": ticker, "year": year, "metric": metric, "code": str(exc).split(":", 1)[0], "detail": str(exc)})
                        continue
                    if metric in computed and metric not in unique:
                        derived[metric] = {"value": str(computed[metric]), "formula": formulas[metric],
                            "dependencies": sorted(derivation_dependencies({metric}) - {metric})}
            missing = sorted(wanted - {"market_cap", "common_shares"} - unique.keys() - derived.keys())
            coverage.append({"ticker": ticker, "year": year,
                "admitted": sorted(metric for metric in unique if not metric.startswith("raw.")),
                "raw_operand_count": sum(metric.startswith("raw.") for metric in unique),
                "derived_preview": derived, "missing": missing})
        for metric in wanted & {"common_shares", "market_cap"}:
            if not any((row.entity_ticker or target) == ticker and row.metric == metric and row.as_of == pricing_date for row in accepted):
                issues.append({"ticker": ticker, "metric": metric, "code": "INPUT_INSTANT_MISSING"})
    dataset.analysis_basis = dataset_basis(accepted)
    session.input_dataset = dataset
    observed_requested_date = bool(args.pricing_date and any(row.as_of == args.pricing_date for _, row in candidates))
    if (pricing_date in common or observed_requested_date) and len(tickers) > 1 and session.draft.peer_pricing_date is None:
        session.draft.peer_pricing_date = pricing_date
        session.draft.peer_pricing_rationale = (
            "按数据任务声明的最近共同可得日政策，从原始行情记录选择" if pricing_date in common else
            "按数据任务显式指定行情日选择；仅部分样本有该日原始记录，缺项样本仍不可用：") + pricing_date.isoformat() + "；保留与估值日的差异，不声称已核对交易日历或属于各市场实际市值之和。"
    outcome = {"status": "partial" if issues or any(row["missing"] for row in coverage) else "inputs_acquired",
        "target": target, "years": sorted(selected_years), "requested_history_years": sorted(set(args.years)),
        "baseline_selection": {**dataset.baseline_selection, "policy": dataset.baseline_selection.get("policy", "latest_available"),
            "period_end": baseline.isoformat() if baseline else None}, "pricing_date": pricing_date.isoformat() if pricing_date else None,
        "observed_common_dates": sorted(day.isoformat() for day in common),
        "available_annual_periods": {ticker: sorted(periods, reverse=True) for ticker, periods in available_periods.items()},
        "new_input_count": len(saved), "active_input_count": len(accepted), "coverage": coverage,
        "issues": issues, "sources": files,
        "instruction": "标准字段及可用原始科目已从原始响应绑定保存，仍是供应商契约输入而非独立审计。derived_preview是确定性公式预览，不是新增原始披露或估值结果；冻结输入时重新验证并推导。无需再次record_inputs重复录入同一原始数值或重新取数。按当前方法check_preparation；缺项只处理对应原文/冲突。已有数值不等于所有方法均可计算。"}
    raw_groups = defaultdict(int)
    for row in accepted:
        if row.metric.startswith("raw.") and row.entity_ticker in tickers and row.period_end:
            raw_groups[(row.entity_ticker, row.period_end)] += 1
    outcome["raw_operand_count"] = sum(raw_groups.values())
    outcome["interpretation_targets"] = [{"ticker": ticker, "period_end": period.isoformat(), "raw_operand_count": count,
        "next_action": {"tool": "inspect_inputs", "arguments": {"ticker": ticker, "period_end": period.isoformat(), "limit": 12}}}
        for (ticker, period), count in sorted(raw_groups.items(), key=lambda item: (item[0][1], item[0][0]), reverse=True)[:12]]
    if raw_groups:
        outcome["instruction"] += " raw.*是已保存且可引用的原始金额，不是模型准入字段，不必read_file重抄。按interpretation_targets查看input_id，随后用record_inputs(calculations=...)声明EBIT、折旧摊销、现金/债务等计算或解释；说明组成是否完整、重复和限制，不把null当0，不把财务费用等同利息。"
    unsupported = [item for item in files if item.get("excluded_by_reason", {}).get("INPUT_PROVIDER_CONTRACT")]
    if not saved:
        outcome["instruction"] = "本次没有新增可入模字段；不等于没有下载到数据，也不能声称已经绑定。查看sources的排除原因及issues；读取已有快照，不重复相同取数。"
    newer = {ticker: max(periods) for ticker, periods in available_periods.items() if periods and int(max(periods)[:4]) > max(selected_years)}
    if newer:
        outcome["newer_unselected_annual_periods"] = newer
        outcome["instruction"] += " 原始快照还包含较新的已披露年度，见newer_unselected_annual_periods；用户未指定旧基期时应选较新年度。用list_input_candidates和record_inputs选择现有来源，不重复下载；明确旧基期研究则保留并披露，不声称最新。"
    if unsupported:
        outcome["instruction"] += " 部分来源暂无直接字段契约，需要LLM读取原始JSON并解释：先inspect_file/read_file(view=records)查看结构，再按原文提取、复核；不能把其数值写入用户数据。"
        outcome["reading_targets"] = [{"file_id": item["file_id"], "ticker": item["ticker"],
            "reason": "INPUT_PROVIDER_CONTRACT", "next_tool": "inspect_file"} for item in unsupported]
    session.input_acquisition = outcome
    runtime.service.store.save_research(session)
    runtime.service.store.append_event(session.session_id, type="inputs.acquired", stage="inputs", status="completed",
        summary="完成有界数据任务，保存标准字段与未解决问题", payload={"request": args.model_dump(mode="json"),
            "saved_input_ids": saved, "outcome": outcome})
    return outcome
