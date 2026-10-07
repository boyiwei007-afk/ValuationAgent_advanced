from valuationagent.application.input_sources import build_source_inputs, dataset_basis
from valuationagent.application.input_workspace import build_user_inputs
from valuationagent.application.input_views import input_overview
from valuationagent.application.input_calculations import save_calculations
from valuationagent.market.tushare import normalize_a_share_ticker
from valuationagent.schemas.inputs import InputDataset


def record_inputs(runtime, args):
    from valuationagent.application.provider_inputs import build_provider_inputs

    session = runtime.session
    dataset = session.input_dataset.model_copy(deep=True) if session.input_dataset else None
    saved = []
    if args.user_values:
        dataset, added = build_user_inputs(runtime, args, dataset)
        saved.extend(added)
    if args.provider_values:
        dataset, added = build_provider_inputs(runtime, args.provider_values, dataset)
        saved.extend(added)
    if args.source_values:
        dataset, added = build_source_inputs(runtime, args.source_values, dataset)
        saved.extend(added)
    if args.comparables:
        target = normalize_a_share_ticker(session.draft.ticker) if session.draft.ticker else ""
        dataset = dataset or InputDataset(entity=session.draft.company or session.draft.ticker, analysis_basis="research")
        seen = set()
        for choice in args.comparables:
            if choice.ticker.startswith("user:"):
                if choice.ticker not in dataset.comparables:
                    raise ValueError("INPUT_PEER_SELECTION: 先用user_values的comparable角色绑定用户实际给出的样本，不凭选样凭空创建数据。")
                ticker = choice.ticker
            else:
                try:
                    ticker = normalize_a_share_ticker(choice.ticker)
                except ValueError:
                    raise ValueError("INPUT_PEER_SELECTION: 可比代码不属于支持的A股。用户提供的虚构样本无需comparables登记或代码：仅提交user_values(role=comparable,entity=原话样本名)，程序自动登记user:名称；不要编造代码或复用目标代码。真实选样须使用范围内的A股代码。") from None
            if ticker == target or ticker in seen:
                raise ValueError("INPUT_PEER_SELECTION: 目标公司不可作为自身可比，同批代码不得重复。")
            seen.add(ticker)
            dataset.comparables[ticker] = choice.model_copy(update={"ticker": ticker})
    if dataset:
        from valuationagent.application.issuer_identity import require_identity

        provider_tickers = {row.entity_ticker for row in dataset.active_records() if row.source.provider_binding}
        for ticker, choice in dataset.comparables.items():
            if choice.enabled and ticker in provider_tickers:
                require_identity(session, ticker, choice.name)
    if args.baseline:
        from valuationagent.application.input_baseline import apply_baseline

        dataset = dataset or InputDataset(entity=session.draft.company or session.draft.ticker, analysis_basis="research")
        apply_baseline(runtime, dataset, args.baseline)
    saved_calculations = save_calculations(dataset, args.calculations) if args.calculations else []
    dataset.analysis_basis = dataset_basis(dataset.active_records())
    if dataset != session.input_dataset:
        session.valuation_methods_override = []
        session.valuation_method_exclusions = {}
        if (session.turn_control and "value" in session.turn_control.decision.actions
                and "calculate" in session.turn_control.effects):
            session.pending_action = "valuation"
    session.input_dataset = dataset
    runtime.service.store.save_research(session)
    runtime.service.store.append_event(session.session_id, type="inputs.recorded", stage="inputs", status="completed",
        summary="保存模型输入；分别保留用户消息、来源解释和供应商契约依据", payload={"input_ids": saved, "calculation_ids": saved_calculations, "analysis_basis": dataset.analysis_basis,
            "comparables": [choice.model_dump(mode="json") for choice in dataset.comparables.values()]})
    overview = input_overview(session, limit=0)
    selected = [row for row in dataset.active_records() if row.input_id in saved]
    return {**{key: overview[key] for key in ("entity", "analysis_basis", "active_count", "source_counts", "periods", "conflicts", "conflicts_omitted")},
        "saved_input_ids": saved, "saved_calculation_ids": saved_calculations, "baseline_selection": dataset.baseline_selection,
        "saved_records": [{**row.model_dump(mode="json", include={"input_id", "entity", "role", "metric", "value", "original_amount", "unit", "period_end", "as_of"}),
            "source_kind": row.source.kind} for row in selected[:12]],
        "saved_records_omitted": max(0, len(selected) - 12), "retrieve_with": "inspect_inputs",
        "instruction": "输入已保存，来源数据不是用户口述、供应商契约校验不是独立审计。按当前方法check_preparation/calculate_valuation，不重复提取已保存字段；用户假设不要求联网证明。更正使用replaces。"}
