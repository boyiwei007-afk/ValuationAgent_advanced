from valuationagent.application.input_sources import validate_source_inputs
from valuationagent.application.input_conflicts import conflict_detail
from valuationagent.core.tools import canonical
from valuationagent.market.tushare import normalize_a_share_ticker


def single_value(rows, ticker, metric):
    if not rows:
        return None
    if len({row.value for row in rows}) != 1:
        raise ValueError(f"INPUT_PEER_CONFLICT: {ticker} 的 {metric} 存在同期间冲突；核对原文位置后用replaces更正，不能挑选或平均。候选："
            + canonical(conflict_detail(metric, rows)))
    return rows[0].value


def prepare_input_peers(session, methods, baseline):
    dataset = session.input_dataset
    if not methods or not dataset.comparables:
        return [], [], []
    valuation_date = session.draft.valuation_date
    pricing_date = session.draft.peer_pricing_date or valuation_date
    cutoff = session.information_cutoff_date or valuation_date
    if (not valuation_date or not pricing_date or not 0 <= (valuation_date - pricing_date).days <= 7
            or pricing_date > cutoff):
        raise ValueError("INPUT_PEER_DATE: 统一行情日须在估值日前七天内且不晚于信息截止日。")
    if pricing_date != valuation_date and len(session.draft.peer_pricing_rationale.strip()) < 12:
        raise ValueError("INPUT_PEER_DATE: 用update_task记录peer_pricing_date及日期选择理由，不改写原始行情日。")
    if baseline is None or baseline > pricing_date or baseline == pricing_date and dataset.analysis_basis != "user_scenario":
        raise ValueError("INPUT_PEER_PERIOD: 先明确目标公司的完整年度财务基期；可比年度须与目标一致且早于定价日。")
    records = dataset.active_records()
    peers, selected, screening = [], [], []
    for ticker, choice in dataset.comparables.items():
        if ticker.startswith("user:"):
            from valuationagent.application.user_peers import prepare_user_peer

            if dataset.analysis_basis != "user_scenario" or ticker != choice.ticker:
                raise ValueError("INPUT_PEER_SELECTION: 用户假设样本只用于用户情景，不冒充真实市场选样。")
            current_rows = [row for row in records if row.role == "comparable" and row.entity_ticker == ticker]
            if any(row.currency != dataset.currency for row in current_rows):
                raise ValueError("INPUT_PEER_CURRENCY: 用户可比与目标币种不一致，不隐式换汇。")
            peer, current, outcome = prepare_user_peer(choice, records, methods, baseline, pricing_date)
            selected.extend(current)
            screening.append(outcome)
            if peer:
                peers.append(peer)
            continue
        if ticker != normalize_a_share_ticker(choice.ticker) or ticker == normalize_a_share_ticker(session.draft.ticker):
            raise ValueError("INPUT_PEER_SELECTION: 可比主体键不一致或使用目标公司自身。")
        from valuationagent.application.input_peer_bridge import prepare_source_peer

        peer, current, outcome = prepare_source_peer(choice, dataset, methods, baseline, pricing_date, cutoff)
        if any(row.source.provider_binding for row in current):
            from valuationagent.application.issuer_identity import require_identity

            require_identity(session, ticker, choice.name)
        validate_source_inputs(session, current)
        selected.extend(current)
        screening.append(outcome)
        if peer:
            peers.append(peer)
    return peers, selected, screening
