from valuationagent.application.input_workspace import input_evidence
from valuationagent.schemas.models import PeerCompany


USER_PEER_POLICY = "user-scenario-peer-method-scope-20261007-v2"


def prepare_user_peer(choice, records, methods, baseline, pricing_date):
    from valuationagent.application.input_peers import single_value

    market_fields = {"market_cap", "market_price", "common_shares"}
    bridge_fields = {"cash_and_non_operating_assets",
        "interest_bearing_debt", "lease_liabilities", "minority_interest", "preferred_equity",
        "unfunded_pension", "non_operating_provisions", "associates_and_non_operating_investments"}
    denominators = {"pe": "net_income_parent", "ps": "revenue", "ev_ebitda": "ebitda"}
    needed = market_fields | {denominators[method] for method in methods}
    if "ev_ebitda" in methods:
        needed |= bridge_fields
    instant = market_fields | bridge_fields
    current = [row for row in records if row.role == "comparable" and row.entity_ticker == choice.ticker
        and row.metric in needed
        and (row.as_of == pricing_date if row.metric in instant else row.period_end == baseline)]
    if any(row.source.kind != "user" for row in current):
        raise ValueError("INPUT_PEER_SOURCE: 用户情景样本不得混入未经身份与期间核验的市场数据。")
    outcome = {**choice.model_dump(mode="json"), "status": "excluded", "methods": [], "reasons": [],
        "selection_basis": "user_scenario", "pricing_date": pricing_date.isoformat(),
        "financial_period_end": baseline.isoformat(), "input_ids": [row.input_id for row in current]}
    if not choice.enabled:
        outcome["reasons"].append("用户显式剔除，保留原始记录")
        return None, current, outcome
    values = {metric: single_value([row for row in current if row.metric == metric], choice.ticker, metric)
        for metric in {row.metric for row in current}}
    evidence = {metric: [input_evidence(row) for row in current if row.metric == metric] for metric in values}
    market_cap = values.get("market_cap")
    price, shares = values.get("market_price"), values.get("common_shares")
    derived_cap = price * shares if price is not None and shares is not None else None
    if derived_cap is not None and market_cap is not None and derived_cap != market_cap:
        raise ValueError(f"INPUT_PEER_CONFLICT: {choice.name} 的市值与股价×股数不一致，不自动挑选。")
    if market_cap is None:
        market_cap = derived_cap
        if derived_cap is not None:
            evidence["market_cap"] = evidence["market_price"] + evidence["common_shares"]
    if market_cap is None or market_cap <= 0 or price is not None and price <= 0 or shares is not None and shares <= 0:
        outcome["reasons"].append("缺少同一估值日正值总市值，或股价与普通股股数；不要求联网补证用户情景")
        return None, current, outcome
    peer = PeerCompany(ticker=choice.ticker, name=choice.name, rationale=choice.rationale,
        peer_tier="user", market_cap=market_cap, as_of_date=pricing_date, financial_period_end=baseline,
        multiple_basis="FY", evidence=evidence, pricing_basis="issuer_market_value",
        calculation_methods={"user_peer_policy": USER_PEER_POLICY,
            "market_cap": "market_price * common_shares" if "market_cap" not in values else "user_input"})
    for method in methods:
        metric = denominators[method]
        denominator = values.get(metric)
        if denominator is None or denominator <= 0:
            outcome["reasons"].append(f"{method}缺少同年度正值{metric}")
            continue
        numerator = market_cap
        dependencies = ["market_cap", metric]
        if method == "ev_ebitda":
            required = {"cash_and_non_operating_assets", "interest_bearing_debt", "lease_liabilities", "minority_interest"}
            missing = required - values.keys()
            if missing:
                outcome["reasons"].append("EV桥接缺少显式输入：" + ",".join(sorted(missing)))
                continue
            additions = [name for name in ("interest_bearing_debt", "lease_liabilities", "minority_interest",
                "preferred_equity", "unfunded_pension", "non_operating_provisions") if name in values]
            deductions = [name for name in ("cash_and_non_operating_assets", "associates_and_non_operating_investments") if name in values]
            if any(values[name] < 0 for name in [*additions, *deductions]):
                raise ValueError("INPUT_PEER_BRIDGE: 可比资本调整项存在负值，需明确专项口径，不把负值改成零。")
            numerator += sum(values[name] for name in additions) - sum(values[name] for name in deductions)
            dependencies += additions + deductions
            peer.calculation_methods["enterprise_value"] = "market_cap + " + " + ".join(additions) + " - " + " - ".join(deductions)
            outcome["reasons"].append("情景EV仅调整已提供项目；租赁另列，不代表未提供科目经核验为零或独立市场估值")
        if numerator <= 0:
            outcome["reasons"].append(f"{method}估值分子非正，不能形成正值可比倍数")
            continue
        setattr(peer, method, numerator / denominator)
        peer.evidence[method] = [entry for name in dependencies for entry in evidence[name]]
        peer.calculation_methods[method] = f"{'enterprise_value' if method == 'ev_ebitda' else 'market_cap'} / {metric}; user_scenario; FY={baseline}"
        outcome["methods"].append(method)
    outcome["status"] = "included" if outcome["methods"] else "excluded"
    return peer if outcome["methods"] else None, current, outcome


def verify_user_peers(request):
    from valuationagent.schemas.inputs import ComparableSelection, InputRecord

    records = [InputRecord.model_validate(row) for row in request.input_records]
    for peer in request.peers:
        if not peer.ticker.startswith("user:"):
            continue
        if peer.calculation_methods.get("user_peer_policy") != USER_PEER_POLICY:
            raise ValueError("INPUT_PEER_POLICY_VERSION: 用户样本冻结口径与当前构造版本不一致，请使用冻结时的源码复算；不自动迁移旧结果。")
        if request.analysis_basis != "user_scenario" or not peer.as_of_date or not request.financials.period_end:
            raise ValueError("INPUT_PEER_REPLAY: 用户样本缺少明确情景、行情日或目标财务年度。")
        expected, _, _ = prepare_user_peer(ComparableSelection(ticker=peer.ticker, name=peer.name,
            rationale=peer.rationale), records, [str(method) for method in request.methods if method != "dcf"],
            request.financials.period_end, peer.as_of_date)
        if expected is None or any(getattr(peer, metric) != getattr(expected, metric)
            for metric in ("market_cap", "pe", "ps", "ev_ebitda", "calculation_methods", "evidence")):
            raise ValueError("INPUT_PEER_REPLAY: 用户样本市值、资本桥接或倍数与冻结原始输入不一致。")
