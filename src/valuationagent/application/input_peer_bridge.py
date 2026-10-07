from decimal import Decimal

from valuationagent.application.input_calculations import apply_calculations, calculation_operands
from valuationagent.application.input_derivations import derivation_dependencies, derive_snapshot_inputs
from valuationagent.application.input_workspace import input_evidence
from valuationagent.schemas.models import PeerCompany


SOURCE_PEER_POLICY = "source-peer-capital-bridge-20261007-v1"
DENOMINATORS = {"pe": "net_income_parent", "ps": "revenue", "ev_ebitda": "ebitda"}
BRIDGE_ADDITIONS = ("interest_bearing_debt", "lease_liabilities", "minority_interest",
    "preferred_equity", "unfunded_pension", "non_operating_provisions")
BRIDGE_DEDUCTIONS = ("cash_and_non_operating_assets", "associates_and_non_operating_investments")
BRIDGE_FIELDS = set(BRIDGE_ADDITIONS + BRIDGE_DEDUCTIONS)
BRIDGE_REQUIRED = {"cash_and_non_operating_assets", "interest_bearing_debt", "lease_liabilities", "minority_interest"}


def prepare_source_peer(choice, dataset, methods, baseline, pricing_date, cutoff):
    from valuationagent.application.input_peers import single_value

    ticker = choice.ticker
    outcome = {**choice.model_dump(mode="json"), "status": "excluded", "methods": [], "reasons": [],
        "selection_basis": "agent_judgment", "pricing_date": pricing_date.isoformat(),
        "financial_period_end": baseline.isoformat()}
    if not choice.enabled:
        outcome["reasons"].append("显式剔除；原始数据与过去冻结结果保留")
        return None, [], outcome
    bridge = choice.capital_bridge if "ev_ebitda" in methods else None
    if bridge and bridge.balance_date > min(pricing_date, cutoff):
        raise ValueError("INPUT_PEER_DATE: 可比桥接余额晚于行情日或信息截止日，不能使用未来资本结构。")
    annual_metrics = derivation_dependencies({DENOMINATORS[method] for method in methods})
    formulas = [item for item in dataset.active_calculations() if item.entity_ticker == ticker
        and (item.metric in annual_metrics and item.period_end == baseline
            or bridge and item.metric in BRIDGE_FIELDS and item.period_end == bridge.balance_date)]
    operand_ids = {row.input_id for item in formulas for _, row in calculation_operands(dataset, item)}
    current = [row for row in dataset.active_records() if row.role == "comparable" and row.entity_ticker == ticker
        and (row.metric == "market_cap" and row.as_of == pricing_date
            or row.metric in annual_metrics and row.period_end == baseline
            or bridge and row.metric in BRIDGE_FIELDS and row.period_end == bridge.balance_date
            or row.input_id in operand_ids)]
    if any(row.currency != dataset.currency for row in current):
        raise ValueError("INPUT_PEER_CURRENCY: 可比与目标币种不一致，不隐式换汇。")
    if any(row.source.kind == "user" for row in current):
        raise ValueError("INPUT_PEER_SOURCE: 真实可比样本不得混入未核验的用户情景数值。")
    outcome["input_ids"] = [row.input_id for row in current]
    outcome["calculation_ids"] = [item.calculation_id for item in formulas]
    market_rows = [row for row in current if row.metric == "market_cap"]
    market_value = single_value(market_rows, ticker, "market_cap")
    if market_value is None or market_value <= 0:
        outcome["reasons"].append(f"缺少{pricing_date}的正值总市值；流通市值不可替代")
        return None, current, outcome
    bases = {row.source.provider_binding.get("pricing_basis") or "issuer_market_value" for row in market_rows}
    if len(bases) != 1:
        raise ValueError("INPUT_PEER_PRICING_BASIS: 同一可比的市值口径不同，不能静默混用。")
    values, evidence = {}, {"market_cap": [input_evidence(row) for row in market_rows]}
    annual_rows = [row for row in current if row.metric in annual_metrics and row.period_end == baseline]
    for metric in {row.metric for row in annual_rows}:
        rows = [row for row in annual_rows if row.metric == metric]
        values[metric] = single_value(rows, ticker, metric)
        evidence[metric] = [input_evidence(row) for row in rows]
    values, evidence, declared = apply_calculations(dataset, baseline, values, evidence, annual_metrics, entity_ticker=ticker)
    values, evidence, derived = derive_snapshot_inputs(values, evidence, {DENOMINATORS[method] for method in methods})
    peer = PeerCompany(ticker=ticker, name=choice.name, rationale=choice.rationale,
        peer_tier="broad", selection_basis="agent_judgment", pricing_basis=next(iter(bases)),
        market_cap=market_value, as_of_date=pricing_date, financial_period_end=baseline, multiple_basis="FY",
        evidence=evidence, calculation_methods={"source_peer_policy": SOURCE_PEER_POLICY, **declared, **derived})
    outcome["pricing_basis"] = peer.pricing_basis
    for method in methods:
        metric = DENOMINATORS[method]
        denominator = values.get(metric)
        if denominator is None or denominator <= 0:
            outcome["reasons"].append(f"{method.upper()}缺少{baseline}的正值{metric}；不补零，不套用TTM或其他年度")
            continue
        numerator = market_value
        dependencies = [*evidence["market_cap"], *evidence[metric]]
        if method == "ev_ebitda":
            if bridge is None:
                outcome["reasons"].append("EV桥接需要record_inputs.comparables.capital_bridge明确余额日、租赁覆盖和现金/投资去重政策；这不是需要重复搜索的金额字段")
                continue
            capital_rows = [row for row in current if row.metric in BRIDGE_FIELDS and row.period_end == bridge.balance_date]
            capital = {name: single_value([row for row in capital_rows if row.metric == name], ticker, name)
                for name in {row.metric for row in capital_rows}}
            refs = {name: [input_evidence(row) for row in capital_rows if row.metric == name] for name in capital}
            capital, refs, expressions = apply_calculations(dataset, bridge.balance_date, capital, refs, BRIDGE_FIELDS, entity_ticker=ticker)
            missing = BRIDGE_REQUIRED - capital.keys()
            if missing:
                outcome["reasons"].append(f"EV桥接缺少{bridge.balance_date}显式输入：" + ",".join(sorted(missing)))
                continue
            if any(capital[name] < 0 for name in BRIDGE_FIELDS & capital.keys()):
                raise ValueError("INPUT_PEER_BRIDGE: 可比资本调整项存在负值，需专项口径，不把负值改成零。")
            declared_lease = capital.get("interest_bearing_debt_includes_leases")
            if declared_lease is not None and declared_lease != bridge.debt_includes_leases:
                raise ValueError("INPUT_PEER_LEASE: 可比债务公式与资本桥接的租赁覆盖标记冲突。")
            adjustments = {name: capital[name] for name in BRIDGE_ADDITIONS if name in capital}
            if bridge.debt_includes_leases:
                if capital["interest_bearing_debt"] < capital["lease_liabilities"]:
                    raise ValueError("INPUT_PEER_LEASE: 含租赁的债务总额小于租赁负债，请核对完整性。")
                adjustments["lease_liabilities"] = Decimal(0)
            adjustments.update({name: -capital[name] for name in BRIDGE_DEDUCTIONS if name in capital})
            if bridge.cash_includes_associates and "associates_and_non_operating_investments" in capital:
                if capital["cash_and_non_operating_assets"] < capital["associates_and_non_operating_investments"]:
                    raise ValueError("INPUT_PEER_BRIDGE: 声明包含投资的现金及非经营资产总额小于投资明细。")
                adjustments["associates_and_non_operating_investments"] = Decimal(0)
            numerator += sum(adjustments.values(), Decimal(0))
            unmeasured = sorted(BRIDGE_FIELDS - capital.keys())
            warnings = ["资本桥接采用已披露余额代理，不等同于各项市场价值；科目范围和EBITDA一致性是建模判断。"]
            if unmeasured:
                warnings.append("未建立的调整项不等于零，本次未调整：" + ",".join(unmeasured))
            if bridge.balance_date != pricing_date:
                warnings.append(f"资本余额日{bridge.balance_date}与行情日{pricing_date}相差{(pricing_date - bridge.balance_date).days}天，不代表期间无资本变动。")
            peer.capital_bridge = {**bridge.model_dump(mode="json"), "values": {key: str(value) for key, value in capital.items()},
                "adjustments": {key: str(value) for key, value in adjustments.items()},
                "unmeasured_items": unmeasured, "warnings": warnings}
            peer.evidence.update(refs)
            dependencies += [ref for name in sorted(BRIDGE_FIELDS) for ref in refs.get(name, [])]
            peer.calculation_methods.update(expressions)
            peer.calculation_methods["enterprise_value"] = "market_cap + signed capital_bridge.adjustments; leases and associates counted once"
            outcome["reasons"].extend(warnings)
        if numerator <= 0:
            outcome["reasons"].append(f"{method}估值分子非正，不形成正值可比倍数")
            continue
        setattr(peer, method, numerator / denominator)
        if method == "ev_ebitda":
            peer.enterprise_value = numerator
        peer.evidence[method] = dependencies
        peer.calculation_methods[method] = f"{'enterprise_value' if method == 'ev_ebitda' else 'market_cap'} / {metric}; pricing_basis={peer.pricing_basis}; FY={baseline}"
        outcome["methods"].append(method)
    outcome["status"] = "included" if outcome["methods"] else "excluded"
    return peer if outcome["methods"] else None, current, outcome


def verify_source_peers(request):
    from valuationagent.schemas.inputs import ComparableSelection, InputDataset, InputRecord, InputCalculation

    choices = {item["ticker"]: ComparableSelection.model_validate({key: item[key] for key in ComparableSelection.model_fields if key in item})
        for item in request.peer_screening if not item["ticker"].startswith("user:")}
    dataset = InputDataset(entity=request.company.name or request.company.ticker, currency=request.company.currency,
        analysis_basis=request.analysis_basis, records=[InputRecord.model_validate(row) for row in request.input_records],
        calculations=[InputCalculation.model_validate(row) for row in request.input_calculations], comparables=choices)
    for peer in request.peers:
        if peer.ticker.startswith("user:") or not (peer.calculation_methods.get("source_peer_policy")
                or any(row.entity_ticker == peer.ticker and row.role == "comparable" for row in dataset.records)):
            continue
        if peer.calculation_methods.get("source_peer_policy") != SOURCE_PEER_POLICY or peer.ticker not in choices:
            raise ValueError("INPUT_PEER_POLICY_VERSION: 真实样本冻结口径或选样依据缺失/版本不一致，使用冻结源码复算，不迁移旧结果。")
        expected, _, _ = prepare_source_peer(choices[peer.ticker], dataset,
            [str(method) for method in request.methods if method != "dcf"], request.financials.period_end,
            peer.as_of_date, request.valuation_date)
        if expected is None or peer.model_dump() != expected.model_dump():
            raise ValueError("INPUT_PEER_REPLAY: 真实样本市值、资本桥接、期间、倍数或来源与冻结输入不一致。")
