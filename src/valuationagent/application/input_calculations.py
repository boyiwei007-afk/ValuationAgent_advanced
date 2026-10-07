from decimal import Decimal

from valuationagent.core.tools import canonical
from valuationagent.schemas.inputs import InputCalculation


CALCULATION_VERSION = "declared-linear-inputs-20261007-v4"
CALCULATION_KINDS = {"ebit": "annual", "depreciation_amortization": "annual",
    "cash_and_non_operating_assets": "instant", "interest_bearing_debt": "instant",
    "lease_liabilities": "instant", "minority_interest": "instant", "operating_nwc": "instant"}
NONNEGATIVE_CALCULATIONS = {"cash_and_non_operating_assets", "interest_bearing_debt", "lease_liabilities", "depreciation_amortization"}
INSTANT_FIELDS = {
    "cash_and_non_operating_assets", "interest_bearing_debt", "common_shares", "diluted_shares",
    "lease_liabilities", "minority_interest", "preferred_equity", "associates_and_non_operating_investments",
    "unfunded_pension", "non_operating_provisions", "market_cap", "market_price", "operating_nwc",
}
AMOUNT_UNITS = {"元", "千元", "万元", "百万元", "亿元"}


def lease_operand(row):
    binding = row.source.provider_binding
    return row.metric == "lease_liabilities" or (row.source.kind == "provider"
        and row.metric == "raw.tushare_balancesheet_lease_liab"
        and binding.get("statement") == "balancesheet" and binding.get("field") == "lease_liab")


def calculation_operands(dataset, calculation):
    records = {row.input_id: row for row in dataset.active_records()}
    selected, ids, identities = [], set(), set()
    for term in calculation.terms:
        row = records.get(term.input_id)
        if row is None:
            raise ValueError("INPUT_CALCULATION_STALE: 公式依赖不存在或已被更正；使用inspect_inputs查看当前输入，并显式替换公式，不自动沿用旧金额。")
        if term.input_id in ids:
            raise ValueError("INPUT_CALCULATION_DUPLICATE: 同一输入只能使用一次，不用重复加减掩盖缺项。")
        ids.add(term.input_id)
        peer = calculation.entity_ticker
        correct_entity = (row.role == "comparable" and row.entity_ticker == peer and peer in dataset.comparables
            if peer else row.role == "historical")
        if not correct_entity or row.scope != "consolidated" or row.currency != dataset.currency or row.unit not in AMOUNT_UNITS or row.metric == "market_price":
            raise ValueError("INPUT_CALCULATION_SCOPE: 公式只使用指定主体、同币种、合并口径金额；可比填entity_ticker，不能混入目标、其他公司、股数、比率或母公司数据。")
        if row.period_end != calculation.period_end or row.as_of is not None:
            raise ValueError("INPUT_CALCULATION_PERIOD: 公式期间与依赖必须一致，不跨期拼接或将独立股数/行情时点当财务期间。")
        kind = row.period_kind
        if kind != CALCULATION_KINDS[calculation.metric]:
            raise ValueError("INPUT_CALCULATION_KIND: EBIT和折旧摊销使用完整年度流量，现金、债务、租赁、少数权益和经营营运资本使用时点存量；原始科目的期间类型尚未明确或不一致。")
        identity = (row.source.kind, row.source.source_id, row.source.locator)
        if row.metric in identities or identity in identities:
            raise ValueError("INPUT_CALCULATION_DUPLICATE: 同指标的多个来源不能相加，同一原始金额也不能通过改名重复计入。")
        identities.update((row.metric, identity))
        selected.append((term, row))
    if calculation.metric == "interest_bearing_debt":
        if calculation.debt_includes_leases is None:
            raise ValueError("INPUT_CALCULATION_LEASE: 债务公式须说明是否已经含租赁负债。")
        if calculation.debt_includes_leases is False and any(lease_operand(row) and term.operation == "add" for term, row in selected):
            raise ValueError("INPUT_CALCULATION_LEASE: 公式已加租赁负债，却声明不含租赁；这会导致桥接重复扣除，请修正口径标记或公式。")
    elif calculation.debt_includes_leases is not None:
        raise ValueError("INPUT_CALCULATION_LEASE: 租赁覆盖标记仅用于有息债务公式。")
    return selected


def calculation_value(operands):
    return sum((row.value * (1 if term.operation == "add" else -1) for term, row in operands), Decimal(0))


def save_calculations(dataset, drafts):
    from valuationagent.application.input_workspace import digest

    if dataset is None:
        raise ValueError("INPUT_CALCULATION_SOURCE: 先保存用户或来源数据，公式只引用实际input_id。")
    saved = []
    for draft in drafts:
        operands = calculation_operands(dataset, draft)
        if draft.metric in NONNEGATIVE_CALCULATIONS and calculation_value(operands) < 0:
            raise ValueError(f"INPUT_CALCULATION_DOMAIN: {draft.metric}不能为负；核查重叠、冲回和符号，不自动取绝对值。")
        payload = draft.model_dump(mode="json")
        identifier = "calculation_" + digest(canonical(payload))[:24]
        if identifier in {row.calculation_id for row in dataset.calculations}:
            continue
        active = {row.calculation_id: row for row in dataset.active_calculations()}
        if any(key not in active or (active[key].entity_ticker, active[key].metric, active[key].period_end) != (draft.entity_ticker, draft.metric, draft.period_end) for key in draft.replaces):
            raise ValueError("INPUT_CALCULATION_REPLACEMENT: 只能显式替换当前有效、同指标同期间的公式。")
        if any((row.entity_ticker, row.metric, row.period_end) == (draft.entity_ticker, draft.metric, draft.period_end) and key not in draft.replaces for key, row in active.items()):
            raise ValueError("INPUT_CALCULATION_CONFLICT: 已有同指标同期间公式，使用replaces说明更正，不静默覆盖。")
        dataset.calculations.append(InputCalculation(calculation_id=identifier, **payload))
        saved.append(identifier)
    return saved


def apply_calculations(dataset, period, values, evidence, required, *, entity_ticker=""):
    from valuationagent.application.input_workspace import input_evidence

    formulas = {}
    values, evidence = dict(values), {key: list(refs) for key, refs in evidence.items()}
    for calculation in dataset.active_calculations():
        if calculation.entity_ticker != entity_ticker or calculation.period_end != period or calculation.metric not in required:
            continue
        operands = calculation_operands(dataset, calculation)
        result = calculation_value(operands)
        if calculation.metric in NONNEGATIVE_CALCULATIONS and result < 0:
            raise ValueError(f"INPUT_CALCULATION_DOMAIN: {calculation.metric}为负，需复核，不冻结错误输入。")
        if calculation.metric in values and values[calculation.metric] != result:
            raise ValueError("INPUT_CALCULATION_CONFLICT: 原始直接值与声明计算不一致；将未调整科目按原始科目录入，不冒充调整后值。")
        values[calculation.metric] = result
        evidence[calculation.metric] = [input_evidence(row) for _, row in operands]
        formulas[calculation.metric] = " ".join(("+ " if term.operation == "add" else "- ") + row.input_id for term, row in operands)
        if calculation.metric == "interest_bearing_debt":
            values["interest_bearing_debt_includes_leases"] = calculation.debt_includes_leases
    if formulas:
        formulas["declared_calculation_policy"] = CALCULATION_VERSION
    return values, evidence, formulas


def verify_frozen_calculations(request):
    from valuationagent.schemas.inputs import ComparableSelection, InputDataset, InputRecord

    if not request.input_calculations:
        return True
    dataset = InputDataset(entity=request.company.name or request.company.ticker, currency=request.company.currency,
        analysis_basis=request.analysis_basis, records=[InputRecord.model_validate(row) for row in request.input_records],
        calculations=[InputCalculation.model_validate(row) for row in request.input_calculations],
        comparables={item["ticker"]: ComparableSelection(ticker=item["ticker"], name=item["name"],
            rationale=item["rationale"]) for item in request.peer_screening})
    snapshots = {row.period_end: row for row in [*request.historical_financials, request.financials] if row is not None}
    for calculation in dataset.active_calculations():
        if calculation.entity_ticker:
            calculation_operands(dataset, calculation)
            continue
        snapshot = snapshots.get(calculation.period_end)
        if snapshot is None or snapshot.calculation_methods.get("declared_calculation_policy") != CALCULATION_VERSION:
            raise ValueError("INPUT_CALCULATION_REPLAY: 缺少相应冻结期间或声明计算版本不匹配。")
        result = calculation_value(calculation_operands(dataset, calculation))
        frozen = snapshot.statement_items.get(calculation.metric, getattr(snapshot, calculation.metric, None))
        if result != frozen:
            raise ValueError("INPUT_CALCULATION_REPLAY: 原始依赖的加减结果与冻结快照不一致。")
        if calculation.metric == "interest_bearing_debt" and snapshot.interest_bearing_debt_includes_leases != calculation.debt_includes_leases:
            raise ValueError("INPUT_CALCULATION_REPLAY: 冻结快照的租赁覆盖标记与声明不一致。")
    return True
