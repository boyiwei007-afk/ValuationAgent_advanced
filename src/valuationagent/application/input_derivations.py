from decimal import Decimal


DERIVATION_VERSION = "unified-snapshot-formulas-20261003-v1"
RAW_INPUT_FIELDS = {
    "ebit", "profit_before_tax", "income_tax_expense", "cash_paid_for_ppe_intangibles",
    "depreciation_fixed_assets", "amortization_intangible_assets",
    "amortization_long_term_deferred_expenses", "depreciation_right_of_use",
    "inventory_decrease", "operating_receivables_decrease", "operating_payables_increase", "operating_nwc",
}
DERIVATIONS = {
    "ebit_margin": (("ebit", "revenue"), "ratio", "ebit / revenue"),
    "tax_rate": (("income_tax_expense", "profit_before_tax"), "ratio", "income_tax_expense / profit_before_tax"),
    "capital_expenditure": (("cash_paid_for_ppe_intangibles",), "absolute", "abs(cash_paid_for_ppe_intangibles)"),
    "depreciation_amortization": (("depreciation_fixed_assets", "amortization_intangible_assets",
        "amortization_long_term_deferred_expenses", "depreciation_right_of_use"), "sum",
        "depreciation_fixed_assets + amortization_intangible_assets + amortization_long_term_deferred_expenses + depreciation_right_of_use"),
    "ebitda": (("ebit", "depreciation_amortization"), "sum", "ebit + depreciation_amortization"),
    "change_operating_nwc": (("inventory_decrease", "operating_receivables_decrease", "operating_payables_increase"),
        "negative_sum", "-(inventory_decrease + operating_receivables_decrease + operating_payables_increase)"),
}


def derivation_dependencies(metrics):
    selected = set(metrics)
    pending = list(metrics)
    while pending:
        rule = DERIVATIONS.get(pending.pop())
        for metric in rule[0] if rule else ():
            if metric not in selected:
                selected.add(metric)
                pending.append(metric)
    return selected


def derive_snapshot_inputs(values, evidence, required):
    values = dict(values)
    evidence = {metric: list(references) for metric, references in evidence.items()}
    methods = {}
    visited = set()

    def derive(metric):
        if metric in visited:
            return
        visited.add(metric)
        rule = DERIVATIONS.get(metric)
        if rule is None:
            return
        dependencies, operation, expression = rule
        for dependency in dependencies:
            derive(dependency)
        if any(dependency not in values for dependency in dependencies):
            return
        operands = [values[dependency] for dependency in dependencies]
        if operation == "ratio":
            if operands[1] <= 0:
                if metric in values:
                    return
                raise ValueError(f"INPUT_DERIVATION_DOMAIN: {metric}的分母{dependencies[1]}须为正；保留原始数据，不补0或外推税率。")
            result = operands[0] / operands[1]
        elif operation == "absolute":
            result = abs(operands[0])
        else:
            result = sum(operands, Decimal(0)) * (-1 if operation == "negative_sum" else 1)
        if metric == "depreciation_amortization" and any(value < 0 for value in operands):
            if metric in values:
                return
            raise ValueError("INPUT_DERIVATION_DOMAIN: 折旧摊销组成出现负值，需解释冲回或口径，不自动取绝对值相加。")
        if metric in values:
            tolerance = max(abs(result) * Decimal("0.000001"), Decimal("0.00000001") if operation == "ratio" else Decimal("0.01"))
            if abs(values[metric] - result) > tolerance:
                raise ValueError(f"INPUT_DERIVATION_CONFLICT: {metric}直接值与{expression}不一致；核对期间、口径或舍入并显式更正，不覆盖或平均。")
            methods["reconciliation." + metric] = expression
            return
        values[metric] = result
        references = {reference.evidence_id: reference for dependency in dependencies for reference in evidence.get(dependency, [])}
        evidence[metric] = list(references.values())
        methods[metric] = expression

    for metric in sorted(required):
        derive(metric)
    if methods:
        methods["derivation_policy"] = DERIVATION_VERSION
    return values, evidence, methods


def verify_frozen_derivations(request):
    for snapshot in [*request.historical_financials, request.financials]:
        if snapshot is None or "derivation_policy" not in snapshot.calculation_methods:
            continue
        if snapshot.calculation_methods["derivation_policy"] != DERIVATION_VERSION:
            raise ValueError("INPUT_DERIVATION_REPLAY: 冻结推导版本与当前实现不一致。")
        targets = set(snapshot.calculation_methods) & DERIVATIONS.keys()
        if "balance_change_policy" in snapshot.calculation_methods:
            targets.discard("change_operating_nwc")
        if any(snapshot.calculation_methods[key] != DERIVATIONS[key][2] for key in targets):
            raise ValueError("INPUT_DERIVATION_REPLAY: 冻结推导表达式与注册公式不一致。")
        reconciliations = {key.removeprefix("reconciliation.") for key in snapshot.calculation_methods if key.startswith("reconciliation.")}
        dependencies = derivation_dependencies(targets | reconciliations)
        values = {key: snapshot.statement_items.get(key, getattr(snapshot, key, None)) for key in dependencies}
        values = {key: value for key, value in values.items() if value is not None and key not in targets}
        recomputed, _, _ = derive_snapshot_inputs(values, {}, targets | reconciliations)
        if any(metric not in recomputed or recomputed[metric] != getattr(snapshot, metric, None) for metric in targets):
            raise ValueError("INPUT_DERIVATION_REPLAY: 原始科目的确定性推导与冻结快照不一致。")
    return True
