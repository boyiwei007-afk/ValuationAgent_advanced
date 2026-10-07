"""Fixed ±10% single-factor stresses; these are scenarios, not probabilities."""
from decimal import Decimal as D
from valuationagent.schemas.models import SensitivityStudy


def relative_sensitivity(finance, request, financials, peers):
    studies = []
    fields = {"pe": "net_income_parent", "ps": "revenue", "ev_ebitda": "ebitda"}
    for method, field in fields.items():
        if method not in request.methods:
            continue
        scoped = request.model_copy(update={"methods": [method]})
        base_rows = finance.relative(scoped, financials, peers)
        if not base_rows:
            continue
        base = base_rows[0]
        if base.status != "success":
            continue
        for axis in ("metric", "multiple"):
            prices = []
            for factor in (D("0.9"), D("1.1")):
                fin = financials.model_copy(update={field: getattr(financials, field) * factor}) if axis == "metric" else financials
                sample = [p.model_copy(update={method: getattr(p, method) * factor if getattr(p, method) is not None else None}) for p in peers] if axis == "multiple" else peers
                trial = scoped.model_copy(deep=True)
                if axis == "multiple" and method in trial.assumptions.relative_multiples:
                    trial.assumptions.relative_multiples[method] *= factor
                value = finance.relative(trial, fin, sample)[0]
                prices.append(value.per_share_value)
            change = max(abs(p - base.per_share_value) for p in prices) / abs(base.per_share_value) if base.per_share_value else None
            studies.append(SensitivityStudy(study_id=f"R-{method}-{axis}", parameter=f"{method.upper()} · {field if axis == 'metric' else '指定倍数' if method in scoped.assumptions.relative_multiples else '可比倍数'}",
                baseline_input=str(getattr(financials, field)) if axis == "metric" else str(scoped.assumptions.relative_multiples.get(method, "原始可比样本")),
                low_input="基准 × 90%", high_input="基准 × 110%", baseline_per_share=base.per_share_value,
                low_per_share=prices[0], high_per_share=prices[1], max_relative_change=change,
                classification="not_available" if change is None else "high" if change >= D("0.2") else "medium" if change >= D("0.1") else "low",
                status="completed", rationale="固定其他输入，单独上下扰动10%；EV/EBITDA保持净债务不变。属于压力测试，不代表发生概率或置信区间。"))
    return studies
