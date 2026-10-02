"""Explicit what-if calculations against an immutable valuation baseline."""
from decimal import Decimal
from typing import Literal

from pydantic import Field

from valuationagent.core.tools import canonical
from valuationagent.schemas.models import ApiModel


class SensitivityRequest(ApiModel):
    method: Literal["dcf", "pe", "ps", "ev_ebitda"]
    parameter: Literal["wacc", "terminal_growth", "multiple", "earnings_scale", "revenue_scale", "ebitda_scale", "shares_scale"]
    values: list[Decimal] = Field(min_length=1, max_length=12, description="wacc/terminal_growth填绝对小数率（8%=0.08）；multiple填绝对倍数，统一替换可比样本作假设；*_scale填基准倍率（下降10%=0.9）。只改变选中的一个因素。")


def analyze_sensitivity(runtime, args):
    if runtime.workspaces is None:
        raise ValueError("SENSITIVITY_BASELINE_REQUIRED: 必须在有已完成估值的工作区试算。")
    workspace = runtime.service.store.workspace_for_research(runtime.session.session_id)
    if workspace is None or not workspace.active_run_id:
        raise ValueError("SENSITIVITY_BASELINE_REQUIRED: 尚无正式计算基准，不生成虚构敏感性价格。")
    record = runtime.service.store.get_run(workspace.active_run_id)
    result = record.result
    if not result or str(record.status) not in {"completed", "completed_with_warnings"} or result.effective_financials is None:
        raise ValueError("SENSITIVITY_BASELINE_REQUIRED: 当前版本未完成或缺少冻结的有效输入。")
    finance = runtime.workspaces.runner.finance
    model_version = finance.model_version_for(record.request) if hasattr(finance, "model_version_for") else finance.version
    if model_version != result.model_version:
        raise ValueError("SENSITIVITY_MODEL_CHANGED: 金融模型版本已变更，须先重新验证估值基准。")
    allowed = {"dcf": {"wacc", "terminal_growth", "shares_scale"},
               "pe": {"multiple", "earnings_scale", "shares_scale"},
               "ps": {"multiple", "revenue_scale", "shares_scale"},
               "ev_ebitda": {"multiple", "ebitda_scale", "shares_scale"}}
    if args.parameter not in allowed[args.method]:
        raise ValueError("SENSITIVITY_PARAMETER_SCOPE: 所选因素不适用于该方法。")
    base = result.dcf if args.method == "dcf" else next((row for row in result.relative if row.method == args.method and row.status == "success"), None)
    if base is None:
        raise ValueError("SENSITIVITY_METHOD_UNAVAILABLE: 基准中该方法没有有效数值结果。")
    values = list(dict.fromkeys(args.values))
    for value in values:
        if not value.is_finite():
            raise ValueError("SENSITIVITY_RANGE: 参数必须是有限数值。")
        valid = (Decimal("0") < value < Decimal("0.5") if args.parameter == "wacc" else
                 Decimal("-0.1") <= value < Decimal("0.2") if args.parameter == "terminal_growth" else
                 Decimal("0") < value <= Decimal("1000") if args.parameter == "multiple" else
                 Decimal("0.1") <= value <= Decimal("3"))
        if not valid:
            raise ValueError("SENSITIVITY_RANGE: 折现率须在(0,0.5)，永续增长在[-0.1,0.2)，倍数在(0,1000]，倍率在[0.1,3]。")
    request = record.request.model_copy(deep=True, update={"methods": [args.method]})
    rows = []
    fields = {"earnings_scale": "net_income_parent", "revenue_scale": "revenue", "ebitda_scale": "ebitda", "shares_scale": "common_shares"}
    for value in values:
        runtime.service._check_execution()
        financials = result.effective_financials.model_copy(deep=True)
        assumptions = result.assumptions.model_copy(deep=True)
        peers = [peer.model_copy(deep=True) for peer in result.effective_peers]
        if args.parameter in {"wacc", "terminal_growth"}:
            assumptions = type(assumptions).model_validate({**assumptions.model_dump(), args.parameter: value})
        elif args.parameter == "multiple":
            peers = [peer.model_copy(update={args.method: value}) for peer in peers]
        else:
            field = fields[args.parameter]
            original = getattr(financials, field)
            if original is None:
                raise ValueError("SENSITIVITY_INPUT_MISSING: 基准缺少所选因素，不能填0试算。")
            changes = {field: original * value}
            if args.parameter == "shares_scale" and financials.diluted_shares is not None:
                changes["diluted_shares"] = financials.diluted_shares * value
            financials = type(financials).model_validate({**financials.model_dump(), **changes})
        if args.method == "dcf":
            stable_roic = assumptions.operating_drivers.get("stable_roic")
            if assumptions.terminal_growth >= assumptions.wacc or stable_roic is not None and assumptions.terminal_growth >= stable_roic:
                rows.append({"input": str(value), "status": "invalid", "reason": "永续增长必须低于WACC及适用的稳定期ROIC。", "per_share_value": None})
                continue
            forecast = finance.forecast(request, financials, assumptions)
            computed = finance.dcf(request, financials, assumptions, forecast)
        else:
            candidates = finance.relative(request, financials, peers)
            computed = next((row for row in candidates if row.method == args.method and row.status == "success"), None)
            if computed is None:
                rows.append({"input": str(value), "status": "invalid", "reason": "情景不满足该方法计算条件。", "per_share_value": None})
                continue
        rows.append({"input": str(value), "status": "completed", "per_share_value": str(computed.per_share_value),
                     "change": str(computed.per_share_value - base.per_share_value)})
    output = {"schema": "valuation-sensitivity-v1", "baseline_run_id": record.run_id,
              "baseline_input_hash": record.input_hash, "effective_input_hash": result.effective_input_hash,
              "model_version": result.model_version, "method": args.method, "parameter": args.parameter,
              "baseline_per_share": str(base.per_share_value), "scenarios": rows,
              "currency": result.currency, "baseline_unchanged": True,
              "notice": "单因素假设试算，不是新的事实或正式估值版本，不是概率区间；其余输入固定，未联网、未改变原模型。multiple表示全部可比样本统一使用假设倍数，EV/EBITDA保持权益桥接不变。"}
    replay = {"request": request.model_dump(mode="json"), "financials": result.effective_financials.model_dump(mode="json"),
              "assumptions": result.assumptions.model_dump(mode="json"), "peers": [peer.model_dump(mode="json") for peer in result.effective_peers]}
    runtime.service._check_execution()
    artifact = runtime.service.store.save_artifact(runtime.session.session_id, {
        "kind": "sensitivity_analysis", "filename": "sensitivity-analysis.json", "media_type": "application/json",
        "status": "scenario_analysis", "numeric_result_available": any(row["status"] == "completed" for row in rows),
        "valuation_run_id": record.run_id, "source_revision": runtime.session.revision,
    }, canonical({**output, "frozen_inputs": replay}).encode("utf-8"))
    return {**output, "artifact": artifact}
