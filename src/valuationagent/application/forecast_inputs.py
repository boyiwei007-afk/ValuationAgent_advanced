from valuationagent.application.input_sources import validate_source_inputs
from valuationagent.application.input_workspace import input_evidence


def forecast_input_evidence(session, evidence_ids):
    selected_ids = [key for key in evidence_ids if key.startswith("input_")]
    records = {row.input_id: row for row in session.input_dataset.active_records()} if session.input_dataset else {}
    missing = [key for key in selected_ids if key not in records]
    if missing:
        raise ValueError("FORECAST_INPUT_CHANGED: 预测引用不存在或已被更正，须引用当前有效input_id重新提出方案：" + ", ".join(missing))
    selected = [records[key] for key in dict.fromkeys(selected_ids)]
    validate_source_inputs(session, selected)
    cutoff = session.draft.information_cutoff_date or session.draft.valuation_date
    if any(row.source.published_at and cutoff and row.source.published_at > cutoff for row in selected):
        raise ValueError("FORECAST_FUTURE_SOURCE: 预测依据披露日晚于信息截止日，不得使用未来信息")
    return [input_evidence(row) for row in selected]


def verify_frozen_forecast_inputs(request):
    from valuationagent.schemas.inputs import InputRecord

    records = {row["input_id"]: InputRecord.model_validate(row) for row in request.input_records}
    superseded = {key for row in records.values() for key in row.supersedes}
    for refs in request.assumption_evidence.values():
        if not any(ref.source == "forecast_assumption" for ref in refs):
            continue
        for ref in refs:
            if not ref.evidence_id.startswith("input_"):
                continue
            row = records.get(ref.evidence_id)
            if row is None or row.input_id in superseded or input_evidence(row) != ref:
                raise ValueError("FORECAST_INPUT_REPLAY: 预测引用与冻结的输入不一致")
