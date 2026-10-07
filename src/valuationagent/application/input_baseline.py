from datetime import date
import hashlib


def apply_baseline(runtime, dataset, instruction):
    from valuationagent.application.input_workspace import bind_user_date
    from valuationagent.llm.context_manager import sensitive_spans

    messages = [item for item in runtime.service.store.list_messages(runtime.session.session_id) if item.role == "user"]
    message = next((item for item in reversed(messages) if not instruction.message_id or item.message_id == instruction.message_id), None)
    if message is None or instruction.user_quote not in message.content or sensitive_spans(instruction.user_quote):
        raise ValueError("INPUT_BASELINE_SOURCE: 基期例外必须引用本工作区用户原话，不引用模型判断、来源数据或凭据。")
    period = instruction.period_end
    cutoff = runtime.session.information_cutoff_date or runtime.session.draft.valuation_date
    if period:
        if (period.month, period.day) != (12, 31) or cutoff and period > cutoff:
            raise ValueError("INPUT_BASELINE_PERIOD: 基期须为信息截止日之前的完整年度期末，不能把中报或季度当作年度。")
        bind_user_date(period, instruction.user_quote, message.content, annual=True)
    dataset.baseline_selection = {"policy": "user_selected" if period else "latest_available",
        "period_end": period.isoformat() if period else None, "message_id": message.message_id,
        "user_quote": instruction.user_quote, "message_sha256": hashlib.sha256(message.content.encode()).hexdigest(),
        "limitation": "保存用户原话及Agent对基期指令的理解，不证明该年度输入完整或资料经过独立审计。"}


def validate_baseline_source(store, session):
    from valuationagent.application.input_workspace import bind_user_date

    selected = session.input_dataset.baseline_selection if session.input_dataset else {}
    if not selected:
        return
    message = next((item for item in store.list_messages(session.session_id) if item.role == "user" and item.message_id == selected.get("message_id")), None)
    if (message is None or hashlib.sha256(message.content.encode()).hexdigest() != selected.get("message_sha256")
            or not selected.get("user_quote") or selected["user_quote"] not in message.content):
        raise ValueError("INPUT_BASELINE_SOURCE_CHANGED: 基期选择与原始用户依据不一致，不得冻结计算。")
    if selected.get("policy") not in {"user_selected", "latest_available"}:
        raise ValueError("INPUT_BASELINE_SOURCE_CHANGED: 未知基期选择政策。")
    if selected.get("policy") == "user_selected":
        period = selected_period(session.input_dataset)
        if period is None or (period.month, period.day) != (12, 31):
            raise ValueError("INPUT_BASELINE_SOURCE_CHANGED: 历史基期必须明确完整年度。")
        bind_user_date(period, selected["user_quote"], message.content, annual=True)


def selected_period(dataset):
    value = dataset.baseline_selection.get("period_end")
    return date.fromisoformat(value) if dataset.baseline_selection.get("policy") == "user_selected" and value else None
