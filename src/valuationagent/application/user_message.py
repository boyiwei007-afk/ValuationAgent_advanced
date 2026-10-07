import re

from pydantic import Field

from valuationagent.core.tools import canonical
from valuationagent.llm.context_manager import redact_context_text, sensitive_spans
from valuationagent.schemas.models import ApiModel


NUMBER = re.compile(r"[+\-−]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?(?!\d)(?:[ \t]*(?:亿元|万元|千元|元/股|元|亿股|万股|股|倍|[%％]))?")


class ReadUserInput(ApiModel):
    message_id: str = Field(default="", description="省略时读取本轮用户消息；旧输入填真实用户消息ID。")
    query: str = Field(default="", max_length=100, description="可选原话关键词；空格、竖线|、顿号、逗号、分号或换行分隔，按任一包含匹配，不是正则。不解释字段。")
    start_line: int = Field(default=1, ge=1)
    limit: int = Field(default=16, ge=1, description="期望行数；服务器每页最多30行，更大请求仍返回有界页面及next_line，不代表后续资料不存在。")


def message_lines(content):
    offset = 0
    rows = []
    for index, raw in enumerate(content.splitlines(keepends=True), 1):
        body = raw.rstrip("\r\n")
        rows.append((index, offset, body))
        offset += len(raw)
    return rows


def selected_line(content, number):
    if number < 1 or number > len(message_lines(content)):
        raise ValueError("USER_INPUT_REF: 行号不在当前用户消息中，先read_user_input。")
    return message_lines(content)[number - 1]


def missing_amount_reference(content, reference):
    number = int(reference.split(":")[0])
    _, offset, body = selected_line(content, number)
    protected = sensitive_spans(content)
    choices = [{"amount_ref": f"{number}:{ordinal}", "text": match[0]}
        for ordinal, match in enumerate(NUMBER.finditer(body), 1)
        if not any(start < offset + match.end() and end > offset + match.start() for start, end in protected)]
    return ValueError(f"USER_INPUT_REF: amount_ref={reference}不存在；当前行原文及真实候选："
        + canonical({"line": number, "text": redact_context_text(body)[:600], "amounts": choices[:12]})
        + "。按原文含义重新选择，不猜序号、不重录已成功输入；这些位置不代表自动字段解释。")


def read_user_input(runtime, args):
    messages = {item.message_id: item for item in runtime.service.store.list_messages(runtime.session.session_id) if item.role == "user"}
    current = runtime.session.turn_control.message_id if runtime.session.turn_control else next(reversed(messages), "")
    message = messages.get(args.message_id or current)
    if message is None:
        raise ValueError("USER_INPUT_REF: 只能读取本工作区真实用户消息。")
    protected = sensitive_spans(message.content)
    terms = list(dict.fromkeys(term for term in re.split(r"[;；|、,，\s]+", args.query) if term))
    selected = [(number, offset, body) for number, offset, body in message_lines(message.content)
        if number >= args.start_line and (not terms or any(term in body for term in terms))]
    lines = []
    page_limit = min(args.limit, 30)
    for number, offset, body in selected[:page_limit]:
        amounts = [{"amount_ref": f"{number}:{index}", "text": match[0]}
            for index, match in enumerate(NUMBER.finditer(body), 1)
            if not any(start < offset + match.end() and end > offset + match.start() for start, end in protected)]
        lines.append({"line": number, "text": redact_context_text(body), "amounts": amounts})
    runtime.user_input_read = True
    selected_numbers = {line["line"] for line in lines}
    context_lines = [{"line": number, "text": redact_context_text(body)[:600], "truncated": len(body) > 600}
        for number, _, body in message_lines(message.content)[:8] if number not in selected_numbers]
    recovery = {"tool": "read_user_input", "arguments": {"message_id": message.message_id,
        "start_line": args.start_line, "limit": page_limit},
        "instruction": "本次关键词在所选范围未命中，不表示用户未提供数据。用此无关键词分页读取原话，不猜行号、不重复相同查询。"} if terms and not lines else None
    return {"message_id": message.message_id, "lines": lines, "total_lines": len(message_lines(message.content)),
        "match_count": len(selected), "page_limit": page_limit, "recovery": recovery,
        "context_lines": context_lines, "query_terms": ["[REDACTED]" if any(term in message.content[start:end]
            for start, end in protected) else redact_context_text(term) for term in terms],
        "context_instruction": "原消息开头用于保留全局阅读背景，不是已解析字段。可引用真实行号；是否适用于当前主体、单位和日期仍由你判断，不能盲套。",
        "next_line": selected[page_limit][0] if len(selected) > page_limit else None,
        "instruction": "这些只是原始数字位置，不是字段解析结果。用record_user_inputs：顶层unit及unit_line声明本批单位依据，periods=[{date:明确年度截止日,line:依据行号}]，as_of={date:明确时点,line:依据行号}；未知日期省略。rows仅metric/amount_ref及可选不同unit。金额引用是行:序号，单位/日期行号是整数。不要向rows填period_ref、as_of_ref或user_basis，不用record_inputs逐项录用户数字。年份和百分数也会被列为数字，不得误作金额。"}


def resolve_user_references(value, basis, content):
    updates = {}
    for field, quote_field in (("unit_ref", "unit_quote"), ("period_ref", "period_quote"), ("as_of_ref", "as_of_quote")):
        number = getattr(value, field) or getattr(basis, field)
        if number is None:
            continue
        _, _, body = selected_line(content, number)
        if sensitive_spans(body):
            raise ValueError("USER_INPUT_REF: 单位或日期整行包含凭据，请使用不含凭据的原话短片段。")
        quote = getattr(value, quote_field) or getattr(basis, quote_field)
        if quote and quote != body:
            raise ValueError("USER_INPUT_REF: 行引用和手写引文冲突；使用引用时省略手写引文。")
        if quote_field == "period_quote" and value.period_end is None or quote_field == "as_of_quote" and value.as_of is None:
            if getattr(value, field) is not None:
                date_field = "period_end" if field == "period_ref" else "as_of"
                raise ValueError(f"INPUT_DATE_REQUIRED: {field}={number}仅定位原文；metric={value.metric}还需填写{date_field}，不静默丢弃已给日期引用。")
            continue
        updates[quote_field] = body
    start = None
    if value.amount_ref:
        line, ordinal = map(int, value.amount_ref.split(":"))
        _, offset, body = selected_line(content, line)
        matches = list(NUMBER.finditer(body))
        if ordinal > len(matches):
            raise missing_amount_reference(content, value.amount_ref)
        match = matches[ordinal - 1]
        start = offset + match.start()
        if any(secret_start < offset + match.end() and secret_end > start for secret_start, secret_end in sensitive_spans(content)):
            raise ValueError("USER_INPUT_REF: 不能将凭据当作财务数据。")
        if value.amount_text and value.amount_text != match[0]:
            raise ValueError("USER_INPUT_REF: amount_ref与手写amount_text不一致；使用引用时省略amount_text，不重写单位。")
        updates["amount_text"] = match[0]
    return value.model_copy(update=updates), start
