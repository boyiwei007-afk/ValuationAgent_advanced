from __future__ import annotations
import json
import copy
from typing import Callable
from pydantic import ValidationError
from valuationagent.core.tools import ToolRegistry, canonical
from valuationagent.llm.client import ContextWindowError, LlmError, ToolProtocolError


def _context_size(messages):
    bounded = []
    for message in messages:
        if isinstance(message.get("content"), list):
            content = [({"type": "image_url", "image_url": "[bounded page image]" + " " * 10000}
                        if part.get("type") == "image_url" else part) for part in message["content"]]
            bounded.append({**message, "content": content})
        else:
            bounded.append(message)
    return len(canonical(bounded))


def omitted_tool_result(content):
    references = []
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        value = None

    def collect(item):
        if len(references) >= 40:
            return
        if isinstance(item, dict):
            reference = {key: item[key] for key in ("file_id", "block_id", "fact_id", "artifact_id", "packet_id", "location", "url", "published_at") if key in item}
            if reference and reference not in references:
                references.append(reference)
            for child in item.values():
                collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(value)
    outcome = {}
    if isinstance(value, dict):
        outcome = {key: value[key] for key in ("ok", "status", "saved_count", "total", "next_offset", "next_line") if key in value}
        if isinstance(value.get("error"), dict):
            outcome["error"] = {key: str(value["error"][key])[:1200] for key in ("code", "message") if key in value["error"]}
        if isinstance(value.get("rows"), list):
            outcome["rows"] = [{key: str(row[key])[:800] for key in ("row", "fact_id", "status", "error", "repair") if key in row}
                               for row in value["rows"][:6] if isinstance(row, dict)]
    return canonical({"context_omitted": True, "outcome": outcome, "retrieval_references": references,
        "instruction": "本条工具正文因上下文预算省略，完整输出仍在审计中。引用列表不是财务证据；按file_id/页码/事实ID小批重读正文再提取或复核，不重新下载，不推断未见内容。"})


def compact_tool_history(messages, budget):
    messages = copy.deepcopy(messages)
    initial = []
    groups = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            groups.append([message])
        elif groups:
            groups[-1].append(message)
        else:
            initial.append(message)
    reserve = min(_context_size(groups[-1]), budget // 3) if groups else 0
    try:
        initial = compact_state_metadata(copy.deepcopy(initial), budget - reserve)
    except LlmError:
        initial = compact_state_metadata(initial, budget)
    omitted = 0
    while groups and _context_size(initial + [item for group in groups for item in group]) > budget:
        if len(groups) > 1:
            groups.pop(0)
            omitted += 1
            continue
        for message in groups[0]:
            if message.get("role") == "tool":
                if _context_size(initial + groups[0]) <= budget:
                    break
                message["content"] = omitted_tool_result(message["content"])
        if _context_size(initial + groups[0]) > budget:
            groups.pop(0)
            omitted += 1
        break
    if omitted:
        notice = {"role": "user", "content": "较早工具轮次已从提示上下文移除；持久化任务与审计记录未删除。需要时用inspect_context/search_file/read_file/read_valuation检索。"}
        if _context_size(initial + [notice] + [item for group in groups for item in group]) <= budget:
            initial.append(notice)
    return initial + [item for group in groups for item in group]


def compact_state_metadata(initial, budget):
    if _context_size(initial) <= budget:
        return initial
    if len(initial) > 1 and initial[1].get("role") == "user":
        try:
            state = json.loads(initial[1]["content"])
        except (TypeError, ValueError):
            state = None
        context = state.get("context") if isinstance(state, dict) else None
        task = context.get("task_state") if isinstance(context, dict) else None
        if isinstance(task, dict):
            context["summary"] = ""
            context["recent_turns"] = [turn for turn in context.get("recent_turns", []) if turn.get("role") == "user"]
            state["context_projection"] = {"metadata_omitted": True,
                "instruction": "上下文压缩仅移除可检索状态和旧助手叙述；任务、用户消息及约束不变。inspect_context检索事实/文件，inspect_requirements检索准备度，read_valuation检索结果。缺少的字段不是不存在或数值为零。"}

            def refresh():
                context["confirmed_fact_ids"] = [fact["fact_id"] for fact in task.get("facts", []) if fact.get("status") == "confirmed"]
                context["evidence_ids"] = list(dict.fromkeys(fact["block_id"] for fact in task.get("facts", []) if fact.get("block_id")))
                initial[1] = {**initial[1], "content": canonical(state)}
                return _context_size(initial)

            for layer in ("recent_searches", "documents", "facts", "gaps", "user_note_ids"):
                rows = task.get(layer, [])
                while isinstance(rows, list) and rows and refresh() > budget:
                    rows.pop(0)
                    if layer in {"documents", "facts"}:
                        task[layer + "_omitted"] = task.get(layer + "_omitted", 0) + 1
            for owner, key, tool in ((state, "research_plan", "inspect_requirements"),
                                    (state, "valuation", "read_valuation"),
                                    (task, "last_issue", "inspect_context"),
                                    (task, "forecast_proposal", "inspect_context")):
                if refresh() > budget and key in owner:
                    owner[key] = {"context_omitted": True, "retrieve_with": tool}
            refresh()
    if _context_size(initial) > budget:
        raise LlmError("CONTEXT_BUDGET: 任务和用户约束本身超出预算，不能静默截断；请缩小当前任务。")
    return initial


def model_tool_result(value):
    if isinstance(value, list):
        return [model_tool_result(item) for item in value]
    if not isinstance(value, dict):
        return value
    if isinstance(value.get("hits"), list) and value.get("provider"):
        value = {**value, "hits": [
            {**hit, "snippet": hit["snippet"][:800], "snippet_truncated": True}
            if isinstance(hit, dict) and isinstance(hit.get("snippet"), str) and len(hit["snippet"]) > 800 else hit
            for hit in value["hits"]],
            "reading_instruction": "这里只是检索线索，较长摘要已缩短；全部来源ID与URL保留。先fetch_search_source取得原件，再search_file定位、read_file精读；完整检索响应在审计中。"}
    duplicate = (isinstance(value.get("text"), str) and isinstance(value.get("lines"), list)
                 and all(isinstance(line, dict) and isinstance(line.get("text"), str) for line in value["lines"])
                 and value["text"].splitlines() == [line["text"] for line in value["lines"]])
    return {key: model_tool_result(item) for key, item in value.items() if not (duplicate and key == "text")}


def run_tool_loop(
    llm, messages: list[dict], registry: ToolRegistry, call: Callable, *, max_rounds=6, max_tokens=900, check_cancel=None, max_context_chars=90000, allow_checkpoint=False, image_resolver=None, state_provider=None, request_adapter=None
):
    """Finite, auditable tool loop with bounded self-correction.

    Free text is never interpreted as a command or valuation. Tool failures are
    returned to the model as safe structured feedback; repeated no-progress
    failures stop and let the application ask the user how to recover.
    """
    messages = list(messages)
    trace = {"rounds": 0, "tools": [], "tool_errors": 0, "protocol_errors": 0, "context_recoveries": 0, "output_recoveries": 0}
    failed_signatures: dict[str, int] = {}
    last_failure = ""
    consecutive_errors = 0
    previous_argument_error = None
    for round_index in range(1, max_rounds + 1):
        if check_cancel:
            check_cancel()
        if state_provider is not None:
            if len(messages) < 2 or messages[1].get("role") != "user":
                raise LlmError("AGENT_STATE_INVALID: 动态工作区状态缺少固定上下文位置。")
            messages[1] = {"role": "user", "content": canonical(state_provider())}
        trace["rounds"] = round_index
        if round_index == max_rounds - 2:
            message = ("当前窗口只剩3次模型调用。若只是预算不足且仍有明确工作，用finish_response(outcome=checkpoint,next_steps=[具体未完成步骤])保存；系统会按实质进展和剩余预算决定同轮续做。不要把预算停止写成资料不可得，也不要编写未计算的估值判断。"
                       if allow_checkpoint else "本轮只剩3次模型调用。请保存任务计划并用finish_response如实总结已完成、未核验和下一步；不要承诺结束后后台自动继续。")
            messages.append({"role": "user", "content": message})
        adapted_messages, adapted_tools = (request_adapter(messages, registry.schemas()) if request_adapter else (messages, registry.schemas()))
        focused_request = adapted_messages is not messages
        messages = compact_tool_history(messages, max_context_chars)
        while True:
            try:
                request_messages, request_tools = (adapted_messages, adapted_tools) if focused_request else (messages, adapted_tools)
                reply = llm.chat(request_messages, tools=request_tools, tool_choice="required", max_tokens=max_tokens)
                break
            except ContextWindowError:
                if focused_request:
                    raise
                if trace["context_recoveries"] >= 2:
                    raise
                smaller_budget = min(int(max_context_chars * .55), int(_context_size(messages) * .75))
                compacted = compact_tool_history(messages, smaller_budget)
                if _context_size(compacted) >= _context_size(messages):
                    raise
                messages, max_context_chars = compacted, smaller_budget
                trace["context_recoveries"] += 1
                if check_cancel:
                    check_cancel()
            except ToolProtocolError:
                if trace["output_recoveries"] >= 2:
                    raise
                trace["output_recoveries"] += 1
                messages.append({"role": "user", "content": "上一条模型输出格式不完整或不符合工具协议，整条未执行，也未保存其中任何候选事实。请重新决定下一步：仅返回一个已注册工具，省略无关可选字段；数据提交每批最多3项。不要重复执行先前已经成功的工具。"})
                if focused_request:
                    adapted_messages = [*adapted_messages, messages[-1]]
                if check_cancel:
                    check_cancel()
        messages = copy.deepcopy(messages)
        for message in messages:
            if isinstance(message.get("content"), list):
                message["content"] = [part for part in message["content"] if part.get("type") != "image_url"]
        if check_cancel:
            check_cancel()
        if not isinstance(reply, dict):
            raise LlmError("TOOL_RESPONSE_INVALID: 模型未返回有效的工具调用消息。")
        calls = reply.get("tool_calls") or []
        if not isinstance(calls, list):
            raise LlmError("TOOL_RESPONSE_INVALID: 工具调用列表格式不正确。")
        if not 1 <= len(calls) <= 6:
            trace["protocol_errors"] += 1
            messages.append(
                {
                    "role": "user",
                    "content": "协议错误：每次返回1至6个已注册工具调用，按顺序执行；依赖前一步结果时请等待其返回。纯文本不能改变任务或产生正式结果。",
                }
            )
            continue
        identities = set()
        for item in calls:
            if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
                raise LlmError("TOOL_RESPONSE_INVALID: 工具调用结构不完整。")
            fn = item["function"]
            if (not isinstance(item.get("id"), str) or not item["id"].strip()
                    or not isinstance(fn.get("name"), str) or not fn["name"].strip()
                    or not isinstance(fn.get("arguments"), str) or item["id"] in identities):
                raise LlmError("TOOL_RESPONSE_INVALID: 工具调用缺少唯一id、名称或JSON参数。")
            identities.add(item["id"])
            if len(fn["arguments"]) > 32000:
                raise LlmError("TOOL_ARGUMENTS_TOO_LARGE: 工具参数过长。")
        clean = {
            "role": "assistant",
            "content": reply.get("content") if isinstance(reply.get("content"), str) else None,
            "tool_calls": [
                {
                    "id": item["id"],
                    "type": "function",
                    "function": {
                        "name": item["function"]["name"],
                        "arguments": item["function"]["arguments"],
                    },
                } for item in calls
            ],
        }
        # Required by providers that use reasoning with tools. It stays only in
        # this in-memory protocol history, never in tool events or UI messages.
        if (isinstance(reply.get("reasoning_content"), str)
                and getattr(getattr(llm, "config", None), "tool_call_format", "native") != "json_content"):
            clean["reasoning_content"] = reply["reasoning_content"]
        messages.append(clean)
        visuals = []
        for item in calls:
            if check_cancel:
                check_cancel()
            fn = item["function"]
            tool_name = fn["name"]
            trace["tools"].append(tool_name)
            result, error_count = _invoke_tool(registry, call, tool_name, fn["arguments"])
            trace["tool_errors"] += error_count
            if image_resolver and isinstance(result, dict) and result.get("page_image"):
                visuals.append(image_resolver(result))
            messages.append({"role": "tool", "tool_call_id": item["id"], "content": canonical(model_tool_result(result))})
            if isinstance(result, dict) and result.get("_terminal"):
                return {**result, "_agent_trace": trace}
            if isinstance(result, dict) and result.get("ok") is False:
                consecutive_errors += 1
                error = result.get("error", {})
                fields = set(error.get("fields", [])) if error.get("code") == "TOOL_ARGUMENTS_INVALID" else set()
                if fields and previous_argument_error and previous_argument_error[0] == tool_name and fields < previous_argument_error[1]:
                    consecutive_errors = 1
                previous_argument_error = (tool_name, fields) if fields else None
                last_failure = str(result.get("error", {}).get("message") or last_failure)
                try:
                    signature = tool_name + ":" + canonical(json.loads(fn["arguments"]))
                except (TypeError, ValueError):
                    signature = tool_name + ":invalid-json"
                failed_signatures[signature] = failed_signatures.get(signature, 0) + 1
                threshold = 3 if signature.endswith(":invalid-json") else 2
                if failed_signatures[signature] >= threshold or consecutive_errors >= 4:
                    raise LlmError("AGENT_NO_PROGRESS: 工具调用重复失败，已停止自动重试并保留当前进度。 最近一次错误：" + last_failure)
            else:
                consecutive_errors = 0
                previous_argument_error = None
        messages.extend(message for message in visuals if message)
    raise LlmError("AGENT_STEP_LIMIT: 未在限定步骤内提交有效决策，请复核后重试。")


def _invoke_tool(registry, call, tool_name, arguments):
    try:
        return call(tool_name, arguments, lambda: registry.invoke(tool_name, arguments)), 0
    except ValidationError as exc:
        errors = exc.errors(include_input=False)
        json_invalid = any(error.get("type") == "json_invalid" for error in errors)
        fields = [".".join(str(part) for part in error["loc"]) for error in errors]
        details = [f"{field}: {error['msg']}" for field, error in zip(fields, errors)]
        message = ("工具JSON未完整闭合，可能因输出截断。请拆成最多4项候选、共享位置并省略不必要quote后重试；不要重复已保存项。"
                   if json_invalid else "工具参数不符合schema；具体错误：" + "；".join(details[:8]))
        return {"ok": False, "error": {"code": "TOOL_JSON_TRUNCATED" if json_invalid else "TOOL_ARGUMENTS_INVALID",
                "message": message, "fields": fields[:12], "recoverable": True}}, 1
    except ValueError as exc:
        return {"ok": False, "error": {"code": "TOOL_PRECONDITION_FAILED", "message": str(exc).strip()[:1000] or "工具前置条件不满足。",
                "recoverable": True}}, 1
