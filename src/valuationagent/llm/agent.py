from __future__ import annotations
import json
import copy
import re
import uuid
from typing import Callable
from pydantic import ValidationError
from valuationagent.core.tools import ToolRegistry, canonical
from valuationagent.llm.client import ContextWindowError, LlmError, ReasoningLimitError, ToolPhaseError, ToolProtocolError


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


def file_focus_context(messages):
    try:
        state = json.loads(messages[1]["content"])
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return isinstance(state, dict) and isinstance(state.get("file_task"), dict)


def workspace_context(messages):
    try:
        state = json.loads(messages[1]["content"])
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return isinstance(state, dict) and isinstance(state.get("context"), dict)


def valuation_checkpoint(value):
    return {**{key: value[key] for key in ("run_id", "status", "run_policy", "input_hash", "error",
        "method_completion", "report_delivery") if key in value},
        "context_omitted": True, "retrieve_with": "read_valuation"}


def research_checkpoint(value):
    checkpoint = {key: copy.deepcopy(value[key]) for key in ("methods",) if key in value}

    def compact_row(row):
        kept = {key: copy.deepcopy(row[key]) for key in (
            "kind", "method", "methods", "status", "tool", "ticker", "period_end", "metric", "metrics",
            "fact_id", "block_id", "required_since", "diagnostics_truncated", "details_omitted") if key in row}
        for key in ("reason", "instruction", "repair_action"):
            if isinstance(row.get(key), str):
                kept[key] = row[key][:320]
                if len(row[key]) > 320:
                    kept["diagnostics_truncated"] = True
        for key in ("arguments", "next_action"):
            if key in row:
                if len(canonical(row[key])) <= 1200:
                    kept[key] = copy.deepcopy(row[key])
                else:
                    kept.setdefault("details_omitted", []).append(key)
        return kept

    if isinstance(value.get("method_readiness"), list):
        checkpoint["method_readiness"] = [compact_row(row) for row in value["method_readiness"]]
    if isinstance(value.get("method_completion"), dict):
        completion = value["method_completion"]
        checkpoint["method_completion"] = {key: copy.deepcopy(completion[key]) for key in (
            "requested_methods", "run_id", "completed_in_run", "current_inputs_match", "completed_methods",
            "all_requested_methods_completed", "diagnostics_truncated") if key in completion}
        if isinstance(completion.get("remaining_methods"), dict):
            checkpoint["method_completion"]["remaining_methods"] = {
                method: reason[:320] if isinstance(reason, str) else copy.deepcopy(reason)
                for method, reason in completion["remaining_methods"].items()}
            if any(isinstance(reason, str) and len(reason) > 320 for reason in completion["remaining_methods"].values()):
                checkpoint["method_completion"]["diagnostics_truncated"] = True
    if isinstance(value.get("next_work"), list):
        checkpoint["next_work"] = [compact_row(row) for row in value["next_work"][:3]]
        omitted = value.get("next_work_omitted", 0) + max(0, len(value["next_work"]) - 3)
        if omitted:
            checkpoint["next_work_omitted"] = omitted
    return {**checkpoint, "context_omitted": True, "retrieve_with": "inspect_requirements"}


def omitted_tool_result(content):
    references = []
    candidate_reads = []
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        value = None

    def collect(item):
        if len(references) >= 40:
            return
        if isinstance(item, dict):
            if item.get("file_id") and isinstance(item.get("input_candidates"), dict):
                candidate_reads.append({"tool": "list_input_candidates", "arguments": {"file_id": item["file_id"]}})
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
        outcome.update({key: value[key] for key in ("run_id", "method_completion", "report_delivery") if key in value})
        if isinstance(value.get("ready_for_review"), bool):
            outcome.update({key: value[key] for key in (
                "ready_for_review", "methods", "requested_methods", "excluded_methods", "degraded",
                "blocking_reason", "instruction") if key in value})
        if isinstance(value.get("error"), dict):
            outcome["error"] = {key: str(value["error"][key])[:1200] for key in ("code", "message") if key in value["error"]}
        if isinstance(value.get("rows"), list):
            outcome["rows"] = [{key: str(row[key])[:800] for key in ("row", "fact_id", "status", "error", "repair") if key in row}
                               for row in value["rows"][:6] if isinstance(row, dict)]
    return canonical({"context_omitted": True, "outcome": outcome, "retrieval_references": references,
        **({"candidate_reads": candidate_reads} if candidate_reads else {}),
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
                "instruction": "仅省略可检索状态与旧助手叙述，保留任务及用户约束。inspect_context读事实/文件，inspect_inputs读输入，inspect_requirements读准备度，read_valuation读结果。省略不表示不存在或为零。"}

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
            dataset = state.get("input_dataset")
            if isinstance(dataset, dict):
                rows = dataset.get("active_records", [])
                while rows and refresh() > budget:
                    rows.pop()
                    dataset["records_omitted"] = dataset.get("records_omitted", 0) + 1
                if refresh() > budget and dataset.get("comparables"):
                    dataset["comparables_omitted"] = len(dataset.pop("comparables"))
                dataset["retrieve_with"] = "inspect_inputs"
            sources = state.get("provider_sources")
            if isinstance(sources, dict):
                while sources.get("sources") and refresh() > budget:
                    sources["sources"].pop(0)
                    sources["omitted"] = sources.get("omitted", 0) + 1
                sources["retrieve_with"] = "list_files"
            acquisition = state.get("input_acquisition")
            if isinstance(acquisition, dict) and refresh() > budget:
                acquisition["issues_omitted"] = acquisition.get("issues_omitted", 0) + len(acquisition.pop("issues", []))
                acquisition["retrieve_with"] = {"tool": "inspect_inputs", "arguments": {"section": "acquisition_issues"}}
            for owner, key, tool in ((state, "research_plan", "inspect_requirements"),
                                    (state, "valuation", "read_valuation"),
                                    (task, "last_issue", "inspect_context"),
                                    (task, "forecast_proposal", "inspect_context")):
                replacement = {"context_omitted": True, "retrieve_with": tool}
                if key == "valuation" and isinstance(owner.get(key), dict):
                    replacement = valuation_checkpoint(owner[key])
                if key == "research_plan" and isinstance(owner.get(key), dict):
                    replacement = research_checkpoint(owner[key])
                if refresh() > budget and key in owner and len(canonical(owner[key])) > len(canonical(replacement)):
                    owner[key] = replacement
            refresh()
    if _context_size(initial) > budget:
        raise LlmError("CONTEXT_BUDGET: 必须保留的系统规则和用户约束超出当前请求空间，不能静默截断；请检查工具目录大小与模型上下文容量。这不是财务数据缺失。")
    return initial


def model_tool_result(value):
    if isinstance(value, list):
        return [model_tool_result(item) for item in value]
    if not isinstance(value, dict):
        return value
    if isinstance(value.get("ready_for_review"), bool) and isinstance(value.get("input_records"), list):
        value = {**{key: item for key, item in value.items() if key != "input_records"},
            "input_records_count": len(value["input_records"]),
            "input_records_retrieve_with": {"tool": "inspect_inputs", "arguments": {"section": "records"}}}
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
    llm, messages: list[dict], registry: ToolRegistry, call: Callable, *, max_rounds=6, max_tokens=900, check_cancel=None, max_context_chars=90000, allow_checkpoint=False, image_resolver=None, state_provider=None, request_adapter=None, budget_instruction=None, response_observer=None, phase_recovery=None, allow_text_answer=False
):
    """Finite, auditable tool loop with bounded self-correction.

    Free text may be offered to finish_response, never executed as a command or valuation. Tool failures are
    returned to the model as safe structured feedback; repeated no-progress
    failures stop and let the application ask the user how to recover.
    """
    messages = list(messages)
    trace = {"rounds": 0, "tools": [], "tool_errors": 0, "protocol_errors": 0, "context_recoveries": 0, "output_recoveries": 0, "reasoning_recoveries": [], "text_answer_candidates": 0}
    config = getattr(llm, "config", None)
    output_ceiling = getattr(config, "max_output_tokens", max_tokens)
    initial_token_budget = min(output_ceiling, max(max_tokens, getattr(config, "output_token_budget", max_tokens)))
    failed_signatures: dict[str, int] = {}
    last_failure = ""
    consecutive_errors = 0
    consecutive_protocol_errors = 0
    previous_argument_error = None
    encountered_preconditions = set()
    focus_budget = max_context_chars
    for round_index in range(1, max_rounds + 1):
        request_token_budget = initial_token_budget
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
            messages.append({"role": "user", "content": budget_instruction or message})
        adapted_messages, adapted_tools = (request_adapter(messages, registry.schemas()) if request_adapter else (messages, registry.schemas()))
        focused_request = adapted_messages is not messages
        schema_size = len(canonical(adapted_tools))
        message_budget = max_context_chars - schema_size
        if message_budget < 1024:
            raise LlmError("TOOL_CONTEXT_LIMIT: 工具定义占满请求预算；须按需加载工具，不能靠截断用户指令或扩大输出解决。")
        compressible = focused_request and (file_focus_context(adapted_messages) or workspace_context(adapted_messages))
        if compressible:
            adapted_messages = compact_tool_history(adapted_messages, min(focus_budget, message_budget))
        messages = compact_tool_history(messages, message_budget)
        returned_to_parent = False
        while True:
            try:
                request_messages, request_tools = (adapted_messages, adapted_tools) if focused_request else (messages, adapted_tools)
                accepts_text = (allow_text_answer and getattr(config, "tool_call_format", "native") in {"native", "qwen3_coder"}
                    and any(tool["function"]["name"] == "finish_response" for tool in request_tools))
                try:
                    reply = llm.chat(request_messages, tools=request_tools, tool_choice="auto" if accepts_text else "required", max_tokens=request_token_budget)
                finally:
                    if response_observer is not None:
                        response_observer(getattr(llm, "last_response_metadata", {}))
                consecutive_protocol_errors = 0
                break
            except ReasoningLimitError:
                if request_token_budget >= output_ceiling or len(trace["reasoning_recoveries"]) >= 2:
                    raise
                expanded = min(output_ceiling, request_token_budget * 2)
                trace["reasoning_recoveries"].append({"from_tokens": request_token_budget, "to_tokens": expanded,
                    "tools_executed_by_truncated_call": False})
                request_token_budget = expanded
                if check_cancel:
                    check_cancel()
            except ContextWindowError:
                if focused_request and not compressible:
                    raise
                if trace["context_recoveries"] >= 2:
                    raise
                current = adapted_messages if focused_request else messages
                current_budget = min(focus_budget, message_budget) if focused_request else message_budget
                smaller_budget = min(int(current_budget * .55), int(_context_size(current) * .75))
                compacted = compact_tool_history(current, smaller_budget)
                if _context_size(compacted) >= _context_size(current):
                    raise
                if focused_request:
                    adapted_messages, focus_budget = compacted, smaller_budget
                else:
                    messages, max_context_chars = compacted, smaller_budget
                trace["context_recoveries"] += 1
                if check_cancel:
                    check_cancel()
            except ToolProtocolError as exc:
                if isinstance(exc, ToolPhaseError) and phase_recovery is not None:
                    if check_cancel:
                        check_cancel()
                    transition = phase_recovery()
                    if transition:
                        messages.append({"role": "user", "content": canonical(transition)})
                        trace.setdefault("phase_returns", []).append(transition.get("file_id"))
                        returned_to_parent = True
                        break
                if consecutive_protocol_errors >= 2:
                    raise
                consecutive_protocol_errors += 1
                trace["output_recoveries"] += 1
                protocol = ("保持Qwen原生标签：<tool_call><function=工具名><parameter=参数名>参数值</parameter></function></tool_call>；不要改成JSON动作包装或代码围栏，object/array参数内部仍用严格JSON。"
                    if getattr(config, "tool_call_format", "native") == "qwen3_coder" else "")
                offered = {tool["function"]["name"] for tool in request_tools}
                recovery = ("目录中的工具未加载时先load_tools。" if "load_tools" in offered else
                    "当前仅为文件阶段；需要其他业务工具时先end_file_task返回主循环。" if "end_file_task" in offered else
                    "当前为定向阶段，只调用这些工具：" + ",".join(sorted(offered)) + "。")
                messages.append({"role": "user", "content": "上一条模型输出格式不完整或不符合工具协议，整条未执行，也未保存其中任何候选事实。具体原因：" + str(exc)[:800] + "。请重新决定下一步：仅返回一个当前可用工具，省略无关可选字段。本次修复可缩小参数；后续正常提交仍按工具批量上限，不必逐项调用。" + recovery + "不要重复执行先前已经成功的工具。" + protocol})
                if focused_request:
                    adapted_messages = [*adapted_messages, messages[-1]]
                if check_cancel:
                    check_cancel()
        if returned_to_parent:
            continue
        messages = copy.deepcopy(messages)
        for message in messages:
            if isinstance(message.get("content"), list):
                message["content"] = [part for part in message["content"] if part.get("type") != "image_url"]
        if check_cancel:
            check_cancel()
        if not isinstance(reply, dict):
            raise LlmError("TOOL_RESPONSE_INVALID: 模型未返回有效的工具调用消息。")
        calls = reply.get("tool_calls") or []
        if not calls and accepts_text and isinstance(reply.get("content"), str) and reply["content"].strip():
            from valuationagent.llm.qwen_tools import DELIMITER

            if not DELIMITER.search(reply["content"]):
                calls = [{"id": "answer_" + uuid.uuid4().hex, "type": "function", "function": {
                    "name": "finish_response", "arguments": canonical({"answer": reply["content"]})}}]
                reply = {"role": "assistant", "content": None, "tool_calls": calls}
                trace["text_answer_candidates"] += 1
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
                condition = error.get("precondition_code")
                if condition and (tool_name, condition) not in encountered_preconditions:
                    encountered_preconditions.add((tool_name, condition))
                    consecutive_errors = 1
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
                encountered_preconditions.clear()
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
        message = str(exc).strip()
        condition = re.match(r"^([A-Z][A-Z0-9_]{2,79}):", message)
        return {"ok": False, "error": {"code": "TOOL_PRECONDITION_FAILED",
                **({"precondition_code": condition[1]} if condition else {}), "message": message[:1000] or "工具前置条件不满足。",
                "recoverable": True}}, 1
