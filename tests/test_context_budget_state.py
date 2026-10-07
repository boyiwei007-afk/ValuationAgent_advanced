import copy
import json

import pytest

from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec, canonical
from valuationagent.llm.agent import compact_tool_history, run_tool_loop
from valuationagent.llm.client import ContextWindowError, LlmError


def test_file_focus_context_recovers_without_repeating_tools_or_losing_current_request():
    from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec

    calls = []
    registry = ToolRegistry([ToolSpec("read_file", "read", NoArguments, lambda _: calls.append("read") or {"text": "document " * 1800}),
        ToolSpec("finish", "done", NoArguments, lambda _: {"_terminal": True})])

    class Model:
        requests = 0
        oversized = None

        def chat(self, messages, **kwargs):
            self.requests += 1
            if self.requests == 3:
                self.oversized = len(canonical(messages))
                raise ContextWindowError("LLM_CONTEXT_LIMIT")
            if self.requests == 4:
                assert len(canonical(messages)) < self.oversized
                assert messages[0]["content"] == "file scope authority"
                assert "keep original date; do not search" in canonical(messages)
                assert {item["function"]["name"] for item in kwargs["tools"]} == {"read_file", "finish"}
            name = "read_file" if self.requests < 3 else "finish"
            return {"tool_calls": [{"id": str(self.requests), "function": {"name": name, "arguments": "{}"}}]}

    def adapt(messages, tools):
        return [{"role": "system", "content": "file scope authority"},
            {"role": "user", "content": canonical({"file_task": {"file_id": "owned"}, "current_request": "keep original date; do not search"})}, *messages[2:]], tools

    model = Model()
    result = run_tool_loop(model, [{"role": "system", "content": "main"}, {"role": "user", "content": "{}"}], registry,
        lambda name, arguments, invoke: invoke(), request_adapter=adapt, max_rounds=3, max_context_chars=90000)
    assert calls == ["read", "read"] and model.requests == 4
    assert result["_agent_trace"]["context_recoveries"] == 1


def test_review_context_limit_never_drops_original_proof_to_force_a_verdict():
    from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec

    packet = {"packets": [{"fact_id": "fact"}], "original_context": {"block": "source " * 500}}
    registry = ToolRegistry([ToolSpec("review_observations", "review", NoArguments, lambda _: pytest.fail("must not execute"))])

    class Model:
        calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            assert json.loads(messages[-1]["content"]) == packet
            raise ContextWindowError("LLM_CONTEXT_LIMIT")

    def adapt(messages, tools):
        return [{"role": "system", "content": "review scope"}, {"role": "user", "content": "{}"},
            {"role": "assistant", "tool_calls": [{"id": "packet", "function": {"name": "prepare_observation_review", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "packet", "content": canonical(packet)}], tools

    model = Model()
    with pytest.raises(ContextWindowError):
        run_tool_loop(model, [], registry, lambda name, arguments, invoke: invoke(), request_adapter=adapt)
    assert model.calls == 1


def workspace_state():
    return {"context": {"summary": "stale narrative " * 100, "recent_turns": [
        {"role": "assistant", "content": "outdated guess " * 100},
        {"role": "user", "content": "仅使用上传文件，不联网、不计算。"}],
        "task_state": {"draft": {"ticker": "600123", "methods": ["pe"]},
            "memory": [{"kind": "constraint", "content": "只准读取，不准对外发送资料"}],
            "data_source_preference": "upload", "information_cutoff_date": "2026-09-30",
            "pending_decision": {"question": "未批准任何计算"},
            "documents": [{"file_id": f"file_{index}", "name": "source " * 500} for index in range(5)],
            "documents_omitted": 0,
            "facts": [{"fact_id": "fact_one", "block_id": "file_0:1", "status": "confirmed", "note": "data " * 200}],
            "facts_omitted": 0}}, "plan": [{"title": "读取文件", "status": "in_progress"}]}


def test_compaction_retrieves_metadata_but_preserves_user_scope_and_original_state():
    state = workspace_state()
    messages = [{"role": "system", "content": "authoritative policy"}, {"role": "user", "content": canonical(state)}]
    before = copy.deepcopy(messages)
    compacted = compact_tool_history(messages, 2300)
    assert len(canonical(compacted)) <= 2300 and messages == before
    projected = json.loads(compacted[1]["content"])
    original, kept = state["context"]["task_state"], projected["context"]["task_state"]
    for key in ["draft", "memory", "data_source_preference", "information_cutoff_date", "pending_decision"]:
        assert kept[key] == original[key]
    assert kept["documents_omitted"] > 0
    assert projected["context"]["recent_turns"] == [state["context"]["recent_turns"][-1]]
    assert projected["context_projection"]["metadata_omitted"]
    assert "outdated guess" not in canonical(compacted)


def test_oversized_user_constraint_fails_closed_instead_of_silently_truncating():
    state = workspace_state()
    state["context"]["task_state"]["memory"][0]["content"] = "user constraint " * 2000
    with pytest.raises(LlmError, match="不能静默截断"):
        compact_tool_history([{"role": "system", "content": "policy"}, {"role": "user", "content": canonical(state)}], 2300)


def test_provider_overflow_recovers_refreshing_state_without_repeating_tool():
    executions = []
    requests = []

    class Model:
        def chat(self, messages, **kwargs):
            requests.append(copy.deepcopy(messages))
            if len(requests) == 2:
                raise ContextWindowError("LLM_CONTEXT_LIMIT")
            if len(requests) >= 3:
                assert len(canonical(messages)) < len(canonical(requests[1]))
                assert "仅使用上传文件" in canonical(messages)
                assert executions == [True]
            name = "read" if len(requests) == 1 else "finish"
            return {"tool_calls": [{"id": str(len(requests)), "function": {"name": name, "arguments": "{}"}}]}

    def read(_):
        executions.append(True)
        return {"text": "original source " * 1000}

    registry = ToolRegistry([ToolSpec("read", "read", NoArguments, read),
        ToolSpec("finish", "finish", NoArguments, lambda _: {"_terminal": True})])
    result = run_tool_loop(Model(), [{"role": "system", "content": "policy"}, {"role": "user", "content": "refresh"}],
        registry, lambda name, args, invoke: invoke(), state_provider=workspace_state, max_context_chars=25000, max_rounds=3)
    assert result["_agent_trace"]["context_recoveries"] == 1
    assert executions == [True]


def test_large_workspace_metadata_cannot_crowd_out_latest_tool_error():
    state = workspace_state()
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": canonical(state)},
        {"role": "assistant", "tool_calls": [{"id": "one", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "one", "content": canonical({"ok": False,
            "error": {"code": "REPEATED_READ", "message": "同一内容已经读取，应修正候选而不是再次查询"},
            "recovery": "very long metadata " * 3000})}]
    compacted = compact_tool_history(messages, 5000)
    assert len(canonical(compacted)) <= 5000
    assert compacted[-1]["role"] == "tool"
    output = json.loads(compacted[-1]["content"])
    assert output["outcome"]["ok"] is False
    assert output["outcome"]["error"]["code"] == "REPEATED_READ"
    assert "仅使用上传文件" in compacted[1]["content"]


def test_omitted_provider_result_retains_typed_candidate_recovery_instead_of_raw_read_loop():
    from valuationagent.llm.agent import omitted_tool_result

    original = {"documents": [{"file_id": "file_abc", "input_candidates": {"items": [
        {"candidate_id": "file_abc@0:revenue", "original_amount": "120", "unit": "元"}]}}]}
    result = json.loads(omitted_tool_result(canonical(original)))
    assert result["candidate_reads"] == [{"tool": "list_input_candidates", "arguments": {"file_id": "file_abc"}}]
    assert "original_amount" not in result


def test_preparation_projection_preserves_action_not_bulk_input_evidence():
    from valuationagent.llm.agent import model_tool_result, omitted_tool_result

    original = {"ready_for_review": True, "status": "ready_for_review", "methods": ["pe"],
        "requested_methods": ["dcf", "pe"], "excluded_methods": {"dcf": "missing cash"},
        "degraded": True, "instruction": "调用calculate_valuation冻结可执行方法；review模式等待批准。",
        "input_records": [{"input_id": f"input_{index}", "source": {"quote": "evidence " * 100}}
                          for index in range(100)], "financials": {"net_income_parent": "100"}}
    wire = model_tool_result(original)
    assert len(canonical(wire)) < 1500
    assert wire["input_records_count"] == 100
    assert "input_records" not in wire
    assert wire["input_records_retrieve_with"] == {"tool": "inspect_inputs", "arguments": {"section": "records"}}
    assert len(original["input_records"]) == 100
    omitted = json.loads(omitted_tool_result(canonical(wire)))
    for key in ("ready_for_review", "methods", "requested_methods", "excluded_methods", "degraded", "instruction"):
        assert omitted["outcome"][key] == original[key]


def test_preparation_projection_keeps_blocked_state_and_does_not_rewrite_valuation_results():
    from valuationagent.llm.agent import model_tool_result, omitted_tool_result

    blocked = {"ready_for_review": False, "status": "building_model", "blocking_reason": "missing shares",
        "instruction": "录入已有股数，不补零。"}
    result = json.loads(omitted_tool_result(canonical(blocked)))["outcome"]
    assert result["ready_for_review"] is False and result["blocking_reason"] == "missing shares"
    request = {"input_records": [{"input_id": "input_original", "value": "100"}]}
    assert model_tool_result(request) == request


def test_compaction_retains_partial_completion_instead_of_erasing_valuation_progress():
    completion = {"requested_methods": ["pe", "ev_ebitda"], "completed_methods": ["pe"],
        "remaining_methods": {"ev_ebitda": "可比B缺少显式租赁负债"}, "all_requested_methods_completed": False}
    state = workspace_state()
    state["valuation"] = {"run_id": "run_partial", "status": "completed_with_warnings",
        "method_completion": completion, "result": {"large_diagnostics": "data " * 10000}}
    before = copy.deepcopy(state)
    compacted = compact_tool_history([{"role": "system", "content": "policy"},
        {"role": "user", "content": canonical(state)}], 2500)
    projected = json.loads(compacted[1]["content"])["valuation"]
    assert projected["method_completion"] == completion
    assert projected["run_id"] == "run_partial" and projected["status"] == "completed_with_warnings"
    assert projected["retrieve_with"] == "read_valuation"
    assert state == before and len(canonical(compacted)) <= 2500


def test_omitted_calculation_keeps_remaining_methods_and_actual_report_receipt():
    from valuationagent.llm.agent import omitted_tool_result

    completion = {"completed_methods": ["pe"], "remaining_methods": {"ev_ebitda": "缺少可比B现金"},
        "all_requested_methods_completed": False}
    delivery = {"status": "saved", "artifact_id": "artifact_actual", "download_url": "/api/workspaces/work/artifacts/artifact_actual"}
    original = {"run_id": "run_partial", "status": "completed_with_warnings", "method_completion": completion,
        "report_delivery": delivery, "result": {"large_diagnostics": "data " * 10000}}
    outcome = json.loads(omitted_tool_result(canonical(original)))["outcome"]
    assert outcome["method_completion"] == completion
    assert outcome["report_delivery"] == delivery
    assert outcome["run_id"] == "run_partial"
