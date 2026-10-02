import copy
import json

import pytest

from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec, canonical
from valuationagent.llm.agent import compact_tool_history, run_tool_loop
from valuationagent.llm.client import ContextWindowError, LlmError


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
