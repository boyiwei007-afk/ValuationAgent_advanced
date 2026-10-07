import copy
import json

import pytest

from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec, canonical
from valuationagent.llm.agent import run_tool_loop
from valuationagent.llm.client import LlmError
from valuationagent.llm.tool_catalog import LoadTools, ToolCatalog, compact_schema


def catalog_tools():
    return [{"type": "function", "function": {"name": name, "description": "description",
        "parameters": {"type": "object", "properties": {"payload": {"type": "string", "description": "d" * 10000}}}}}
        for name in ["load_tools", "finish_response", "extract", "search", "calculate"]]


def test_catalog_loads_only_selected_definitions_and_never_executes_business():
    catalog = ToolCatalog()
    tools = catalog_tools()
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "request"}]
    before = copy.deepcopy((messages, tools))
    projected, selected = catalog.adapt(messages, tools)
    assert {tool["function"]["name"] for tool in selected} == {"load_tools", "finish_response"}
    assert all(name in projected[0]["content"] for name in ["extract", "search", "calculate"])
    assert len(canonical(selected)) < len(canonical(tools)) / 2
    assert (messages, tools) == before
    assert catalog.load(LoadTools(names=["extract"]))["executed"] is False
    _, selected = catalog.adapt(messages, tools)
    assert {tool["function"]["name"] for tool in selected} == {"load_tools", "finish_response", "extract"}
    catalog.load(LoadTools(names=["calculate"]))
    _, selected = catalog.adapt(messages, tools)
    assert "extract" not in {tool["function"]["name"] for tool in selected}


def test_catalog_never_restores_removed_permission_or_mutates_property_named_title():
    catalog = ToolCatalog()
    tools = catalog_tools()
    catalog.adapt([], tools)
    catalog.load(LoadTools(names=["search"]))
    _, selected = catalog.adapt([], [tool for tool in tools if tool["function"]["name"] != "search"])
    assert "search" not in {tool["function"]["name"] for tool in selected}
    with pytest.raises(ValueError, match="TOOL_UNAVAILABLE"):
        catalog.load(LoadTools(names=["search"]))
    schema = {"title": "Model", "properties": {"title": {"title": "Field", "type": "string"}}}
    assert compact_schema(schema) == {"properties": {"title": {"type": "string"}}}


def test_catalog_deferred_tool_explains_prerequisite_without_granting_access():
    catalog = ToolCatalog()
    deferred = {"record_user_inputs": "先read_user_input取得原文行号，再录入。"}
    messages, selected = catalog.adapt([], catalog_tools(), deferred=deferred)
    assert "先read_user_input" in messages[0]["content"]
    assert "record_user_inputs" not in {tool["function"]["name"] for tool in selected}
    with pytest.raises(ValueError, match="TOOL_PREREQUISITE.*read_user_input"):
        catalog.load(LoadTools(names=["record_user_inputs"]))
    assert catalog.loaded == []
    catalog.adapt([], catalog_tools())
    with pytest.raises(ValueError, match="TOOL_UNAVAILABLE"):
        catalog.load(LoadTools(names=["record_user_inputs"]))


def test_tool_schemas_count_toward_request_budget_before_model_call():
    registry = ToolRegistry([ToolSpec("large", "definition" * 1000, NoArguments, lambda _: pytest.fail("no execution"))])

    class Model:
        def chat(self, *args, **kwargs):
            pytest.fail("oversized schema must not reach provider")

    with pytest.raises(LlmError, match="TOOL_CONTEXT_LIMIT"):
        run_tool_loop(Model(), [{"role": "user", "content": "a company"}], registry,
            lambda name, args, invoke: invoke(), max_context_chars=4000)


def test_protocol_recovery_respects_the_current_stage_tool_set():
    from valuationagent.llm.client import ToolProtocolError

    class Model:
        calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ToolProtocolError("TOOL_QWEN_INVALID: tool is not exposed in this stage")
            assert "end_file_task" in messages[-1]["content"]
            assert "load_tools" not in messages[-1]["content"]
            return {"tool_calls": [{"id": "end", "function": {"name": "end_file_task", "arguments": "{}"}}]}

    registry = ToolRegistry([ToolSpec("end_file_task", "exit focus", NoArguments, lambda _: {"_terminal": True})])
    result = run_tool_loop(Model(), [], registry, lambda name, args, invoke: invoke())
    assert result["_agent_trace"]["output_recoveries"] == 1


def test_lazy_workspace_projection_can_recover_server_context_limit():
    from valuationagent.llm.client import ContextWindowError
    from test_context_budget_state import workspace_state

    catalog = ToolCatalog()
    registry = ToolRegistry([
        ToolSpec("load_tools", "load", LoadTools, catalog.load),
        ToolSpec("finish_response", "finish", NoArguments, lambda _: {"_terminal": True}),
    ])
    sizes = []

    class Model:
        def chat(self, messages, **kwargs):
            sizes.append(len(canonical(messages)))
            if len(sizes) == 1:
                raise ContextWindowError("LLM_CONTEXT_LIMIT")
            assert "仅使用上传文件" in canonical(messages)
            return {"tool_calls": [{"id": "done", "function": {"name": "finish_response", "arguments": "{}"}}]}

    result = run_tool_loop(Model(), [{"role": "system", "content": "policy"},
        {"role": "user", "content": canonical(workspace_state())}], registry,
        lambda name, args, invoke: invoke(), request_adapter=catalog.adapt)
    assert sizes[1] < sizes[0]
    assert result["_agent_trace"]["context_recoveries"] == 1


def test_separated_protocol_errors_do_not_spend_a_lifetime_failure_quota():
    from valuationagent.llm.client import ToolProtocolError

    class Model:
        calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls in {1, 3, 5}:
                raise ToolProtocolError("TOOL_QWEN_INVALID: load next tool")
            name = "finish_response" if self.calls == 6 else "work"
            return {"tool_calls": [{"id": str(self.calls), "function": {"name": name, "arguments": "{}"}}]}

    executions = []
    registry = ToolRegistry([ToolSpec("work", "work", NoArguments, lambda _: executions.append(True) or {"saved_count": 1}),
        ToolSpec("finish_response", "done", NoArguments, lambda _: {"_terminal": True})])
    result = run_tool_loop(Model(), [{"role": "user", "content": "task"}], registry,
        lambda name, args, invoke: invoke(), max_rounds=4)
    assert executions == [True, True]
    assert result["_agent_trace"]["output_recoveries"] == 3
