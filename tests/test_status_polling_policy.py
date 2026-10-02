import json

from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec
from valuationagent.llm.agent import run_tool_loop
from test_agent_recovery import runtime_fixture


def test_status_tools_are_hidden_after_repeated_reads_and_restored_by_new_evidence(tmp_path):
    from valuationagent.schemas.research import FactCandidate

    runtime, _ = runtime_fixture(tmp_path)
    tools = [ToolSpec(name, name, NoArguments, lambda _: {}).schema() for name in ["inspect_requirements", "search_sources"]]
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "{}"}]
    for _ in range(2):
        runtime.call("inspect_requirements", "{}", lambda: {"readiness": "not ready"})
    _, selected = runtime.adapt_request(messages, tools)
    assert [tool["function"]["name"] for tool in selected] == ["search_sources"]
    assert runtime.exhausted_status_tools() == ["inspect_requirements"]
    runtime.session.plan = [{"title": "Only cosmetic plan text changed", "status": "in_progress"}]
    assert runtime.exhausted_status_tools() == ["inspect_requirements"]
    runtime.session.facts.append(FactCandidate(metric="revenue", period="2025", raw_value="100", quote="100", block_id="source:1"))
    _, selected = runtime.adapt_request(messages, tools)
    assert len(selected) == 2
    assert runtime.exhausted_status_tools() == []


def test_tool_only_adapter_is_honored_without_replacing_message_objects():
    available = ToolRegistry([
        ToolSpec("read_status", "read", NoArguments, lambda _: {}),
        ToolSpec("finish", "finish", NoArguments, lambda _: {"_terminal": True}),
    ])

    class Model:
        def chat(self, messages, *, tools, **kwargs):
            assert [tool["function"]["name"] for tool in tools] == ["finish"]
            return {"tool_calls": [{"id": "done", "function": {"name": "finish", "arguments": json.dumps({})}}]}

    def adapt(messages, tools):
        return messages, [tool for tool in tools if tool["function"]["name"] == "finish"]

    result = run_tool_loop(Model(), [{"role": "system", "content": "policy"}, {"role": "user", "content": "task"}],
                           available, lambda name, arguments, invoke: invoke(), request_adapter=adapt)
    assert result["_terminal"]
