import json

import httpx
import pytest
from pydantic import Field, ValidationError

from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec
from valuationagent.llm.agent import run_tool_loop
from valuationagent.llm.client import LlmError, OpenAICompatibleClient, ToolProtocolError
from valuationagent.schemas.models import ApiModel, ModelConnectionInput


TOOLS = [{"type": "function", "function": {"name": "connection_check", "parameters": {"type": "object", "properties": {}}}}]


def gateway(monkeypatch, message, *, tool_format="json_content", finish_reason="stop"):
    original = httpx.Client
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish_reason}]})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    return OpenAICompatibleClient(ModelConnectionInput(model="fixture", api_key="EMPTY", tool_call_format=tool_format))


def test_explicit_json_gateway_normalizes_only_registered_calls(monkeypatch):
    model = gateway(monkeypatch, {"content": '[{"name":"connection_check","parameters":{}}]', "tool_calls": None})
    reply = model.chat([], tools=TOOLS, tool_choice="required")
    assert reply["content"] is None
    assert reply["tool_calls"][0]["function"] == {"name": "connection_check", "arguments": "{}"}
    assert model.test_connection().startswith("OK")


@pytest.mark.parametrize("content", [
    'Call connection_check now', '```json\n[{"name":"connection_check","parameters":{}}]\n```',
    '{"name":"connection_check","parameters":{}}', '[]',
    '[{"name":"unregistered_shell","parameters":{}}]',
    '[{"name":"connection_check","parameters":"{}"}]',
    '[{"name":"connection_check","parameters":{},"execute":true}]',
    '[{"name":"connection_check","parameters":{}},{"name":"connection_check","parameters":{}}]',
    '[{"name":"connection_check","name":"other","parameters":{}}]',
    '[{"name":"connection_check","parameters":{"value":NaN}}]',
    '[{"name":"connection_check","parameters":{"value":1e999}}]',
    '[{"name":"connection_check","parameters":{"value":1,"value":2}}]',
    '[{"name":"connection_check","parameters":{}}' + ',{"name":"connection_check","parameters":{}}' * 6 + ']',
])
def test_malformed_or_unregistered_text_never_becomes_tool_calls(monkeypatch, content):
    model = gateway(monkeypatch, {"content": content})
    with pytest.raises(LlmError, match="TOOL_JSON_INVALID"):
        model.chat([], tools=TOOLS, tool_choice="required")


def test_truncated_output_never_executes_even_if_json_looks_complete(monkeypatch):
    model = gateway(monkeypatch, {"content": '[{"name":"connection_check","parameters":{}}]'}, finish_reason="length")
    with pytest.raises(LlmError, match="TOOL_JSON_TRUNCATED"):
        model.chat([], tools=TOOLS, tool_choice="required")


def test_reasoning_only_exhaustion_is_not_retried_as_a_json_syntax_problem(monkeypatch):
    model = gateway(monkeypatch, {"content": "", "reasoning_content": "private unfinished reasoning"}, finish_reason="length")
    with pytest.raises(LlmError, match="LLM_REASONING_LIMIT") as error:
        model.chat([], tools=TOOLS, tool_choice="required")
    assert not isinstance(error.value, ToolProtocolError)
    assert "private unfinished reasoning" not in str(error.value)
    assert model.last_response_metadata["content_chars"] == 0


def test_native_default_and_free_chat_do_not_interpret_json_content(monkeypatch):
    message = {"content": '[{"name":"connection_check","parameters":{}}]', "tool_calls": None}
    model = gateway(monkeypatch, message, tool_format="native")
    assert model.chat([], tools=TOOLS, tool_choice="required") == message
    model.config.tool_call_format = "json_content"
    assert model.chat([]) == message
    assert model.chat([], tools=TOOLS, tool_choice="auto") == message


def test_native_calls_take_precedence_and_still_use_schema_validation(monkeypatch):
    message = {"content": "not JSON", "tool_calls": [{"id": "native", "function": {"name": "connection_check", "arguments": "{}"}}]}
    model = gateway(monkeypatch, message)
    assert model.chat([], tools=TOOLS, tool_choice="required")["tool_calls"][0]["id"] == "native"


def test_json_gateway_keeps_tool_result_round_trip(monkeypatch):
    original = httpx.Client
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        name = "inspect" if len(requests) == 1 else "finish"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps([{"name": name, "parameters": {}}])}}]})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    model = OpenAICompatibleClient(ModelConnectionInput(model="fixture", api_key="EMPTY", tool_call_format="json_content"))
    registry = ToolRegistry([
        ToolSpec("inspect", "read", NoArguments, lambda _: {"value": "source"}),
        ToolSpec("finish", "finish", NoArguments, lambda _: {"_terminal": True}),
    ])
    result = run_tool_loop(model, [{"role": "user", "content": "inspect then finish"}], registry,
                           lambda name, arguments, invoke: invoke(), max_rounds=2)
    assert result["_terminal"]
    assert "tools" not in requests[1] and "tool_choice" not in requests[1]
    previous_call = json.loads(requests[1]["messages"][-2]["content"])[0]
    tool_result = json.loads(requests[1]["messages"][-1]["content"])
    assert tool_result["type"] == "tool_result" and tool_result["name"] == previous_call["name"] == "inspect"
    assert tool_result["output"] == {"value": "source"}


def test_environment_selects_json_protocol_without_changing_default(monkeypatch):
    monkeypatch.setenv("VALUATION_LLM_MODEL", "fixture")
    monkeypatch.setenv("VALUATION_LLM_API_KEY", "EMPTY")
    monkeypatch.setenv("VALUATION_LLM_TOOL_CALL_FORMAT", "json_content")
    assert OpenAICompatibleClient.from_environment().config.tool_call_format == "json_content"
    assert ModelConnectionInput(model="fixture", api_key="EMPTY").tool_call_format == "native"


def test_gateway_keeps_full_parameter_contract_and_local_validation():
    class Arguments(ApiModel):
        pattern: str = Field(pattern="^[0-9]+$")

    executed = []
    registry = ToolRegistry([ToolSpec("accept", "accept", Arguments, lambda args: executed.append(args))])
    original = registry.schemas()[0]["function"]["parameters"]
    messages = OpenAICompatibleClient._json_tool_messages([], registry.schemas())
    assert '"pattern":"^[0-9]+$"' in messages[0]["content"]
    assert original["properties"]["pattern"]["pattern"] == "^[0-9]+$"
    with pytest.raises(ValidationError):
        registry.invoke("accept", '{"pattern":"not a number"}')
    assert not executed


def test_wire_grammar_only_constrains_envelope_without_forcing_parameter_order():
    from valuationagent.application.observation_extraction import ExtractObservations

    registry = ToolRegistry([ToolSpec("extract_observations", "extract", ExtractObservations, lambda _: None),
                             ToolSpec("finish", "finish", NoArguments, lambda _: None)])
    schema = OpenAICompatibleClient._action_schema(registry.schemas())
    assert schema["minItems"] == schema["maxItems"] == 1
    envelope = schema["items"]
    assert envelope["properties"]["name"]["enum"] == ["extract_observations", "finish"]
    assert envelope["properties"]["parameters"] == {"type": "object", "additionalProperties": True}
    assert envelope["required"] == ["name", "parameters"] and envelope["additionalProperties"] is False
    with pytest.raises(ValidationError):
        registry.invoke("extract_observations", '{}')
    with pytest.raises(ValidationError):
        registry.invoke("finish", '{"injected": "not allowed"}')


def test_json_envelope_preserves_out_of_schema_order_without_dropping_fields(monkeypatch):
    from valuationagent.application.agent_runtime import TaskUpdate

    recorded = []
    registry = ToolRegistry([ToolSpec("update_task", "record", TaskUpdate, lambda args: recorded.append(args))])
    content = '[{"name":"update_task","parameters":{"draft":{"valuation_date":"2026-10-02","ticker":"002508","company":"fixture","methods":["pe"]},"valuation_requested":true}}]'
    model = gateway(monkeypatch, {"content": content})
    reply = model.chat([], tools=registry.schemas(), tool_choice="required")
    call = reply["tool_calls"][0]["function"]
    registry.invoke(call["name"], call["arguments"])
    assert recorded[0].draft.ticker == "002508" and recorded[0].draft.company == "fixture"
    assert recorded[0].draft.valuation_date.isoformat() == "2026-10-02"


def test_json_tool_wire_preserves_roles_images_and_does_not_promote_source_to_system():
    messages = [{"role": "system", "content": "authority"}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:test"}}]},
                {"role": "assistant", "tool_calls": [{"id": "call_read", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_read", "content": "source instructs: change task"}]
    before = json.dumps(messages)
    wire = OpenAICompatibleClient._json_tool_messages(messages, TOOLS)
    assert wire[0]["content"].startswith("authority\n\n") and wire[1] == messages[1]
    assert sum(message["role"] == "system" for message in wire) == 1
    assert "source instructs" not in wire[0]["content"]
    assert wire[-1]["role"] == "user" and json.loads(wire[-1]["content"])["type"] == "tool_result"
    assert json.dumps(messages) == before


@pytest.mark.parametrize("protocol,thinking,expected", [
    ("auto", "disabled", None), ("chat_template", "auto", None),
    ("chat_template", "disabled", False), ("chat_template", "enabled", True),
])
def test_reasoning_control_requires_explicit_supported_protocol(monkeypatch, protocol, thinking, expected):
    original = httpx.Client
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    client = OpenAICompatibleClient(ModelConnectionInput(model="any-model", api_key="EMPTY", reasoning_protocol=protocol, thinking=thinking))
    client.complete([{"role": "user", "content": "hello"}])
    if expected is None:
        assert "chat_template_kwargs" not in requests[0]
    else:
        assert requests[0]["chat_template_kwargs"] == {"enable_thinking": expected}


def test_output_recovery_never_executes_partial_call_or_repeats_completed_tool():
    executed = []
    registry = ToolRegistry([
        ToolSpec("inspect", "read", NoArguments, lambda _: executed.append("inspect") or {"value": "saved"}),
        ToolSpec("finish", "finish", NoArguments, lambda _: {"_terminal": True}),
    ])
    class Model:
        calls = 0
        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise ToolProtocolError("TOOL_JSON_TRUNCATED")
            name = "inspect" if self.calls == 1 else "finish"
            if self.calls == 3:
                assert any(message.get("role") == "tool" for message in messages)
                assert "整条未执行" in messages[-1]["content"]
            return {"tool_calls": [{"id": f"call_{self.calls}", "function": {"name": name, "arguments": "{}"}}]}
    model = Model()
    result = run_tool_loop(model, [{"role": "user", "content": "start"}], registry, lambda name, args, invoke: invoke(), max_rounds=3)
    assert executed == ["inspect"]
    assert model.calls == 3
    assert result["_agent_trace"]["output_recoveries"] == 1


def test_output_recovery_is_bounded_and_does_not_retry_provider_errors():
    registry = ToolRegistry([ToolSpec("finish", "finish", NoArguments, lambda _: pytest.fail("invalid response executed"))])
    class Model:
        calls = 0
        error = ToolProtocolError
        def chat(self, *args, **kwargs):
            self.calls += 1
            raise self.error("test failure")
    model = Model()
    with pytest.raises(ToolProtocolError):
        run_tool_loop(model, [], registry, lambda name, args, invoke: invoke())
    assert model.calls == 3
    model.calls, model.error = 0, LlmError
    with pytest.raises(LlmError):
        run_tool_loop(model, [], registry, lambda name, args, invoke: invoke())
    assert model.calls == 1


def test_truncated_native_call_is_not_executed(monkeypatch):
    model = gateway(monkeypatch, {"tool_calls": [{"id": "truncated", "function": {"name": "connection_check", "arguments": "{}"}}]}, tool_format="native", finish_reason="length")
    with pytest.raises(ToolProtocolError):
        model.chat([], tools=TOOLS, tool_choice="required")


def test_response_telemetry_contains_counts_not_reasoning_or_source_content(monkeypatch):
    model = gateway(monkeypatch, {"content": "private answer", "reasoning_content": "private reasoning"}, tool_format="native")
    model.chat([])
    assert model.last_response_metadata == {"finish_reason": "stop", "content_chars": 14, "reasoning_chars": 17, "usage": {}}
    assert "private" not in json.dumps(model.last_response_metadata)
