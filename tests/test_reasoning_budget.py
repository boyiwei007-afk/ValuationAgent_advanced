import json

import httpx
import pytest
from pydantic import ValidationError

from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec
from valuationagent.llm.agent import run_tool_loop
from valuationagent.llm.client import ModelSessionRegistry, OpenAICompatibleClient, ReasoningLimitError
from valuationagent.schemas.models import ModelConnectionInput


def reasoning_gateway(monkeypatch, *, always_truncated=False, **settings):
    original = httpx.Client
    requests = []

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        truncated = always_truncated or len(requests) == 1
        message = {"content": "", "reasoning_content": "PRIVATE"} if truncated else {
            "content": '[{"name":"finish","parameters":{}}]'}
        return httpx.Response(200, json={"choices": [{"message": message,
            "finish_reason": "length" if truncated else "stop"}]})

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    config = ModelConnectionInput(model="fixture", api_key="EMPTY", thinking="enabled",
        reasoning_protocol="chat_template", tool_call_format="json_content", **settings)
    return OpenAICompatibleClient(config), requests


def execute(model, *, check_cancel=None):
    executed = []
    registry = ToolRegistry([ToolSpec("finish", "finish", NoArguments,
        lambda _: executed.append("finish") or {"_terminal": True})])
    result = run_tool_loop(model, [{"role": "user", "content": "仅使用我给的数据，不联网。"}], registry,
        lambda name, arguments, invoke: invoke(), max_tokens=1800, check_cancel=check_cancel)
    return result, executed


def test_reasoning_budget_expands_before_any_tool_without_changing_mode(monkeypatch):
    model, requests = reasoning_gateway(monkeypatch)
    result, executed = execute(model)
    assert [request["max_tokens"] for request in requests] == [8192, 16384]
    assert executed == ["finish"]
    assert requests[0]["messages"] == requests[1]["messages"]
    assert all(request["chat_template_kwargs"] == {"enable_thinking": True} for request in requests)
    assert all(request["model"] == "fixture" for request in requests)
    assert result["_agent_trace"]["reasoning_recoveries"] == [
        {"from_tokens": 8192, "to_tokens": 16384, "tools_executed_by_truncated_call": False}]
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("initial,ceiling,expected", [(8192, 16384, [8192, 16384]),
    (8192, 8192, [8192]), (1024, 16384, [1800, 3600, 7200])])
def test_reasoning_retry_is_bounded(monkeypatch, initial, ceiling, expected):
    model, requests = reasoning_gateway(monkeypatch, always_truncated=True,
        output_token_budget=initial, max_output_tokens=ceiling)
    with pytest.raises(ReasoningLimitError, match="LLM_REASONING_LIMIT"):
        execute(model)
    assert [request["max_tokens"] for request in requests] == expected


def test_cancel_prevents_second_generation(monkeypatch):
    model, requests = reasoning_gateway(monkeypatch)

    def cancelled():
        if requests:
            raise ValueError("EXECUTION_CANCELLED")

    with pytest.raises(ValueError, match="EXECUTION_CANCELLED"):
        execute(model, check_cancel=cancelled)
    assert len(requests) == 1


def test_expanded_budget_is_scoped_to_one_decision_not_every_later_tool():
    class Model:
        config = ModelConnectionInput(model="fixture", api_key="EMPTY")

        def __init__(self):
            self.budgets = []

        def chat(self, *args, **kwargs):
            self.budgets.append(kwargs["max_tokens"])
            if len(self.budgets) == 1:
                raise ReasoningLimitError("LLM_REASONING_LIMIT: fixture")
            return {"tool_calls": [{"id": str(len(self.budgets)), "function": {
                "name": "step" if len(self.budgets) == 2 else "finish", "arguments": "{}"}}]}

    model = Model()
    registry = ToolRegistry([ToolSpec("step", "step", NoArguments, lambda _: {"saved": True}),
        ToolSpec("finish", "finish", NoArguments, lambda _: {"_terminal": True})])
    result = run_tool_loop(model, [], registry, lambda name, arguments, invoke: invoke())
    assert result["_terminal"]
    assert model.budgets == [8192, 16384, 8192]


def test_budget_settings_roundtrip_without_credentials(monkeypatch):
    monkeypatch.setenv("VALUATION_LLM_MODEL", "fixture")
    monkeypatch.setenv("VALUATION_LLM_API_KEY", "not-public")
    monkeypatch.setenv("VALUATION_LLM_OUTPUT_TOKEN_BUDGET", "12288")
    monkeypatch.setenv("VALUATION_LLM_MAX_OUTPUT_TOKENS", "24576")
    monkeypatch.setenv("VALUATION_LLM_TIMEOUT_SECONDS", "300")
    config = OpenAICompatibleClient.from_environment().config
    public = ModelSessionRegistry().create(config)
    assert public.output_token_budget == 12288 and public.max_output_tokens == 24576
    assert public.timeout_seconds == 300
    assert "not-public" not in public.model_dump_json()
    with pytest.raises(ValidationError, match="初始输出预算"):
        ModelConnectionInput(model="fixture", api_key="EMPTY", output_token_budget=8192, max_output_tokens=4096)


def test_explicit_sampling_is_sent_and_public_without_changing_thinking(monkeypatch):
    model, requests = reasoning_gateway(monkeypatch, temperature=1.0, top_p=.95, presence_penalty=1.5, top_k=20)
    result, executed = execute(model)
    assert executed == ["finish"]
    assert all({key: request[key] for key in ("temperature", "top_p", "presence_penalty", "top_k")} ==
        {"temperature": 1.0, "top_p": .95, "presence_penalty": 1.5, "top_k": 20} for request in requests)
    assert all(request["chat_template_kwargs"] == {"enable_thinking": True} for request in requests)
    public = ModelSessionRegistry().create(model.config)
    assert public.presence_penalty == 1.5 and public.top_p == .95 and public.top_k == 20
    assert "PRIVATE" not in json.dumps(result)


def test_optional_sampling_is_not_sent_without_explicit_configuration(monkeypatch):
    model, requests = reasoning_gateway(monkeypatch)
    execute(model)
    assert all(not {"top_p", "presence_penalty", "top_k"} & request.keys() for request in requests)


def test_model_observer_only_receives_usage_and_protocol_metadata(monkeypatch):
    model, _ = reasoning_gateway(monkeypatch)
    observed = []
    registry = ToolRegistry([ToolSpec("finish", "done", NoArguments, lambda _: {"_terminal": True})])
    run_tool_loop(model, [{"role": "user", "content": "Only my data"}], registry,
        lambda name, arguments, invoke: invoke(), response_observer=lambda metadata: observed.append(dict(metadata)))
    assert len(observed) == 2
    assert all(item["request_message_chars"] > 0 and item["request_schema_chars"] > 0 for item in observed)
    assert observed[0]["finish_reason"] == "length"
    assert observed[-1]["parsed_tool_call_count"] == 1
    assert "PRIVATE" not in json.dumps(observed)


def test_sampling_from_environment_is_not_lost(monkeypatch):
    monkeypatch.setenv("VALUATION_LLM_MODEL", "fixture")
    monkeypatch.setenv("VALUATION_LLM_API_KEY", "not-public")
    monkeypatch.setenv("VALUATION_LLM_TOP_P", "0.95")
    monkeypatch.setenv("VALUATION_LLM_PRESENCE_PENALTY", "1.5")
    monkeypatch.setenv("VALUATION_LLM_TOP_K", "20")
    config = OpenAICompatibleClient.from_environment().config
    assert config.top_p == .95 and config.presence_penalty == 1.5 and config.top_k == 20


def test_file_contracts_are_loaded_only_when_documents_exist(tmp_path):
    from test_input_workspace import fixture
    from valuationagent.schemas.research import DocumentSummary

    _, runtime = fixture(tmp_path)
    names = {"record_inputs", "read_file", "extract_observations", "review_observations", "finish_response"}
    catalog = [{"type": "function", "function": {"name": name, "parameters": {}}} for name in names]
    messages = [{"role": "user", "content": "不要联网，只使用已有输入。"}]
    _, tools = runtime.adapt_request(messages, catalog)
    assert {tool["function"]["name"] for tool in tools} == {"finish_response"}
    runtime.user_input_read = True
    _, tools = runtime.adapt_request(messages, catalog)
    assert {tool["function"]["name"] for tool in tools} == {"record_inputs", "finish_response"}
    runtime.session.documents.append(DocumentSummary(file_id="source", name="fixture.pdf", role="historical", block_count=1))
    _, tools = runtime.adapt_request(messages, catalog)
    assert {tool["function"]["name"] for tool in tools} == names


def test_offline_reading_projects_reference_inputs_without_mutating_registry(tmp_path):
    from test_input_workspace import fixture
    from valuationagent.schemas.inputs import RecordInputs

    _, runtime = fixture(tmp_path)
    runtime.user_input_read = True
    original = RecordInputs.model_json_schema()
    catalog = [{"type": "function", "function": {"name": "record_inputs", "parameters": original}}]
    _, tools = runtime.adapt_request([], catalog)
    projected = tools[0]["function"]["parameters"]
    assert "amount_ref" in projected["$defs"]["UserInputValue"]["required"]
    assert "amount_text" not in projected["$defs"]["UserInputValue"]["properties"]
    assert "provider_values" not in projected["properties"]
    assert "comparables" not in projected["properties"]
    assert "SelectedProviderInput" not in projected["$defs"]
    assert projected["properties"]["user_values"]["maxItems"] == 8
    assert "amount_text" in original["$defs"]["UserInputValue"]["properties"]
    assert original["properties"]["user_values"]["maxItems"] == 64


def test_new_valuation_task_exposes_required_date_in_model_schema(tmp_path):
    from datetime import date
    from test_input_workspace import fixture
    from valuationagent.application.agent_runtime import TaskUpdate

    _, runtime = fixture(tmp_path)
    runtime.session.draft.methods = []
    original = TaskUpdate.model_json_schema()
    catalog = [{"type": "function", "function": {"name": "update_task", "parameters": original}}]
    _, tools = runtime.adapt_request([], catalog)
    schema = tools[0]["function"]["parameters"]["$defs"]["ResearchDraft"]
    assert {"valuation_date", "methods"} <= set(schema["required"])
    assert schema["properties"]["valuation_date"]["type"] == "string"
    assert "required" not in original["$defs"]["ResearchDraft"]
    runtime.session.draft.valuation_date = date(2025, 12, 31)
    runtime.session.draft.methods = ["pe"]
    _, tools = runtime.adapt_request([], catalog)
    assert tools[0]["function"]["parameters"] == original


@pytest.mark.parametrize("codes,passes", [(["INPUT_UNIT", "INPUT_DATE_SOURCE", "INPUT_METRIC", "INPUT_DIMENSION"], True),
    (["INPUT_UNIT", "INPUT_DATE_SOURCE"] * 4, False)])
def test_newly_exposed_preconditions_can_be_repaired_but_cycles_stop(codes, passes):
    from valuationagent.llm.client import LlmError
    from valuationagent.schemas.models import ApiModel

    class Attempt(ApiModel):
        candidate: int

    class Model:
        calls = 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            return {"tool_calls": [{"id": f"call_{self.calls}", "function": {"name": "attempt",
                "arguments": json.dumps({"candidate": self.calls})}}]}

    def attempt(args):
        if args.candidate <= len(codes):
            raise ValueError(codes[args.candidate - 1] + ": fixture validation")
        return {"_terminal": True}

    model = Model()
    registry = ToolRegistry([ToolSpec("attempt", "attempt", Attempt, attempt)])
    if passes:
        result = run_tool_loop(model, [], registry, lambda name, arguments, invoke: invoke(), max_rounds=10)
        assert result["_terminal"] and model.calls == 5
    else:
        with pytest.raises(LlmError, match="AGENT_NO_PROGRESS"):
            run_tool_loop(model, [], registry, lambda name, arguments, invoke: invoke(), max_rounds=10)
        assert model.calls <= 6
