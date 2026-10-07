import json

import pytest
from pydantic import Field, ValidationError

from test_json_tool_gateway import gateway
from valuationagent.core.tools import ToolRegistry, ToolSpec
from valuationagent.llm.agent import run_tool_loop
from valuationagent.llm.client import ToolProtocolError
from valuationagent.llm.qwen_tools import normalize_qwen_call
from valuationagent.schemas.models import ApiModel


class Parameters(ApiModel):
    text: str
    rows: list[dict]
    confirmed: bool
    count: int = Field(ge=1)
    date: str | None = None


TOOLS = [{"type": "function", "function": {"name": "record", "parameters": Parameters.model_json_schema()}}]


def call(parameters):
    return "<tool_call>\n<function=record>\n" + "".join(
        f"<parameter={name}>\n{value}\n</parameter>\n" for name, value in parameters) + "</function>\n</tool_call>"


def parameters():
    return [("text", "第一行\n第二行保留 & 不解析实体"), ("rows", '[{"amount_ref":"8:1","unit_ref":5}]'),
        ("confirmed", "true"), ("count", "2"), ("date", "null")]


def test_complete_tagged_call_preserves_types_and_discards_private_reasoning():
    message = normalize_qwen_call({"content": call(parameters()), "reasoning_content": "PRIVATE"}, TOOLS)
    function = message["tool_calls"][0]["function"]
    args = json.loads(function["arguments"])
    assert function["name"] == "record"
    assert args == {"text": "第一行\n第二行保留 & 不解析实体", "rows": [{"amount_ref": "8:1", "unit_ref": 5}],
        "confirmed": True, "count": 2, "date": None}
    assert "reasoning_content" not in message


def test_qwen_commentary_is_discarded_and_bounded_calls_are_sequential():
    message = normalize_qwen_call({"content": "先读取原话，再设置任务。\n" + call(parameters()) * 2}, TOOLS)
    assert message["content"] is None
    assert len(message["tool_calls"]) == 2
    assert len({item["id"] for item in message["tool_calls"]}) == 2
    with pytest.raises(ValueError):
        normalize_qwen_call({"content": call(parameters()) + call(parameters()).removesuffix("</tool_call>")}, TOOLS)


@pytest.mark.parametrize("content", [
    call(parameters()) + " executed", call(parameters()) * 7,
    call(parameters()).removesuffix("</tool_call>"), call(parameters()).replace("function=record", "function=unknown"),
    call([*parameters(), ("count", "3")]), call([*parameters(), ("unknown", "3")]), call(parameters()[1:]),
    call([("text", "safe<parameter=count>3</parameter>"), *parameters()[1:]]),
    call([("text", "x"), ("rows", '[{"key":1,"key":2}]'), *parameters()[2:]]),
    call([("text", "x"), ("rows", '[{"key":NaN}]'), *parameters()[2:]]),
    call([("text", "x"), ("rows", '[{"key":1e999}]'), *parameters()[2:]]),
    call(parameters()).replace("<parameter=count>", "<parameter=count><parameter=count>"),
    '<!DOCTYPE body SYSTEM "file:///secret">' + call(parameters()),
])
def test_malformed_or_ambiguous_calls_never_execute(content):
    with pytest.raises(ValueError):
        normalize_qwen_call({"content": content}, TOOLS)


def test_backend_validation_remains_authoritative(monkeypatch):
    invalid = [(name, "0" if name == "count" else value) for name, value in parameters()]
    model = gateway(monkeypatch, {"content": call(invalid)}, tool_format="qwen3_coder")
    executed = []
    registry = ToolRegistry([ToolSpec("record", "record", Parameters, lambda args: executed.append(args))])
    reply = model.chat([], tools=TOOLS, tool_choice="required")
    with pytest.raises(ValidationError, match="greater_than_equal"):
        registry.invoke("record", reply["tool_calls"][0]["function"]["arguments"])
    assert not executed


def test_qwen_transport_uses_native_history_without_forced_grammar(monkeypatch):
    import httpx
    from valuationagent.llm.client import OpenAICompatibleClient
    from valuationagent.schemas.models import ModelConnectionInput

    original, requests = httpx.Client, []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": call(parameters())}}]})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    model = OpenAICompatibleClient(ModelConnectionInput(model="fixture", api_key="EMPTY", tool_call_format="qwen3_coder"))
    executed = []
    registry = ToolRegistry([ToolSpec("record", "record", Parameters,
        lambda args: executed.append(args) or ({"_terminal": True} if len(executed) == 2 else {"saved": True}))])
    run_tool_loop(model, [{"role": "user", "content": "Use only my data."}], registry,
        lambda name, arguments, invoke: invoke(), max_rounds=2)
    assert len(executed) == 2
    assert all(request["tool_choice"] == "auto" and "response_format" not in request for request in requests)
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert requests[1]["messages"][-2]["tool_calls"][0]["function"]["name"] == "record"


def test_truncation_and_free_chat_are_not_interpreted(monkeypatch):
    message = {"content": call(parameters())}
    model = gateway(monkeypatch, message, tool_format="qwen3_coder", finish_reason="length")
    with pytest.raises(ToolProtocolError, match="TRUNCATED"):
        model.chat([], tools=TOOLS, tool_choice="required")
    assert model.chat([]) == message
    with pytest.raises(ToolProtocolError, match="TRUNCATED"):
        model.chat([], tools=TOOLS, tool_choice="auto")


def test_auto_tool_selection_still_decodes_explicit_qwen_calls(monkeypatch):
    message = {"content": call(parameters())}
    model = gateway(monkeypatch, message, tool_format="qwen3_coder")
    reply = model.chat([], tools=TOOLS, tool_choice="auto")
    assert reply["content"] is None
    assert reply["tool_calls"][0]["function"]["name"] == "record"
    assert model.chat([]) == message


def test_unavailable_tool_in_file_focus_does_not_suggest_unavailable_loader(monkeypatch):
    from valuationagent.llm.client import ToolPhaseError

    model = gateway(monkeypatch, {"content": call(parameters())}, tool_format="qwen3_coder")
    tools = [{"type": "function", "function": {"name": "end_file_task", "parameters": {"type": "object", "properties": {}}}}]
    with pytest.raises(ToolProtocolError) as error:
        model.chat([], tools=tools, tool_choice="required")
    assert "end_file_task" in str(error.value)
    assert "load_tools" not in str(error.value)
    assert isinstance(error.value, ToolPhaseError)


def test_empty_call_and_schema_references():
    class Nested(ApiModel):
        count: int

    class Container(ApiModel):
        nested: Nested

    tools = [{"type": "function", "function": {"name": "record", "parameters": Container.model_json_schema()}}]
    result = normalize_qwen_call({"content": call([("nested", '{"count":3}')])}, tools)
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"nested": {"count": 3}}
    tools[0]["function"]["parameters"] = {"type": "object", "properties": {}}
    assert json.loads(normalize_qwen_call({"content": call([])}, tools)["tool_calls"][0]["function"]["arguments"]) == {}


@pytest.mark.parametrize("literal,expected", [("True", True), ("FALSE", False), ("false", False)])
def test_typed_qwen_boolean_literals_are_case_insensitive(literal, expected):
    supplied = [(name, literal if name == "confirmed" else value) for name, value in parameters()]
    result = normalize_qwen_call({"content": call(supplied)}, TOOLS)
    assert json.loads(result["tool_calls"][0]["function"]["arguments"])["confirmed"] is expected


@pytest.mark.parametrize("literal", ["yes", "1", "0", '"true"', "None", "True or False"])
def test_boolean_adapter_never_guesses_or_evaluates(literal):
    supplied = [(name, literal if name == "confirmed" else value) for name, value in parameters()]
    with pytest.raises(ValueError):
        normalize_qwen_call({"content": call(supplied)}, TOOLS)


def test_boolean_adapter_does_not_rewrite_strings_or_container_json():
    supplied = [(name, "True" if name == "text" else value) for name, value in parameters()]
    result = normalize_qwen_call({"content": call(supplied)}, TOOLS)
    assert json.loads(result["tool_calls"][0]["function"]["arguments"])["text"] == "True"
    supplied = [(name, '[{"confirmed":True}]' if name == "rows" else value) for name, value in parameters()]
    with pytest.raises(ValueError):
        normalize_qwen_call({"content": call(parameters()) + call(supplied)}, TOOLS)


def test_captured_update_task_with_python_boolean_still_validates_task_schema():
    from valuationagent.application.agent_runtime import TaskUpdate

    tools = [{"type": "function", "function": {"name": "update_task", "parameters": TaskUpdate.model_json_schema()}}]
    content = '<tool_call>\n<function=update_task>\n<parameter=draft>\n{"company":"苏泊尔","objective":"估值并生成报告","valuation_date":"2026-10-07"}\n</parameter>\n<parameter=valuation_requested>\nTrue\n</parameter>\n</function>\n</tool_call>'
    result = normalize_qwen_call({"content": content}, tools)
    arguments = result["tool_calls"][0]["function"]["arguments"]
    assert TaskUpdate.model_validate_json(arguments).valuation_requested is True


def test_nullable_none_literal_never_changes_a_quoted_string():
    for literal, expected in [("None", None), ('"None"', "None")]:
        supplied = [(name, literal if name == "date" else value) for name, value in parameters()]
        result = normalize_qwen_call({"content": call(supplied)}, TOOLS)
        assert json.loads(result["tool_calls"][0]["function"]["arguments"])["date"] == expected


@pytest.mark.parametrize("supplied", [
    [("file_id", "file_example")],
    [("file_id", "file_example"), ("view", "pdf_geometry"), ("page", "226")],
    [("file_id", "file_example"), ("view", "sheet"), ("sheet", "Financials"), ("cell_range", "A1:D12")],
    [("file_id", "file_example"), ("view", "records"), ("record_path", "/data"), ("record_filters", '{"year":["2025"]}')],
    [("file_id", "file_example"), ("view", "records")],
])
def test_read_file_union_schema_accepts_real_pdf_excel_and_json_calls(supplied):
    from valuationagent.application.file_workspace import FileRead

    tools = [{"type": "function", "function": {"name": "record", "parameters": FileRead.model_json_schema()}}]
    result = normalize_qwen_call({"content": call(supplied)}, tools)
    assert FileRead.model_validate_json(result["tool_calls"][0]["function"]["arguments"]).file_id == "file_example"


@pytest.mark.parametrize("supplied", [
    [("file_id", "file_example"), ("view", "records"), ("record_path", "/data"), ("page", "2")],
    [("file_id", "file_example"), ("view", "pdf_geometry"), ("record_path", "/data")],
    [("file_id", "file_example"), ("view", "unknown")],
])
def test_union_schema_rejects_missing_or_cross_branch_parameters(supplied):
    from valuationagent.application.file_workspace import FileRead

    tools = [{"type": "function", "function": {"name": "record", "parameters": FileRead.model_json_schema()}}]
    with pytest.raises(ValueError):
        normalize_qwen_call({"content": call(supplied)}, tools)


def test_protocol_repair_message_names_schema_fields_not_raw_content(monkeypatch):
    from valuationagent.application.file_workspace import FileRead

    tools = [{"type": "function", "function": {"name": "record", "parameters": FileRead.model_json_schema()}}]
    model = gateway(monkeypatch, {"content": call([("wrong_field", "SECRET_BODY")])}, tool_format="qwen3_coder")
    with pytest.raises(ToolProtocolError) as error:
        model.chat([], tools=tools, tool_choice="required")
    assert "file_id" in str(error.value) and "record_path" in str(error.value)
    assert "SECRET_BODY" not in str(error.value) + json.dumps(model.last_response_metadata)


def test_oneof_ambiguity_never_executes_a_guessed_branch():
    branch = {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}
    tools = [{"type": "function", "function": {"name": "record", "parameters": {"oneOf": [branch, branch]}}}]
    with pytest.raises(ValueError):
        normalize_qwen_call({"content": call([("value", "3")])}, tools)
