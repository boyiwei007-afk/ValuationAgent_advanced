import json

import pytest

from test_json_tool_gateway import gateway
from valuationagent.application.agent_runtime import AgentResponse
from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec
from valuationagent.llm.agent import run_tool_loop
from valuationagent.llm.client import LlmError, ToolProtocolError


def finish_registry(handler=None):
    return ToolRegistry([ToolSpec("finish_response", "finish", AgentResponse,
        handler or (lambda args: {"_terminal": True, "answer": args.answer}))])


def test_qwen_auto_returns_public_answer_without_private_reasoning(monkeypatch):
    registry = finish_registry()
    model = gateway(monkeypatch, {"content": "这是普通方法讨论。", "reasoning_content": "PRIVATE"}, tool_format="qwen3_coder")
    result = run_tool_loop(model, [], registry, lambda name, args, invoke: invoke(), allow_text_answer=True)
    assert result["answer"] == "这是普通方法讨论。"
    assert result["_agent_trace"]["text_answer_candidates"] == 1
    assert model.last_response_metadata["tool_transport"] == "text_answer_candidate"
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("text", ["", " \n ", "<tool_call><function=finish_response>", "<parameter=answer>not a complete call"])
def test_empty_or_partial_tools_cannot_be_final_answers(monkeypatch, text):
    model = gateway(monkeypatch, {"content": text, "reasoning_content": "PRIVATE"}, tool_format="qwen3_coder")
    with pytest.raises(ToolProtocolError):
        model.chat([], tools=finish_registry().schemas(), tool_choice="auto")


def test_length_truncation_is_never_accepted_as_final_answer(monkeypatch):
    model = gateway(monkeypatch, {"content": "half an answer"}, tool_format="qwen3_coder", finish_reason="length")
    with pytest.raises(ToolProtocolError, match="TRUNCATED"):
        model.chat([], tools=finish_registry().schemas(), tool_choice="auto")


def test_required_control_and_unoffered_finish_still_need_real_tools(monkeypatch):
    model = gateway(monkeypatch, {"content": "set_turn_plan now"}, tool_format="qwen3_coder")
    registry = ToolRegistry([ToolSpec("set_turn_plan", "control", NoArguments, lambda _: pytest.fail("not executed"))])
    with pytest.raises(ToolProtocolError):
        run_tool_loop(model, [], registry, lambda name, args, invoke: invoke(), allow_text_answer=True)


def test_final_text_uses_same_delivery_validation_and_can_recover():
    attempts = []

    def finish(args):
        attempts.append(args.answer)
        if len(attempts) == 1:
            raise ValueError("REPORT_NOT_DELIVERED: save the requested report")
        return {"_terminal": True, "answer": args.answer}

    class Model:
        def chat(self, messages, **kwargs):
            if attempts:
                assert "REPORT_NOT_DELIVERED" in messages[-1]["content"]
            return {"content": "保存完成" if attempts else "下一步保存"}

    result = run_tool_loop(Model(), [], finish_registry(finish), lambda name, args, invoke: invoke(), allow_text_answer=True)
    assert attempts == ["下一步保存", "保存完成"]
    assert result["_agent_trace"]["tool_errors"] == 1


def test_text_cannot_bypass_ready_calculation_and_report_requirements(tmp_path):
    from test_input_workspace import fixture, values
    from valuationagent.application.record_inputs import record_inputs

    _, runtime = fixture(tmp_path)
    record_inputs(runtime, values())

    class Model:
        def chat(self, *args, **kwargs):
            return {"content": "已完成，价格40元。"}

    with pytest.raises(LlmError, match="VALUATION_READY_NOT_EXECUTED"):
        run_tool_loop(Model(), [], finish_registry(runtime.finish), lambda name, args, invoke: invoke(), allow_text_answer=True)
    assert not runtime.session.valuation_run_id
