from datetime import date
import json
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from valuationagent.schemas.agent import SearchQuery
from valuationagent.search.providers import MockSearchProvider, UnavailableSearchProvider
from valuationagent.llm.client import ContextWindowError, OpenAICompatibleClient
from valuationagent.llm.agent import model_tool_result, run_tool_loop
from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec
from valuationagent.schemas.models import ModelConnectionInput


def test_search_query_normalizes_domains_and_mock_never_calls_network():
    query = SearchQuery(
        query="贵州茅台 可比公司",
        ticker=" 600519 ",
        purpose="comparables",
        allowed_domains=["https://example.com/", "example.com", "CNINFO.CN"],
        candidate_limit=2,
    )
    provider = MockSearchProvider(
        [
            {
                "source_id": "peer_1",
                "title": "贵州茅台行业资料",
                "url": "https://example.com/a",
                "domain": "example.com",
                "snippet": "可比公司与白酒行业",
            },
            {
                "source_id": "other_1",
                "title": "宏观资料",
                "url": "https://example.com/b",
                "domain": "example.com",
                "snippet": "宏观经济",
            },
        ]
    )
    result = provider.search(query)
    assert query.ticker == "600519"
    assert query.allowed_domains == ["example.com", "cninfo.cn"]
    assert result.status == "completed"
    assert result.hits[0].source_id == "peer_1"
    assert result.provider == "mock-search"


def test_search_unavailable_is_explicit_and_policy_needs_finance_review():
    query = SearchQuery(query="600519 年报", information_cutoff=date(2026, 9, 19))
    result = UnavailableSearchProvider().search(query)
    assert result.status == "not_configured"
    assert result.hits == []


def test_search_query_rejects_empty_or_oversized_budget():
    with pytest.raises(ValidationError):
        SearchQuery(query="", candidate_limit=8)
    with pytest.raises(ValidationError):
        SearchQuery(query="ok", candidate_limit=51)


def test_openai_compatible_client_normalizes_object_tool_arguments():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "connection_check",
                                        "arguments": {},
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    original = httpx.Client
    config = ModelConnectionInput(model="mock", api_key="not-real")
    with patch(
        "valuationagent.llm.client.httpx.Client",
        side_effect=lambda **kwargs: original(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    ):
        assert OpenAICompatibleClient(config).test_connection().startswith("OK")


def test_tool_loop_recovers_from_two_truncated_json_calls_with_smaller_retry():
    class TruncatedThenValid:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            arguments = "{" if self.calls < 3 else "{}"
            return {"tool_calls": [{
                "id": f"call_{self.calls}",
                "type": "function",
                "function": {"name": "finish", "arguments": arguments},
            }]}

    model = TruncatedThenValid()
    registry = ToolRegistry([
        ToolSpec("finish", "finish", NoArguments, lambda _: {"_terminal": True, "answer": "ok"}),
    ])
    result = run_tool_loop(
        model,
        [{"role": "system", "content": "test"}],
        registry,
        lambda _name, _arguments, invoke: invoke(),
        max_rounds=4,
    )
    assert result["answer"] == "ok"
    assert result["_agent_trace"]["tool_errors"] == 2
    assert model.calls == 3


def test_tool_batch_executes_in_order_with_all_results_returned_before_next_model_call():
    visited = []
    class Model:
        calls = 0
        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 2:
                outputs = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
                assert [output["position"] for output in outputs] == [1, 2]
                assert visited == [1, 2]
            names = ["step", "step"] if self.calls == 1 else ["finish"]
            return {"tool_calls": [{"id": f"call_{self.calls}_{index}", "function": {"name": name, "arguments": "{}"}}
                                   for index, name in enumerate(names)]}
    def step(_):
        visited.append(len(visited) + 1)
        return {"position": len(visited)}
    registry = ToolRegistry([ToolSpec("step", "step", NoArguments, step),
                             ToolSpec("finish", "finish", NoArguments, lambda _: {"_terminal": True})])
    result = run_tool_loop(Model(), [{"role": "system", "content": "test"}], registry,
                           lambda name, args, invoke: invoke(), max_rounds=3)
    assert result["_agent_trace"]["protocol_errors"] == 0
    assert result["_agent_trace"]["tools"] == ["step", "step", "finish"]


def test_tool_batch_validates_all_identifiers_before_any_side_effect():
    from valuationagent.llm.client import LlmError
    class Model:
        def chat(self, *args, **kwargs):
            return {"tool_calls": [{"id": "duplicate", "function": {"name": "step", "arguments": "{}"}}] * 2}
    visited = []
    registry = ToolRegistry([ToolSpec("step", "step", NoArguments, lambda _: visited.append(True))])
    with pytest.raises(LlmError, match="唯一id"):
        run_tool_loop(Model(), [], registry, lambda name, args, invoke: invoke())
    assert not visited


def test_prompt_projection_retains_numbered_lines_and_does_not_mutate_audit():
    original = {"blocks": [{"block_id": "file:1", "text": "first\nsecond\n", "location": {"page": 2},
                            "lines": [{"line": 1, "text": "first"}, {"line": 2, "text": "second"}]}]}
    projected = model_tool_result(original)
    assert "text" not in projected["blocks"][0]
    assert projected["blocks"][0]["lines"] == original["blocks"][0]["lines"]
    assert original["blocks"][0]["text"] == "first\nsecond\n"
    assert model_tool_result({"text": "different", "lines": []})["text"] == "different"


def test_context_recovery_never_reexecutes_tools_or_removes_system_scope():
    visited = []
    class Model:
        calls = 0
        def chat(self, messages, **kwargs):
            self.calls += 1
            assert messages[0]["content"] == "authoritative scope"
            if self.calls == 2:
                raise ContextWindowError("LLM_CONTEXT_LIMIT")
            if self.calls == 3:
                assert "context_omitted" in messages[-1]["content"]
                assert visited == [True]
            name = "read" if self.calls == 1 else "finish"
            return {"tool_calls": [{"id": str(self.calls), "function": {"name": name, "arguments": "{}"}}]}
    def read(_):
        visited.append(True)
        return {"text": "large source " * 1000}
    registry = ToolRegistry([ToolSpec("read", "read", NoArguments, read), ToolSpec("finish", "finish", NoArguments, lambda _: {"_terminal": True})])
    result = run_tool_loop(Model(), [{"role": "system", "content": "authoritative scope"}], registry,
                           lambda name, args, invoke: invoke(), max_rounds=3, max_context_chars=20000)
    assert result["_agent_trace"]["context_recoveries"] == 1
    assert visited == [True]


def test_context_error_is_classified_without_echoing_vendor_content(monkeypatch):
    original = httpx.Client
    def handler(request):
        return httpx.Response(400, json={"message": "The input (35010 tokens) is longer than the model's context length (32768 tokens). private-content"})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    model = OpenAICompatibleClient(ModelConnectionInput(model="test", api_key="fixture"))
    with pytest.raises(ContextWindowError) as error:
        model.chat([])
    assert "private-content" not in str(error.value)
