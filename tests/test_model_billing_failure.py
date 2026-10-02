import httpx
import pytest

from valuationagent.llm.client import LlmError, OpenAICompatibleClient
from valuationagent.schemas.models import ModelConnectionInput
from valuationagent.schemas.research import ResearchTurn
from test_evidence_recovery import configured_service


def test_402_is_single_request_and_never_echoes_provider_body(monkeypatch):
    calls = []
    original = httpx.Client

    def handler(request):
        calls.append(request)
        return httpx.Response(402, json={"error": "private-provider-body-should-not-appear"})

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    model = OpenAICompatibleClient(ModelConnectionInput(base_url="https://example.com", api_key="fixture", model="fixture"))
    with pytest.raises(LlmError, match="LLM_HTTP_402") as caught:
        model.chat([{"role": "user", "content": "test"}])
    assert len(calls) == 1
    assert "额度不足" in str(caught.value)
    assert "private-provider-body" not in str(caught.value)


def test_billing_failure_keeps_outcome_without_retry_loop(tmp_path):
    class BillingFailure:
        model = "fixture"
        provider = "fixture"

        def chat(self, *args, **kwargs):
            raise LlmError("LLM_HTTP_402: 模型账户额度不足；请处理计费后重新连接。")

    service, session, _ = configured_service(tmp_path, BillingFailure())
    state = service.turn(session.session_id, ResearchTurn(content="开始自动化DCF估值"))
    assert state["session"]["last_issue"]["retryable"] is False
    assert "question" not in state["session"]
    assert state["execution"]["status"] == "failed"
    assert "LLM_HTTP_402" in state["messages"][-1]["content"]
    assert state["result_document"]["status"] == "insufficient_data"


def test_connection_failure_replaces_stale_checkpoint_reason(tmp_path):
    class UnavailableModel:
        def chat(self, *args, **kwargs):
            raise LlmError("LLM_NETWORK_FAILED: 模型请求传输失败。")

    service, session, _ = configured_service(tmp_path, UnavailableModel())
    session.resume_context = {"reason": "TOOL_JSON_TRUNCATED", "instruction": "obsolete issue"}
    service.store.save_research(session)
    state = service.turn(session.session_id, ResearchTurn(content="继续"))
    checkpoint = state["session"]["resume_context"]
    assert checkpoint["reason"] == "LLM_NETWORK_FAILED" and checkpoint["request_id"]
    assert "不自动切换模型" in checkpoint["instruction"]
    assert "obsolete issue" not in str(checkpoint)
