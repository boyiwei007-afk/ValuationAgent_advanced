"""Explicit, bounded live integration check. Credentials are environment-only."""
import json
import os
import tempfile
from pathlib import Path

import httpx

from valuationagent.api.main import create_app
from valuationagent.llm.client import LlmError, OpenAICompatibleClient
from valuationagent.schemas.agent import SearchQuery
from valuationagent.schemas.models import ModelConnectionInput
from valuationagent.schemas.research import ResearchTurn
from valuationagent.search.providers import TavilySearchProvider


def main():
    api_key = os.environ["DEEPSEEK_API_KEY"]
    search_key = os.environ["TAVILY_API_KEY"]
    model_name = os.getenv("DEEPSEEK_MODEL")
    if not model_name:
        response = httpx.get(
            "https://api.deepseek.com/models",
            headers={"Authorization": "Bearer " + api_key}, timeout=30,
        )
        response.raise_for_status()
        names = [item["id"] for item in response.json()["data"]]
        model_name = next((name for name in names if "flash" in name),
                          next((name for name in names if "chat" in name), names[0]))
    client = OpenAICompatibleClient(ModelConnectionInput(
        provider="openai_compatible", base_url="https://api.deepseek.com",
        model=model_name, api_key=api_key, thinking="disabled",
    ))
    provider = TavilySearchProvider(search_key)
    search = provider.search(SearchQuery(
        query="贵州茅台 上市公司 官方网站", purpose="company_profile", candidate_limit=2,
    ))
    if search.status != "completed" or not search.hits:
        raise RuntimeError("Live search did not return usable results: " + search.status)

    class BudgetedModel:
        config = client.config
        calls = 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            if self.calls > 12:
                raise LlmError("LIVE_TEST_BUDGET: stop after 12 model calls")
            return client.chat(*args, **kwargs)

    bounded = BudgetedModel()
    with tempfile.TemporaryDirectory(prefix="valuation-live-") as directory:
        app = create_app(Path(directory))
        service = app.state.workspaces
        workspace = service.create(llm=bounded, data_source_preference="web")
        service.research.attach_search(workspace.research_session_id, provider)
        service.message(workspace.workspace_id, ResearchTurn(
            content="你好，简短解释 DCF 和 P/E 的差别。这是方法讨论，不要启动估值。",
            time_budget_seconds=180,
        ))
        service.message(workspace.workspace_id, ResearchTurn(
            content="请使用 search_sources 查找贵州茅台的官方网站，purpose 用 company_profile。只检索一次并给出一个来源链接，不下载文件，不估值。",
            time_budget_seconds=240,
        ))
        snapshot = service.snapshot(workspace.workspace_id)
        events = service.store.list_events(workspace.research_session_id)
        assert any(event.tool == "search_sources" and event.type == "tool.completed" for event in events)
        assert snapshot["execution"]["status"] == "completed", snapshot["execution"]["status"]
        assert snapshot["active_run"] is None
        serialized = json.dumps(snapshot, ensure_ascii=False)
        assert api_key not in serialized and search_key not in serialized
        print(json.dumps({
            "model": model_name, "model_calls": bounded.calls,
            "search_status": search.status, "search_hits": len(search.hits),
            "conversation_messages": len(snapshot["messages"]),
            "tool_sequence": [event.tool for event in events if event.type == "tool.completed"],
            "credentials_persisted": False, "passed": True,
        }, ensure_ascii=True))


if __name__ == "__main__":
    main()
