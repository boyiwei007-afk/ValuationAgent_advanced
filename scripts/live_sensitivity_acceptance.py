"""Opt-in real-LLM follow-up test on an explicitly synthetic valuation baseline."""
import argparse
import json
import time
from datetime import date
from pathlib import Path

from valuationagent.api.main import create_app
from valuationagent.application.ledgers import build_model_spec
from valuationagent.llm.client import LlmError, OpenAICompatibleClient
from valuationagent.schemas.models import ValuationRequest
from valuationagent.schemas.research import ResearchTurn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.directory.exists():
        parser.error("Use a new isolated directory; never overwrite a user workspace.")
    client = OpenAICompatibleClient.from_environment()
    started = time.monotonic()
    class BoundedModel:
        config = client.config
        calls = 0

        def chat(self, *positional, **keyword):
            if self.calls >= 25:
                raise LlmError("LIVE_TEST_BUDGET: follow-up evaluation reached 25 model calls")
            self.calls += 1
            response = client.chat(*positional, **keyword)
            print(json.dumps({"call": self.calls, "seconds": round(time.monotonic() - started),
                              "tools": [item["function"]["name"] for item in response.get("tool_calls", [])]}, ensure_ascii=False), flush=True)
            return response
    model = BoundedModel()
    service = create_app(args.directory).state.workspaces
    workspace = service.create(llm=model, run_policy="automatic", data_source_preference="upload", title="Synthetic sensitivity acceptance")
    record = service.runner.create_run(ValuationRequest(
        company={"name": "Synthetic fixture — NOT a real issuer", "currency": "CNY"},
        valuation_date=date(2026, 9, 30), data_source="structured", assumption_source="automatic",
        mode="demo", forecast_years=5, methods=["dcf", "pe"], user_goal="Explicit synthetic follow-up evaluation",
    ))
    spec = build_model_spec(service.store, workspace, record, None, service.runner.finance.version)
    service.store.save_workspace_record(workspace.workspace_id, "model_spec", spec, immutable=True)
    session = service.store.get_research(workspace.research_session_id)
    session.valuation_run_id = record.run_id
    service.store.save_research(session)
    service.store.freeze_run_sources(record.run_id, session)
    workspace.active_run_id = record.run_id
    service.store.save_workspace(workspace)
    service.execute(record.run_id)
    baseline = service.store.get_run(record.run_id).model_dump(mode="json")
    prompts = [
        "这是明确的合成测试基准，不对应真实公司。请保持现有估值版本，做PE归母净利润为基准的90%、100%、110%的敏感性分析，保存并回读试算报告。不要联网或重新估值。",
        "现在只聊聊：为什么敏感性区间不是置信区间？不要运行新试算，不修改模型。",
    ]
    failure = None
    turn_statuses = []
    for prompt in prompts:
        try:
            service.message(workspace.workspace_id, ResearchTurn(content=prompt, time_budget_seconds=600))
            current = service.snapshot(workspace.workspace_id)
            status = (current.get("execution") or {}).get("status")
            turn_statuses.append(status)
            if status == "failed":
                failure = "FOLLOWUP_EXECUTION_FAILED"
                break
        except (ValueError, LlmError) as exc:
            failure = str(exc)[:500]
            break
    snapshot = service.snapshot(workspace.workspace_id)
    artifacts = service.store.list_artifacts(workspace.research_session_id)
    calculations = [artifact for artifact in artifacts if artifact["kind"] == "sensitivity_analysis"]
    events = service.store.list_events(workspace.research_session_id)
    tools = [event.tool for event in events if event.type == "tool.completed"]
    unchanged = baseline == service.store.get_run(record.run_id).model_dump(mode="json") and service.get(workspace.workspace_id).active_run_id == record.run_id
    result = {"synthetic_baseline": True, "real_company_valuation_test": False,
              "model": client.config.model, "model_calls": model.calls, "failure": failure,
              "turn_statuses": turn_statuses,
              "baseline_unchanged": unchanged, "sensitivity_artifacts": len(calculations),
              "tools": tools, "latest_reply": snapshot["messages"][-1]["content"],
              "answer_quality": "ungraded: workflow success does not certify the model's financial or statistical explanations",
              "workflow_passed": unchanged and len(calculations) == 1 and "read_artifact" in tools
                        and "search_sources" not in tools and "calculate_valuation" not in tools
                        and (snapshot.get("execution") or {}).get("status") != "failed" and failure is None}
    (args.directory / "acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
