import argparse
import hashlib
import json
import mimetypes
import time
from pathlib import Path

from valuationagent.api.main import create_app
from valuationagent.application.evidence_status import observation_verified
from valuationagent.llm.client import OpenAICompatibleClient
from valuationagent.schemas.models import ModelConnectionInput
from valuationagent.schemas.research import ResearchTurn


def source_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]:
        digest.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Opt-in real LLM file extraction; not valuation or independent audit certification.")
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--objective", default="仅阅读我上传的文件，不联网、不估值。提取文件明确提供的最近三个完整年度营业收入和归母净利润，保留单位、年度与原文位置，保存研究笔记。没有的年度或指标明确说明，不能猜测。")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    initial, started = source_digest(), time.monotonic()

    class ObservedClient(OpenAICompatibleClient):
        calls = 0

        def chat(self, *positional, **keywords):
            self.calls += 1
            if self.calls > 65:
                raise ValueError("LIVE_TEST_BUDGET: maximum 65 calls")
            try:
                return super().chat(*positional, **keywords)
            finally:
                print(json.dumps({"call": self.calls, "seconds": round(time.monotonic() - started),
                    "response": self.last_response_metadata}, ensure_ascii=False), flush=True)

    model = ObservedClient(ModelConnectionInput(base_url="http://127.0.0.1:18000/v1", api_key="EMPTY",
        model="qwen36-teacher", tool_call_format="qwen3_coder", reasoning_protocol="chat_template",
        thinking=args.thinking, temperature=1 if args.thinking == "enabled" else .7,
        top_p=.95 if args.thinking == "enabled" else .8, presence_penalty=1.5, top_k=20, timeout_seconds=600))
    app = create_app(args.directory / "workspace")
    workspace = app.state.workspaces.create(llm=model, run_policy="automatic", data_source_preference="upload")
    upload = app.state.store.save_upload(args.file.name, "evidence",
        mimetypes.guess_type(args.file.name)[0] or "application/octet-stream", args.file.read_bytes())
    failure = None
    try:
        app.state.workspaces.message(workspace.workspace_id, ResearchTurn(content=args.objective,
            file_ids=[upload["file_id"]], time_budget_seconds=args.seconds))
    except Exception as exc:
        failure = app.state.research._redact_text(str(exc))
    session = app.state.store.get_research(workspace.research_session_id)
    events = app.state.store.list_events(session.session_id)
    tools = [event.tool for event in events if event.type == "tool.completed"]
    bound = [fact for fact in session.facts if fact.status != "rejected" and observation_verified(fact)]
    values = [{"metric": fact.standard_metric or fact.metric, "period": fact.period,
        "value": fact.normalized_value, "unit": fact.unit, "block_id": fact.block_id, "status": fact.status,
        "warnings": fact.warnings} for fact in bound]
    distinct = {(fact.standard_metric or fact.metric, fact.period) for fact in bound
                if (fact.standard_metric or fact.metric) in {"revenue", "net_income_parent"}}
    checks = {"real_file_read": "read_file" in tools, "observations_saved": "extract_observations" in tools and len(distinct) >= 6,
        "no_network": not set(tools) & {"search_sources", "fetch_search_source", "fetch_financial_history", "acquire_financial_inputs"},
        "no_unsolicited_valuation": not session.valuation_run_id, "note_saved": "write_research_note" in tools,
        "no_execution_failure": not failure and not session.last_issue, "source_revision_stable": initial == source_digest()}
    receipt = {"passed": all(checks.values()), "checks": checks,
        "scope": "Single uploaded file extraction and note; not a numerical valuation or semantic audit acceptance.",
        "input_sha256": upload["sha256"], "source_digest": initial, "thinking": args.thinking,
        "model_calls": model.calls, "seconds": round(time.monotonic() - started), "tools": tools,
        "observations": values, "failure": failure, "last_issue": session.last_issue.model_dump(mode="json") if session.last_issue else None,
        "answer": app.state.store.list_messages(session.session_id)[-1].content}
    (args.directory / "acceptance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("passed", "checks", "seconds", "model_calls", "last_issue")}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
