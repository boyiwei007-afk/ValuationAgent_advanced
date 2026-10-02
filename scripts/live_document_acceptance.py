"""Opt-in live valuation evaluation; credentials are supplied through environment only."""
import argparse
import hashlib
import json
import mimetypes
import os
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx

from valuationagent.api.main import create_app
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.llm.client import LlmError, OpenAICompatibleClient
from valuationagent.schemas.models import ModelConnectionInput
from valuationagent.schemas.research import ResearchTurn
from valuationagent.search.providers import TavilySearchProvider


def acceptance_checks(snapshot, failure=None):
    run = snapshot.get("active_run") or {}
    result = run.get("result") or {}
    methods = [result.get("dcf"), *result.get("relative", [])]
    numeric_methods = []
    for method in methods:
        if not method or method.get("status") != "success":
            continue
        try:
            value = Decimal(str(method.get("per_share_value")))
        except InvalidOperation:
            continue
        if value.is_finite():
            numeric_methods.append(method)
    calculated = bool(run.get("run_id") and run.get("status") in {"completed", "completed_with_warnings"}
                      and (run.get("request") or {}).get("mode") == "snapshot" and numeric_methods)
    matching_report = bool(calculated and any(
        artifact.get("kind") == "result_report" and artifact.get("numeric_result_available") is True
        and artifact.get("valuation_run_id") == run["run_id"]
        for artifact in snapshot.get("artifacts", [])))
    return {"numeric_valuation_completed": calculated, "numeric_report_matches_active_run": matching_report,
            "execution_has_no_failure": not failure and (snapshot.get("execution") or {}).get("status") not in {"failed", "cancelled"}}


def acceptance_exit_code(checks):
    return 0 if checks and all(checks.values()) else 1


def source_revision(root):
    selected = [root / "pyproject.toml", root / "requirements.lock", root / "scripts/live_document_acceptance.py"]
    selected.extend(path for path in (root / "src/valuationagent").rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(set(selected)):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            count += 1
    return {"sha256": digest.hexdigest(), "file_count": count}


def main():
    source_root = Path(__file__).resolve().parents[1]
    revision_before = source_revision(source_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume only a single-workspace evaluation directory with acceptance.json.")
    parser.add_argument("--objective")
    parser.add_argument("--base-url", default=os.getenv("VALUATION_LLM_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("VALUATION_LLM_MODEL"))
    parser.add_argument("--tool-call-format", choices=["native", "json_content"],
                        default=os.getenv("VALUATION_LLM_TOOL_CALL_FORMAT", "native"))
    parser.add_argument("--reasoning-protocol", choices=["auto", "chat_template"],
                        default=os.getenv("VALUATION_LLM_REASONING_PROTOCOL", "auto"))
    parser.add_argument("--file", type=Path, action="append", default=[])
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--calls", type=int, default=80)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    previous = None
    if args.resume:
        try:
            previous = json.loads((args.directory / "acceptance.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            parser.error("--resume requires a prior acceptance.json from this evaluator, not a user workspace.")
        if previous.get("company") != args.company or not previous.get("workspace_id"):
            parser.error("The evaluation company must match the previous acceptance record.")
    elif args.directory.exists():
        parser.error("Use a new evaluation directory; never run against a user's existing workspace.")
    if not 1 <= args.calls <= 160 or not 1 <= args.turns <= 4 or not 30 <= args.seconds <= 1800:
        parser.error("Bounded limits: calls 1..160, turns 1..4, seconds 30..1800")
    key, search_key = os.getenv("VALUATION_LLM_API_KEY", ""), os.getenv("TAVILY_API_KEY", "")
    if not args.base_url or not key:
        parser.error("Set VALUATION_LLM_BASE_URL and VALUATION_LLM_API_KEY (EMPTY only for a server without authentication).")
    if not args.upload_only and not search_key:
        parser.error("Set TAVILY_API_KEY for online evaluation, or use --upload-only with --file.")
    if len(args.file) > 8 or any(not path.is_file() or path.stat().st_size > 50 * 1024 * 1024 for path in args.file):
        parser.error("Provide at most 8 existing files, each at most 50 MB.")
    if args.upload_only and not args.file and not args.resume:
        parser.error("--upload-only requires --file.")
    model_name = args.model
    if not model_name:
        response = httpx.get(args.base_url.rstrip("/") + "/models", headers={"Authorization": "Bearer " + key}, timeout=30)
        response.raise_for_status()
        names = [item["id"] for item in response.json()["data"]]
        if len(names) != 1:
            parser.error("Endpoint does not expose exactly one model; specify --model from /models.")
        model_name = names[0]
    client = OpenAICompatibleClient(ModelConnectionInput(provider="openai_compatible", base_url=args.base_url,
        model=model_name, api_key=key, thinking=os.getenv("VALUATION_LLM_THINKING", "auto"),
        tool_call_format=args.tool_call_format, reasoning_protocol=args.reasoning_protocol,
        temperature=float(os.getenv("VALUATION_LLM_TEMPERATURE", "0")), timeout_seconds=180))
    started = time.monotonic()

    class BudgetedModel:
        config = client.config
        calls = 0

        def chat(self, *positional, **keyword):
            if self.calls >= args.calls:
                raise LlmError("LIVE_TEST_BUDGET: evaluation model-call budget reached")
            self.calls += 1
            message = {}
            try:
                message = client.chat(*positional, **keyword)
            finally:
                print(json.dumps({"call": self.calls, "seconds": round(time.monotonic() - started),
                                  "tools": [call.get("function", {}).get("name") for call in message.get("tool_calls", [])],
                                  "response": client.last_response_metadata}, ensure_ascii=False), flush=True)
            return message

    model = BudgetedModel()
    app = create_app(args.directory)
    service = app.state.workspaces
    if args.resume:
        existing = service.list()
        if len(existing) != 1 or existing[0].workspace_id != previous["workspace_id"]:
            parser.error("Resume is restricted to the evaluator's recorded single workspace.")
        workspace = existing[0]
        saved = service.store.get_research(workspace.research_session_id)
        if saved.data_source_preference != ("upload" if args.upload_only else "web"):
            parser.error("Resume cannot change source permissions; use the original --upload-only setting.")
        service.research.attach(workspace.research_session_id, model)
    else:
        workspace = service.create(llm=model, title=args.company,
            data_source_preference="upload" if args.upload_only else "web", run_policy="automatic")
    if not args.upload_only:
        service.research.attach_search(workspace.research_session_id, TavilySearchProvider(search_key))
    uploads = [service.store.save_upload(path.name, "evidence", mimetypes.guess_type(path.name)[0] or "application/octet-stream", path.read_bytes())["file_id"] for path in args.file]
    objective = args.objective or ("从检查点继续未完成任务，先处理已有资料，不重复相同失败调用。" if args.resume else
        f"请对{args.company}进行估值，使用当前可得公开资料，选择适用方法完成估值草案并输出报告。")
    failure = None
    for turn in range(args.turns):
        try:
            service.message(workspace.workspace_id, ResearchTurn(content=objective if turn == 0 else
                "从检查点继续未完成任务，先处理已有资料，不重复相同失败调用。",
                file_ids=uploads if turn == 0 else [], time_budget_seconds=args.seconds))
        except (LlmError, ValueError) as exc:
            failure = str(exc)[:500]
        snapshot = service.snapshot(workspace.workspace_id)
        latest_reply = next((item["content"] for item in reversed(snapshot.get("messages", [])) if item["role"] == "assistant"), "")
        if (snapshot.get("execution") or {}).get("status") == "failed" or latest_reply.startswith(("LLM_", "TOOL_", "AGENT_", "LIVE_TEST_BUDGET")):
            failure = latest_reply
        if all(acceptance_checks(snapshot, failure).values()) or model.calls >= args.calls or failure:
            break
        session = snapshot["research"]["session"]
        if session.get("pending_decision") or not session.get("resume_context"):
            break
    snapshot = service.snapshot(workspace.workspace_id)
    session = snapshot["research"]["session"]
    events = service.store.list_events(workspace.research_session_id)
    facts = session.get("facts", [])
    checks = acceptance_checks(snapshot, failure)
    calculated = checks["numeric_valuation_completed"]
    checks.update(report_integrity=False, frozen_input_replay=False)
    validation_error = None
    if calculated:
        try:
            run_id = snapshot["active_run"]["run_id"]
            reports = [artifact for artifact in snapshot.get("artifacts", [])
                       if artifact.get("kind") == "result_report" and artifact.get("numeric_result_available") is True
                       and artifact.get("valuation_run_id") == run_id]
            checks["report_integrity"] = bool(reports) and all(
                bool(service.store.get_artifact(workspace.research_session_id, artifact["artifact_id"])[1]) for artifact in reports)
            checks["frozen_input_replay"] = replay_bundle(build_valuation_bundle(service.store, service.store.get_run(run_id)))["passed"]
        except (ValueError, KeyError, OSError) as exc:
            validation_error = str(exc)[:500]
    distinct = {(fact.get("peer_ticker", ""), fact["role"], fact["standard_metric"], fact["period"], fact["scope"], fact["normalized_value"])
                for fact in facts if fact["status"] == "confirmed" and not fact.get("warnings")}
    revision_after = source_revision(source_root)
    checks["source_revision_stable"] = revision_before == revision_after
    report = {"company": args.company, "workspace_id": workspace.workspace_id, "model": model_name,
        "source_revision_before": revision_before, "source_revision_after": revision_after,
        "tool_call_format": args.tool_call_format, "thinking": client.config.thinking, "reasoning_protocol": args.reasoning_protocol,
        "temperature": client.config.temperature,
        "model_calls": model.calls, "elapsed_seconds": round(time.monotonic() - started),
        "fact_statuses": dict(Counter(fact["status"] for fact in facts)),
        "distinct_confirmed_values": len(distinct),
        "confirmed_metrics": sorted({fact["standard_metric"] for fact in facts if fact["status"] == "confirmed"}),
        "confirmed_periods": sorted({fact["period"] for fact in facts if fact["status"] == "confirmed"}),
        "warnings": dict(Counter(warning for fact in facts for warning in fact.get("warnings", []))),
        "documents": len(session.get("documents", [])), "valuation_calculated": calculated,
        "valuation_and_report_complete": acceptance_exit_code(checks) == 0,
        "acceptance_checks": checks, "validation_error": validation_error,
        "execution": snapshot.get("execution"),
        "latest_reply": next((item["content"] for item in reversed(snapshot.get("messages", [])) if item["role"] == "assistant"), ""),
        "artifacts": len(snapshot.get("artifacts", [])), "failure": failure,
        "tools": dict(Counter(event.tool for event in events if event.type == "tool.completed")),
        "checkpoint": session.get("resume_context"), "prompt_version": session.get("prompt_version"),
        "limitation": "Live observed behavior, not proof of universal accuracy; facts still need independent spot checks."}
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if any(secret and secret != "EMPTY" and secret in encoded for secret in (key, search_key)):
        raise RuntimeError("Credential leak in evaluation summary")
    filename = f"acceptance-resume-{time.time_ns()}.json" if args.resume else "acceptance.json"
    with (args.directory / filename).open("x", encoding="utf-8") as output:
        output.write(encoded)
    print(encoded, flush=True)
    return acceptance_exit_code(checks)


if __name__ == "__main__":
    raise SystemExit(main())
