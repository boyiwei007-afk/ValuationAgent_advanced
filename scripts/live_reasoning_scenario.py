import argparse
import hashlib
import json
import time
from decimal import Decimal
from pathlib import Path

from valuationagent.api.main import create_app
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.llm.client import OpenAICompatibleClient
from valuationagent.schemas.models import ModelConnectionInput
from valuationagent.schemas.research import ResearchTurn


def fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]:
        digest.update(str(path.relative_to(root)).encode() + path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Real LLM, exact user scenario, no market/network tools; not production certification.")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--tool-call-format", choices=["json_content", "native_json", "native", "qwen3_coder"], default="json_content")
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--output-token-budget", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--temperature", type=float, default=.6)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    prompt = (Path(__file__).resolve().parents[1] / "tests/fixtures/user_manufacturing_scenario.txt").read_text(encoding="utf-8")
    initial = fingerprint()
    started = time.monotonic()

    class ObservedClient(OpenAICompatibleClient):
        calls = 0

        def chat(self, *positional, **keywords):
            self.calls += 1
            if self.calls > 100:
                raise ValueError("LIVE_TEST_BUDGET: maximum 100 calls")
            reply = None
            try:
                reply = super().chat(*positional, **keywords)
                return reply
            finally:
                print(json.dumps({"call": self.calls, "seconds": round(time.monotonic() - started),
                    "tools": [call["function"]["name"] for call in (reply or {}).get("tool_calls", [])],
                    "response": self.last_response_metadata}, ensure_ascii=False), flush=True)

    model = ObservedClient(ModelConnectionInput(base_url="http://127.0.0.1:18000/v1", model="qwen36-teacher",
        api_key="EMPTY", tool_call_format=args.tool_call_format, reasoning_protocol="chat_template",
        thinking=args.thinking, temperature=args.temperature, top_p=args.top_p,
        presence_penalty=args.presence_penalty, top_k=args.top_k, timeout_seconds=args.timeout_seconds,
        output_token_budget=args.output_token_budget, max_output_tokens=args.max_output_tokens))
    app = create_app(args.directory / "workspace")
    workspace = app.state.workspaces.create(llm=model, run_policy="automatic", data_source_preference="upload")
    failure = None
    try:
        app.state.workspaces.message(workspace.workspace_id, ResearchTurn(content=prompt, time_budget_seconds=args.seconds))
    except Exception as exc:
        failure = app.state.research._redact_text(str(exc))
    session = app.state.store.get_research(workspace.research_session_id)
    events = app.state.store.list_events(session.session_id)
    completed = [event.tool for event in events if event.type == "tool.completed"]
    record = app.state.store.get_run(session.valuation_run_id) if session.valuation_run_id else None
    results = record.result.model_dump(mode="json") if record and record.result else {}
    artifacts = app.state.store.list_artifacts(session.session_id)
    reports = [item for item in artifacts if item.get("numeric_result_available") and item.get("valuation_run_id") == session.valuation_run_id]
    for report in reports:
        metadata, data = app.state.store.get_artifact(session.session_id, report["artifact_id"])
        (args.directory / f"{metadata['number']}-{Path(metadata['filename']).name}").write_bytes(data)
    replay = replay_bundle(build_valuation_bundle(app.state.store, record)) if record and record.result else {}
    dcf = results.get("dcf") or {}
    relative = {item["method"]: item for item in results.get("relative", [])}
    expected = {"pe": ("22.5", "20.25", "24.75"), "ps": ("15", "13.5", "16.5"), "ev_ebitda": ("26", "23.5", "28.5")}
    checks = {"real_tools_executed": "set_turn_plan" in completed and bool({"record_inputs", "record_user_inputs"} & set(completed)),
        "no_network_tools": not set(completed) & {"search_sources", "fetch_search_source", "download_source", "acquire_financial_inputs", "fetch_financial_history"},
        "scenario_not_demo": bool(record and record.request.analysis_basis == "user_scenario" and record.request.mode != "demo"),
        "dcf_matches_independent_math": dcf.get("status") == "success" and abs(Decimal(str(dcf.get("per_share_value", 0))) - Decimal(16)) < Decimal("0.0001"),
        "all_relative_methods": {item["method"] for item in results.get("relative", []) if item["status"] == "success"} >= {"pe", "ps", "ev_ebitda"},
        "relative_medians_and_intervals": all(method in relative and all(
            abs(Decimal(str(relative[method].get(field) or 0)) - Decimal(value)) < Decimal(".0001")
            for field, value in zip(("per_share_value", "range_low", "range_high"), values)) for method, values in expected.items()),
        "explicit_fcff_paths": len(results.get("forecast", [])) == 10 and all(
            Decimal(str(row["fcff"])) == 1500000 and Decimal(str(row["tax_rate"])) == Decimal(".25")
            and Decimal(str(row["revenue"])) == 10000000 for row in results.get("forecast", [])),
        "dates_not_dropped": bool(record and str(record.request.valuation_date) == "2025-12-31"
            and str(record.request.financials.period_end) == "2025-12-31"
            and str(record.request.financials.common_shares_as_of or record.request.financials.diluted_shares_as_of) == "2025-12-31"),
        "matching_numeric_report": bool(reports), "replay": bool(replay.get("passed")),
        "source_revision_stable": initial == fingerprint(), "no_execution_failure": not failure and not session.last_issue}
    receipt = {"passed": all(checks.values()), "checks": checks, "scope": "Exact user synthetic scenario; checks independent FCFF, DCF and relative median/interval arithmetic. Not real-company accuracy certification.",
        "model": model.config.model, "thinking": args.thinking, "model_calls": model.calls,
        "tool_call_format": model.config.tool_call_format,
        "sampling": {key: getattr(model.config, key) for key in ("temperature", "top_p", "presence_penalty", "top_k")},
        "output_token_budget": model.config.output_token_budget, "max_output_tokens": model.config.max_output_tokens,
        "seconds": round(time.monotonic() - started), "workspace_id": workspace.workspace_id,
        "source_digest": initial, "tools": completed, "failure": failure,
        "last_issue": session.last_issue.model_dump(mode="json") if session.last_issue else None,
        "result": results, "replay": replay, "answer": app.state.store.list_messages(session.session_id)[-1].content}
    (args.directory / "acceptance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("passed", "checks", "model_calls", "seconds", "last_issue")}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
