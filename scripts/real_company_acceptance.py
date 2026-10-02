"""Opt-in real-disclosure matrix, isolated from user research and credentials.

Download with --download (public CNINFO catalogue); then --live for real LLM
research. This script never approves facts or forecasts automatically. A saved
formal plan must be inspected before a separate reviewer-controlled calculation.
"""
import argparse
import getpass
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

from valuationagent.application.research import ResearchService
from valuationagent.application.research_export import build_research_export
from valuationagent.application.runner import ValuationRunner
from valuationagent.application.workspaces import ValuationWorkspaceService
from valuationagent.core.documents import parse_document
from valuationagent.finance.team_model import FinanceTeamModel
from valuationagent.llm.client import OpenAICompatibleClient
from valuationagent.schemas.models import ModelConnectionInput
from valuationagent.schemas.research import ResearchDraft, ResearchTurn
from valuationagent.search.providers import CninfoAnnouncementProvider, TavilySearchProvider
from valuationagent.storage.sqlite import SQLiteRunStore


CASES = {
    "600519": ("贵州茅台", "食品饮料 / 白酒", "600519.SH"),
    "600887": ("伊利股份", "食品饮料 / 乳制品", "600887.SH"),
    "603288": ("海天味业", "食品饮料 / 调味品", "603288.SH"),
    "000333": ("美的集团", "家用电器", "000333.SZ"),
    "300750": ("宁德时代", "电力设备 / 电池", "300750.SZ"),
}
CUTOFF = date(2025, 6, 30)


def acceptance_checks(state, *, report_before_export, elapsed_seconds, budget):
    """Report delivery and numerical valuation are separate acceptance gates."""
    session = state["research"]["session"]
    outcome = state["research"].get("result_document") or {}
    return {
        "valuation_goal_retained": session.get("pending_action") == "valuation" or bool(session.get("valuation_run_id")),
        "report_available_without_export_side_effect": bool(report_before_export),
        "report_has_integrity_id": bool(outcome.get("report_id")),
        "review_policy_preserved": state["workspace"]["run_policy"] == "review",
        # The configured deadline is cooperative; one in-flight model/download
        # call and export may finish after it. Keep that overrun visible.
        "within_budget_plus_60_seconds": elapsed_seconds <= budget + 60,
        "numeric_valuation_completed": outcome.get("numeric_result_available") is True,
        "outcome_status": outcome.get("status", "missing"),
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def download_case(root, code):
    name, industry, ticker = CASES[code]
    folder = root / "sources" / code
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.json"
    if manifest.exists():
        result = json.loads(manifest.read_text(encoding="utf-8"))
        if hashlib.sha256((folder / "annual-2024.pdf").read_bytes()).hexdigest() != result["sha256"]:
            raise ValueError("Cached public source hash mismatch")
        return result
    started = time.monotonic()
    found = CninfoAnnouncementProvider().search_annual_reports(ticker, [2024], cutoff=CUTOFF, company_name=name)
    write_json(folder / "catalogue.json", found.model_dump(mode="json"))
    if not found.hits:
        raise ValueError(f"No official annual report for {code}: {found.status}")
    hit = found.hits[0]
    store = SQLiteRunStore(folder / "parse-store")
    service = ResearchService(store)
    url, content, media = service._download_disclosure_pdf(hit.url)
    (folder / "annual-2024.pdf").write_bytes(content)
    meta = store.save_upload(name + "2024年年度报告.pdf", "historical_financials", media, content)
    blocks, warnings = parse_document(store.get_file(meta["file_id"]))
    write_json(folder / "parsed-blocks.json", blocks)
    result = {"company": name, "ticker": ticker, "year": 2024, "source_url": url,
              "published_at": str(hit.published_at), "sha256": hashlib.sha256(content).hexdigest(),
              "downloaded_at": datetime.now(timezone.utc).isoformat(), "bytes": len(content),
              "pages_read": len({b['location'].get('page') for b in blocks}), "blocks": len(blocks),
              "warnings": warnings, "elapsed_seconds": round(time.monotonic() - started, 2)}
    write_json(manifest, result)
    return result


def live_case(root, run_label, code, model_key, search_key, budget, mode):
    name, industry, ticker = CASES[code]
    folder = root / run_label / code
    if folder.exists():
        raise FileExistsError("Choose a new --run-label; acceptance evidence is immutable")
    folder.mkdir(parents=True)
    store = SQLiteRunStore(folder)
    llm = OpenAICompatibleClient(ModelConnectionInput(
        base_url="https://api.deepseek.com", model="deepseek-flash",
        api_key=model_key, timeout_seconds=45,
    ))
    research = ResearchService(store, search_provider=TavilySearchProvider(search_key) if search_key else None)
    service = ValuationWorkspaceService(store, research, ValuationRunner(store, FinanceTeamModel()))
    workspace = service.create(llm=llm, data_source_preference="web", run_policy="review")
    session = store.get_research(workspace.research_session_id)
    session.draft = ResearchDraft(
        company=name, ticker=ticker, industry=industry, methods=["dcf"],
        valuation_date=CUTOFF, objective="真实公开年报软件验收；非投资建议；正式方案待复核",
    )
    store.save_research(session)
    file_ids = []
    if mode == "upload":
        content = (root / "sources" / code / "annual-2024.pdf").read_bytes()
        source = json.loads((root / "sources" / code / "manifest.json").read_text(encoding="utf-8"))
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError("Public fixture changed")
        meta = store.save_upload(name + "2024年年度报告.pdf", "historical_financials", "application/pdf", content)
        file_ids = [meta["file_id"]]
        store.append_event(
            session.session_id, type="test.public_fixture_uploaded", stage="input", status="completed",
            summary="独立验收：官方年报上传附件", payload={**source, "file_id": meta["file_id"]},
        )
    started = time.monotonic()
    service.message(workspace.workspace_id, ResearchTurn(
        file_ids=file_ids, time_budget_seconds=budget,
        content=(
            "开始自动化DCF估值。优先读取已有附件，无附件或缺项时自行检索官方披露。以2024完整年度为基期；"
            "多年历史不足可提出有依据、明确区分事实和观点的十年三情景预测与WACC/g。不要补零、不要把股本金额当股数。"
            "通过 calculate_valuation 提交审阅方案，等待用户批准。必要证据仍不可得，则交付说明报告。"
        ),
    ))
    state = service.snapshot(workspace.workspace_id)
    report_before_export = bool(state["research"].get("result_document"))
    for format_name in ("json", "html", "pdf"):
        content, _ = build_research_export(research, session.session_id, format_name)
        (folder / f"outcome.{format_name}").write_bytes(content if isinstance(content, bytes) else content.encode())
    saved = state["research"]["session"]
    elapsed = round(time.monotonic() - started, 2)
    events = store.list_events(session.session_id)
    summary = {
        "company": name, "ticker": ticker, "session_id": session.session_id,
        "workspace_id": workspace.workspace_id, "mode": mode,
        "prompt_version": saved["prompt_version"], "elapsed_seconds": elapsed,
        "documents": len(saved["documents"]), "facts": len(saved["facts"]),
        "clean_candidates": sum(fact["status"] != "rejected" and not fact["warnings"] for fact in saved["facts"]),
        "warning_candidates": sum(bool(fact["warnings"]) and fact["status"] != "rejected" for fact in saved["facts"]),
        "searches": len(saved["search_history"]), "issue": saved.get("last_issue"),
        "outcome": state["research"].get("result_document"),
        "checks": acceptance_checks(state, report_before_export=report_before_export, elapsed_seconds=elapsed, budget=budget),
        "tool_calls": dict(Counter(event.tool for event in events if event.type == "tool.started")),
        "event_counts": dict(Counter(event.type for event in events)),
        "scope": "Real public documents and real model; review policy; not investment research",
    }
    write_json(folder / "acceptance.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("var/real-company-20260926"))
    parser.add_argument("--companies", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--mode", choices=("upload", "zero-upload"), default="upload")
    parser.add_argument("--run-label", default="baseline")
    parser.add_argument("--budget", type=int, default=240)
    parser.add_argument("--require-numeric", action="store_true",
                        help="Fail acceptance unless every case has a completed numerical valuation; a missing-data report does not pass this gate")
    args = parser.parse_args()
    if not args.download and not args.live:
        parser.error("--download or --live is required for network calls")
    failures = 0
    if args.download:
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {pool.submit(download_case, args.root, code): code for code in args.companies}
            for job in as_completed(jobs):
                try:
                    print(json.dumps(job.result(), ensure_ascii=False), flush=True)
                except Exception as exc:
                    failures += 1
                    print(json.dumps({"ticker": jobs[job], "download_failed": str(exc)}, ensure_ascii=False), flush=True)
    if args.live:
        model_key = getpass.getpass("Model key (hidden): ")
        search_key = getpass.getpass("Tavily key (hidden, optional): ")
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(
                live_case, args.root, args.run_label, code, model_key,
                search_key, args.budget, args.mode,
            ): code for code in args.companies}
            results = []
            for job in as_completed(jobs):
                try:
                    result = job.result()
                    results.append(result)
                    if args.require_numeric and not result["checks"]["numeric_valuation_completed"]:
                        failures += 1
                    print(json.dumps({k: result[k] for k in ("company", "session_id", "elapsed_seconds", "facts", "searches", "clean_candidates", "warning_candidates")}, ensure_ascii=False), flush=True)
                except Exception as exc:
                    failures += 1
                    results.append({
                        "ticker": jobs[job],
                        "unhandled_exception": type(exc).__name__,
                        "error": str(exc)[:500],
                    })
                    print(json.dumps(results[-1]), flush=True)
            write_json(args.root / args.run_label / "matrix.json", results)
            print(json.dumps({"execution_errors_or_failed_numeric_gates": failures,
                              "numeric_valuations": sum(bool(r.get("checks", {}).get("numeric_valuation_completed")) for r in results),
                              "cases": len(results), "require_numeric": args.require_numeric}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
