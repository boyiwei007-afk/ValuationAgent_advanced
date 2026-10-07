from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import httpx

from valuationagent.api.main import create_app
from valuationagent.llm.client import LlmError, OpenAICompatibleClient
from valuationagent.schemas.models import ModelConnectionInput
from valuationagent.schemas.research import ResearchTurn


def fingerprint():
    root = Path(__file__).resolve().parents[1]
    checksum = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]:
        checksum.update(str(path.relative_to(root)).encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Real provider acquisition and LLM reading; not a numerical valuation acceptance gate.")
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--company", action="append", help="Name:ticker; may be repeated")
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    args = parser.parse_args()
    key = os.getenv("INFOWAY_API_KEY", "")
    if not key:
        parser.error("Supply INFOWAY_API_KEY through the process environment, not command arguments or files.")
    if args.directory.exists():
        parser.error("Use a fresh evaluation directory, never an existing user workspace.")
    with httpx.Client(trust_env=False, timeout=30) as probe:
        response = probe.get(args.base_url.rstrip("/") + "/models")
        response.raise_for_status()
        models = [entry["id"] for entry in response.json()["data"]]
    if len(models) != 1:
        parser.error("The local endpoint must expose exactly one model for this evaluator.")
    args.directory.mkdir(parents=True)
    app = create_app(args.directory / "workspace")
    client = OpenAICompatibleClient(ModelConnectionInput(base_url=args.base_url, model=models[0], api_key="EMPTY",
        tool_call_format="json_content", reasoning_protocol="chat_template", thinking="disabled", temperature=0, timeout_seconds=180))
    revision = fingerprint()
    results = []
    for subject in args.company or ["瑞芯微:603893", "青岛啤酒:600600", "比亚迪:002594"]:
        company, ticker = subject.split(":", 1)
        started = time.monotonic()

        class BudgetedModel:
            config = client.config
            calls = 0

            def chat(self, *positional, **keyword):
                if self.calls >= 32:
                    raise LlmError("LIVE_TEST_BUDGET: acquisition model-call limit reached")
                self.calls += 1
                message = client.chat(*positional, **keyword)
                print(json.dumps({"company": company, "call": self.calls,
                    "tools": [call.get("function", {}).get("name") for call in message.get("tool_calls", [])]}, ensure_ascii=False), flush=True)
                return message

        model = BudgetedModel()
        workspace = app.state.workspaces.create(llm=model, run_policy="automatic", data_source_preference="web")
        failure = None
        content = (f"读取{company}（{ticker}）最近三个完整年度的利润表和股数资料。"
            "仅使用已经连接的结构化财务API，允许发起这些API请求，不做网页搜索或PDF下载。"
            "阅读返回的原始记录，解释收入、利润、股数各字段的含义、期间与缺失的元数据；没有依据不要自动认作归母利润或期末总股数。"
            "不做估值，不计算价格，也不要求你把缺少口径的数据强行入模。生成带原始记录引用的可下载研究笔记，然后说明哪些资料已取得、哪些还需要核对。")
        try:
            app.state.workspaces.message(workspace.workspace_id, ResearchTurn(content=content, time_budget_seconds=600))
        except Exception as exc:
            failure = f"{type(exc).__name__}: {str(exc).replace(key, '[redacted]')[:400]}"
        session = app.state.store.get_research(workspace.research_session_id)
        events = app.state.store.list_events(session.session_id)
        tools = [event.tool for event in events if event.type == "tool.completed"]
        documents = [document for document in session.documents if document.provider.startswith("infoway:")]
        blocks = [block for document in documents for block in app.state.store.research_blocks(session.session_id, document.file_id)]
        years = sorted({block["location"]["period_end"][:4] for block in blocks})
        complete_sources = all(hashlib.sha256(Path(app.state.store.get_file(document.file_id)["storage_path"]).read_bytes()).hexdigest() == document.sha256 for document in documents)
        notes = [item for item in app.state.store.list_artifacts(session.session_id) if item.get("kind") == "research_note"]
        for note in notes:
            metadata, raw = app.state.store.get_artifact(session.session_id, note["artifact_id"])
            if key.encode() in raw:
                raise RuntimeError("Credential leak in generated artifact")
            (args.directory / f"{ticker}-{metadata['number']}-{Path(metadata['filename']).name}").write_bytes(raw)
        checks = {"provider_sources_saved": len(documents) >= 2 and len(years) >= 3,
            "llm_read_sources": any(tool in tools for tool in ("read_file", "search_file", "read_financial_evidence")),
            "source_bytes_intact": bool(documents) and complete_sources,
            "research_note_saved": bool(notes), "no_valuation_or_web_search": not session.valuation_run_id and not session.search_history,
            "source_citations_present": bool(notes) and all(note.get("basis") == "source_analysis" and
                len({reference["block_id"].rsplit(":", 1)[0] for reference in note.get("evidence_refs", [])}) >= 2 for note in notes),
            "execution_finished": failure is None and session.last_issue is None}
        results.append({"company": company, "ticker": ticker, "checks": checks, "passed": all(checks.values()),
            "workspace_id": workspace.workspace_id, "model_calls": model.calls, "seconds": round(time.monotonic() - started),
            "years": years, "documents": len(documents), "blocks": len(blocks), "tools": tools, "failure": failure,
            "citation_counts": [len(note.get("evidence_refs", [])) for note in notes],
            "last_issue": session.last_issue.model_dump(mode="json") if session.last_issue else None,
            "answer": app.state.store.list_messages(session.session_id)[-1].content})
    report = {"model": models[0], "source_revision": revision, "source_revision_stable": revision == fingerprint(),
        "results": results, "scope": "真实API取数、LLM阅读与引用笔记；不是字段准确率或数值估值生产验收。"}
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if key in encoded:
        raise RuntimeError("Credential leak in evaluation report")
    (args.directory / "result.json").write_text(encoded, encoding="utf-8")
    print(json.dumps({"results": [{"company": row["company"], "passed": row["passed"], "checks": row["checks"]} for row in results],
        "source_revision_stable": report["source_revision_stable"]}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["source_revision_stable"] and all(row["passed"] for row in results) else 1)


if __name__ == "__main__":
    main()
