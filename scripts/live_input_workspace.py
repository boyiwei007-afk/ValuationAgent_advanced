from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from decimal import Decimal
from pathlib import Path

from valuationagent.api.main import create_app
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.llm.client import OpenAICompatibleClient
from valuationagent.schemas.models import ModelConnectionInput
from valuationagent.schemas.research import ResearchTurn


CASES = [
    ("pe_anonymous", "仅按我给的数据进行估值试算：归母净利润10亿元，普通股5亿股，PE取20倍。不要联网，不需要找实际公司，生成可下载的估值报告。", "pe", Decimal(40)),
    ("ps_named", "给一家名为示例制造的非金融企业做情景估值，营业收入200亿元，普通股10亿股，PS指定3倍。只使用我给的数据，不要联网。算出结果并生成PDF报告。", "ps", Decimal(60)),
    ("pe_units", "为示例科技按我的数值试算：归母净利润25000万元，总普通股50000万股，市盈率按12倍。不要联网，请生成报告。", "pe", Decimal(6)),
]


def fingerprint():
    root = Path(__file__).resolve().parents[1]
    checksum = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]:
        checksum.update(str(path.relative_to(root)).encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def known_monetary_values(record, price):
    financials = record.request.financials
    values = {value for key, value in financials if isinstance(value, Decimal)}
    if price:
        values.update({price.per_share_value, price.equity_value})
        field = {"pe": "net_income_parent", "ps": "revenue", "ev_ebitda": "ebitda"}.get(str(price.method))
        shares = financials.diluted_shares or financials.common_shares
        numerator = getattr(financials, field, None) if field else None
        if numerator is not None and shares:
            values.add(numerator / shares)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="qwen36-teacher")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--case", choices=[case[0] for case in CASES])
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=False)
    app = create_app(directory / "workspace")
    model = OpenAICompatibleClient(ModelConnectionInput(base_url=args.base_url, model=args.model, api_key="EMPTY",
        tool_call_format="json_content", reasoning_protocol="chat_template", thinking=args.thinking, temperature=0.6, timeout_seconds=300))
    revision = fingerprint()
    results = []
    for name, content, method, expected in CASES:
        if args.case and name != args.case:
            continue
        workspace = app.state.workspaces.create(llm=model, run_policy="automatic", data_source_preference="web")
        started = time.monotonic()
        try:
            app.state.workspaces.message(workspace.workspace_id, ResearchTurn(content=content, time_budget_seconds=600))
            session = app.state.store.get_research(workspace.research_session_id)
            events = app.state.store.list_events(session.session_id)
            tools = [event.tool for event in events if event.type == "tool.completed"]
            record = app.state.store.get_run(session.valuation_run_id) if session.valuation_run_id else None
            price = next((item for item in record.result.relative if item.method == method and item.status == "success"), None) if record and record.result else None
            artifacts = app.state.store.list_artifacts(session.session_id)
            reports = [item for item in artifacts if item.get("numeric_result_available") and item.get("valuation_run_id") == session.valuation_run_id]
            for report in reports:
                metadata, content_bytes = app.state.store.get_artifact(session.session_id, report["artifact_id"])
                (directory / f"{name}-{metadata['number']}-{Path(metadata['filename']).name}").write_bytes(content_bytes)
            replay = replay_bundle(build_valuation_bundle(app.state.store, record)) if record and record.result else {}
            answer = app.state.store.list_messages(session.session_id)[-1].content
            known_amounts = known_monetary_values(record, price) if record else set()
            scales = {"亿元": Decimal(100000000), "万元": Decimal(10000), "元": Decimal(1)}
            unsupported_amounts = [match[0] for match in re.finditer(r"([0-9][0-9,.]*)\s*(亿元|万元|元)", answer)
                if Decimal(match[1].replace(",", "")) * scales[match[2]] not in known_amounts]
            row = {"case": name, "passed": bool(price and price.per_share_value == expected and reports and replay.get("passed")
                and not session.search_history and session.last_issue is None and record.request.analysis_basis == "user_scenario"
                and record.request.mode != "demo" and not record.result.effective_peers and not unsupported_amounts
                and record.request.financials.period_end is None and record.request.financials.common_shares_as_of is None and "sandbox:" not in answer),
                "tools": tools, "per_share_value": str(price.per_share_value) if price else None,
                "unmatched_money_mentions": unsupported_amounts,
                "reports": reports, "replay": replay, "last_issue": session.last_issue.model_dump(mode="json") if session.last_issue else None,
                "answer": answer,
                "workspace_id": workspace.workspace_id}
        except Exception as exc:
            row = {"case": name, "passed": False, "error": app.state.research._redact_text(str(exc))}
        row["seconds"] = round(time.monotonic() - started, 2)
        results.append(row)
        print(json.dumps({key: value for key, value in row.items() if key not in {"answer", "reports", "replay"}}, ensure_ascii=False), flush=True)
        (directory / "result.json").write_text(json.dumps({"model": args.model, "thinking": args.thinking, "source_digest": revision,
            "source_revision_stable": revision == fingerprint(), "scope": "用户输入情景；不是公开资料公司估值验收", "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        if name == "pe_anonymous" and row["passed"] and not args.baseline_only:
            baseline_id = record.run_id
            baseline_result = record.result.model_dump(mode="json")
            for followup_name, prompt in [
                ("explain_only", "只解释为什么会得到这个数值，暂时不要重算，不要联网，也不要修改数据。"),
                ("multiple_sensitivity", "基于当前结果，分析PE分别为15倍、25倍的敏感性，其他条件不变，不要联网，不要改变基准。"),
                ("correct_earnings", "把归母净利润更正为12亿元，仍是5亿股、20倍PE。不要联网，重新计算并生成报告。"),
            ]:
                before = len(app.state.store.list_events(session.session_id))
                try:
                    app.state.workspaces.message(workspace.workspace_id, ResearchTurn(content=prompt, time_budget_seconds=600))
                    session = app.state.store.get_research(session.session_id)
                    current = app.state.store.get_run(session.valuation_run_id)
                    followup_tools = [event.tool for event in app.state.store.list_events(session.session_id)[before:] if event.type == "tool.completed"]
                    passed = session.last_issue is None and not session.search_history
                    if followup_name == "explain_only":
                        passed = passed and current.run_id == baseline_id and not set(followup_tools) & {"calculate_valuation", "record_inputs", "analyze_sensitivity"}
                    elif followup_name == "multiple_sensitivity":
                        artifact = next(item for item in app.state.store.list_artifacts(session.session_id) if item["kind"] == "sensitivity_analysis")
                        metadata, payload = app.state.store.get_artifact(session.session_id, artifact["artifact_id"])
                        study = json.loads(payload)
                        passed = passed and current.run_id == baseline_id and [Decimal(item["per_share_value"]) for item in study["scenarios"]] == [30, 50]
                        (directory / "pe-sensitivity.json").write_bytes(payload)
                    else:
                        current_price = next(item for item in current.result.relative if item.method == "pe")
                        passed = passed and current.run_id != baseline_id and current_price.per_share_value == 48 and replay_bundle(build_valuation_bundle(app.state.store, current))["passed"]
                        passed = passed and any(item.get("numeric_result_available") and item.get("valuation_run_id") == current.run_id for item in app.state.store.list_artifacts(session.session_id))
                    passed = passed and app.state.store.get_run(baseline_id).result.model_dump(mode="json") == baseline_result
                    followup = {"case": followup_name, "passed": bool(passed), "tools": followup_tools,
                        "last_issue": session.last_issue.model_dump(mode="json") if session.last_issue else None,
                        "answer": app.state.store.list_messages(session.session_id)[-1].content}
                except Exception as exc:
                    followup = {"case": followup_name, "passed": False, "error": app.state.research._redact_text(str(exc))}
                results.append(followup)
                print(json.dumps({key: value for key, value in followup.items() if key != "answer"}, ensure_ascii=False), flush=True)
                (directory / "result.json").write_text(json.dumps({"model": args.model, "thinking": args.thinking, "source_digest": revision,
                    "source_revision_stable": revision == fingerprint(), "scope": "用户输入情景；不是公开资料公司估值验收", "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if all(row["passed"] for row in results) and revision == fingerprint() else 1)


if __name__ == "__main__":
    main()
