from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from valuationagent.api.main import create_app
from valuationagent.application.turn_control import prepare_turn
from valuationagent.llm.client import OpenAICompatibleClient
from valuationagent.schemas.models import ModelConnectionInput
from valuationagent.schemas.research import ResearchTurn


CASES = [
    ("explain", "先解释DCF和PE的区别，不要继续估值，不要联网。", set(), {"network", "calculate", "sensitivity", "inputs"}),
    ("given_inputs", "仅按我给的数据试算：归母净利润10亿元，普通股5亿股，PE取20倍。不要联网。", {"inputs", "calculate"}, {"network"}),
    ("extract", "请提取这个Excel里的财务数据，但先不要计算估值，也不要联网。", {"files", "inputs"}, {"network", "calculate"}),
    ("api_read", "读取当前已连接的财务API中最近三个完整年度的报表，允许这些API请求，但不搜索网页或下载PDF；先不估值。", {"network", "structured_data", "files"}, {"web", "calculate"}),
    ("api_only", "只用Infoway API获取这家公司的收入、利润和股数并写研究笔记，不要网页检索，也不要计算价格。", {"network", "structured_data", "artifacts"}, {"web", "calculate"}),
    ("sensitivity", "基于已经完成的估值，分析WACC从7%到10%的敏感性，不重新搜集资料。", {"sensitivity"}, {"network", "inputs"}),
    ("report", "把已经算好的估值导出报告，不要改参数或重新计算。", {"artifacts"}, {"network", "calculate", "inputs"}),
    ("change_method", "把后续估值方法改为PE，暂时不要计算，也不要联网。", {"task"}, {"network", "calculate"}),
    ("pause", "暂停估值，我先考虑一下，不要继续执行。", set(), {"network", "calculate", "inputs", "files"}),
    ("normal_value", "请估值青岛啤酒600600，用公开资料并生成报告。", {"network", "calculate", "artifacts"}, set()),
    ("unspecified_company", "请估值一家非金融A股公司，用公开资料并生成报告。", set(), {"network", "calculate"}),
]


def fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="qwen36-teacher")
    args = parser.parse_args()
    directory = Path(args.directory)
    if directory.exists():
        raise SystemExit("Use a new acceptance directory; do not overwrite previous evidence.")
    directory.mkdir(parents=True)
    app = create_app(directory / "workspace")
    started_revision = fingerprint()
    service = app.state.research
    model = OpenAICompatibleClient(ModelConnectionInput(base_url=args.base_url, model=args.model, api_key="EMPTY",
        tool_call_format="json_content", reasoning_protocol="chat_template", thinking="disabled", timeout_seconds=180))
    results = []
    for name, content, required, forbidden in CASES:
        workspace = app.state.workspaces.create(llm=model, run_policy="automatic", data_source_preference="web")
        session = app.state.store.get_research(workspace.research_session_id)
        session.pending_action = "valuation"
        session.draft.company = "Synthetic scope for instruction evaluation"
        app.state.store.add_message(session.session_id, "user", content, "agent")
        started = time.monotonic()
        try:
            control = prepare_turn(service, session, model)
            effects = set(control["effects"])
            passed = required <= effects and not effects & forbidden
            row = {"case": name, "passed": passed, "control": control,
                "missing": sorted(required - effects), "unexpected": sorted(forbidden & effects)}
        except Exception as exc:
            row = {"case": name, "passed": False, "error": service._redact_text(str(exc))}
        row["seconds"] = round(time.monotonic() - started, 2)
        results.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "control"}, ensure_ascii=False), flush=True)

    workspace = app.state.workspaces.create(llm=model, data_source_preference="web")
    session = app.state.store.get_research(workspace.research_session_id)
    for name, content, allowed in [
        ("persistent_deny", "估值青岛啤酒600600，但只用我提供的材料，不要联网。", False),
        ("continue_keeps_deny", "继续", False),
        ("explicit_permission_change", "现在允许联网，搜索尚缺的财务资料。", True),
        ("deny_again", "后续只使用已有数据，不允许再访问网络。", False),
        ("natural_permission_change", "刚才的联网限制取消，可以查外部资料补齐缺口了。", True),
    ]:
        app.state.store.add_message(session.session_id, "user", content, "agent")
        try:
            control = prepare_turn(service, session, model)
            row = {"case": name, "passed": control["permissions"]["network"] is allowed
                and (allowed or "network" not in control["effects"]), "control": control}
        except Exception as exc:
            row = {"case": name, "passed": False, "error": service._redact_text(str(exc))}
        results.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "control"}, ensure_ascii=False), flush=True)

    workspace = app.state.workspaces.create(llm=model, run_policy="automatic", data_source_preference="web")
    session = app.state.store.get_research(workspace.research_session_id)
    session.pending_action = "valuation"
    app.state.store.save_research(session)
    app.state.workspaces.message(workspace.workspace_id,
        ResearchTurn(content="先只解释PE估值的含义，不要联网，不要计算，也不要读文件。", time_budget_seconds=180))
    current = app.state.store.get_research(session.session_id)
    executed = [event.tool for event in app.state.store.list_events(session.session_id) if event.type == "tool.completed"]
    answer = app.state.store.list_messages(session.session_id)[-1].content
    forbidden_tools = {"search_sources", "calculate_valuation", "read_file", "analyze_sensitivity", "update_task"}
    results.append({"case": "full_runtime_explain", "passed": current.last_issue is None and not set(executed) & forbidden_tools
        and "finish_response" in executed and not app.state.store.list_runs(), "tools": executed, "answer": answer})
    ended_revision = fingerprint()
    output = {"model": args.model, "base_url": args.base_url, "thinking": "disabled", "tool_call_format": "json_content",
        "tested_at": datetime.now(timezone.utc).isoformat(), "source_digest": started_revision,
        "source_revision_stable": started_revision == ended_revision,
        "passed": all(row["passed"] for row in results) and started_revision == ended_revision,
        "scope": "Instruction and execution-control acceptance only; not real-company valuation acceptance.", "results": results}
    (directory / "result.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": output["passed"], "result": str(directory / "result.json")}), flush=True)
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
