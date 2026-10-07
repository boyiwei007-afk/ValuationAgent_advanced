"""Rich presentation for the unified workspace, independent of agent decisions."""
import time
from concurrent.futures import ThreadPoolExecutor

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from valuationagent.application.result_views import input_source_label

console = Console(highlight=False, theme=Theme({
    "accent": "#5EEAD4", "muted": "#94A3B8", "good": "#6EE7B7",
    "warn": "#FBBF24", "bad": "#FB7185", "title": "bold #E2E8F0",
}))
BACKGROUND = "#E2E8F0 on #101B2D"
TOOL_LABELS = {
    "extract_observations": "按原文片段提交LLM解释", "prepare_observation_review": "读取原文复核包",
    "review_observations": "复核主体、金额与会计口径", "inspect_extraction_progress": "选择未尝试的取证策略",
    "list_files": "浏览统一文件工作区", "inspect_file": "检查文件结构与可用视图",
    "read_file": "按页或单元格读取原文", "view_pdf_page": "查看原始页图",
    "begin_file_task": "聚焦单份原文理解", "end_file_task": "返回估值主流程",
    "search_file": "全文检索并定位原文页码",
    "analyze_sensitivity": "按指定情景试算并保留原模型",
    "write_workspace_report": "生成可追溯报告", "write_research_note": "保存未审阅研究笔记",
    "list_artifacts": "查看文件交付清单", "read_artifact": "回读已生成文件",
    "planning": "整理任务与续做检查点", "inspect_requirements": "核对模型字段与年度覆盖",
    "search_sources": "检索公开披露", "fetch_search_source": "下载并保存原文",
    "read_document": "定位原文与年度列", "parse_document": "解析附件",
    "fetch_financial_history": "批量取得多年财务快照", "read_financial_evidence": "批量读取相关原文",
    "corroborate_facts": "交叉核对多源字段",
    "reject_candidates": "撤回错误候选并保留依据",
    "check_preparation": "检查确定性输入", "update_plan": "更新任务计划",
    "propose_forecast": "提出预测假设", "calculate_valuation": "冻结输入并计算",
    "read_valuation": "复核计算结果", "finish_response": "整理本轮结论", "saved": "进度已保存",
}
WORDMARK = (
    ' _    __      __            __  _             ___                    __',
    '| |  / /___ _/ /_  ______ _/ /_(_)___  ____  /   | ____ ____  ____  / /_',
    '| | / / __ `/ / / / / __ `/ __/ / __ \\/ __ \\/ /| |/ __ `/ _ \\/ __ \\/ __/',
    '| |/ / /_/ / / /_/ / /_/ / /_/ / /_/ / / / / ___ / /_/ /  __/ / / / /_',
    '|___/\\__,_/_/\\__,_/\\__,_/\\__/_/\\____/_/ /_/_/  |_\\__, /\\___/_/ /_/\\__/',
    '                                                /____/',
)


def panel(body, title="", **kwargs):
    return Panel(body, title=Text(title, style="#94A3B8"), border_style="#33465F",
                 style=BACKGROUND, box=box.ROUNDED, padding=(1, 2), **kwargs)


class Welcome:
    def __rich_console__(self, target, options):
        width = min(options.max_width, 116)
        logo = Group(*(Text(line, style="#5EEAD4", no_wrap=True) for line in WORDMARK)) if width >= 94 and target.height >= 28 else Text("V / A\nValuationAgent", justify="center", style="bold #5EEAD4")
        workflow = "01  任务与计划    →  02  多年证据\n03  事实核验      →  04  假设与计算\n05  风险复核      →  06  报告与复算" if width >= 56 else "01  任务与计划\n02  多年证据\n03  事实核验\n04  假设与计算\n05  风险复核\n06  报告与复算"
        yield Align.center(panel(Group(
            Align.center(logo), Text(""),
            Text("Agent-powered valuation research", justify="center", style="bold #E2E8F0"),
            Text("自由对话 · 原文取证 · 确定性估值", justify="center", style="#94A3B8"),
            Text(""), Text("WORKFLOW  /  可追溯工作台", style="#5EEAD4"),
            Text(workflow, style="#CBD5E1"),
            Text(""), Text("直接描述目标即可开始。模型和搜索在终端内配置。", style="#94A3B8"),
        ), "Welcome to ValuationAgent", width=width))


def welcome():
    return Welcome()


def conversation(content, role="assistant"):
    body = Markdown(content, hyperlinks=False) if role == "assistant" else Text(content)
    return panel(body, "ValuationAgent · 回答与依据" if role == "assistant" else "你 · 目标与补充")


def decision_view(decision):
    rows = [Text(decision["question"], style="bold #E2E8F0")]
    for index, option in enumerate(decision["options"], 1):
        rows.extend([Text(f"{index} / {chr(64 + index)}  {option['label']}", style="#5EEAD4"),
                     Text(option["description"], style="#94A3B8")])
    rows.append(Text("输入序号或字母选择，可追加说明；也可以直接输入自己的方案。", style="#FBBF24"))
    return panel(Group(*rows), "方案选择 · 保留自由对话")


def workbench(snapshot, elapsed=0):
    research = snapshot.get("research", {})
    session = research.get("session", {})
    execution = snapshot.get("execution") or research.get("execution", {})
    facts = session.get("facts", [])
    docs = session.get("documents", [])
    leads = sum(doc.get("provenance_type") in {"search_snippet", "official_index"} for doc in docs)
    verified = sum(fact.get("status") == "confirmed" and not fact.get("warnings") for fact in facts)
    pending = sum(fact.get("status") == "proposed" for fact in facts)
    stage = execution.get("stage", "")
    heading = Text(TOOL_LABELS.get(stage, stage or "等待你的下一条消息"), style="bold #5EEAD4")
    if elapsed:
        heading.append(f"  ·  {int(elapsed)}s", style="#94A3B8")
    bound = snapshot.get("research_plan", {}).get("evidence_counts", {}).get("observations_verified", 0)
    rows = [heading, Text(f"{len(docs) - leads} 份原文 / {leads} 条线索  ·  {bound} 原文已绑定 / {verified} 通过字段准入 / {pending} 候选待处理", style="#CBD5E1")]
    if dataset := session.get("input_dataset"):
        superseded = {key for item in dataset["records"] for key in item.get("supersedes", [])}
        active = [item for item in dataset["records"] if item["input_id"] not in superseded]
        label = "模型输入工作区 · 来源数据与用户假设分别保留" if any(item["source"]["kind"] != "user" for item in active) else "用户输入情景 · 未经外部事实核验，无需先搜索年报"
        rows.append(Text(f"{label} · {len(active)} 项有效输入", style="#5EEAD4"))
        rows.extend(Text(f"{item.get('entity_ticker') or item['entity']} · {item['metric']}：{item['original_amount']} · {item.get('period_end') or item.get('as_of') or '期间未指定'} · {input_source_label(item['source'])}", style="#CBD5E1") for item in active[:8])
        for peer in dataset.get("comparables", {}).values():
            rows.append(Text(f"可比 {peer['name']}（{peer['ticker']}） · {'候选' if peer['enabled'] else '已剔除'} · Agent判断待独立复核：{peer['rationale']}", style="#94A3B8"))
    counts = snapshot.get("research_plan", {}).get("evidence_counts", {})
    if counts.get("semantic_review_pending") or counts.get("semantic_review_supported"):
        rows.append(Text(f"LLM语义复核：{counts.get('semantic_review_pending', 0)} 待复核 / {counts.get('semantic_review_supported', 0)} 已支持；原文定位不等于语义正确或独立审计。", style="#FBBF24"))
    for recovery in snapshot.get("research_plan", {}).get("extraction_recovery", {}).get("files", [])[-2:]:
        rows.append(Text(f"读取恢复：{recovery['file_id']} · {recovery['failure_count']} 次失败 · 换视图或补证，不重复下载。", style="#94A3B8"))
    corroborated = sum(fact.get("status") == "confirmed" and fact.get("verification", {}).get("source_assessment", {}).get("admission") == "corroborated_draft" for fact in facts)
    if corroborated:
        rows.append(Text(f"其中 {corroborated} 项为跨源一致的草案输入；保留第三方来源限制，不等于官方核验。", style="#FBBF24"))
    if version := snapshot.get("runtime", {}).get("agent_version"):
        rows.append(Text(version, style="#94A3B8"))
    plan = snapshot.get("plan") or session.get("plan", [])
    for step in plan:
        marker, color = {"completed": ("✓", "#6EE7B7"), "in_progress": ("◉", "#5EEAD4")}.get(step.get("status"), ("○", "#94A3B8"))
        rows.append(Text(f"{marker}  {step['title']}", style=color))
    coverage = snapshot.get("research_plan", {}).get("annual_coverage", [])
    if coverage:
        rows.append(Text("年度覆盖（候选 / 原文绑定 / 字段准入；非模型完整度）", style="#94A3B8"))
        rows.append(Text("  /  ".join(f"{row['year']}: {len(row.get('candidate_metrics', []))}/{len(row.get('bound_metrics', []))}/{len(row['confirmed_metrics'])}" for row in coverage), style="#CBD5E1"))
    for group in snapshot.get("research_plan", {}).get("table_repairs", [])[:2]:
        rows.append(Text(f"共性取证问题影响 {group['affected_count']} 条：{group['issue']}；补读原文或换视图后修正解释。", style="#FBBF24"))
    for method in snapshot.get("research_plan", {}).get("method_readiness", []):
        label = "输入准备通过" if method["status"] == "inputs_ready" else "输入未就绪"
        rows.append(Text(f"{method['method'].upper()} · {label} · {method['reason'][:120]}", style="#94A3B8"))
    events = research.get("events", [])[-5:]
    if events:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(ratio=3)
        table.add_column(justify="right")
        for event in events:
            table.add_row(Text(TOOL_LABELS.get(event.get("tool"), event.get("summary", ""))[:100]), Text(event.get("status", ""), style="#94A3B8"))
        rows.extend([Text("最近动作 · 原始记录可审计", style="#94A3B8"), table])
    if session.get("resume_context"):
        rows.append(Text("续做检查点已保存；下轮优先重检事实与未覆盖年度。", style="#FBBF24"))
    return panel(Group(*rows), "WORKSPACE / 流程工作台")


def execute_with_display(service, workspace_id, turn, target=None):
    target = target or console
    if not target.is_terminal:
        return service.message(workspace_id, turn)
    session_id = service.get(workspace_id).research_session_id
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(service.message, workspace_id, turn)
        try:
            with Live(console=target, refresh_per_second=4, transient=True) as live:
                while not future.done():
                    session = service.store.get_research(session_id)
                    state = {"research": {"session": session.model_dump(mode="json"),
                             "execution": service.research.execution_state(session_id),
                             "events": service.store.event_page(session_id, limit=5)["events"]}}
                    live.update(workbench(state, time.monotonic() - started))
                    time.sleep(.5)
                return future.result()
        except KeyboardInterrupt:
            service.research.cancel_turn(session_id)
            target.print("正在停止当前工具并保存进度…", style="yellow")
            future.result()
            raise
