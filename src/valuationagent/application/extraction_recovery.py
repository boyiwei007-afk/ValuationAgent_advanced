"""Persistent, source-local recovery choices, not a second extraction engine."""
import hashlib
from pathlib import Path

from valuationagent.core.tools import canonical


def record_attempt(session, tool, arguments, result):
    facts = {fact.fact_id: fact for fact in session.facts}
    items = [*result.get("rows", []), *result.get("reviews", [])]
    selected_ids = [*arguments.get("fact_ids", []), *[review.get("fact_id") for review in arguments.get("reviews", [])],
                    *[item.get("fact_id") for item in items]]
    file_ids = ([arguments["file_id"]] if arguments.get("file_id") else
                sorted({facts[fact_id].block_id.rsplit(":", 1)[0] for fact_id in selected_ids if fact_id in facts}) or [""])
    for file_id in file_ids:
        selected = {fact_id for fact_id in selected_ids if fact_id in facts and facts[fact_id].block_id.rsplit(":", 1)[0] == file_id}
        local_items = [item for item in items if item.get("fact_id") in selected or not item.get("fact_id")]
        failures, failed_ids = [], set()
        for item in local_items:
            issues = ([item["error"]] if item.get("error") else [])
            issues.extend(warning for warning in item.get("warnings", []) if warning.startswith("MODEL_"))
            if item.get("semantic_review") == "needs_evidence":
                issues.append("SEMANTIC_REVIEW_NEEDS_EVIDENCE: " + "; ".join(item.get("warnings", [])))
            failures.extend(issues)
            if issues and item.get("fact_id"):
                failed_ids.add(item["fact_id"])
        unbound_failure = bool(any(item.get("error") for item in local_items) or result.get("ok") is False and not failed_ids)
        if result.get("ok") is False:
            failures.append((result.get("error") or {}).get("message", "提取未推进"))
        effective_view = result.get("view", arguments.get("view", "")) if tool == "read_file" else arguments.get("view", "")
        attempt = {"file_id": file_id, "tool": tool, "view": effective_view,
                   "page": arguments.get("page"), "signature": hashlib.sha256(canonical([tool, arguments]).encode()).hexdigest(),
                   "status": "failed" if failures else "completed", "issues": list(dict.fromkeys(failures))[:4],
                   "failed_fact_ids": sorted(failed_ids), "unbound_failure": unbound_failure}
        session.reading_attempts = [*session.reading_attempts[-127:], attempt]


def active_failure(attempt, facts):
    if attempt.get("unbound_failure"):
        return True
    if "failed_fact_ids" in attempt:
        return any(fact.fact_id in attempt["failed_fact_ids"] and fact.status != "rejected" and fact.warnings for fact in facts)
    if facts and attempt["tool"] in {"review_observations", "prepare_observation_review", "extract_observations"}:
        if all(fact.status == "rejected" or fact.status == "confirmed" and not fact.warnings for fact in facts):
            return not all(issue.startswith(("MODEL_", "SEMANTIC_REVIEW_")) for issue in attempt["issues"])
    return True


def recovery_plan(session, image_enabled=False, *, file_ids=None):
    files = []
    for document in session.documents:
        if file_ids is not None and document.file_id not in file_ids:
            continue
        attempts = [attempt for attempt in session.reading_attempts if attempt["file_id"] == document.file_id]
        if not attempts:
            continue
        historical_failures = [attempt for attempt in attempts if attempt["status"] == "failed"]
        facts = [fact for fact in session.facts if fact.block_id.rsplit(":", 1)[0] == document.file_id]
        failed = [attempt for attempt in historical_failures if active_failure(attempt, facts)]
        if not failed:
            continue
        strategies = []
        tried_views = {attempt["view"] for attempt in attempts if attempt["tool"] == "read_file"}
        suffix = Path(document.name).suffix.lower()
        if suffix == ".pdf":
            strategies.extend({"tool": "read_file", "arguments": {"file_id": document.file_id, "view": view},
                "instruction": "结合原文位置填写page；pdf_geometry按字形坐标解码，可分开文本层粘连列，不解释财务语义。不重复下载整份报告。"} for view in ("pdf_geometry", "pdf_plain", "pdf_layout") if view not in tried_views)
            if image_enabled and not any(attempt["tool"] == "view_pdf_page" for attempt in attempts):
                strategies.append({"tool": "view_pdf_page", "arguments": {"file_id": document.file_id}, "instruction": "填写目标page；看图识别布局，不把视觉猜测当作确定性核验。"})
        elif suffix == ".xlsx" and "sheet" not in tried_views:
            strategies.append({"tool": "read_file", "arguments": {"file_id": document.file_id, "view": "sheet"}, "instruction": "先inspect_file，再填写sheet/cell_range，保留原始单元格。"})
        elif suffix in {".txt", ".csv", ".tsv", ".json", ".html", ".htm", ".md"} and "raw_text" not in tried_views:
            strategies.append({"tool": "read_file", "arguments": {"file_id": document.file_id, "view": "raw_text"}, "instruction": "按原始行读取未重排的文本，定位后重提观察。"})
        latest_issues = list(dict.fromkeys(failed[-1]["issues"]))
        amount_failure = any("AMOUNT_" in issue for issue in latest_issues)
        reference_failure = any("ANCHOR_" in issue for issue in latest_issues)
        model_failure = any(issue.startswith("MODEL_") for issue in latest_issues)
        repair = {"tool": "extract_observations", "instruction": (
            "这是引用参数错误，不是文件缺失。value_ref/其他refs必须是anchors字典的键名（如rev_row），不是数值或引文。"
            "anchors可只填block_id、start_line/end_line，省略quote以直接引用保存原文；不要反复换文件。"
            if reference_failure else
            "这是已保存解释的模型类型问题，不是阅读器故障。按MODEL错误核对period_kind/日期/单位/口径等具体字段，"
            "用replaces更正当前观察；复核不能清除模型约束，不先重读相同页或重下载。"
            if model_failure else
            "金额定位失败不是资料缺失。raw_value填模型选定的完整数值；value_ref可直接引用整行，工具在行内定位该值。"
            "相同数值多次出现时指定value_occurrence；分隔正常的多列不要用value_segments。检查错误返回中的source_quote，修正参数而非重下载。"
            if amount_failure else "已有原文时优先修正锚点/期间/单位解释；用replaces更正原候选，不把解析错误描述为资料缺失。")}
        format_failure = any("AMOUNT_NOT_EXPLICIT" in issue or "AMOUNT_UNIT" in issue for issue in latest_issues)
        if model_failure or format_failure:
            strategies = [repair]
        elif amount_failure or reference_failure:
            strategies.insert(0, repair)
        else:
            strategies.append(repair)
        if any("PAGE_UNSUPPORTED" in issue or "请求的块内行窗口超出" in issue for issue in latest_issues):
            strategies = [{"tool": "read_file", "arguments": {"file_id": document.file_id, "view": "text"},
                           "instruction": "使用text视图并省略page及start_line/line_count，先读现有块；再按返回的block_id、total_lines与next_line定位。网页没有PDF页码，不把跨块偏移量当作块内行号。"}]
        elif any("VIEW_MISMATCH" in issue for issue in latest_issues):
            strategies = [{"tool": "read_file", "arguments": {"file_id": document.file_id, "view": "text"},
                           "instruction": "这是读取参数组合错误，不是解析失败；重读已存块时只填原block_id与行窗口，省略page并使用text视图。按页重新解码时填page/view但省略block_id。不要换来源或提取猜测值。"}]
        elif any("PAGE_NOT_FOUND" in issue for issue in latest_issues):
            strategies = [{"tool": "inspect_file", "arguments": {"file_id": document.file_id},
                           "instruction": "核对当前文件真实页数；block_id后缀不是页码。需要正文位置时用search_file查关键词再按返回的location.page读取，不反复加减猜页码。"}]
        if session.data_source_preference != "upload" and not format_failure:
            strategies.append({"tool": "search_sources", "instruction": "若上述视图仍不能消歧，定向查该字段/期间的其他来源，不重搜当前文件；保留来源等级。"})
        files.append({"file_id": document.file_id, "failure_count": len(failed), "historical_failure_count": len(historical_failures), "latest_issues": latest_issues,
                      "repeated_identical_failures": sum(attempt["signature"] == failed[-1]["signature"] for attempt in failed),
                      "next_choices": strategies})
    pending = [fact.fact_id for fact in session.facts if fact.status == "proposed"
               and (file_ids is None or fact.block_id.rsplit(":", 1)[0] in file_ids)
               and fact.verification.get("semantic_review", {}).get("status") in {"pending", "needs_evidence"}]
    return {"files": files[-8:], "pending_semantic_reviews": pending[:24],
            "instruction": "优先复核已定位观察；策略由主LLM选择，不自动扩大网络权限、不重置时间预算。确无可用策略时交付具体缺口，不要求用户反复说继续。"}
