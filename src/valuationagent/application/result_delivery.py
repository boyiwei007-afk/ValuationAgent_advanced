import re
import unicodedata
from decimal import Decimal

from valuationagent.application.result_views import money


DELIVERY_SCHEMA = "frozen-valuation-delivery-v1"
METHOD_NAMES = {"dcf": "DCF", "pe": "PE", "ps": "PS", "ev_ebitda": "EV/EBITDA"}


def method_completion(record):
    requested = list(record.request.requested_methods or record.request.methods)
    completed = []
    if record.result:
        if record.result.dcf and record.result.dcf.status == "success":
            completed.append("dcf")
        completed.extend(item.method for item in record.result.relative if item.status == "success")
    remaining = {method: record.request.excluded_methods.get(method, "尚无本方法成功的确定性计算结果")
        for method in requested if method not in completed}
    return {"requested_methods": requested, "completed_methods": completed, "remaining_methods": remaining,
        "all_requested_methods_completed": not remaining,
        "instruction": ("本次仅部分方法完成；先核对剩余方法缺项是否已在用户原话或已有来源中提供，补录后重新check_preparation/calculate_valuation并更新报告。"
            "不重录已成功输入、不删减原请求方法、不心算补齐。确实不可得用insufficient_data/needs_input，需续做用checkpoint，不能标为全部完成。"
            if remaining else "所选方法均已完成确定性计算；报告与本次run_id匹配后交付。该状态不等于输入已独立审计或预测准确。")}


def bridge_amount(key, value, currency):
    return f"{Decimal(str(value)):,.0f} 股" if key == "diluted_or_common_shares" else money(value, currency)


def label(value):
    return str(value or "未指定").replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def price(value, currency):
    return "未计算" if value is None else f"{Decimal(str(value)):,.4f} {'元' if currency == 'CNY' else currency}/股"


def qualitative_only(text):
    normalized = unicodedata.normalize("NFKC", text)
    return not re.search(r"\d|https?://|sandbox:|file:|/api/|www\.|百分之|"
        r"[零〇一二三四五六七八九十百千万两几数半点]+\s*(?:亿元|万元|元|亿股|万股|股|倍|亿|万)|"
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\s+(?:yuan|dollars?|shares?|times|percent)\b",
        normalized, re.IGNORECASE)


def render_delivery(store, session, record, workspace, artifacts, commentary):
    result, request = record.result, record.request
    completion = method_completion(record)
    financials = result.effective_financials or request.financials
    reports = [item for item in artifacts.values() if item.get("kind") == "result_report"
        and item.get("numeric_result_available") and item.get("valuation_run_id") == record.run_id]
    latest = {}
    for item in sorted(reports, key=lambda item: str(item.get("created_at", ""))):
        latest[item["filename"]] = item
    title = "估值草案" if completion["all_requested_methods_completed"] else "部分估值草案"
    lines = [f"## {label(request.company.name or request.company.ticker)} · {title}", "",
        f"估值日：{request.valuation_date}；财务基期：{financials.period_end or '用户未指定年度'}。", "",
        "以下数值直接来自已完成的冻结计算，不由模型重抄或重新估算。",
        "", "| 方法 | 点估值 | 每股区间 | 口径 |", "| --- | --- | --- | --- |"]
    methods = []
    if result.dcf and result.dcf.status == "success":
        methods.append(("dcf", result.dcf, "基准情景；区间为模型情景结果"))
    for item in result.relative:
        if item.status == "success":
            basis = f"用户指定倍数 {item.selected_multiple}" if item.valuation_basis == "explicit_multiple" else f"{item.statistic}；有效样本 {item.sample_size} 家"
            methods.append((item.method, item, basis))
    for method, item, basis in methods:
        lines.append(f"| {METHOD_NAMES[method]} | {price(item.per_share_value, result.currency)} | "
            f"{price(item.range_low, result.currency)} — {price(item.range_high, result.currency)} | {label(basis)} |")
    lines.extend(["", "各方法独立展示，不机械平均；相对估值区间不冒充DCF情景区间。"])
    if result.dcf and result.dcf.status == "success":
        lines.extend([f"DCF企业价值：{money(result.dcf.enterprise_value, result.currency)}；普通股权益价值：{money(result.dcf.equity_value, result.currency)}。",
            "企业价值到权益的逐项桥接：" + "；".join(f"{key}={bridge_amount(key, value, result.currency)}" for key, value in result.dcf.bridge.items())])
    peers = [peer for peer in request.peers if any(getattr(peer, field) is not None for field in ("pe", "ps", "ev_ebitda"))]
    if peers:
        lines.extend(["", "### 冻结可比样本", "", "| 公司 | 财务期 / 行情日 | 市值 | PE | PS | EV/EBITDA |", "| --- | --- | --- | --- | --- | --- |"])
        for peer in peers[:12]:
            multiples = [f"{getattr(peer, field):.4f}" if getattr(peer, field) is not None else "未采用" for field in ("pe", "ps", "ev_ebitda")]
            lines.append(f"| {label(peer.name)}（{label(peer.ticker)}） | {peer.financial_period_end or '未指定'} / {peer.as_of_date or '未指定'} | {money(peer.market_cap, result.currency)} | " + " | ".join(multiples) + " |")
        lines.extend(["", "倍数及样本身份取自冻结请求；业务可比性仍是研究判断，供应商数据不是独立审计。完整输入及股份定价口径见报告。"])
    if result.analysis_basis == "user_scenario":
        lines.append("这是用户提供数据及假设的情景计算，不是现实公司的核验结论或投资建议。")
    if result.warnings:
        lines.extend(["", "### 主要限制", ""])
        lines.extend(f"- {label(warning)}" for warning in result.warnings[:6])
        if len(result.warnings) > 6:
            lines.append("其余限制详见完整报告。")
    usable_commentary = qualitative_only(commentary)
    if commentary.strip() and usable_commentary:
        lines.extend(["", "### Agent 定性说明（非独立核验）", "", commentary])
    elif commentary.strip():
        lines.append("模型重述中含未绑定数值或链接，未作为最终定量结论采用；以本页计算表和原始报告为准。")
    links = []
    for item in latest.values():
        store.get_artifact(session.session_id, item["artifact_id"])
        url = f"/api/workspaces/{workspace.workspace_id}/artifacts/{item['artifact_id']}"
        links.append({"artifact_id": item["artifact_id"], "filename": item["filename"], "sha256": item["sha256"], "url": url})
    if links:
        lines.extend(["", "### 下载报告", ""])
        lines.extend(f"- [{label(item['filename'])}]({item['url']})" for item in links)
    else:
        lines.append("当前没有与该计算版本匹配的数值报告文件；不编造下载链接。")
    lines.append(f"计算版本：`{record.run_id}`；输入哈希：`{record.input_hash}`。")
    selection = request.baseline_selection
    if selection.get("policy") == "user_selected":
        lines.append("基期采用用户明确指定的历史年度，不声称最新可得；原话与选择依据见报告。")
    return "\n".join(lines), {"schema": DELIVERY_SCHEMA, "run_id": record.run_id,
        "method_completion": completion,
        "input_hash": record.input_hash, "narrative_status": "qualitative_only" if usable_commentary else "unbound_not_published",
        "methods": [method for method, _, _ in methods], "reports": links, "baseline_selection": selection,
        "limitation": "确定性绑定只证明展示与冻结计算一致，不证明原始资料、业务判断或预测准确。"}
