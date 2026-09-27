"""Deterministic, versioned outcomes even when no defensible price exists.

The document records a completed assessment, not a successful valuation. It
never promotes a search snippet, candidate, or illustrative assumption to fact.
"""
import hashlib
import html
import json
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from decimal import Decimal

from valuationagent.application.research_valuation import METRIC_LABELS, _period
from valuationagent.application.valuation_plan import preview_session, _build_for_methods
from valuationagent.schemas.models import required_financial_metrics

METHODS = {"dcf": "DCF 现金流折现", "pe": "P/E 市盈率", "ps": "P/S 市销率", "ev_ebitda": "EV/EBITDA 企业价值倍数"}
STATUS = {
    "insufficient_data": "数据不足，已形成说明报告",
    "awaiting_review": "方案待确认，已形成阶段报告",
    "ready": "具备提交条件，尚未执行计算",
    "calculating": "正式计算进行中",
    "review_required": "计算需复核，已保留诊断",
    "valued": "估值已完成",
}
FORMULAS = {
    "dcf": "FCFF = EBIT × (1 - 税率) + 折旧摊销 - 资本开支 - 营运资本增加额；逐期折现并加入终值，再调整净债务。",
    "pe": "每股价值 = 归母净利润 × 同期 FY 可比市盈率 / 普通股股数。",
    "ps": "每股价值 = 营业收入 × 同期 FY 可比市销率 / 普通股股数。",
    "ev_ebitda": "每股价值 = (EBITDA × 可比倍数 + 现金及非经营资产 - 有息债务) / 普通股股数。",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_result_document(service, session):
    selected = list(session.draft.methods)
    preview = preview_session(session)
    methods = []
    for method in selected or METHODS:
        try:
            request = _build_for_methods(preview, service.valuation_assembler, [method])
            reason = "所需输入已具备；正式计算仍需完成确认与财务审核。"
            status = "ready"
            if request.data_source == "ticker":
                reason = "已具备在线取数条件；数据尚未取得，不能据此宣称估值已完成。"
                status = "pending_retrieval"
        except ValueError as exc:
            reason, status = str(exc), "missing"
        methods.append({"method": method, "label": METHODS[method], "status": status,
                        "reason": reason, "selected": method in selected,
                        "required_inputs": [METRIC_LABELS.get(key, key) for key in sorted(required_financial_metrics([method]))],
                        "formula": FORMULAS[method]})
    executable = [m["method"] for m in methods if m["selected"] and m["status"] == "ready"]
    record = None
    if session.valuation_run_id:
        try:
            record = service.store.get_run(session.valuation_run_id)
        except KeyError:
            pass
    result = record.result if record and str(record.status) in {"completed", "completed_with_warnings"} else None
    numeric = bool(result and (result.dcf or any(r.status == "success" for r in result.relative)))
    if numeric:
        status = "valued"
        conclusion = result.executive_summary
    elif record and str(record.status) in {"created", "running"}:
        status = "calculating"
        conclusion = "正式估值任务已提交，计算尚未完成；本报告不含未完成的数值结论。"
    elif record:
        status = "review_required"
        issue = record.review or record.error
        conclusion = f"正式计算未形成有效估值。{issue.get('message', '') if issue else '请检查计算任务中的复核说明。'}"
    elif executable:
        status = "awaiting_review" if session.question or any(f.status == "proposed" and not f.warnings for f in session.facts) else "ready"
        conclusion = "以下方法已具备提交条件：" + " / ".join(METHODS[m] for m in executable) + "。尚未生成数值估值，确认方案后进入确定性计算。"
    else:
        status = "insufficient_data"
        conclusion = "目前没有足够的可核验输入支持数值估值。已交付方法适用性、资料缺口与执行记录；价格区间及数值敏感性标记为未计算，不以零值或模型猜测代替。"
    counts = {
        "verified": sum(f.status == "confirmed" and not f.warnings for f in session.facts),
        "staged": sum(f.status == "proposed" and not f.warnings for f in session.facts),
        "needs_repair": sum(f.status != "rejected" and bool(f.warnings) for f in session.facts),
        "sources": len(session.documents),
        "searches": len(session.search_history),
    }
    facts = [{"metric": f.metric, "label": METRIC_LABELS.get(f.metric, f.metric), "value": f.normalized_value,
              "period": f.period, "unit": f.unit, "raw_value": f.raw_value, "scope": f.scope,
              "source_id": f.block_id, "source_url": f.source_url, "sha256": f.source_sha256}
             for f in session.facts if f.status == "confirmed" and not f.warnings and f.role == "historical"]
    sources = []
    for doc in session.documents:
        # Only the source manifest is copied; large raw blocks stay in the audit package.
        location = service.store.research_source_location(session.session_id, doc.file_id)
        sources.append({**doc.model_dump(mode="json"), "source_url": location.get("source_url") or location.get("url") or "",
                        "published_at": location.get("published_at")})
    gaps = list(dict.fromkeys([m["reason"] for m in methods if m["selected"] and m["status"] == "missing"] + session.gaps))
    steps = (["确认页面上的估值方案，系统将继续计算区间、敏感性并生成完整报告。"] if executable else
             ["可直接下载本说明报告；已有资料和检索记录会保留。", "补齐方法表中列出的关键缺项后继续；只需补缺失项，无需重新上传全部资料。"])
    if not session.draft.company and not session.draft.ticker:
        steps.insert(0, "提供公司名称或代码，系统才能定位对应的正式披露。")
    if not service._clients.get(session.session_id) and not numeric:
        steps.append("连接推理模型后，可自动理解需求、定位公开资料并提取字段；报告下载本身不依赖模型服务。")
    if session.data_source_preference == "upload" and not session.documents:
        steps.append("当前选择了上传模式；若不提供文件，可切换为公开资料检索。")
    if session.question:
        steps.append(session.question.title)
    limitations = [
        "结论只覆盖所列公司、估值日和方法；资料缺失不等于相应指标为零。",
        "搜索结果属于来源线索；只有通过来源、年度、单位与口径核验并确认的字段才能进入计算。",
        "增长率、利润率、WACC和永续增长率属于估计或观点，不能替代历史收入、债务、现金和股数。",
        "本系统面向非金融企业；银行、保险等需专门估值模型。扫描件OCR及复杂表格仍需补充可读取资料。",
    ]
    if result:
        limitations.extend(result.warnings)
    issue = session.last_issue
    # A source-risk inventory is a diagnostic, not a promoted financial fact.
    # Before a final baseline exists, use the latest eligible candidate year
    # solely to locate risk-bearing rows and disclose that selection explicitly.
    periods = [_period(f.period) for f in session.facts if f.role == "historical" and f.status != "rejected"]
    periods = [period for period in periods if period and period.month == 12 and period.day == 31
               and (not session.draft.valuation_date or period <= session.draft.valuation_date)]
    source_risks = service.valuation_assembler.source_risks(session, max(periods)) if periods else None
    if source_risks:
        source_risks["baseline_selection"] = "最新候选年度，仅用于原文风险定位；不表示已确认财务基期或已完成估值。"
    document = {
        "schema": "valuation-outcome-v1", "source_revision": session.revision,
        "session_id": session.session_id, "generated_at": str(session.updated_at),
        "company": session.draft.company or session.draft.ticker or "研究对象待确定",
        "ticker": session.draft.ticker, "valuation_date": str(session.draft.valuation_date or "待确定"),
        "status": status, "status_label": STATUS[status], "conclusion": conclusion,
        "numeric_result_available": numeric, "counts": counts, "methods": methods,
        "valuation_run_id": session.valuation_run_id,
        "valuation_run_updated_at": str(record.updated_at) if record else None,
        "valuation_result": result.model_dump(mode="json") if numeric else None,
        "verified_facts": facts, "sources": sources,
        "unconfirmed_candidates": [{"metric": f.metric, "period": f.period, "raw_value": f.raw_value, "unit": f.unit,
                                    "source_id": f.block_id, "warnings": f.warnings,
                                    "state": "待补证，不可计算" if f.warnings else "通过来源校验，尚待最终方案确认"}
                                   for f in session.facts if f.status == "proposed"],
        "research_summary": session.summary,
        "pending_review": session.question.model_dump(mode="json") if session.question else None,
        "forecast_assumptions": session.forecast_proposal.model_dump(mode="json") if session.forecast_proposal else None,
        "gaps": gaps, "next_steps": steps if not numeric else ["使用完整估值任务导出 Excel 底稿、PDF 报告和 JSON 离线复算包。"],
        "limitations": list(dict.fromkeys(limitations)), "searches": session.search_history,
        "latest_issue": {"code": issue.code, "message": issue.message, "status": issue.status} if issue else None,
        "source_risk_review": source_risks,
        "model": {"provider": session.model_provider, "name": session.model_name, "prompt_version": session.prompt_version},
        "sensitivity_status": "详见正式结果" if numeric else "未计算：尚无有效数值基准；不生成虚构价格敏感性表。",
    }
    document = service._redact_value(document)
    document["report_id"] = _digest(document)
    return document


def ensure_result_document(service, session=None):
    if session is None:
        raise ValueError("报告需要研究会话")
    previous = service.store.research_report(session.session_id)
    run_stamp = None
    if session.valuation_run_id:
        try:
            run_stamp = str(service.store.get_run(session.valuation_run_id).updated_at)
        except KeyError:
            pass
    if previous and previous["source_revision"] == session.revision and previous.get("valuation_run_updated_at") == run_stamp:
        if _digest({key: value for key, value in previous.items() if key != "report_id"}) != previous["report_id"]:
            raise ValueError("已保存报告的完整性校验失败，请管理员检查存储；未交付可能损坏的报告。")
        return previous
    document = build_result_document(service, session)
    summary = {key: document[key] for key in ("report_id", "source_revision", "generated_at", "company", "status", "status_label", "conclusion", "numeric_result_available", "counts", "valuation_run_id")}
    if service.store.save_research_report(session.session_id, document, summary):
        service.store.append_event(session.session_id, type="report.generated", stage="reporting", status="completed",
            summary="已生成估值结果或数据缺失说明报告", payload={"report_id": document["report_id"], "source_revision": session.revision, "outcome": document["status"]})
    return document


def _forecast_lines(proposal):
    """Readable paths in PDF/HTML; the full precision lives in the JSON."""
    if not proposal:
        return ["暂无已提出的预测假设。未擅自填入增长、利润率或折现率。"]
    inputs = proposal["inputs"]
    labels = {"pessimistic": "审慎", "base": "基准", "optimistic": "乐观"}
    lines = [f"状态：{'已确认' if proposal['status'] == 'confirmed' else '待集中确认'}。以下均为估计或观点，不是历史事实。",
             proposal["rationale"]]
    for key, title in (("revenue_growth_scenarios", "收入增长率"),
                       ("ebit_margin_scenarios", "EBIT利润率")):
        for scenario, label in labels.items():
            path = (inputs.get(key) or {}).get(scenario)
            if path:
                values = "、".join(f"{Decimal(str(value)) * 100}%" for value in path)
                lines.append(f"{title}·{label}（第1至{len(path)}年）：{values}")
    lines.append(f"WACC：{Decimal(str(inputs['wacc'])) * 100}%；永续增长率：{Decimal(str(inputs['terminal_growth'])) * 100}%")
    lines.extend("风险：" + risk for risk in proposal["risks"])
    return lines


def document_sections(doc):
    """One content model for HTML and PDF so their conclusions cannot drift."""
    sections = [
        ("结论与适用范围", [doc["conclusion"], f"公司：{doc['company']}　代码：{doc['ticker'] or '待确定'}　估值日：{doc['valuation_date']}"]),
        ("已核验事实", [f"{f['period']} · {f['label']}：{f['raw_value']} {f['unit']}；来源 {f['source_id']}" for f in doc["verified_facts"]] or ["暂无已确认的历史财务事实。暂存候选及搜索摘要不作为事实列入。"]),
        ("方法适用性与最小输入", [f"{m['label']}（{'已选' if m['selected'] else '仅供评估'}）：{m['reason']}\n必需输入：{'、'.join(m['required_inputs'])}\n{m['formula']}" + ("\n相对估值还需至少3家同日、FY口径可比公司倍数；优先5家。" if m['method'] != 'dcf' else "\n自动预测需连续历史；历史不足时可提出有依据的十年三情景假设。") for m in doc["methods"]]),
    ]
    if doc["valuation_result"]:
        result = doc["valuation_result"]
        values = []
        if result.get("dcf"):
            dcf = result["dcf"]
            values.append(f"DCF：每股基准 {dcf['per_share_value']}；区间 {dcf['range_low']} - {dcf['range_high']} {result['currency']}/股。")
        values += [f"{r['method'].upper()}：每股基准 {r['per_share_value']}；区间 {r['range_low']} - {r['range_high']} {result['currency']}/股。" for r in result["relative"] if r["status"] == "success"]
        sections.append(("估值计算结果", values))
    if doc.get("unconfirmed_candidates"):
        sections.append(("已提取候选（未确认为事实）", [f"{f['period']} · {f['metric']}：{f['raw_value']} {f['unit']} · {f['state']}\n来源 {f['source_id']}" + ("\n待核验：" + "；".join(f['warnings']) if f['warnings'] else "") for f in doc["unconfirmed_candidates"]]))
    assumptions = doc["forecast_assumptions"]
    source_risks = doc.get("source_risk_review")
    if source_risks:
        sections.append(("原文风险科目检查", [source_risks["baseline_selection"], *source_risks["limitations"],
            *[f"{r['label']} · 来源 {r['block_id']} · 第{r['page'] or '?'}页 · "
              + ("已绑定财务字段，仍需评估估值调整" if r['resolved'] else "尚未绑定当前基期，不可忽略")
              for r in source_risks["matches"]]]))
    sections += [
        ("假设与观点", _forecast_lines(assumptions)),
        ("敏感性分析", [doc["sensitivity_status"]]),
        ("缺口与下一步", doc["gaps"] + doc["next_steps"]),
        ("来源清单", [f"{s['name']} · {s['role']} · {s['block_count']} 原文块\n{s.get('source_url') or '本地资料'}\nSHA-256: {s['sha256'] or '未记录'}" + ("\n读取限制：" + "；".join(s['warnings']) if s.get('warnings') else "") for s in doc["sources"]] or ["暂无取得的原始资料。用户无需先上传文件；公开资料仍须实际检索、下载并核验。"]),
        ("检索与异常记录", [f"{s.get('query', '')} · {s.get('purpose', '')} · {s.get('status', 'attempted')} · {s.get('provider', '')}\n{s.get('attempted_at', '')}" for s in doc["searches"]] or ["尚无已记录的网络检索请求；不宣称已经查遍公开来源。"]),
        ("风险与边界", doc["limitations"]),
        ("复核信息", [f"报告标识 SHA-256: {doc['report_id']}", f"研究版本：v{doc['source_revision']}；生成时间：{doc['generated_at']}", "模型：" + "/".join([doc["model"]["provider"] or "未连接", doc["model"]["name"] or "未连接"]), "原始资料、工具调用和更完整审计见 JSON 复核包。"]),
    ]
    if doc["latest_issue"]:
        sections[-3][1].append(f"最近异常：{doc['latest_issue']['code']} · {doc['latest_issue']['message']}")
    if doc.get("research_summary"):
        sections.insert(1, ("研究说明（非数值计算结论）", [doc["research_summary"]]))
    if doc.get("pending_review"):
        question = doc["pending_review"]
        sections.insert(-1, ("待确认事项", [question["title"], *[option["label"] + "：" + option.get("description", "") for option in question["options"]]]))
    return sections


def render_html(doc):
    esc = lambda value: html.escape(str(value))
    sections = "".join(f"<section><h2>{esc(title)}</h2>" + "".join(f"<p>{esc(line)}</p>" for line in lines) + "</section>" for title, lines in document_sections(doc))
    if doc['valuation_run_id']:
        sections += f"<p>正式金融模型任务：{esc(doc['valuation_run_id'])}</p>"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(doc['company'])} · 估值结果报告</title><style>
*{{box-sizing:border-box}}body{{font:15px/1.8 'Segoe UI','Microsoft YaHei',sans-serif;color:#193649;background:#f2f6f5;margin:0;padding:40px 20px}}main{{max-width:960px;margin:auto;background:white;padding:48px;border:1px solid #dbe7e3;border-radius:18px}}header{{border-bottom:2px solid #168576;padding-bottom:24px}}.brand{{font-size:12px;letter-spacing:.16em;color:#168576}}h1{{font-size:30px;line-height:1.4;margin:12px 0}}.status{{display:inline-block;background:#eef7f3;color:#156658;padding:7px 12px;border-radius:8px}}h2{{font-size:19px;color:#156658;margin:28px 0 10px}}p{{white-space:pre-wrap;overflow-wrap:anywhere;margin:9px 0}}section{{border-bottom:1px solid #e6edeb;padding-bottom:15px}}footer{{margin-top:24px;font-size:12px;color:#617885}}@media(max-width:600px){{body{{padding:12px}}main{{padding:22px}}h1{{font-size:24px}}}}@media print{{body{{padding:0;background:white}}main{{border:0;padding:12px}}h2{{break-after:avoid}}p{{orphans:3;widows:3}}}}
</style></head><body><main><header><div class="brand">VALUATIONAGENT / RESULT DOCUMENT</div><h1>{esc(doc['company'])} · 估值结果报告</h1><span class="status">{esc(doc['status_label'])}</span><p>{esc(doc['generated_at'])} · v{doc['source_revision']}</p></header>{sections}<footer>事实、假设与计算结论分开披露。{'No formal valuation submitted.' if not doc['valuation_run_id'] else 'Linked formal valuation: ' + esc(doc['valuation_run_id'])}</footer></main></body></html>'''


@lru_cache(maxsize=1)
def _report_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont, TTFError
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.is_file():
            try:
                pdfmetrics.registerFont(TTFont("OutcomeCN", str(path), subfontIndex=0))
                return "OutcomeCN"
            except (OSError, ValueError, TTFError):
                continue
    if "STSong-Light" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def render_pdf(doc, *, sections=None):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise ValueError("PDF组件未安装，可先下载HTML报告；管理员可安装 valuationagent[reports]。") from exc
    font = _report_font()
    compact = sections is not None
    body = ParagraphStyle("OutcomeBody", fontName=font, fontSize=9.5 if compact else 10, leading=15 if compact else 17, wordWrap="CJK", spaceAfter=6 if compact else 9, textColor=colors.HexColor("#193649"))
    heading = ParagraphStyle("OutcomeHeading", parent=body, fontSize=13 if compact else 15, leading=19 if compact else 22, spaceBefore=12 if compact else 16, spaceAfter=6 if compact else 8, keepWithNext=True, textColor=colors.HexColor("#147665"))
    title = ParagraphStyle("OutcomeTitle", parent=heading, fontSize=23 if compact else 25, leading=30 if compact else 35, spaceAfter=12 if compact else 16)
    p = lambda value, style=body: Paragraph(html.escape(str(value)).replace("\n", "<br/>"), style)
    story = [p("ValuationAgent / 估值结果报告", title), p(doc["company"], heading), p(doc["status_label"]), Spacer(1, 10)]
    for label, lines in document_sections(doc) if sections is None else sections:
        story.append(p(label, heading))
        story.extend(p(line) for line in lines)
    stream = BytesIO()
    def footer(canvas, pdf):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#617885"))
        canvas.drawString(42, 27, f"ValuationAgent · v{doc['source_revision']} · {doc['report_id'][:16]}")
        canvas.drawRightString(A4[0] - 42, 27, str(pdf.page))
        canvas.restoreState()
    SimpleDocTemplate(stream, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=36, bottomMargin=44,
                      title=f"{doc['company']} - 估值结果报告", author="ValuationAgent").build(story, onFirstPage=footer, onLaterPages=footer)
    return stream.getvalue()


def build_run_diagnostic(record, store=None):
    """A stopped calculation has an outcome, but never a replayable valuation."""
    issue = record.review or record.error or {}
    events = [e.model_dump(mode="json") for e in store.list_events(record.run_id)] if store else []
    report = {"schema": "valuation-diagnostic-v1", "run_id": record.run_id,
              "source_revision": record.revision, "generated_at": str(record.updated_at),
              "company": record.request.company.name or record.request.company.ticker or "待确定",
              "status_label": "未形成数值估值，已保存执行诊断", "status": str(record.status),
              "numeric_result_available": False, "valuation_result": None,
              "input_hash": record.input_hash, "submitted_request": record.request.model_dump(mode="json"),
              "issue": issue, "events": events,
              "limitation": "本文件是未完成估值的诊断说明，不是数值结果或可离线复算的估值报告。"}
    report["report_id"] = _digest(report)
    return report


def render_run_diagnostic(report):
    request = report["submitted_request"]
    financials = request.get("financials") or {}
    submitted = [f"{METRIC_LABELS[key]}：{value}" for key, value in financials.items() if key in METRIC_LABELS and value is not None]
    if submitted:
        submitted.insert(0, f"期间：{financials.get('period_end', '未确定')}；币种：{financials.get('currency', '未确定')}；金额按原始输入基础单位列示。")
    assumptions = {key: value for key, value in request.get("assumptions", {}).items() if value is not None and value not in ({}, [], "")}
    issue_lines = [report["issue"].get("message") or "当前计算尚未完成，不能据此输出价格结论。"]
    if report["issue"].get("code"):
        issue_lines.append("问题代码：" + str(report["issue"]["code"]))
    sections = [
        ("结论与任务范围", [report["limitation"], f"估值日：{request['valuation_date']}；任务状态：{report['status']}；运行ID：{report['run_id']}"]),
        ("未完成原因", issue_lines),
        ("已提交输入（不代表核验通过）", submitted or ["未提交结构化财务输入；在线取数或读取附件尚未完成。"]),
        ("估值方法与假设", [" / ".join(METHODS.get(m, m) for m in request["methods"]), "已提交假设：" + json.dumps(assumptions, ensure_ascii=False) if assumptions else "未提交显式预测假设；不据此假定模型已具备计算条件。"]),
        ("区间与敏感性", ["未计算：没有通过审核的完整结果，缺失值不按零处理。"]),
        ("最近执行记录", [f"{e.get('sequence')} · {e['type']} · {e['status']} · {e.get('summary', '')}" for e in report["events"][-20:]] or ["暂无执行记录。"]),
        ("后续处理", ["优先按未完成原因补齐数据或更正口径；旧输入和日志会保留。", "无需反复恢复同一个缺数任务；只有输入、连接或来源条件变化后才有必要重试。", "JSON诊断包包含提交输入与执行记录；核对完成后通过新版本计算。"]),
        ("审计信息", [f"输入SHA-256：{report['input_hash']}", f"报告SHA-256：{report['report_id']}", f"生成时间：{report['generated_at']}"]),
    ]
    return render_pdf(report, sections=sections)
