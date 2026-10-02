"""Shared CLI / API outcome report and detailed evidence package."""
import json

from valuationagent.application.result_document import ensure_result_document, render_html, render_pdf


def build_research_export(service, session_id, format="json", *, session=None):
    if format not in {"json", "html", "pdf"}:
        raise ValueError("支持 PDF、HTML 结果报告和 JSON 复核包。")
    session = session or service.store.get_research(session_id)
    document = ensure_result_document(service, session)
    if format == "pdf":
        return render_pdf(document), "application/pdf"
    if format == "html":
        return render_html(document), "text/html"
    snapshot = service.snapshot(session_id)
    snapshot["session"] = session.model_dump(mode="json")
    snapshot["report_kind"] = "research_preparation"
    snapshot["result_document"] = document
    run_id = session.valuation_run_id
    snapshot["financial_model_status"] = "not_submitted"
    if run_id:
        record = service.store.get_run(run_id)
        snapshot["financial_model_status"] = str(record.status)
        snapshot["valuation_result"] = record.result.model_dump(mode="json") if record.result else None
    return json.dumps(snapshot, ensure_ascii=False, indent=2), "application/json"
