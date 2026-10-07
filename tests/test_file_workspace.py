import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook
from pypdf import PdfWriter

from valuationagent.api.main import create_app
from valuationagent.application.file_workspace import (
    FileList, FileReference, FileRead, PageView, inspect_file, list_files, read_file, render_page, source_bytes,
)
from valuationagent.application.workspace_artifacts import (
    ArtifactRead, NoteWrite, ReportWrite, read_artifact, save_local_artifact, write_note, write_report,
)
from valuationagent.core.documents import parse_document
from valuationagent.schemas.research import DocumentSummary
from test_multisource_extraction import attach, runtime_at
from test_unified_workspace_agent import ScriptedModel


def attach_bytes(runtime, name, raw):
    meta = runtime.service.store.save_upload(name, "evidence", "application/octet-stream", raw)
    blocks, warnings = parse_document(runtime.service.store.get_file(meta["file_id"]))
    runtime.service.store.save_research_blocks(runtime.session.session_id, meta["file_id"], blocks)
    runtime.session.documents.append(DocumentSummary(file_id=meta["file_id"], name=name, role="evidence",
        block_count=len(blocks), sha256=meta["sha256"], warnings=warnings, provenance_type="user_upload"))
    return meta


def blank_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=600, height=800)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_geometry_view_separates_positioned_columns_without_financial_templates(tmp_path):
    from reportlab.pdfgen.canvas import Canvas
    runtime = runtime_at(tmp_path)
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(600, 800))
    canvas.setFont("Helvetica", 9)
    canvas.drawString(30, 750, "Issuer 600123 | consolidated | CNY")
    canvas.drawString(150, 725, "2025")
    canvas.drawString(235, 725, "2024")
    canvas.drawString(30, 700, "Revenue")
    canvas.drawString(150, 700, "4,402,090,888.24")
    canvas.drawString(235, 700, "3,136,370,678.42")
    canvas.save()
    meta = attach_bytes(runtime, "positioned.pdf", stream.getvalue())
    args = FileRead(file_id=meta["file_id"], view="pdf_geometry", page=1)
    result = read_file(runtime.service.store, runtime.session, args)
    text = "\n".join(block["text"] for block in result["blocks"])
    assert "4,402,090,888.24 3,136,370,678.42" in text
    assert result["blocks"][0]["location"]["decoder"] == "pdfplumber"
    assert result == read_file(runtime.service.store, runtime.session, args)
    assert not runtime.session.facts
    with pytest.raises(ValueError, match="PAGE_NOT_FOUND"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="pdf_geometry", page=2))


def test_blank_geometry_view_never_synthesizes_observations(tmp_path):
    runtime = runtime_at(tmp_path)
    meta = attach_bytes(runtime, "blank-geometry.pdf", blank_pdf())
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="pdf_geometry"))
    assert result["blocks"] == [] and result["reading_status"] == "no_text"
    assert not runtime.session.facts


def test_explicit_page_without_view_reads_only_that_pdf_page(tmp_path):
    from reportlab.pdfgen.canvas import Canvas
    runtime = runtime_at(tmp_path)
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(600, 800))
    for number in range(1, 4):
        canvas.drawString(30, 750, f"Page {number} revenue {number * 100}")
        canvas.showPage()
    canvas.save()
    meta = attach_bytes(runtime, "three-pages.pdf", stream.getvalue())
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], page=2))
    assert result["view"] == "pdf_geometry"
    assert {block["location"]["page"] for block in result["blocks"]} == {2}
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="text", page=3))
    assert result["view"] == "text"
    assert {block["location"]["page"] for block in result["blocks"]} == {3}
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], page=2, query="no-such-keyword"))
    assert result["reading_status"] == "readable" and result["query_matches"] == 0
    assert "revenue 200" in result["blocks"][0]["text"]
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="text", query="no-such-keyword"))
    assert result["reading_status"] == "no_match" and result["blocks"] == []


@pytest.mark.parametrize("public", [False, True])
def test_same_file_tools_read_uploaded_and_downloaded_sources(tmp_path, public):
    runtime = runtime_at(tmp_path)
    block = attach(runtime, "arbitrary.txt", "不规则说明\n单位是千元\n全年收入 105", public=public)[0]
    file_id = block["file_id"]
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=file_id))
    assert result["blocks"][0]["lines"][2]["text"] == "全年收入 105"
    assert result["source_sha256"]
    assert "不可信" in result["trust"]
    assert list_files(runtime.service.store, runtime.session, FileList(query="arbitrary"))["total"] == 1


def test_file_scope_and_original_integrity_checked_before_read(tmp_path):
    runtime = runtime_at(tmp_path)
    block = attach(runtime, "original.txt", "immutable original")[0]
    outsider = runtime.service.store.save_upload("outside.txt", "evidence", "text/plain", b"private")
    for file_id in [outsider["file_id"], "../../secrets"]:
        with pytest.raises(ValueError, match="FILE_OUT_OF_SCOPE"):
            read_file(runtime.service.store, runtime.session, FileRead(file_id=file_id))
    meta = runtime.service.store.get_file(block["file_id"])
    Path(meta["storage_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=block["file_id"]))


def test_raw_view_keeps_physical_lines_and_does_not_execute_html(tmp_path):
    runtime = runtime_at(tmp_path)
    block = attach(runtime, "page.html", '<h1>说明</h1>\n<script>stealSecrets()</script>\n<table><tr><td>105</td></tr></table>')[0]
    args = FileRead(file_id=block["file_id"], view="raw_text", start_line=2, line_count=1)
    first = read_file(runtime.service.store, runtime.session, args)
    second = read_file(runtime.service.store, runtime.session, args)
    assert first == second
    assert first["blocks"][0]["location"]["line_start"] == 2
    assert "<script>" in first["blocks"][0]["text"]
    assert len(runtime.session.facts) == 0


def test_spreadsheet_range_keeps_coordinates_blanks_and_formula_cache(tmp_path):
    runtime = runtime_at(tmp_path)
    book = Workbook()
    sheet = book.active
    sheet.title = "任意名称"
    sheet["B2"], sheet["D2"] = "营业收入", 123
    sheet["D3"] = "=D2*2"
    stream = io.BytesIO()
    book.save(stream)
    meta = attach_bytes(runtime, "input.xlsx", stream.getvalue())
    structure = inspect_file(runtime.service.store, runtime.session, FileReference(file_id=meta["file_id"]))
    assert structure["sheets"][0]["name"] == "任意名称"
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="sheet", sheet="任意名称", cell_range="B2:D3"))
    assert result["blocks"][0]["location"]["cells"] == ["营业收入", "", "123"]
    details = result["blocks"][0]["location"]["cell_details"]
    assert [cell["address"] for cell in details] == ["B2", "C2", "D2"]
    assert "公式未执行" in result["blocks"][1]["text"]
    assert "缓存值: 缺失" in result["blocks"][1]["text"]
    for invalid in ["A1:XFD100000", "A:A", "A0:B2", "D3:A1"]:
        with pytest.raises(ValueError, match="SHEET_RANGE_LIMIT"):
            read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="sheet", sheet="任意名称", cell_range=invalid))


def test_pdf_empty_page_is_not_zero_and_views_have_page_limits(tmp_path):
    runtime = runtime_at(tmp_path)
    meta = attach_bytes(runtime, "scan.pdf", blank_pdf())
    structure = inspect_file(runtime.service.store, runtime.session, FileReference(file_id=meta["file_id"]))
    assert structure["pages"] == 1
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="pdf_plain"))
    assert result["reading_status"] == "no_text"
    assert "不能填0" in result["next_action"]
    with pytest.raises(ValueError, match="PAGE_NOT_FOUND"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="pdf_layout", page=2))


def test_renderer_produces_bounded_image_without_changing_source(tmp_path):
    pytest.importorskip("pypdfium2")
    runtime = runtime_at(tmp_path)
    raw = blank_pdf()
    meta = attach_bytes(runtime, "scan.pdf", raw)
    payload, reference = render_page(runtime.service.store, runtime.session, PageView(file_id=meta["file_id"], page=1))
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as picture:
        assert max(picture.size) <= 1800
    assert reference["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert source_bytes(runtime.service.store, runtime.session, meta["file_id"])[1] == raw


def test_cancellation_does_not_publish_partial_file_view(tmp_path):
    runtime = runtime_at(tmp_path)
    block = attach(runtime, "sample.txt", "first\nsecond")[0]
    from valuationagent.llm.client import LlmError

    def cancel():
        raise LlmError("EXECUTION_CANCELLED")

    with pytest.raises(LlmError, match="EXECUTION_CANCELLED"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=block["file_id"], view="raw_text"), cancel)
    assert runtime.service.store.research_blocks(runtime.session.session_id, block["file_id"]) == [block]


def test_images_are_opt_in_and_sent_to_main_model_once_not_tool_audit(tmp_path, monkeypatch):
    runtime = runtime_at(tmp_path)
    args = PageView(file_id="file_any", page=1)
    with pytest.raises(ValueError, match="MODEL_VISION_DISABLED"):
        runtime.view_page(args)
    reference = {"file_id": args.file_id, "page": 1, "source_sha256": "source", "image_sha256": "image"}
    monkeypatch.setattr("valuationagent.application.agent_runtime.render_page", lambda *args: (b"fake-image-for-transport-only", reference))
    model = ScriptedModel(("view_pdf_page", args.model_dump()), ("inspect_context", {}), ("finish_response", {"answer": "只查看原始页面，未自动入模。"}))
    model.config = SimpleNamespace(supports_images=True)
    runtime.run(model)
    second_call = model.calls[1]
    assert "data:image/png;base64," in str(second_call)
    assert "data:image/png;base64," not in str(model.calls[2])
    assert "base64" not in str(runtime.service.store.list_events(runtime.session.session_id))
    assert not runtime.session.facts


@pytest.mark.parametrize("format_name", ["md", "json", "html", "pdf"])
def test_report_is_immutable_and_not_a_numeric_valuation(tmp_path, format_name):
    runtime = runtime_at(tmp_path)
    runtime.service.store.save_research(runtime.session)
    first = write_report(runtime.service, runtime.session, ReportWrite(format=format_name))
    second = write_report(runtime.service, runtime.session, ReportWrite(format=format_name))
    assert first["artifact_id"] == second["artifact_id"]
    assert first["numeric_result_available"] is False
    assert first["status"] == "insufficient_data"
    assert len(runtime.service.store.list_artifacts(runtime.session.session_id)) == 1
    metadata, content = runtime.service.store.get_artifact(runtime.session.session_id, first["artifact_id"])
    assert hashlib.sha256(content).hexdigest() == metadata["sha256"]
    if format_name == "md":
        assert "未计算" in content.decode()
        assert read_artifact(runtime.service.store, runtime.session.session_id, ArtifactRead(artifact_id=first["artifact_id"], limit=50))["next_offset"] == 50


def test_note_references_and_generated_files_do_not_become_sources(tmp_path):
    runtime = runtime_at(tmp_path)
    with pytest.raises(ValueError, match="NOTE_REFERENCES_REQUIRED"):
        write_note(runtime.service, runtime.session, NoteWrite(title="资料分析", body="没有引用的公司数据不能这样交付。"))
    concept = write_note(runtime.service, runtime.session, NoteWrite(title="方法概念", body="只是概念说明。", basis="concept_note"))
    assert concept["basis"] == "concept_note" and not concept["evidence_refs"]
    block = attach(runtime, "source.txt", "原文数字 123")[0]
    with pytest.raises(ValueError, match="NOTE_REFERENCE_INVALID"):
        write_note(runtime.service, runtime.session, NoteWrite(title="笔记", body="说明", evidence_ids=["another_workspace:1"]))
    artifact = write_note(runtime.service, runtime.session, NoteWrite(title="研究", body="需要进一步核对。", evidence_ids=[block["block_id"]]))
    assert artifact["status"] == "unreviewed"
    assert len(runtime.session.documents) == 1
    assert not runtime.session.facts
    target = save_local_artifact(runtime.service.store, runtime.session.session_id, artifact["artifact_id"], tmp_path / "outputs")
    assert "不是确定性估值报告" in target.read_text(encoding="utf-8")
    target.write_bytes(b"user edited")
    with pytest.raises(ValueError, match="ARTIFACT_EXISTS"):
        save_local_artifact(runtime.service.store, runtime.session.session_id, artifact["artifact_id"], target.parent)


def test_artifact_atomic_storage_scope_quota_and_corruption(tmp_path):
    runtime = runtime_at(tmp_path)
    store = runtime.service.store
    session_id = runtime.session.session_id
    metadata = {"kind": "research_note", "filename": "research-note.md", "media_type": "text/markdown"}
    artifact = store.save_artifact(session_id, metadata, b"test")
    with pytest.raises(ValueError, match="ARTIFACT_NOT_FOUND"):
        store.get_artifact("other", artifact["artifact_id"])
    with pytest.raises(ValueError, match="ARTIFACT_SIZE_LIMIT"):
        store.save_artifact(session_id, metadata, b"x" * (10 * 1024 * 1024 + 1))
    for index in range(99):
        store.save_artifact(session_id, {**metadata, "index": index}, str(index).encode())
    with pytest.raises(ValueError, match="ARTIFACT_QUOTA"):
        store.save_artifact(session_id, metadata, b"new output")
    with store._connect() as connection:
        connection.execute("UPDATE workspace_artifacts SET content=? WHERE artifact_id=?", (b"corrupted", artifact["artifact_id"]))
    with pytest.raises(ValueError, match="ARTIFACT_INTEGRITY"):
        store.get_artifact(session_id, artifact["artifact_id"])
    with pytest.raises(ValueError, match="ARTIFACT_INTEGRITY"):
        store.save_artifact(session_id, metadata, b"test")


def test_web_export_persists_downloadable_artifact_and_rejects_other_workspace(tmp_path):
    app = create_app(tmp_path)
    service = app.state.workspaces
    workspace = service.create()
    other = service.create()
    with TestClient(app) as client:
        response = client.get(f"/api/workspaces/{workspace.workspace_id}/export?format=md")
        assert response.status_code == 200, response.text
        artifact_id = response.headers["x-artifact-id"]
        assert client.get(f"/api/workspaces/{workspace.workspace_id}/artifacts").json()["artifacts"][0]["artifact_id"] == artifact_id
        repeated = client.get(f"/api/workspaces/{workspace.workspace_id}/artifacts/{artifact_id}")
        assert repeated.content == response.content
        assert repeated.headers["x-content-type-options"] == "nosniff"
        assert client.get(f"/api/workspaces/{other.workspace_id}/artifacts/{artifact_id}").status_code == 404
        assert client.get(f"/api/workspaces/{workspace.workspace_id}/files/not_owned").status_code == 422


def test_agent_can_write_then_read_generated_report_without_model_prices(tmp_path):
    runtime = runtime_at(tmp_path)
    model = ScriptedModel(("write_workspace_report", {"format": "md"}), ("list_artifacts", {}), ("finish_response", {"answer": "已生成缺口报告，尚无估值数值。"}))
    runtime.run(model)
    assert runtime.service.store.list_artifacts(runtime.session.session_id)[0]["numeric_result_available"] is False
    assert not runtime.session.valuation_run_id


def test_downloaded_file_survives_initial_decoder_failure_and_is_not_redownloaded(tmp_path, monkeypatch):
    runtime = runtime_at(tmp_path)
    source_id = "web_example"
    runtime.session.documents.append(DocumentSummary(file_id=source_id, name="披露线索", role="evidence", block_count=1, provenance_type="search_snippet"))
    runtime.service.store.save_research_blocks(runtime.session.session_id, source_id, [{"block_id": source_id + ":1", "text": "官方文件", "location": {"source_type": "web_search", "url": "https://static.cninfo.com.cn/report.pdf"}}])
    downloads = []
    def download(url):
        downloads.append(url)
        return url, blank_pdf(), "application/pdf"
    def broken_decoder(*args, **kwargs):
        raise ValueError("synthetic decoder failure")
    monkeypatch.setattr(runtime.service, "_download_disclosure_pdf", download)
    monkeypatch.setattr("valuationagent.application.research.parse_document", broken_decoder)
    first = runtime.service._fetch_search_source(runtime.session, source_id)
    assert first["block_count"] == 0
    assert inspect_file(runtime.service.store, runtime.session, FileReference(file_id=first["file_id"]))["pages"] == 1
    second = runtime.service._fetch_search_source(runtime.session, source_id)
    assert second["status"] == "already_fetched"
    assert second["file_id"] == first["file_id"]
    original_id = runtime.service._fetch_search_source(runtime.session, first["file_id"])
    assert original_id["status"] == "already_fetched" and original_id["file_id"] == first["file_id"]
    assert len(downloads) == 1


def test_pdf_initial_window_can_be_extended_without_changing_original(tmp_path):
    runtime = runtime_at(tmp_path)
    writer = PdfWriter()
    for page_number in range(30):
        writer.add_blank_page(width=600, height=800)
    writer.add_outline_item("财务报表", 26)
    stream = io.BytesIO()
    writer.write(stream)
    meta = attach_bytes(runtime, "long.pdf", stream.getvalue())
    structure = inspect_file(runtime.service.store, runtime.session, FileReference(file_id=meta["file_id"]))
    assert structure["outline"] == [{"title": "财务报表", "page": 27}]
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="pdf_plain", page=27))
    assert result["reading_status"] == "no_text"
