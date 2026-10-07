import io

import pytest
from reportlab.pdfgen.canvas import Canvas

from valuationagent.application.file_workspace import FileRead, read_file
from test_file_workspace import attach_bytes, blank_pdf
from test_multisource_extraction import runtime_at


def financial_table_pdf(*, ruled=True, merged=False):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(600, 800))
    canvas.setFont("Helvetica", 10)
    canvas.drawString(40, 765, "Example issuer 600123; consolidated; CNY thousand")
    if ruled:
        for horizontal in (720, 690, 660, 630):
            canvas.line(40, horizontal, 550, horizontal)
        for vertical in (40, 200, 370, 550):
            canvas.line(vertical, 630, vertical, 690 if merged and vertical == 370 else 720)
    canvas.drawString(45, 700, "Metric")
    canvas.drawString(205, 700, "Annual periods" if merged else "2025")
    if not merged:
        canvas.drawString(375, 700, "2024")
    canvas.drawString(45, 670, "Revenue")
    canvas.drawString(205, 670, "123,456.78")
    canvas.drawString(375, 670, "98,765.43")
    canvas.drawString(45, 640, "Other")
    canvas.drawString(205, 640, "(1,234.50)")
    canvas.save()
    return stream.getvalue()


def table_view(tmp_path, **kwargs):
    runtime = runtime_at(tmp_path)
    meta = attach_bytes(runtime, "unfamiliar-layout.pdf", financial_table_pdf(**kwargs))
    args = FileRead(file_id=meta["file_id"], view="pdf_tables", page=1, limit=12)
    return runtime, args, read_file(runtime.service.store, runtime.session, args)


def test_ruled_table_preserves_raw_cells_positions_and_negative_signs(tmp_path):
    runtime, args, result = table_view(tmp_path)
    assert result["reading_status"] == "readable"
    layout = result["table_layout"]
    assert layout["tables"][0] == {"table": 1, "bbox": [40, 80, 550, 170], "rows": 3, "columns": 3}
    assert layout["semantic_verified"] is False
    assert layout["text_layer_only"] is True
    revenue = result["blocks"][1]
    assert revenue["text"] == "Revenue | 123,456.78 | 98,765.43"
    assert revenue["location"]["cell_details"][1] == {"address": "T1R2C2", "row": 2, "column": 2,
        "value": "123,456.78", "bbox": [200, 110, 370, 140], "status": "text"}
    assert result["blocks"][2]["location"]["cells"] == ["Other", "(1,234.50)", ""]
    assert result["blocks"][2]["location"]["cell_details"][2]["status"] == "empty"
    assert result == read_file(runtime.service.store, runtime.session, args)
    assert not runtime.session.facts
    assert runtime.session.input_dataset is None


def test_merged_cell_is_not_filled_into_adjacent_column(tmp_path):
    _, _, result = table_view(tmp_path, merged=True)
    cells = result["blocks"][0]["location"]["cell_details"]
    assert cells[1]["value"] == "Annual periods"
    assert cells[1]["bbox"] == [200, 80, 550, 110]
    assert cells[2]["value"] is None
    assert cells[2]["bbox"] is None
    assert cells[2]["status"] == "no_separate_cell"


def test_unruled_table_has_explicit_alternative_not_silent_strategy_switch(tmp_path):
    runtime, args, result = table_view(tmp_path, ruled=False)
    assert result["reading_status"] == "no_tables"
    assert result["table_layout"]["strategy"] == "lines"
    assert "不是数据缺失" in result["next_action"]
    text_result = read_file(runtime.service.store, runtime.session, args.model_copy(update={"table_strategy": "text"}))
    assert text_result["table_layout"]["tables"]
    assert any("123,456.78" in block["text"] for block in text_result["blocks"])
    assert all(block["location"]["table_strategy"] == "text" for block in text_result["blocks"])


def test_table_rows_are_pageable_and_can_be_read_back_by_block(tmp_path):
    runtime, args, result = table_view(tmp_path)
    first = read_file(runtime.service.store, runtime.session, args.model_copy(update={"limit": 1}))
    assert first["next_offset"] == 1
    second_args = FileRead.model_validate(first["continue_reads"][-1])
    second = read_file(runtime.service.store, runtime.session, second_args)
    assert second["blocks"][0]["block_id"] == result["blocks"][1]["block_id"]
    saved = read_file(runtime.service.store, runtime.session,
        FileRead(file_id=args.file_id, block_id=second["blocks"][0]["block_id"]))
    assert saved["blocks"][0] == second["blocks"][0]


def test_table_parameters_are_scoped_and_page_bounds_enforced(tmp_path):
    runtime, args, _ = table_view(tmp_path)
    inferred = read_file(runtime.service.store, runtime.session, FileRead(file_id=args.file_id, table_index=1))
    assert inferred["view"] == "pdf_tables"
    for update, code in [({"table_index": 2}, "TABLE_NOT_FOUND"), ({"page": 2}, "PAGE_NOT_FOUND")]:
        with pytest.raises(ValueError, match=code):
            read_file(runtime.service.store, runtime.session, args.model_copy(update=update))
    with pytest.raises(ValueError, match="VIEW_MISMATCH"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=args.file_id, view="pdf_geometry", table_index=1))
    with pytest.raises(ValueError, match="VIEW_MISMATCH"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=args.file_id, block_id=f"{args.file_id}:1", table_index=1))


def test_blank_page_and_cancel_do_not_admit_values(tmp_path):
    runtime = runtime_at(tmp_path)
    meta = attach_bytes(runtime, "scan.pdf", blank_pdf())
    args = FileRead(file_id=meta["file_id"], view="pdf_tables")
    result = read_file(runtime.service.store, runtime.session, args)
    assert result["reading_status"] == "no_tables"
    assert result["blocks"] == []
    assert "OCR" in result["next_action"]
    from valuationagent.llm.client import LlmError

    def cancel():
        raise LlmError("EXECUTION_CANCELLED")

    with pytest.raises(LlmError, match="EXECUTION_CANCELLED"):
        read_file(runtime.service.store, runtime.session, args, cancel)
    assert not runtime.session.facts


def test_table_view_checks_original_hash_before_decode(tmp_path):
    from pathlib import Path

    runtime, args, _ = table_view(tmp_path)
    meta = runtime.service.store.get_file(args.file_id)
    Path(meta["storage_path"]).write_bytes(b"changed original")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        read_file(runtime.service.store, runtime.session, args)
