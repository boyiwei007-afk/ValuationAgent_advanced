import pytest

from valuationagent.application.file_workspace import FileRead, read_file
from test_multisource_extraction import attach, runtime_at


def test_block_id_read_preserves_original_line_numbers_and_stored_text(tmp_path):
    runtime = runtime_at(tmp_path)
    block = attach(runtime, "source.txt", "issuer\nunit\nperiod\nvalue 1200\nfootnote")[0]
    args = FileRead(file_id=block["file_id"], block_id=block["block_id"], start_line=3, line_count=2)
    result = read_file(runtime.service.store, runtime.session, args)
    selected = result["blocks"][0]
    assert selected["lines"] == [{"line": 3, "text": "period"}, {"line": 4, "text": "value 1200"}]
    assert selected["next_line"] == 5 and selected["total_lines"] == 5
    assert selected["windowed"] and selected["block_id"] == block["block_id"]
    assert result["continue_reads"] == [{"file_id": block["file_id"], "block_id": block["block_id"], "start_line": 5, "line_count": 2}]
    following = read_file(runtime.service.store, runtime.session, FileRead(**result["continue_reads"][0]))
    assert following["blocks"][0]["text"] == "footnote"
    original = runtime.service.store.research_blocks(runtime.session.session_id, block["file_id"])[0]
    assert original["text"] == block["text"]
    assert read_file(runtime.service.store, runtime.session, FileRead(file_id=block["file_id"], block_id=block["block_id"]))["blocks"][0]["text"] == block["text"]


def test_block_read_cannot_mix_page_or_cross_file(tmp_path):
    runtime = runtime_at(tmp_path)
    first = attach(runtime, "one.txt", "one")[0]
    other = attach(runtime, "two.txt", "two")[0]
    with pytest.raises(ValueError, match="BLOCK_NOT_FOUND"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=first["file_id"], block_id=other["block_id"]))
    with pytest.raises(ValueError, match="VIEW_MISMATCH"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=first["file_id"], block_id=first["block_id"], page=1))


def test_empty_window_is_not_fabricated_or_renumbered(tmp_path):
    runtime = runtime_at(tmp_path)
    block = attach(runtime, "source.txt", "only one line")[0]
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=block["file_id"], block_id=block["block_id"], start_line=50))
    assert result["blocks"][0]["lines"] == []
    assert result["blocks"][0]["next_line"] is None
    assert result["ok"] is False and result["reading_status"] == "empty_window"
    assert result["error"]["code"] == "READ_WINDOW_EMPTY"


def test_html_page_argument_cannot_silently_hide_all_existing_text(tmp_path):
    runtime = runtime_at(tmp_path)
    block = attach(runtime, "source.html", "<p>Existing web source with facts 1200</p>")[0]
    with pytest.raises(ValueError, match="PAGE_UNSUPPORTED"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=block["file_id"], page=1))
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=block["file_id"], view="text"))
    assert result["reading_status"] == "readable" and "1200" in result["blocks"][0]["text"]


def test_raw_text_continuation_does_not_skip_unreturned_lines(tmp_path):
    runtime = runtime_at(tmp_path)
    block = attach(runtime, "source.txt", "\n".join(f"source line {number}" for number in range(1, 10)))[0]
    args = FileRead(file_id=block["file_id"], view="raw_text", line_count=6, limit=3)
    first = read_file(runtime.service.store, runtime.session, args)
    assert first["next_line"] is None and first["next_offset"] == 3
    second = read_file(runtime.service.store, runtime.session, FileRead(**first["continue_reads"][0]))
    assert [row["text"] for row in second["blocks"]] == ["source line 4", "source line 5", "source line 6"]
    assert second["next_line"] == 7 and second["next_offset"] is None
    third = read_file(runtime.service.store, runtime.session, FileRead(**second["continue_reads"][0]))
    assert [row["text"] for row in third["blocks"]] == ["source line 7", "source line 8", "source line 9"]
    assert not third["continue_reads"]
