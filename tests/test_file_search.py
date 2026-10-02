import io

import pytest
from reportlab.pdfgen.canvas import Canvas

from valuationagent.application.file_search import FileSearch, search_file
from test_file_workspace import attach_bytes, blank_pdf
from test_multisource_extraction import runtime_at


def indexed_fixture(runtime):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(600, 800))
    for number in range(1, 36):
        canvas.drawString(30, 750, f"Page {number}")
        if number in {28, 34}:
            canvas.drawString(30, 700, f"Ordinary shares {number * 100}")
        canvas.showPage()
    canvas.save()
    meta = attach_bytes(runtime, "long.pdf", stream.getvalue())
    store = runtime.service.store
    initial = store.research_blocks(runtime.session.session_id, meta["file_id"])
    store.save_research_blocks(runtime.session.session_id, meta["file_id"],
                              [block for block in initial if block["location"]["page"] <= 25])
    return meta


def test_search_discovers_unloaded_pages_and_paginates_without_interpreting(tmp_path):
    runtime = runtime_at(tmp_path)
    meta = indexed_fixture(runtime)
    args = FileSearch(file_id=meta["file_id"], queries=["Ordinary shares"], limit=1)
    result = search_file(runtime.service.store, runtime.session, args)
    assert result["coverage"]["complete_file"]
    assert result["total_matches"] == 2 and result["next_offset"] == 1
    assert result["matches"][0]["location"]["page"] == 28
    assert result["source_sha256"] == meta["sha256"]
    assert [row["page"] for row in result["page_hits"]] == [28, 34]
    original = runtime.service.store.research_blocks(runtime.session.session_id, meta["file_id"])
    match = result["matches"][0]
    block = next(block for block in original if block["block_id"] == match["block_id"])
    assert block["text"].splitlines()[match["matched_line"] - 1] == "Ordinary shares 2800"
    repeated = search_file(runtime.service.store, runtime.session, args)
    assert repeated == result
    assert runtime.service.store.research_blocks(runtime.session.session_id, meta["file_id"]) == original
    second = search_file(runtime.service.store, runtime.session, args.model_copy(update={"offset": 1}))
    assert second["matches"][0]["location"]["page"] == 34
    assert second["next_offset"] is None
    assert not runtime.session.facts


def test_search_reports_partial_coverage_and_empty_pages(tmp_path):
    runtime = runtime_at(tmp_path)
    meta = indexed_fixture(runtime)
    result = search_file(runtime.service.store, runtime.session,
                         FileSearch(file_id=meta["file_id"], queries=["Ordinary"], page_limit=25))
    assert result["total_matches"] == 0
    assert result["coverage"]["next_page"] == 26
    assert not result["coverage"]["complete_file"]
    meta = attach_bytes(runtime, "empty.pdf", blank_pdf())
    result = search_file(runtime.service.store, runtime.session,
                         FileSearch(file_id=meta["file_id"], queries=["revenue"]))
    assert result["coverage"]["pages_without_text"] == [1]
    assert not result["coverage"]["complete_file"]
    assert not result["matches"] and not runtime.session.facts


def test_search_obeys_scope_hash_query_and_cancellation(tmp_path):
    runtime = runtime_at(tmp_path)
    meta = indexed_fixture(runtime)
    args = FileSearch(file_id=meta["file_id"], queries=["Ordinary"])
    with pytest.raises(ValueError, match="FILE_OUT_OF_SCOPE"):
        search_file(runtime.service.store, runtime.session, args.model_copy(update={"file_id": "outside"}))
    with pytest.raises(ValueError, match="SEARCH_QUERY_INVALID"):
        search_file(runtime.service.store, runtime.session, args.model_copy(update={"queries": [" "]}))
    def cancelled():
        raise ValueError("CANCELLED")
    with pytest.raises(ValueError, match="CANCELLED"):
        search_file(runtime.service.store, runtime.session, args, cancelled)
    from pathlib import Path
    Path(runtime.service.store.get_file(meta["file_id"])["storage_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        search_file(runtime.service.store, runtime.session, args)


def test_search_ranks_specific_terms_without_financial_templates(tmp_path):
    runtime = runtime_at(tmp_path)
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(600, 800))
    for number in range(1, 8):
        canvas.drawString(30, 750, f"Common heading {number}")
        if number == 7:
            canvas.drawString(30, 700, "Rare detail Common")
        canvas.showPage()
    canvas.save()
    meta = attach_bytes(runtime, "generic.pdf", stream.getvalue())
    args = FileSearch(file_id=meta["file_id"], queries=["Common", "Rare detail"], limit=1)
    ranked = search_file(runtime.service.store, runtime.session, args)
    assert ranked["matches"][0]["location"]["page"] == 7
    assert ranked["order"] == "relevance" and ranked["total_matches"] == 8
    document = search_file(runtime.service.store, runtime.session, args.model_copy(update={"order": "document"}))
    assert document["matches"][0]["location"]["page"] == 1
    both = search_file(runtime.service.store, runtime.session, args.model_copy(update={"match_mode": "all"}))
    assert both["total_matches"] == 1 and len(both["matches"][0]["queries"]) == 2
    assert not runtime.session.facts


def test_document_order_does_not_depend_on_previous_page_scan_order(tmp_path):
    runtime = runtime_at(tmp_path)
    meta = indexed_fixture(runtime)
    args = FileSearch(file_id=meta["file_id"], queries=["Page"], order="document")
    search_file(runtime.service.store, runtime.session, args.model_copy(update={"start_page": 26}))
    whole = search_file(runtime.service.store, runtime.session, args)
    assert whole["matches"][0]["location"]["page"] == 1
