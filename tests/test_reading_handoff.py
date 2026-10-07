import json

import pytest

from test_multisource_extraction import runtime_at
from test_file_workspace import attach_bytes
from valuationagent.application.file_workspace import FileRead, FileReference, inspect_file, read_file
from valuationagent.core.tools import NoArguments, ToolSpec
from valuationagent.llm.tool_catalog import LoadTools


def provider_file(tmp_path, *, provider="tushare", fields=None, rows=None):
    runtime = runtime_at(tmp_path)
    fields = fields or ["ts_code", "end_date", "money_cap", "lease_liab"]
    rows = rows or [["600123.SH", "20251231", "123456789.123456789", None],
        ["600123.SH", "20241231", "100", "0"]]
    payload = {"data": {"fields": fields, "items": rows}} if provider == "tushare" else {"data": [dict(zip(fields, row)) for row in rows]}
    meta = attach_bytes(runtime, "supplier.json", json.dumps(payload).encode())
    document = runtime.session.documents[-1]
    document.provenance_type = "structured_provider"
    document.provider = provider + ":600123.SH:balancesheet:test"
    return runtime, meta


@pytest.mark.parametrize("provider,path,columns", [("tushare", "/data/items", "/data/fields"), ("infoway", "/data", None)])
def test_known_transport_reads_without_model_guessing_paths(tmp_path, provider, path, columns):
    runtime, meta = provider_file(tmp_path, provider=provider)
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="records", limit=1))
    assert result["records"]["record_path"] == path
    assert result["records"]["record_columns_path"] == columns
    block = result["blocks"][0]
    assert json.loads(block["text"])["money_cap"] == "123456789.123456789"
    assert json.loads(block["text"])["lease_liab"] is None
    assert block["location"]["json_pointer"] == path + "/0"
    assert result["records"]["selector_binding"]["basis"] == "provider_transport_contract"
    assert "published_at" not in block["location"]
    next_page = read_file(runtime.service.store, runtime.session, FileRead(**result["continue_reads"][-1]))
    assert next_page["blocks"][0]["location"]["json_pointer"] == path + "/1"


def test_inspection_returns_executable_default_reader_not_only_a_path_directory(tmp_path):
    runtime, meta = provider_file(tmp_path)
    inspected = inspect_file(runtime.service.store, runtime.session, FileReference(file_id=meta["file_id"]))
    outcome = read_file(runtime.service.store, runtime.session, FileRead(**inspected["default_read"]))
    assert outcome["blocks"]
    assert not runtime.session.facts and runtime.session.input_dataset is None


def test_generic_unique_object_array_is_readable_but_column_pairing_is_not_guessed(tmp_path):
    runtime = runtime_at(tmp_path)
    obj = attach_bytes(runtime, "objects.json", b'{"data":[{"amount":123}]}')
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=obj["file_id"], view="records"))
    assert result["blocks"][0]["location"]["json_pointer"] == "/data/0"
    assert result["records"]["selector_binding"]["basis"] == "unique_object_array"
    ambiguous = attach_bytes(runtime, "columns.json", b'{"data":{"fields":["amount"],"items":[[123]]}}')
    with pytest.raises(ValueError, match="JSON_RECORDS"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=ambiguous["file_id"], view="records"))


@pytest.mark.parametrize("payload", [
    {"first": [{"amount": 123}], "second": [{"amount": 456}]},
    {"visible": [{"amount": 123}], "nested": {"one": {"two": {"three": {"hidden": [{"amount": 456}]}}}}},
    {"visible": [{"amount": 123}], **{f"extra{index}": [] for index in range(30)}},
])
def test_multiple_or_incompletely_indexed_arrays_require_selection(tmp_path, payload):
    runtime = runtime_at(tmp_path)
    meta = attach_bytes(runtime, "partial.json", json.dumps(payload).encode())
    inspected = inspect_file(runtime.service.store, runtime.session, FileReference(file_id=meta["file_id"]))
    assert "default_read" not in inspected
    with pytest.raises(ValueError, match="JSON_RECORDS"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="records"))


def test_filename_and_source_text_cannot_claim_a_provider_contract(tmp_path):
    runtime = runtime_at(tmp_path)
    payload = b'{"provider":"tushare:600123.SH:balancesheet:test","data":{"fields":["value"],"items":[[123]]}}'
    meta = attach_bytes(runtime, "tushare-provider.json", payload)
    with pytest.raises(ValueError, match="JSON_RECORDS"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="records"))


def test_explicit_wrong_selector_and_malformed_contract_are_not_silently_repaired(tmp_path):
    runtime, meta = provider_file(tmp_path)
    with pytest.raises(ValueError, match="JSON_POINTER"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="records", record_path="/missing"))
    with pytest.raises(ValueError, match="JSON_RECORDS"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="records", record_path=""))
    malformed, bad = provider_file(tmp_path / "invalid", rows=[[1, 2]])
    with pytest.raises(ValueError, match="JSON_ROW_WIDTH"):
        read_file(malformed.service.store, malformed.session, FileRead(file_id=bad["file_id"], view="records"))


def test_large_page_request_is_bounded_without_a_model_repair_turn(tmp_path):
    runtime, meta = provider_file(tmp_path, rows=[["600123.SH", str(index), "1", None] for index in range(30)])
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="records", limit=100))
    assert len(result["blocks"]) == 12 and result["next_offset"] == 12 and result["page_limit"] == 12


def test_record_projection_does_not_inherit_single_field_location(tmp_path):
    runtime, meta = provider_file(tmp_path)
    stored = runtime.service.store.research_blocks(runtime.session.session_id, meta["file_id"])
    stored[0]["location"].update(json_pointer="/data/items/0", value_pointer="/data/items/0/3",
        provider_field="lease_liab", period_end="2025-12-31", published_at="2026-04-03")
    runtime.service.store.save_research_blocks(runtime.session.session_id, meta["file_id"], stored)
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="records", record_fields=["money_cap"]))
    location = result["blocks"][0]["location"]
    assert "value_pointer" not in location and "provider_field" not in location
    assert location["column_pointers"] == {"money_cap": "/data/items/0/2"}
    assert location["published_at"] == "2026-04-03"


def tools_for(runtime):
    return [ToolSpec(name, name, LoadTools if name == "load_tools" else NoArguments, lambda _: {}).schema()
        for name in ("load_tools", "read_file", "extract_observations", "prepare_observation_review", "record_inputs")]


def visible(runtime):
    _, selected = runtime.adapt_request([{"role": "system", "content": "policy"}, {"role": "user", "content": "{}"}], tools_for(runtime))
    return {tool["function"]["name"] for tool in selected}


def test_successful_read_exposes_extraction_without_executing_or_requiring_load(tmp_path):
    runtime, meta = provider_file(tmp_path)
    assert "extract_observations" not in visible(runtime)
    output = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"]))
    result = runtime.progress_advisory("read_file", output)
    assert "extract_observations" in visible(runtime)
    assert result["reading_handoff"]["executed"] is False
    assert not runtime.session.facts and runtime.session.input_dataset is None
    runtime.progress_advisory("extract_observations", {"saved_count": 1})
    assert "prepare_observation_review" in visible(runtime)
    runtime.progress_advisory("review_observations", {"reviews": [{"semantic_review": "supported"}]})
    assert "record_inputs" in visible(runtime)


def test_failed_or_empty_read_does_not_promote_extraction(tmp_path):
    runtime, _ = provider_file(tmp_path)
    for output in [{"ok": False, "error": {}}, {"blocks": [], "reading_status": "no_match"}]:
        runtime.progress_advisory("read_file", output)
        assert "extract_observations" not in visible(runtime)


def test_read_handoff_never_expands_user_permissions(tmp_path):
    runtime, meta = provider_file(tmp_path)
    runtime.progress_advisory("read_file", read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"])))
    runtime.session.turn_control.decision.actions = ["read"]
    runtime.session.turn_control.effects = ["files"]
    assert "extract_observations" not in visible(runtime)
