import json

import pytest

from valuationagent.application.file_workspace import FileRead, FileReference, inspect_file, read_file
from valuationagent.application.json_records import pointer_value
from test_file_workspace import attach_bytes
from test_multisource_extraction import runtime_at


def fixture(tmp_path):
    runtime = runtime_at(tmp_path)
    raw = b'{"data":[{"field":"income","period":"2025","value":1234567890.123456789},{"field":"income","period":"2024","value":null},{"field":"income","period":"2023","value":12},{"field":"expense","period":"2025","value":34}]}'
    meta = attach_bytes(runtime, "generic-records.json", raw)
    return runtime, meta


def test_json_inventory_exposes_generic_array_and_columns(tmp_path):
    runtime, meta = fixture(tmp_path)
    result = inspect_file(runtime.service.store, runtime.session, FileReference(file_id=meta["file_id"]))
    assert "records" in result["views"]
    assert result["records"]["arrays"][0]["record_path"] == "/data"
    assert result["records"]["arrays"][0]["fields"] == ["field", "period", "value"]


def test_record_projection_preserves_precision_nulls_source_positions_and_pagination(tmp_path):
    runtime, meta = fixture(tmp_path)
    args = FileRead(file_id=meta["file_id"], view="records", record_path="/data",
        record_filters={"field": ["income"]}, record_fields=["period", "value"], limit=2)
    first = read_file(runtime.service.store, runtime.session, args)
    assert first["total"] == 3 and first["next_offset"] == 2
    assert json.loads(first["blocks"][0]["text"]) == {"period": "2025", "value": "1234567890.123456789"}
    assert json.loads(first["blocks"][1]["text"])["value"] is None
    assert first["blocks"][0]["location"]["json_pointer"] == "/data/0"
    second = read_file(runtime.service.store, runtime.session, FileRead(**first["continue_reads"][-1]))
    assert second["total"] == 3 and second["next_offset"] is None
    assert json.loads(second["blocks"][0]["text"]) == {"period": "2023", "value": 12}
    assert read_file(runtime.service.store, runtime.session, args)["blocks"] == first["blocks"]
    assert first["source_sha256"] == meta["sha256"]


@pytest.mark.parametrize("changes,code", [
    ({"record_path": "/missing"}, "JSON_POINTER"),
    ({"record_path": "/data/0"}, "JSON_RECORDS"),
    ({"record_path": "__import__('os')"}, "JSON_POINTER"),
    ({"record_fields": ["invented"]}, "JSON_FIELDS"),
    ({"record_filters": {"invented": ["value"]}}, "JSON_FIELDS"),
    ({"query": "income"}, "JSON_FILTER"),
    ({"view": "text"}, "VIEW_MISMATCH"),
])
def test_record_view_rejects_invalid_operations(tmp_path, changes, code):
    runtime, meta = fixture(tmp_path)
    values = {"file_id": meta["file_id"], "view": "records", "record_path": "/data", **changes}
    with pytest.raises(ValueError, match=code):
        read_file(runtime.service.store, runtime.session, FileRead(**values))


def test_json_pointer_escaping_and_no_negative_index():
    assert pointer_value({"a/b": {"~key": [12]}}, "/a~1b/~0key/0") == 12
    with pytest.raises(ValueError, match="JSON_POINTER"):
        pointer_value([12], "/-1")


def test_empty_match_is_not_zero_or_an_unreadable_file(tmp_path):
    runtime, meta = fixture(tmp_path)
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="records",
        record_path="/data", record_filters={"field": ["missing"]}))
    assert not result["blocks"] and result["total"] == 0
    assert result["records"]["source_record_count"] == 4
    assert result["reading_status"] == "no_match"
    assert result["records"]["filter_value_samples"] == {"field": ["income", "expense"]}
    assert "不自动转换日期" in result["next_action"]


def test_explicit_record_parameters_select_record_view_when_view_omitted(tmp_path):
    runtime, meta = fixture(tmp_path)
    args = FileRead(file_id=meta["file_id"], record_path="/data", record_fields=["value"], limit=2)
    result = read_file(runtime.service.store, runtime.session, args)
    assert result["view"] == "records"
    assert result["blocks"][0]["location"]["json_pointer"] == "/data/0"
    assert json.loads(result["blocks"][0]["text"])["value"] == "1234567890.123456789"
    assert args.view == "text" and "view" not in args.model_fields_set
    next_page = read_file(runtime.service.store, runtime.session, FileRead(**result["continue_reads"][-1]))
    assert next_page["view"] == "records" and next_page["next_offset"] is None


@pytest.mark.parametrize("extra", [{"page": 1}, {"block_id": "placeholder"}, {"sheet": "sheet1"}])
def test_record_view_inference_rejects_ambiguous_positions(tmp_path, extra):
    runtime, meta = fixture(tmp_path)
    with pytest.raises(ValueError, match="VIEW_MISMATCH"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], record_path="/data", **extra))


def test_missing_record_path_selects_only_unambiguous_object_array(tmp_path):
    runtime, meta = fixture(tmp_path)
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], record_fields=["value"]))
    assert result["records"]["record_path"] == "/data"


def test_model_reading_schema_requires_complete_record_selector():
    content, records = FileRead.model_json_schema()["anyOf"]
    assert "records" not in content["properties"]["view"]["enum"]
    assert "record_fields" not in content["properties"]
    assert records["properties"]["view"]["const"] == "records"
    assert set(records["required"]) == {"file_id", "view"}
    assert not {"block_id", "page", "sheet", "query"} & records["properties"].keys()


def test_record_view_does_not_inherit_another_rows_publication_date(tmp_path):
    runtime, meta = fixture(tmp_path)
    original = runtime.service.store.research_blocks(runtime.session.session_id, meta["file_id"])
    original[0]["location"]["published_at"] = "2026-04-20"
    runtime.service.store.save_research_blocks(runtime.session.session_id, meta["file_id"], original)
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="records", record_path="/data"))
    assert all("published_at" not in block["location"] for block in result["blocks"])


def test_columnar_json_table_preserves_original_cell_positions_and_precision(tmp_path):
    runtime = runtime_at(tmp_path)
    raw = b'{"data":{"fields":["period","revenue","profit"],"items":[["2025",123456789012.123456789,null],["2024",123,2]]}}'
    meta = attach_bytes(runtime, "columnar.json", raw)
    inventory = inspect_file(runtime.service.store, runtime.session, FileReference(file_id=meta["file_id"]))
    assert {item["record_path"] for item in inventory["records"]["arrays"]} >= {"/data/fields", "/data/items"}
    result = read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="records",
        record_path="/data/items", record_columns_path="/data/fields", record_filters={"period": ["2025"]}, record_fields=["revenue", "profit"]))
    block = result["blocks"][0]
    assert json.loads(block["text"]) == {"revenue": "123456789012.123456789", "profit": None}
    assert block["location"]["json_pointer"] == "/data/items/0"
    assert block["location"]["column_pointers"] == {"revenue": "/data/items/0/1", "profit": "/data/items/0/2"}
    assert result["total"] == 1 and result["source_sha256"] == meta["sha256"]


def test_record_columns_are_reused_only_after_successful_explicit_binding(tmp_path):
    runtime = runtime_at(tmp_path)
    meta = attach_bytes(runtime, "remember-columns.json",
        b'{"data":{"fields":["year","value"],"items":[["2025",12],["2024",null]],"other":[["2023",7]]}}')
    arguments = {"file_id": meta["file_id"], "view": "records", "record_path": "/data/items", "limit": 1}
    with pytest.raises(ValueError, match='JSON_COLUMNS.*data/fields'):
        read_file(runtime.service.store, runtime.session, FileRead(**arguments))
    initial = read_file(runtime.service.store, runtime.session,
        FileRead(**arguments, record_columns_path="/data/fields"))
    continued = read_file(runtime.service.store, runtime.session, FileRead(**arguments, offset=1))
    assert continued["records"]["column_binding"]["basis"] == "prior_successful_read"
    assert continued["records"]["record_columns_path"] == "/data/fields"
    assert json.loads(continued["blocks"][0]["text"]) == {"year": "2024", "value": None}
    assert continued["source_sha256"] == initial["source_sha256"]
    with pytest.raises(ValueError, match="JSON_COLUMNS"):
        read_file(runtime.service.store, runtime.session, FileRead(**{**arguments, "record_path": "/data/other"}))
    unrelated = attach_bytes(runtime, "unbound-columns.json",
        b'{"data":{"fields":["year","value"],"items":[["2023",1]]}}')
    with pytest.raises(ValueError, match="JSON_COLUMNS"):
        read_file(runtime.service.store, runtime.session, FileRead(**{**arguments, "file_id": unrelated["file_id"]}))


def test_multiple_successful_column_bindings_require_explicit_selection(tmp_path):
    runtime = runtime_at(tmp_path)
    meta = attach_bytes(runtime, "ambiguous-columns.json", b'{"fields":["year","value"],"alternative":["date","amount"],"rows":[["2025",12]]}')
    arguments = {"file_id": meta["file_id"], "view": "records", "record_path": "/rows"}
    for path in ("/fields", "/alternative"):
        read_file(runtime.service.store, runtime.session, FileRead(**arguments, record_columns_path=path))
    with pytest.raises(ValueError, match="JSON_COLUMNS_AMBIGUOUS"):
        read_file(runtime.service.store, runtime.session, FileRead(**arguments))
    explicit = read_file(runtime.service.store, runtime.session, FileRead(**arguments, record_columns_path="/fields"))
    assert json.loads(explicit["blocks"][0]["text"]) == {"year": "2025", "value": 12}


@pytest.mark.parametrize("network_allowed", [True, False])
def test_undownloaded_search_source_recovers_to_fetch_not_another_search(tmp_path, network_allowed):
    from valuationagent.application.extraction_recovery import record_attempt, recovery_plan

    runtime, meta = fixture(tmp_path)
    runtime.session.data_source_preference = "web" if network_allowed else "upload"
    runtime.session.turn_control = None
    record_attempt(runtime.session, "read_file", {"file_id": meta["file_id"], "view": "text"},
        {"ok": False, "error": {"message": "SOURCE_NOT_DOWNLOADED: 先下载"}})
    choices = recovery_plan(runtime.session)["files"][0]["next_choices"]
    assert [choice["tool"] for choice in choices] == (["fetch_search_source"] if network_allowed else ["finish_response"])


@pytest.mark.parametrize("fields,items", [
    (["value", "value"], [[1, 2]]),
    (["value", "date"], [[1]]),
    (["value", "date"], [[1, 2, 3]]),
    ([12], [[1]]),
])
def test_columnar_json_does_not_guess_headers_or_truncate_rows(tmp_path, fields, items):
    runtime = runtime_at(tmp_path)
    meta = attach_bytes(runtime, "invalid-table.json", json.dumps({"fields": fields, "rows": items}).encode())
    with pytest.raises(ValueError, match="JSON_COLUMNS|JSON_ROW_WIDTH"):
        read_file(runtime.service.store, runtime.session, FileRead(file_id=meta["file_id"], view="records",
            record_path="/rows", record_columns_path="/fields"))


def test_invalid_json_fields_recover_to_actual_schema_not_raw_text_or_network(tmp_path):
    runtime, meta = fixture(tmp_path)
    args = FileRead(file_id=meta["file_id"], view="records", record_path="/data", record_fields=["invented"])
    result = runtime.call("read_file", args.model_dump(mode="json"), lambda: read_file(runtime.service.store, runtime.session, args))
    assert result["ok"] is False
    assert "当前真实列：field, period, value" in result["error"]["message"]
    choices = result["recovery"]["files"][0]["next_choices"]
    assert [choice["tool"] for choice in choices] == ["inspect_file"]
    assert choices[0]["arguments"]["file_id"] == meta["file_id"]


def test_columnar_inventory_shows_actual_headers_without_guessing_a_mapping():
    from valuationagent.application.json_records import record_inventory

    payload = {"data": {"fields": ["ts_code", "end_date", "n_income_attr_p"],
        "items": [["002032.SZ", "20251231", None]], "alternative": ["entity", "date", "amount"]}}
    inventory = record_inventory(json.dumps(payload))
    headers = next(item for item in inventory["arrays"] if item["record_path"] == "/data/fields")
    assert headers["string_values"] == payload["data"]["fields"]
    assert headers["string_values_truncated"] is False
    rows = next(item for item in inventory["arrays"] if item["record_path"] == "/data/items")
    assert {item["record_columns_path"] for item in rows["column_candidates"]} == {"/data/fields", "/data/alternative"}
    assert rows["fields"] == []
    assert "record_columns_path" not in rows
    assert rows["column_candidates"][0]["requires_confirmation"] is True


def test_columnar_inventory_bounds_headers_and_rejects_invalid_candidates():
    from valuationagent.application.json_records import record_inventory

    payload = {"headers": [f"column_{index}" for index in range(120)], "rows": [list(range(120))],
        "duplicate": ["same", "same"], "bad_width": [[1, 2], [1]], "empty": [],
        "other": {"names": [f"other_{index}" for index in range(120)]}}
    inventory = record_inventory(json.dumps(payload))
    rows = next(item for item in inventory["arrays"] if item["record_path"] == "/rows")
    assert [item["record_columns_path"] for item in rows["column_candidates"]] == ["/headers"]
    assert len(rows["column_candidates"][0]["fields"]) == 60
    headers = next(item for item in inventory["arrays"] if item["record_path"] == "/headers")
    assert len(headers["string_values"]) == 60 and headers["string_values_truncated"]
    for path in ("/bad_width", "/empty"):
        assert not next(item for item in inventory["arrays"] if item["record_path"] == path).get("column_candidates")
