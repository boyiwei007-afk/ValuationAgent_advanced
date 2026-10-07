from datetime import date

from test_input_views import large_dataset
from test_input_workspace import fixture, values
from valuationagent.application.input_views import InspectInputs, inspect_inputs
from valuationagent.application.record_inputs import record_inputs
from valuationagent.core.tools import canonical


def test_compact_records_keep_normalized_and_original_units_and_source_cautions(tmp_path):
    _, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    dataset = runtime.session.input_dataset
    for row in dataset.records:
        row.source.limitations = ["用户情景数值，未进行外部来源核验。"]
    before = dataset.model_dump(mode="json")
    result = inspect_inputs(runtime.session, InspectInputs())
    profit = next(row for row in result["items"] if row["metric"] == "net_income_parent")
    assert profit["value"] == "1000000000" and profit["value_unit"] == "元"
    assert profit["original_amount"] == "10亿元" and profit["unit"] == "亿元"
    assert profit["source"]["kind"] == "user" and "quote" not in profit["source"]
    assert result["source_notes"] == ["用户情景数值，未进行外部来源核验。"]
    assert all(row["source"]["note_indexes"] == [0] for row in result["items"])
    assert dataset.model_dump(mode="json") == before
    details = inspect_inputs(runtime.session, InspectInputs.model_validate(result["evidence_action"]["arguments"]))
    assert result["evidence_action"]["tool"] == "inspect_inputs"
    assert all(row["source"]["quote"] and row["source"]["sha256"] for row in details["items"])


def test_compact_view_does_not_repeat_long_evidence_in_every_record(tmp_path):
    runtime = large_dataset(tmp_path)
    for row in runtime.session.input_dataset.records:
        row.source.quote = "这是实际保存在输入中的较长原文引用。" * 60
        row.source.limitations = ["同一供应商的重复记录不是独立佐证；原始数值不等于最终模型指标。"]
    compact = inspect_inputs(runtime.session, InspectInputs(limit=12))
    detailed = inspect_inputs(runtime.session, InspectInputs(limit=12, include_evidence=True))
    assert len(canonical(compact)) < len(canonical(detailed)) * 0.7
    assert [row["input_id"] for row in compact["items"]] == [row["input_id"] for row in detailed["items"]]
    assert compact["source_notes"] == ["同一供应商的重复记录不是独立佐证；原始数值不等于最终模型指标。"]


def test_pagination_returns_executable_action_and_clamps_large_page(tmp_path):
    runtime = large_dataset(tmp_path)
    query = InspectInputs(ticker="600000", limit=24)
    response = inspect_inputs(runtime.session, query)
    assert response["page_limit"] == 12 and response["offset"] == 0
    assert len(response["items"]) == 10 and response["next_action"] is None
    query = InspectInputs(period_end=date(2025, 12, 31), limit=6)
    first = inspect_inputs(runtime.session, query)
    action = first["next_action"]
    assert action["tool"] == "inspect_inputs" and action["arguments"]["period_end"] == "2025-12-31"
    assert action["arguments"]["offset"] == 6 and action["arguments"]["limit"] == 6
    second = inspect_inputs(runtime.session, InspectInputs.model_validate(action["arguments"]))
    assert second["next_action"] is None
    assert len({row["input_id"] for row in first["items"] + second["items"]}) == 10


def test_exact_identifier_query_distinguishes_unreturned_and_unmatched_ids(tmp_path):
    runtime = large_dataset(tmp_path)
    identifiers = [row.input_id for row in runtime.session.input_dataset.records[:7]]
    response = inspect_inputs(runtime.session, InspectInputs(input_ids=[*identifiers, "input_not_present"]))
    assert response["total"] == 7 and len(response["items"]) == 6
    assert response["unmatched_input_ids"] == ["input_not_present"]
    next_page = inspect_inputs(runtime.session, InspectInputs.model_validate(response["next_action"]["arguments"]))
    assert len(next_page["items"]) == 1 and next_page["next_action"] is None
    assert next_page["unmatched_input_ids"] == ["input_not_present"]


def test_metric_mismatch_lists_only_actual_metrics_for_selected_entity_period(tmp_path):
    runtime = large_dataset(tmp_path)
    existing_metric = runtime.session.input_dataset.records[0].metric
    runtime.session.input_dataset.records[1].metric = "raw.unrelated_peer_field"
    response = inspect_inputs(runtime.session, InspectInputs(ticker="600000", period_end=date(2016, 12, 31), metrics=["bad_name"]))
    assert response["items"] == []
    assert response["available_metrics"] == [existing_metric]
    assert response["unmatched_metrics"] == ["bad_name"]
    assert response["next_action"] is None


def test_source_details_are_opt_in_without_permission_or_state_changes(tmp_path):
    runtime = large_dataset(tmp_path)
    before = runtime.session.model_dump(mode="json")
    result = inspect_inputs(runtime.session, InspectInputs(include_evidence=True, limit=30))
    assert result["page_limit"] == 12 and len(result["items"]) == 12
    assert result["next_action"]["arguments"]["include_evidence"] is True
    assert result["evidence_action"] is None
    assert runtime.session.model_dump(mode="json") == before


def test_duplicate_sources_are_marked_not_merged_or_summed(tmp_path):
    _, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    original = runtime.session.input_dataset.records[0]
    duplicate = original.model_copy(deep=True, update={"input_id": "second_source"})
    duplicate.source.source_id = "another_source"
    runtime.session.input_dataset.records.append(duplicate)
    response = inspect_inputs(runtime.session, InspectInputs(metrics=[original.metric]))
    assert len(response["items"]) == 2
    assert {frozenset(group) for group in response["same_value_input_groups"]} == {frozenset({original.input_id, duplicate.input_id})}
    assert all(row["value"] == str(original.value) for row in response["items"])
    duplicate.period_end = date(2025, 12, 31)
    assert inspect_inputs(runtime.session, InspectInputs(metrics=[original.metric]))["same_value_input_groups"] == []


def test_date_filter_explains_existing_share_record_without_relabeling_it(tmp_path):
    _, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    shares = next(row for row in runtime.session.input_dataset.records if row.metric == "common_shares")
    shares.as_of = date(2025, 12, 31)
    response = inspect_inputs(runtime.session, InspectInputs(metrics=["common_shares"], period_end=date(2025, 12, 31)))
    assert response["items"] == []
    dates = response["date_filtered_metrics"][0]
    assert dates["metric"] == "common_shares"
    assert dates["period_ends"] == [] and dates["as_of_dates"] == ["2025-12-31"]
    assert shares.period_end is None and shares.as_of == date(2025, 12, 31)
