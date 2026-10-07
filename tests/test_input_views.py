import copy
import json
from datetime import date

import pytest

from test_input_workspace import fixture, values
from valuationagent.application.input_views import InspectInputs, input_overview, inspect_inputs
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.turn_control import guard_tool
from valuationagent.core.tools import canonical
from valuationagent.llm.agent import compact_tool_history
from valuationagent.schemas.inputs import ComparableSelection, RecordInputs


def large_dataset(tmp_path):
    _, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    dataset = runtime.session.input_dataset
    original = dataset.records[0]
    dataset.records = [original.model_copy(update={"input_id": f"input_{year}_{number}",
        "entity_ticker": f"600{number:03d}.SH", "entity": f"Peer {number}",
        "role": "comparable", "period_end": date(year, 12, 31)}) for year in range(2016, 2026) for number in range(10)]
    dataset.comparables = {f"600{number:03d}.SH": ComparableSelection(ticker=f"600{number:03d}.SH",
        name=f"Peer {number}", rationale="公开业务范围说明与尚待复核的可比差异。" * 70) for number in range(10)}
    return runtime


def test_record_receipt_does_not_repeat_the_whole_dataset(tmp_path):
    _, runtime = fixture(tmp_path)
    result = record_inputs(runtime, values())
    assert "active_records" not in result
    assert len(result["saved_records"]) == 3
    assert all(row["source_kind"] == "user" for row in result["saved_records"])
    assert all("source" not in row for row in result["saved_records"])
    assert len(record_inputs(runtime, values())["saved_records"]) == 0
    assert all(row["source"]["quote"] for row in inspect_inputs(runtime.session, InspectInputs(include_evidence=True))["items"])


def test_large_dataset_has_bounded_preview_and_readable_omitted_records(tmp_path):
    runtime = large_dataset(tmp_path)
    overview = input_overview(runtime.session)
    assert overview["active_count"] == 100
    assert len(overview["active_records"]) == 12 and overview["records_omitted"] == 88
    assert overview["comparables"][0]["rationale_truncated"]
    assert len(canonical(overview)) < 11000
    args = InspectInputs(ticker="600000", period_end=date(2016, 12, 31))
    result = inspect_inputs(runtime.session, args)
    assert result["total"] == 1 and result["items"][0]["input_id"] == "input_2016_0"
    assert result["items"][0]["source"]["kind"] == "user"
    assert "provider_binding" not in result["items"][0]["source"]
    selected = inspect_inputs(runtime.session, InspectInputs(section="comparables", ticker="600000"))
    assert len(selected["items"][0]["rationale"]) > 180


def test_paging_is_stable_without_research_or_state_mutation(tmp_path):
    runtime = large_dataset(tmp_path)
    before = runtime.session.model_dump(mode="json")
    identifiers = []
    offset = 0
    while offset is not None:
        result = inspect_inputs(runtime.session, InspectInputs(offset=offset, limit=12))
        identifiers.extend(row["input_id"] for row in result["items"])
        offset = result["next_offset"]
    assert len(set(identifiers)) == len(identifiers) == 100
    assert runtime.session.model_dump(mode="json") == before
    guard_tool(runtime.session, "inspect_inputs", {})


def test_superseded_inputs_are_explicit_not_returned_as_active(tmp_path):
    _, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    previous = runtime.session.input_dataset.records[0]
    args = RecordInputs(user_values=[{"metric": "net_income_parent", "amount_text": "10亿元", "unit": "亿元",
        "replaces": [previous.input_id]}])
    record_inputs(runtime, args)
    assert inspect_inputs(runtime.session, InspectInputs(input_ids=[previous.input_id]))["total"] == 0
    result = inspect_inputs(runtime.session, InspectInputs(input_ids=[previous.input_id], include_superseded=True))
    assert result["items"][0]["active"] is False


def test_acquisition_directories_have_real_bounded_recovery(tmp_path):
    _, runtime = fixture(tmp_path)
    runtime.session.input_acquisition = {"status": "partial", "issues": [{"code": f"MISSING_{number}"} for number in range(20)],
        "coverage": [{"ticker": "600000.SH", "year": 2025}], "sources": [{"ticker": "600000.SH", "file_id": "real"}]}
    result = inspect_inputs(runtime.session, InspectInputs(section="acquisition_issues", offset=6))
    assert result["items"][0]["code"] == "MISSING_6" and result["next_offset"] == 12
    assert inspect_inputs(runtime.session, InspectInputs(section="acquisition_sources", ticker="600000"))["items"][0]["file_id"] == "real"
    with pytest.raises(ValueError, match="INPUT_QUERY_SCOPE"):
        inspect_inputs(runtime.session, InspectInputs(section="comparables", metrics=["revenue"]))


def test_large_input_state_can_compact_without_losing_user_request_or_source_truth(tmp_path):
    runtime = large_dataset(tmp_path)
    state = runtime.working_state()
    state["context"]["current_request"] = {"message_id": "real_user", "content": "不要联网；只分析已有输入，不修改原数值。"}
    state["input_acquisition"] = {"status": "partial", "issues": [{"code": "MISSING", "detail": "detail" * 100} for _ in range(50)]}
    state["provider_sources"] = {"sources": [{"file_id": f"real_{index}", "ticker": "600000.SH"} for index in range(60)], "omitted": 0}
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": canonical(state)}]
    before = copy.deepcopy(messages)
    compacted = compact_tool_history(messages, 5000)
    assert len(canonical(compacted)) <= 5000 and messages == before
    kept = json.loads(compacted[1]["content"])
    assert kept["input_dataset"]["active_count"] == 100
    assert kept["input_dataset"]["retrieve_with"] == "inspect_inputs"
    assert kept["context"]["current_request"] == state["context"]["current_request"]
    assert kept["input_dataset"]["records_omitted"] >= 88
    assert kept["input_acquisition"]["issues_omitted"] == 50
    kept["research_plan"] = {"coverage": "已有来源的年度覆盖状态。" * 600}
    next_messages = [{**compacted[0]}, {**compacted[1], "content": canonical(kept)}]
    second = compact_tool_history(next_messages, 5000)
    assert len(canonical(second)) <= 5000
    second_state = json.loads(second[1]["content"])
    assert second_state["input_acquisition"]["issues_omitted"] == 50
    assert second_state["context"]["task_state"]["forecast_proposal"] is None
    assert second_state["context"]["task_state"]["last_issue"] == state["context"]["task_state"]["last_issue"]
    assert inspect_inputs(runtime.session, InspectInputs())["total"] == 100
