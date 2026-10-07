from decimal import Decimal

import pytest

from test_input_workspace import fixture
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.user_message import ReadUserInput, read_user_input
from valuationagent.schemas.inputs import RecordInputs


def test_user_numeric_refs_preserve_source_without_copying_or_choosing_financial_meaning(tmp_path):
    text = "不要联网。\n金额单位为万元。\n2023、2024、2025三个完整年度相同。\n收入1000，归母净利润150。\n估值日2025年12月31日。\n股数100万股，股价20元/股。"
    _, runtime = fixture(tmp_path, text)
    source = read_user_input(runtime, ReadUserInput())
    assert source["lines"][3]["amounts"] == [{"amount_ref": "4:1", "text": "1000"}, {"amount_ref": "4:2", "text": "150"}]
    record_inputs(runtime, RecordInputs(user_basis={"unit_ref": 2, "period_ref": 3, "as_of_ref": 5}, user_values=[
        {"metric": "net_income_parent", "amount_ref": "4:2", "unit": "万元", "period_end": "2025-12-31"},
        {"metric": "common_shares", "amount_ref": "6:1", "unit": "万股", "as_of": "2025-12-31"},
        {"metric": "market_price", "amount_ref": "6:2", "unit": "元", "as_of": "2025-12-31"}]))
    rows = runtime.session.input_dataset.records
    assert [row.value for row in rows] == [Decimal(1500000), Decimal(1000000), Decimal(20)]
    assert [row.original_amount for row in rows] == ["150", "100万股", "20元/股"]
    assert all(row.source.kind == "user" for row in rows)


def test_numeric_refs_do_not_resolve_other_messages_or_credentials(tmp_path):
    text = "不要联网。\ntoken=12345678901234567890\n利润10亿元。"
    _, runtime = fixture(tmp_path, text)
    result = read_user_input(runtime, ReadUserInput())
    assert result["lines"][1]["amounts"] == []
    assert "12345678901234567890" not in str(result)
    for reference in ("2:1", "3:9", "99:1"):
        with pytest.raises(ValueError, match="USER_INPUT_REF"):
            record_inputs(runtime, RecordInputs(user_values=[{"metric": "net_income_parent", "amount_ref": reference, "unit": "亿元"}]))
    with pytest.raises(ValueError, match="USER_INPUT_REF"):
        read_user_input(runtime, ReadUserInput(message_id="not-in-this-workspace"))


def test_paged_read_keeps_original_header_context_without_assigning_metadata(tmp_path):
    text = "不要联网。\n估值日2025年12月31日\n金额万元，股数万股\ntoken=12345678901234567890\n说明\n说明\n说明\n说明\n现金100\n股数200"
    _, runtime = fixture(tmp_path, text)
    result = read_user_input(runtime, ReadUserInput(query="股数200", limit=1))
    assert [row["line"] for row in result["lines"]] == [10]
    assert any(row["line"] == 2 and "2025" in row["text"] for row in result["context_lines"])
    assert any(row["line"] == 3 and "万元" in row["text"] for row in result["context_lines"])
    assert "12345678901234567890" not in str(result)
    assert runtime.session.input_dataset is None


@pytest.mark.parametrize("separator", ["；", "|", " ", "、", ",", "，", "\t"])
def test_multiple_search_terms_use_or_and_keep_original_references(tmp_path, separator):
    _, runtime = fixture(tmp_path, "不要联网，仅用用户数据\n所得税费用50万元\n经营性营运资本增加额0万元\n其他数据10万元")
    result = read_user_input(runtime, ReadUserInput(query=separator.join(["所得税费用", "经营性营运资本增加额"])))
    assert [line["line"] for line in result["lines"]] == [2, 3]
    assert result["lines"][1]["amounts"] == [{"amount_ref": "3:1", "text": "0万元"}]
    assert result["query_terms"] == ["所得税费用", "经营性营运资本增加额"]
    assert runtime.session.input_dataset is None


def test_large_read_request_is_bounded_and_pageable_not_a_schema_failure(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网\n" + "\n".join(f"数据{index}万元" for index in range(40)))
    first = read_user_input(runtime, ReadUserInput(limit=100))
    assert len(first["lines"]) == 30 and first["page_limit"] == 30 and first["next_line"] == 31
    second = read_user_input(runtime, ReadUserInput(start_line=first["next_line"], limit=100))
    assert len(second["lines"]) == 11 and second["lines"][0]["line"] == 31
    assert second["next_line"] is None


def test_unmatched_query_provides_actual_unfiltered_read_instead_of_missing_data_claim(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网\n现金100万元\n负债0万元")
    output = read_user_input(runtime, ReadUserInput(query="[其他].*", start_line=2))
    assert not output["lines"] and output["match_count"] == 0
    assert output["recovery"]["tool"] == "read_user_input"
    args = ReadUserInput.model_validate(output["recovery"]["arguments"])
    recovered = read_user_input(runtime, args)
    assert [line["line"] for line in recovered["lines"]] == [2, 3]
    assert runtime.session.input_dataset is None


def test_missing_amount_reference_returns_safe_actual_line_choices(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网\n收入100万元，净利润20万元，token=12345678901234567890")
    with pytest.raises(ValueError, match="USER_INPUT_REF") as failure:
        record_inputs(runtime, RecordInputs(user_values=[{"metric": "net_income_parent", "amount_ref": "2:9", "unit": "万元"}]))
    detail = str(failure.value)
    assert "2:9" in detail and "2:1" in detail and "2:2" in detail
    assert "12345678901234567890" not in detail and "2:3" not in detail
    assert runtime.session.input_dataset is None


def test_query_metadata_does_not_echo_a_credential_substring(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网\ntoken=12345678901234567890\n收入100万元")
    result = read_user_input(runtime, ReadUserInput(query="12345678901234567890；收入"))
    assert "12345678901234567890" not in str(result)
    assert result["query_terms"] == ["[REDACTED]", "收入"]


def test_ref_cannot_relabel_a_number_or_discard_its_sign(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。\n净利润-10亿元。")
    with pytest.raises(ValueError, match="USER_INPUT_REF"):
        record_inputs(runtime, RecordInputs(user_values=[{"metric": "net_income_parent", "amount_ref": "2:1", "amount_text": "10亿元", "unit": "亿元"}]))
    record_inputs(runtime, RecordInputs(user_values=[{"metric": "net_income_parent", "amount_ref": "2:1", "unit": "亿元"}]))
    assert runtime.session.input_dataset.records[0].value == -1000000000


def test_numeric_refs_do_not_consume_a_partial_year_after_a_comma(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。t=1，2035年末；金额1,234.56万元。")
    result = read_user_input(runtime, ReadUserInput())
    assert [amount["text"] for amount in result["lines"][0]["amounts"]] == ["1", "2035", "1,234.56万元"]


@pytest.mark.parametrize("shared", [False, True])
def test_date_reference_without_date_is_not_silently_dropped(tmp_path, shared):
    _, runtime = fixture(tmp_path, "不要联网。\n2025完整年度净利润150万元。")
    value = {"metric": "net_income_parent", "amount_ref": "2:2", "unit": "万元"}
    kwargs = {"user_basis": {"period_ref": 2}} if shared else {}
    if not shared:
        value["period_ref"] = 2
    with pytest.raises(ValueError, match="INPUT_DATE_REQUIRED"):
        record_inputs(runtime, RecordInputs(user_values=[value], **kwargs))
    assert runtime.session.input_dataset is None


def test_shared_date_is_explicit_and_still_verified_against_original_message(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。\n2025完整年度净利润150万元。\n2025年12月31日股数100万股。")
    result = record_inputs(runtime, RecordInputs(user_basis={"period_ref": 2, "period_end": "2025-12-31"},
        user_values=[{"metric": "net_income_parent", "amount_ref": "2:2", "unit": "万元"}]))
    assert result["saved_records"][0]["period_end"] == "2025-12-31"
    with pytest.raises(ValueError, match="INPUT_DATE_SOURCE"):
        record_inputs(runtime, RecordInputs(user_basis={"period_ref": 2, "period_end": "2024-12-31"},
            user_values=[{"metric": "net_income_parent", "amount_ref": "2:2", "unit": "万元"}]))
