from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from test_input_workspace import fixture
from valuationagent.application.user_input_table import UserInputTable, record_user_table


def test_runtime_explains_read_prerequisite_and_reveals_table_after_read(tmp_path):
    from valuationagent.application.user_message import ReadUserInput, read_user_input
    from valuationagent.core.tools import ToolSpec
    from valuationagent.llm.tool_catalog import LoadTools

    _, runtime = fixture(tmp_path)
    tools = [ToolSpec("load_tools", "load", LoadTools, runtime.tool_catalog.load).schema(),
        ToolSpec("read_user_input", "read", ReadUserInput, lambda args: read_user_input(runtime, args)).schema(),
        ToolSpec("record_user_inputs", "save", UserInputTable, lambda args: record_user_table(runtime, args)).schema()]
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "{}"}]
    _, selected = runtime.adapt_request(messages, tools)
    assert "record_user_inputs" not in {tool["function"]["name"] for tool in selected}
    with pytest.raises(ValueError, match="TOOL_PREREQUISITE.*read_user_input"):
        runtime.tool_catalog.load(LoadTools(names=["record_user_inputs"]))
    read_user_input(runtime, ReadUserInput())
    _, selected = runtime.adapt_request(messages, tools)
    assert "record_user_inputs" in {tool["function"]["name"] for tool in selected}
    assert runtime.tool_catalog.load(LoadTools(names=["record_user_inputs"]))["executed"] is False
    assert runtime.session.input_dataset is None


def test_shared_units_and_explicit_years_expand_without_inventing_dates(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。金额单位万元。\n2023、2024、2025三个完整年度数据相同。\n营业收入1000\n归母净利润150")
    result = record_user_table(runtime, UserInputTable(unit="万元", unit_line=1,
        periods=[{"date": f"{year}-12-31", "line": 2} for year in (2023, 2024, 2025)],
        rows=[{"metric": "revenue", "amount_ref": "3:1"}, {"metric": "net_income_parent", "amount_ref": "4:1"}]))
    assert result["active_count"] == 6
    records = runtime.session.input_dataset.active_records()
    assert {row.period_end for row in records} == {date(year, 12, 31) for year in (2023, 2024, 2025)}
    assert all(row.value == (10000000 if row.metric == "revenue" else 1500000) for row in records)
    with pytest.raises(ValueError, match="INPUT_DATE_SOURCE"):
        record_user_table(runtime, UserInputTable(unit="万元", unit_line=1,
            periods=[{"date": "2022-12-31", "line": 2}], rows=[{"metric": "revenue", "amount_ref": "3:1"}]))
    assert len(runtime.session.input_dataset.records) == 6


def test_date_and_reference_are_required_together_in_the_model_schema():
    with pytest.raises(ValidationError):
        UserInputTable(unit="万元", periods=[{"line": 1}], rows=[{"metric": "revenue", "amount_ref": "3:1"}])
    with pytest.raises(ValidationError):
        UserInputTable(unit="万元", as_of={"date": "2025-12-31"}, rows=[{"metric": "revenue", "amount_ref": "3:1"}])


def test_unknown_dates_stay_unknown_and_unit_override_preserves_dimensions(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。净利润10亿元，普通股5亿股，PE20倍。")
    record_user_table(runtime, UserInputTable(unit="亿元", unit_line=None, rows=[
        {"metric": "net_income_parent", "amount_ref": "1:1"},
        {"metric": "common_shares", "amount_ref": "1:2", "unit": "亿股"},
        {"metric": "pe_multiple", "amount_ref": "1:3", "unit": "ratio"}]))
    records = runtime.session.input_dataset.active_records()
    assert all(row.as_of is None and row.period_end is None for row in records)
    assert {row.metric: row.value for row in records} == {
        "net_income_parent": Decimal(1000000000), "common_shares": Decimal(500000000), "pe_multiple": Decimal(20)}


def test_wrong_scale_or_nonexistent_reference_does_not_partially_save(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。单位万元\n收入1000，净利润150")
    with pytest.raises(ValueError, match="INPUT_UNIT"):
        record_user_table(runtime, UserInputTable(unit="元", unit_line=1,
            rows=[{"metric": "revenue", "amount_ref": "2:1"}]))
    with pytest.raises(ValueError, match="USER_INPUT_REF"):
        record_user_table(runtime, UserInputTable(unit="万元", unit_line=1,
            rows=[{"metric": "revenue", "amount_ref": "2:1"}, {"metric": "net_income_parent", "amount_ref": "2:9"}]))
    assert runtime.session.input_dataset is None


def test_literal_units_need_no_redundant_llm_conversion(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。收入100万元，股数20万股，税率25%，PE10倍")
    record_user_table(runtime, UserInputTable(unit="万元", unit_line=None, rows=[
        {"metric": "revenue", "amount_ref": "1:1"}, {"metric": "common_shares", "amount_ref": "1:2"},
        {"metric": "tax_rate", "amount_ref": "1:3"}, {"metric": "pe_multiple", "amount_ref": "1:4"}]))
    assert {row.metric: (row.value, row.unit) for row in runtime.session.input_dataset.records} == {
        "revenue": (Decimal(1000000), "万元"), "common_shares": (Decimal(200000), "万股"),
        "tax_rate": (Decimal(".25"), "%"), "pe_multiple": (Decimal(10), "ratio")}
    with pytest.raises(ValueError, match="INPUT_UNIT"):
        record_user_table(runtime, UserInputTable(unit="万元", unit_line=None, rows=[
            {"metric": "revenue", "amount_ref": "1:1", "unit": "元"}]))


def test_naked_zero_requires_explicit_shared_unit_and_table_error_names(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。金额万元\n现金0")
    with pytest.raises(ValidationError):
        UserInputTable(unit="万元", rows=[{"metric": "cash_and_non_operating_assets", "amount_ref": "2:1"}])
    with pytest.raises(ValueError, match="USER_TABLE_UNIT_LINE.*unit_line"):
        record_user_table(runtime, UserInputTable(unit="万元", unit_line=None,
            rows=[{"metric": "cash_and_non_operating_assets", "amount_ref": "2:1"}]))
    assert runtime.session.input_dataset is None


def test_table_schema_exposes_canonical_fields_and_typed_raw_items(tmp_path):
    schema = UserInputTable.model_json_schema()["$defs"]["UserValueRow"]["properties"]
    assert "profit_before_tax" in schema["metric"]["anyOf"][0]["enum"]
    assert "ebt" not in schema["metric"]["anyOf"][0]["enum"]
    assert "period_kind" in schema
    _, runtime = fixture(tmp_path, "不要联网。金额万元\n2025完整年度\n利息费用2")
    with pytest.raises(ValueError, match="INPUT_PERIOD_KIND"):
        record_user_table(runtime, UserInputTable(unit="万元", unit_line=1,
            periods=[{"date": "2025-12-31", "line": 2}],
            rows=[{"metric": "raw.interest_expense", "amount_ref": "3:1"}]))
    record_user_table(runtime, UserInputTable(unit="万元", unit_line=1,
        periods=[{"date": "2025-12-31", "line": 2}],
        rows=[{"metric": "raw.interest_expense", "amount_ref": "3:1", "period_kind": "annual"}]))
    row = runtime.session.input_dataset.active_records()[0]
    assert row.period_kind == "annual" and row.value == Decimal(20000)


def test_multi_year_replacement_preserves_individual_year_lineage(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。金额万元\n2023、2024、2025完整年度同值\n收入100\n更正收入120")
    periods = [{"date": f"{year}-12-31", "line": 2} for year in (2023, 2024, 2025)]
    record_user_table(runtime, UserInputTable(unit="万元", unit_line=1, periods=periods,
        rows=[{"metric": "revenue", "amount_ref": "3:1"}]))
    before = {row.input_id: row for row in runtime.session.input_dataset.active_records()}
    record_user_table(runtime, UserInputTable(unit="万元", unit_line=1, periods=periods,
        rows=[{"metric": "revenue", "amount_ref": "4:1", "replaces": list(before)}]))
    active = runtime.session.input_dataset.active_records()
    assert len(active) == 3
    assert all(row.value == Decimal(1200000) for row in active)
    assert all(len(row.supersedes) == 1 and before[row.supersedes[0]].period_end == row.period_end for row in active)
    with pytest.raises(ValueError, match="USER_TABLE_REPLACEMENT"):
        record_user_table(runtime, UserInputTable(unit="万元", unit_line=1, periods=periods[:2],
            rows=[{"metric": "revenue", "amount_ref": "4:1", "replaces": [row.input_id for row in active]}]))
