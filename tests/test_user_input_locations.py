import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from test_input_workspace import fixture
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.application.workspace_artifacts import ReportWrite, write_report
from valuationagent.schemas.inputs import RecordInputs, UserInputValue


def one_value(amount_text="10亿元", **changes):
    return RecordInputs(user_values=[{"metric": "net_income_parent", "amount_text": amount_text, "unit": "亿元", **changes}])


def test_ps_user_words_require_no_rewritten_quote_or_network(tmp_path):
    text = "不要联网。营业收入200亿元，普通股10亿股，PS指定3倍，按这些数值试算并出报告。"
    app, runtime = fixture(tmp_path, text)
    runtime.session.draft.methods = ["ps"]
    args = RecordInputs(user_values=[{"metric": "revenue", "amount_text": "200亿元", "unit": "亿元"},
        {"metric": "common_shares", "amount_text": "10亿股", "unit": "亿股", "scope": "issuer"},
        {"metric": "ps_multiple", "amount_text": "3倍", "unit": "ratio", "scope": "assumption"}])
    record_inputs(runtime, args)
    outcome = runtime.calculate(CalculateValuation())
    run = app.state.store.get_run(outcome["run_id"])
    assert run.result.relative[0].per_share_value == Decimal(60)
    assert run.result.analysis_basis == "user_scenario"
    assert replay_bundle(build_valuation_bundle(app.state.store, run))["passed"]
    assert write_report(app.state.research, runtime.session, ReportWrite(format="md"))["numeric_result_available"]
    assert not runtime.session.search_history and not runtime.session.documents
    for row in runtime.session.input_dataset.records:
        locator = json.loads(row.source.locator)
        assert text[locator["amount_start"]:locator["amount_end"]] == row.original_amount
        assert text[locator["quote_start"]:locator["quote_end"]] == row.source.quote


def test_user_schema_does_not_offer_or_accept_model_written_quotes():
    assert "quote" not in UserInputValue.model_json_schema()["properties"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        one_value(quote="用户明确给出净利润10亿元")


def test_target_market_price_is_a_reference_not_a_total_financial_amount(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。测试市场价格20元/股，不用于倒推假设。")
    args = RecordInputs(user_values=[{"metric": "market_price", "amount_text": "20元/股", "unit": "元"}])
    record_inputs(runtime, args)
    row = runtime.session.input_dataset.records[0]
    assert row.value == 20 and row.original_amount == "20元/股"
    args.user_values[0].metric = "revenue"
    with pytest.raises(ValueError, match="INPUT_DIMENSION"):
        record_inputs(runtime, args)


def test_shared_annual_context_and_exact_value_selector(tmp_path):
    text = "不要联网。金额单位为万元。2023、2024、2025三个完整年度均相同：收入1000；净利润150；资本开支150。"
    _, runtime = fixture(tmp_path, text)
    record_inputs(runtime, RecordInputs(user_basis={"unit_quote": "金额单位为万元",
        "period_quote": "2023、2024、2025三个完整年度"}, user_values=[{
            "metric": "net_income_parent", "amount_text": "150", "unit": "万元",
            "value_context": "净利润150", "period_end": f"{year}-12-31"} for year in (2023, 2024, 2025)]))
    records = runtime.session.input_dataset.records
    assert len(records) == 3 and all(row.value == 1500000 for row in records)
    for row in records:
        location = json.loads(row.source.locator)
        assert text[location["amount_start"]:location["amount_end"]] == "150"
        assert location["amount_start"] == text.index("净利润150") + 3
    with pytest.raises(ValueError, match="INPUT_DATE_SOURCE"):
        record_inputs(runtime, RecordInputs(user_basis={"unit_quote": "金额单位为万元",
            "period_quote": "2023、2024、2025三个完整年度"}, user_values=[{
                "metric": "revenue", "amount_text": "1000", "unit": "万元", "period_end": "2022-12-31"}]))


@pytest.mark.parametrize("text,amount,code", [
    ("不要联网。利润110亿元。", "10亿元", "INPUT_AMOUNT_BOUNDARY"),
    ("不要联网。利润-10亿元。", "10亿元", "INPUT_AMOUNT_BOUNDARY"),
    ("不要联网。利润−10亿元。", "10亿元", "INPUT_AMOUNT_BOUNDARY"),
    ("不要联网。利润（10亿元）。", "10亿元", "INPUT_AMOUNT_BOUNDARY"),
    ("不要联网。利润10亿元。", "10", "INPUT_AMOUNT_BOUNDARY"),
    ("不要联网。利润10.50亿元。", "50亿元", "INPUT_AMOUNT_BOUNDARY"),
    ("不要联网。利润1e10亿元。", "10亿元", "INPUT_AMOUNT_BOUNDARY"),
    ("不要联网。利润10亿元。", "11亿元", "INPUT_SOURCE"),
])
def test_numeric_substrings_cannot_change_sign_scale_or_magnitude(tmp_path, text, amount, code):
    _, runtime = fixture(tmp_path, text)
    with pytest.raises(ValueError, match=code):
        record_inputs(runtime, one_value(amount))
    assert runtime.session.input_dataset is None


def test_shared_unit_is_anchored_and_does_not_accept_yuan_inside_ten_thousand(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。金额单位为万元；利润10。")
    with pytest.raises(ValueError, match="INPUT_UNIT"):
        record_inputs(runtime, one_value("10", unit="元", unit_quote="金额单位为万元"))
    with pytest.raises(ValueError, match="INPUT_UNIT"):
        record_inputs(runtime, one_value("10", unit="万元"))
    record_inputs(runtime, one_value("10", unit="万元", unit_quote="金额单位为万元"))
    row = runtime.session.input_dataset.records[0]
    assert row.value == Decimal(100000)
    assert json.loads(row.source.locator)["unit_quote"] == "金额单位为万元"


def test_repeated_values_require_explicit_occurrence_with_original_context(tmp_path):
    text = "不要联网。收入10亿元；归母净利润10亿元。"
    _, runtime = fixture(tmp_path, text)
    with pytest.raises(ValueError, match="INPUT_AMOUNT_AMBIGUOUS"):
        record_inputs(runtime, one_value())
    with pytest.raises(ValueError, match="INPUT_AMOUNT_OCCURRENCE"):
        record_inputs(runtime, one_value(amount_occurrence=2))
    record_inputs(runtime, one_value(amount_occurrence=1))
    row = runtime.session.input_dataset.records[0]
    location = json.loads(row.source.locator)
    assert location["amount_start"] == text.rindex("10亿元")
    assert location["amount_occurrence"] == 1


def test_complete_match_ignores_numeric_substring_elsewhere(tmp_path):
    text = "不要联网。收入110亿元，归母净利润10亿元。"
    _, runtime = fixture(tmp_path, text)
    record_inputs(runtime, one_value())
    location = json.loads(runtime.session.input_dataset.records[0].source.locator)
    assert location["amount_start"] == text.rindex("10亿元")


def test_bounded_excerpt_is_exact_unicode_slice_and_not_a_paraphrase(tmp_path):
    text = "不要联网。" + "背景说明😀" * 400 + "归母净利润10亿元。" + "后续约束😀" * 400
    _, runtime = fixture(tmp_path, text)
    record_inputs(runtime, one_value())
    row = runtime.session.input_dataset.records[0]
    location = json.loads(row.source.locator)
    assert len(row.source.quote) < 400
    assert text[location["quote_start"]:location["quote_end"]] == row.source.quote
    assert text[location["amount_start"]:location["amount_end"]] == "10亿元"


def test_assistant_or_other_workspace_cannot_supply_user_input(tmp_path):
    app, runtime = fixture(tmp_path, "不要联网。解释PE，不提供利润数字。")
    assistant = app.state.store.add_message(runtime.session.session_id, "assistant", "利润10亿元。", "agent")
    for message_id in ("", assistant.message_id, "not_this_workspace"):
        with pytest.raises(ValueError, match="INPUT_SOURCE"):
            record_inputs(runtime, one_value(message_id=message_id))
    assert runtime.session.input_dataset is None


def test_explicit_negative_and_percentage_are_preserved(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。归母净利润-10亿元；税率25%。")
    record_inputs(runtime, RecordInputs(user_values=[
        {"metric": "net_income_parent", "amount_text": "-10亿元", "unit": "亿元"},
        {"metric": "tax_rate", "amount_text": "25%", "unit": "%"}]))
    assert [row.value for row in runtime.session.input_dataset.records] == [Decimal(-1000000000), Decimal("0.25")]


def test_automatic_excerpt_does_not_copy_adjacent_credentials(tmp_path):
    secret = "sk-synthetic-test-secret-value"
    text = f"不要联网。token={secret} 归母净利润10亿元。password=synthetic-password 继续解释。"
    _, runtime = fixture(tmp_path, text)
    record_inputs(runtime, one_value())
    row = runtime.session.input_dataset.records[0]
    locator = json.loads(row.source.locator)
    assert row.source.quote == text[locator["quote_start"]:locator["quote_end"]]
    assert secret not in row.model_dump_json() and "synthetic-password" not in row.model_dump_json()
    assert "归母净利润10亿元" in row.source.quote


def test_numeric_credential_cannot_be_used_as_financial_input(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。token=12345678901234567890 只解释方法，未给出利润。")
    with pytest.raises(ValueError, match="INPUT_SOURCE"):
        record_inputs(runtime, one_value("12345678901234567890", unit="ratio", metric="pe_multiple"))


def test_unit_suffix_separates_amount_from_following_year_or_sentence(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。归母净利润10亿元，2025年度数据。")
    record_inputs(runtime, one_value())
    assert runtime.session.input_dataset.records[0].value == Decimal(1000000000)


def test_date_reference_cannot_copy_credentials_into_financial_proof(tmp_path):
    text = "不要联网。归母净利润10亿元。2025年12月31日 token=synthetic-private-value"
    _, runtime = fixture(tmp_path, text)
    with pytest.raises(ValueError, match="INPUT_DATE_SOURCE"):
        record_inputs(runtime, one_value(period_end="2025-12-31", period_quote="2025年12月31日 token=synthetic-private-value"))


@pytest.mark.parametrize("quote", ["2025年", "token=synthetic-private-value"])
def test_unused_date_reference_is_not_saved_as_evidence(tmp_path, quote):
    _, runtime = fixture(tmp_path, "不要联网。归母净利润10亿元。" + quote)
    with pytest.raises(ValueError, match="INPUT_DATE_SOURCE"):
        record_inputs(runtime, one_value(period_quote=quote))
    assert runtime.session.input_dataset is None


def test_ambiguous_amount_recovery_does_not_echo_adjacent_credentials(tmp_path):
    secret = "sk-synthetic-test-secret-value"
    _, runtime = fixture(tmp_path, f"不要联网。收入10亿元。token={secret} 利润10亿元。")
    with pytest.raises(ValueError, match="INPUT_AMOUNT_AMBIGUOUS") as error:
        record_inputs(runtime, one_value())
    assert secret not in str(error.value)
