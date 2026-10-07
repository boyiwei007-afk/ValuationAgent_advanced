import pytest

from test_input_workspace import fixture
from valuationagent.application.input_workspace import bind_user_date
from valuationagent.application.user_input_table import UserInputTable, record_user_table


def test_wrong_amount_line_as_date_returns_existing_dated_line_and_keeps_atomicity(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。金额万元。\n估值日2025年12月31日。\n现金、有息债务均为0。")
    args = UserInputTable(unit="万元", unit_line=1, as_of={"date": "2025-12-31", "line": 3},
        rows=[{"metric": "cash_and_non_operating_assets", "amount_ref": "3:1"}])
    with pytest.raises(ValueError, match="INPUT_DATE_SOURCE") as error:
        record_user_table(runtime, args)
    assert '"line":2' in str(error.value)
    assert "估值日2025年12月31日" in str(error.value)
    assert runtime.session.input_dataset is None
    args.as_of.line = 2
    record_user_table(runtime, args)
    row = runtime.session.input_dataset.active_records()[0]
    assert str(row.as_of) == "2025-12-31" and row.value == 0


def test_date_candidates_do_not_convert_quarters_or_leak_credentials():
    from datetime import date

    text = "2025年第三季度\n日期2025年12月31日 密钥sk-" + "x" * 40 + "\n2024完整年度\n没有可用的日期"
    with pytest.raises(ValueError, match="INPUT_DATE_SOURCE") as error:
        bind_user_date(date(2025, 12, 31), "没有可用的日期", text, annual=True)
    assert "sk-" not in str(error.value)
    assert '"date_candidates":[]' in str(error.value)


def test_shared_annual_date_candidates_keep_actual_line_numbers(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网。金额万元\n2023、2024、2025三个完整年度同值\n经营性营运资本变动0；2022年末余额200")
    with pytest.raises(ValueError, match="INPUT_DATE_SOURCE") as error:
        record_user_table(runtime, UserInputTable(unit="万元", unit_line=1,
            periods=[{"date": "2025-12-31", "line": 3}],
            rows=[{"metric": "change_operating_nwc", "amount_ref": "3:1"}]))
    assert '"line":2' in str(error.value)
    assert "完整年度同值" in str(error.value)
    assert runtime.session.input_dataset is None


@pytest.mark.parametrize("secret", ["sk-" + "x" * 40, "tvly-dev-" + "x" * 40, "x" * 32 + "-infoway"])
def test_credentials_adjacent_to_chinese_text_are_redacted(secret):
    from valuationagent.llm.context_manager import redact_context_text, sensitive_spans

    prefix = "这是密钥"
    text = prefix + secret + "请不要外泄"
    assert sensitive_spans(text) == [(len(prefix), len(prefix) + len(secret))]
    assert redact_context_text(text) == "这是密钥[REDACTED]请不要外泄"
