from decimal import Decimal

import pytest

from valuationagent.application.extraction_recovery import record_attempt, recovery_plan
from valuationagent.application.observation_extraction import observation_amount
from test_multisource_extraction import runtime_at
from test_observation_extraction import example, review, submit


@pytest.mark.parametrize("literal,unit,expected", [
    ("7.71亿元", "亿元", "7.71"), ("−2.96 亿元", "亿元", "-2.96"),
    ("1,200千元", "千元", "1200"), ("（12.5） 万元", "万元", "-12.5"),
    ("799,882,879股", "股", "799882879"), ("12倍", "ratio", "12"),
    ("12.5%", "%", "12.5"),
])
def test_numeric_suffix_only_matches_explicit_declared_unit(literal, unit, expected):
    assert observation_amount(literal, unit) == Decimal(expected)


@pytest.mark.parametrize("literal,unit", [
    ("7.71亿元", "元"), ("7.71万元", "亿元"), ("1000万股", "万元"),
    ("12倍", "元"), ("约7.71亿元", "亿元"), ("7至8亿元", "亿元"),
    ("--亿元", "亿元"), ("7.71e3亿元", "亿元"),
])
def test_units_do_not_enable_scaling_guessing_ranges_or_missing_values(literal, unit):
    with pytest.raises(ValueError, match="AMOUNT_"):
        observation_amount(literal, unit)


def test_unit_suffix_preserves_raw_input_and_reviewable_numeric_anchor(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime, amount="1,200千元")
    result = submit(runtime, args)
    assert result["saved_count"] == 1
    fact = runtime.session.facts[0]
    assert fact.raw_value == "1,200千元" and fact.quote == "1,200"
    assert fact.normalized_value == "1200000" and fact.status == "proposed"
    review(runtime)
    assert fact.status == "confirmed" and not fact.warnings


def test_unit_mismatch_recovery_does_not_redirect_to_html_source_or_network(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime, amount="1,200亿元")
    result = submit(runtime, args)
    assert not result["saved_count"] and "AMOUNT_UNIT" in result["rows"][0]["error"]
    record_attempt(runtime.session, "extract_observations", args, result)
    choices = recovery_plan(runtime.session)["files"][0]["next_choices"]
    assert [choice["tool"] for choice in choices] == ["extract_observations"]
