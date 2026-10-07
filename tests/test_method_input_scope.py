from decimal import Decimal

import pytest

from test_input_workspace import TEXT, fixture, values
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.record_inputs import record_inputs
from valuationagent.schemas.inputs import RecordInputs


def conflicting_workspace(tmp_path):
    app, runtime = fixture(tmp_path, TEXT + "\n资本开支候选1亿元与2亿元；WACC候选10%与12%。")
    record_inputs(runtime, values())
    record_inputs(runtime, RecordInputs(user_values=[
        {"metric": "capital_expenditure", "amount_text": amount, "unit": "亿元"} for amount in ("1亿元", "2亿元")]))
    record_inputs(runtime, RecordInputs(user_values=[
        {"metric": "wacc", "amount_text": amount, "unit": "%", "scope": "assumption"} for amount in ("10%", "12%")]))
    return app, runtime


def test_unused_dcf_conflicts_do_not_block_pe_or_enter_its_frozen_inputs(tmp_path):
    app, runtime = conflicting_workspace(tmp_path)
    before = runtime.session.input_dataset.model_dump()
    request = prepare_dataset(runtime.session, ["pe"])
    assert request.financials.capital_expenditure is None
    assert request.assumptions.wacc is None
    assert {row["metric"] for row in request.input_records} == {"net_income_parent", "common_shares", "pe_multiple"}
    assert runtime.session.input_dataset.model_dump() == before
    outcome = runtime.calculate(CalculateValuation())
    record = app.state.store.get_run(outcome["run_id"])
    assert record.result.relative[0].per_share_value == Decimal(40)
    assert len(runtime.session.input_dataset.active_records()) == 7


@pytest.mark.parametrize("methods", [["dcf"], ["pe", "dcf"]])
def test_requested_method_conflicts_are_not_silently_dropped(tmp_path, methods):
    _, runtime = conflicting_workspace(tmp_path)
    with pytest.raises(ValueError, match="INPUT_CONFLICT"):
        prepare_dataset(runtime.session, methods)


def test_ps_does_not_resolve_or_use_conflicting_pe_earnings(tmp_path):
    _, runtime = fixture(tmp_path, TEXT + "\n另有净利润候选12亿元；营业收入30亿元；PS为3倍。")
    runtime.session.draft.methods = ["pe", "ps"]
    record_inputs(runtime, values())
    record_inputs(runtime, RecordInputs(user_values=[
        {"metric": "net_income_parent", "amount_text": "12亿元", "unit": "亿元"},
        {"metric": "revenue", "amount_text": "30亿元", "unit": "亿元"},
        {"metric": "ps_multiple", "amount_text": "3倍", "unit": "ratio", "scope": "assumption"}]))
    request = prepare_dataset(runtime.session, ["ps"])
    assert request.financials.revenue == Decimal(3000000000)
    assert request.financials.net_income_parent is None
    assert request.assumptions.relative_multiples == {"ps": Decimal(3)}
    with pytest.raises(ValueError, match="INPUT_CONFLICT"):
        prepare_dataset(runtime.session, ["pe"])


def test_conflict_feedback_contains_actual_input_ids_and_source_positions(tmp_path):
    _, runtime = fixture(tmp_path, TEXT + "\n归母净利润另有候选12亿元。")
    record_inputs(runtime, values())
    receipt = record_inputs(runtime, RecordInputs(user_values=[
        {"metric": "net_income_parent", "amount_text": "12亿元", "unit": "亿元"}]))
    ids = {row.input_id for row in runtime.session.input_dataset.active_records() if row.metric == "net_income_parent"}
    assert len(receipt["conflicts"]) == 1
    assert {row["input_id"] for row in receipt["conflicts"][0]["candidates"]} == ids
    assert receipt["conflicts"][0]["metric"] == "net_income_parent"
    with pytest.raises(ValueError, match="INPUT_CONFLICT") as error:
        prepare_dataset(runtime.session, ["pe"])
    assert all(identity in str(error.value) for identity in ids)
    assert "10亿元" in str(error.value) and "12亿元" in str(error.value)


def test_unrelated_forecast_assumptions_do_not_enter_equity_only_request(tmp_path):
    _, runtime = conflicting_workspace(tmp_path)
    request = prepare_dataset(runtime.session, ["pe"], assumptions={"wacc": Decimal(".13"),
        "terminal_growth": Decimal(".04"), "relative_multiples": {"ps": Decimal(3)}})
    assert request.assumptions.wacc is None and request.assumptions.terminal_growth is None
    assert request.assumptions.relative_multiples == {"pe": Decimal(20)}


def test_prior_year_conflict_is_retained_but_does_not_block_current_pe(tmp_path):
    _, runtime = fixture(tmp_path, TEXT + "\n2023、2024两个完整年度；旧年净利润12亿元。")
    record_inputs(runtime, values())
    rows = runtime.session.input_dataset.active_records()
    profit = next(row for row in rows if row.metric == "net_income_parent")
    record_inputs(runtime, RecordInputs(user_values=[
        {"metric": "net_income_parent", "amount_text": "10亿元", "unit": "亿元",
            "period_end": "2023-12-31", "period_quote": "2023、2024两个完整年度", "replaces": [profit.input_id]},
        {"metric": "net_income_parent", "amount_text": "12亿元", "unit": "亿元",
            "period_end": "2023-12-31", "period_quote": "2023、2024两个完整年度"},
        {"metric": "net_income_parent", "amount_text": "10亿元", "unit": "亿元",
            "period_end": "2024-12-31", "period_quote": "2023、2024两个完整年度"}]))
    request = prepare_dataset(runtime.session, ["pe"])
    assert str(request.financials.period_end) == "2024-12-31"
    assert request.financials.net_income_parent == Decimal(1000000000)
    assert all(row["period_end"] != "2023-12-31" for row in request.input_records)
    assert len([row for row in runtime.session.input_dataset.active_records() if str(row.period_end) == "2023-12-31"]) == 2
