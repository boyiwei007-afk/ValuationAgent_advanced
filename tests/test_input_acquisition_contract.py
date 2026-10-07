from datetime import date

import pytest

from test_input_acquisition import acquisition_workspace
from test_provider_raw_inputs import raw_workspace
from valuationagent.application.input_acquisition import AcquireFinancialInputs, acquire_financial_inputs
from valuationagent.application.input_calculations import CALCULATION_KINDS
from valuationagent.application.input_derivations import DERIVATIONS, RAW_INPUT_FIELDS
from valuationagent.application.input_workspace import FINANCIAL_FIELDS
from valuationagent.market.tushare_contracts import CONTRACTS


@pytest.mark.parametrize("metric", ["total_market_capitalization", "operate_profit", "money_capital",
    "depreciation_and_amortization", "raw.tushare_balancesheet_invented"])
def test_unknown_acquisition_metric_rejected_during_argument_validation(metric):
    with pytest.raises(ValueError, match="INPUT_ACQUISITION_METRIC") as failure:
        AcquireFinancialInputs(years=[2025], metrics=["revenue", metric])
    message = str(failure.value)
    assert metric in message and "market_cap" in message and "raw.tushare_balancesheet_money_cap" in message


def test_acquisition_metric_schema_matches_model_and_actual_provider_contracts():
    metrics = AcquireFinancialInputs.model_json_schema()["properties"]["metrics"]["items"]
    expected = (FINANCIAL_FIELDS | RAW_INPUT_FIELDS | DERIVATIONS.keys() | CALCULATION_KINDS.keys()
        | {contract[0] for contracts in CONTRACTS.values() for contract in contracts.values()})
    assert set(metrics.get("enum", [])) == expected
    assert "money_capital" not in metrics["enum"] and "operate_profit" not in metrics["enum"]
    for metric in expected:
        assert AcquireFinancialInputs(years=[2025], metrics=[metric]).metrics == [metric]


def test_metric_validation_rechecks_nonvalidating_copies_before_acquisition_side_effects(tmp_path):
    app, runtime, args, calls = acquisition_workspace(tmp_path)
    args = args.model_copy(update={"metrics": ["money_capital"]})
    before = runtime.session.model_dump()
    with pytest.raises(ValueError, match="INPUT_ACQUISITION_METRIC"):
        acquire_financial_inputs(runtime, args)
    assert not calls and runtime.session.model_dump() == before
    assert app.state.store.get_research(runtime.session.session_id).input_dataset is None


@pytest.mark.parametrize("methods", [["pe"], ["dcf", "ev_ebitda"]])
@pytest.mark.parametrize("metric,statement", [
    ("raw.tushare_income_fin_exp_int_exp", "income"),
    ("raw.tushare_balancesheet_money_cap", "balancesheet"),
    ("raw.tushare_cashflow_use_right_asset_dep", "cashflow"),
])
def test_explicit_raw_query_selects_source_without_forcing_peer_market_day(tmp_path, methods, metric, statement):
    _, runtime, args, calls = raw_workspace(tmp_path)
    runtime.session.draft.methods = methods
    args.metrics = [metric]
    outcome = acquire_financial_inputs(runtime, args)
    assert {call["api_name"] for call in calls} == {"stock_basic", "income", statement}
    assert not any(issue["code"].startswith("INPUT_MARKET") or issue["code"] == "INPUT_NO_COMMON_MARKET_DATE"
        for issue in outcome["issues"])
    records = runtime.session.input_dataset.active_records()
    assert {row.metric for row in records} == {metric}
    assert any(row.role == "comparable" and row.period_end == date(2025, 12, 31) for row in records)
    assert runtime.session.draft.peer_pricing_date is None


@pytest.mark.parametrize("metric,operands", [
    ("tax_rate", {"profit_before_tax", "income_tax_expense"}),
    ("capital_expenditure", {"cash_paid_for_ppe_intangibles"}),
])
def test_explicit_deterministic_derivation_does_not_capture_unrelated_raw_fields(tmp_path, metric, operands):
    _, runtime, args, _ = raw_workspace(tmp_path)
    args.comparables = []
    args.metrics = [metric]
    outcome = acquire_financial_inputs(runtime, args)
    assert {row.metric for row in runtime.session.input_dataset.active_records()} == operands
    assert all(metric in row["derived_preview"] and not row["missing"] for row in outcome["coverage"])
    assert outcome["raw_operand_count"] == 0


def test_explicit_raw_request_cannot_bypass_no_network_permission(tmp_path):
    _, runtime, args, calls = raw_workspace(tmp_path)
    args.metrics = ["raw.tushare_balancesheet_money_cap"]
    with pytest.raises(ValueError, match="network"):
        runtime.call("acquire_financial_inputs", args.model_dump_json(), lambda: acquire_financial_inputs(runtime, args))
    assert not calls and runtime.session.input_dataset is None


@pytest.mark.parametrize("metric,statement", [("depreciation_fixed_assets", "cashflow"),
    ("amortization_intangible_assets", "cashflow"), ("depreciation_right_of_use", "cashflow"),
    ("operating_receivables_decrease", "cashflow"), ("diluted_shares", "daily_basic"),
    ("market_price", "daily_basic"), ("preferred_equity", "balancesheet")])
def test_valid_not_directly_admitted_fields_route_to_relevant_statement(tmp_path, metric, statement):
    _, runtime, args, calls = raw_workspace(tmp_path)
    runtime.session.draft.methods = ["pe"]
    args.comparables = []
    args.metrics = [metric]
    outcome = acquire_financial_inputs(runtime, args)
    assert statement in {call["api_name"] for call in calls}
    assert any(metric in row["missing"] for row in outcome["coverage"])
    assert not any(row.metric == metric for row in runtime.session.input_dataset.active_records())


def test_explicit_relative_data_request_collects_paired_share_market_cap_dates(tmp_path):
    _, runtime, args, _ = acquisition_workspace(tmp_path)
    args.metrics = ["net_income_parent"]
    outcome = acquire_financial_inputs(runtime, args)
    assert outcome["pricing_date"] == "2026-09-30"
    assert not outcome["issues"]
    records = runtime.session.input_dataset.active_records()
    assert any(row.role == "historical" and row.metric == "common_shares" for row in records)
    assert sum(row.role == "comparable" and row.metric == "market_cap" for row in records) == 5


def test_explicit_raw_null_is_still_missing_not_zero_or_final_cash(tmp_path):
    _, runtime, args, _ = raw_workspace(tmp_path, {"balancesheet": {"money_cap": None}})
    runtime.session.draft.methods = ["pe"]
    args.comparables = []
    args.metrics = ["raw.tushare_balancesheet_money_cap"]
    outcome = acquire_financial_inputs(runtime, args)
    assert all("raw.tushare_balancesheet_money_cap" in row["missing"] for row in outcome["coverage"])
    assert not runtime.session.input_dataset.active_records()
    assert any(source["excluded_by_reason"].get("INPUT_PROVIDER_AMOUNT") for source in outcome["sources"])
