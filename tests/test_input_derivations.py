from datetime import date
from decimal import Decimal

import pytest

from test_input_workspace import fixture
from valuationagent.application.input_derivations import DERIVATION_VERSION, derivation_dependencies, derive_snapshot_inputs
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.record_inputs import record_inputs
from valuationagent.schemas.inputs import RecordInputs
from valuationagent.schemas.models import EvidenceRef


def raw_values():
    return {metric: Decimal(value) for metric, value in {
        "revenue": "1000", "ebit": "200", "profit_before_tax": "160", "income_tax_expense": "40",
        "cash_paid_for_ppe_intangibles": "-80", "depreciation_fixed_assets": "10",
        "amortization_intangible_assets": "7", "amortization_long_term_deferred_expenses": "3",
        "depreciation_right_of_use": "0", "inventory_decrease": "-20",
        "operating_receivables_decrease": "-30", "operating_payables_increase": "10",
    }.items()}


def test_deterministic_formulas_keep_raw_values_and_transitive_evidence():
    original = raw_values()
    evidence = {metric: [EvidenceRef(evidence_id=metric, source="synthetic")] for metric in original}
    values, references, formulas = derive_snapshot_inputs(original, evidence,
        {"ebit_margin", "tax_rate", "capital_expenditure", "ebitda", "change_operating_nwc"})
    assert values["ebit_margin"] == Decimal("0.2")
    assert values["tax_rate"] == Decimal("0.25")
    assert values["capital_expenditure"] == 80 and values["cash_paid_for_ppe_intangibles"] == -80
    assert values["depreciation_amortization"] == 20 and values["ebitda"] == 220
    assert values["change_operating_nwc"] == 40
    assert {ref.evidence_id for ref in references["ebitda"]} == {
        "ebit", "depreciation_fixed_assets", "amortization_intangible_assets",
        "amortization_long_term_deferred_expenses", "depreciation_right_of_use"}
    assert formulas["derivation_policy"] == DERIVATION_VERSION
    assert original == raw_values() and set(evidence) == set(original)
    assert "depreciation_amortization" in derivation_dependencies({"ebitda"})


def test_missing_depreciation_component_is_not_zero_and_profit_is_not_ebit():
    original = raw_values()
    original.pop("depreciation_right_of_use")
    original.pop("ebit")
    values, _, _ = derive_snapshot_inputs(original, {}, {"ebit_margin", "ebitda"})
    assert "ebit_margin" not in values and "ebitda" not in values
    assert "depreciation_amortization" not in values


@pytest.mark.parametrize("field,value,error", [
    ("ebit_margin", "0.9", "INPUT_DERIVATION_CONFLICT"),
    ("profit_before_tax", "0", "INPUT_DERIVATION_DOMAIN"),
    ("profit_before_tax", "-10", "INPUT_DERIVATION_DOMAIN"),
    ("amortization_intangible_assets", "-7", "INPUT_DERIVATION_DOMAIN"),
])
def test_conflicts_and_invalid_derivations_do_not_overwrite(field, value, error):
    original = raw_values()
    original[field] = Decimal(value)
    with pytest.raises(ValueError, match=error):
        derive_snapshot_inputs(original, {}, {"ebit_margin", "tax_rate", "depreciation_amortization"})
    assert original[field] == Decimal(value)


def test_direct_values_reconcile_and_unrelated_methods_do_not_derive():
    original = raw_values() | {"tax_rate": Decimal("0.25")}
    _, _, methods = derive_snapshot_inputs(original, {}, {"tax_rate"})
    assert "reconciliation.tax_rate" in methods and "tax_rate" not in methods
    original["profit_before_tax"] = Decimal(0)
    assert derive_snapshot_inputs(original, {}, {"tax_rate"})[0]["tax_rate"] == Decimal("0.25")
    assert derive_snapshot_inputs(original, {}, {"net_income_parent"})[2] == {}


def test_raw_user_fields_keep_periods_separate_and_do_not_trigger_network(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网，2024年EBIT 100元；2025年营业收入1000元、归母净利润50元；普通股10股，PE取20倍。")
    record_inputs(runtime, RecordInputs(user_values=[
        {"metric": "ebit", "amount_text": "100元", "unit": "元", "period_end": "2024-12-31", "period_quote": "2024年"},
        {"metric": "revenue", "amount_text": "1000元", "unit": "元", "period_end": "2025-12-31", "period_quote": "2025年"},
        {"metric": "net_income_parent", "amount_text": "50元", "unit": "元", "period_end": "2025-12-31", "period_quote": "2025年"},
        {"metric": "common_shares", "amount_text": "10股", "unit": "股", "scope": "issuer"},
        {"metric": "pe_multiple", "amount_text": "20倍", "unit": "ratio", "scope": "assumption"},
    ]))
    request = prepare_dataset(runtime.session, ["pe"])
    assert request.financials.period_end == date(2025, 12, 31)
    assert request.financials.ebit_margin is None and "ebit" not in request.financials.statement_items
    with pytest.raises(ValueError, match="INPUTS_MISSING.*ebit_margin"):
        prepare_dataset(runtime.session, ["dcf"])
    assert not runtime.session.search_history


def test_raw_user_fields_reject_wrong_dimensions(tmp_path):
    _, runtime = fixture(tmp_path, "不要联网，所得税费用10股。")
    with pytest.raises(ValueError, match="INPUT_DIMENSION"):
        record_inputs(runtime, RecordInputs(user_values=[{"metric": "income_tax_expense", "amount_text": "10股", "unit": "股"}]))
    assert runtime.session.input_dataset is None


def test_four_year_raw_user_scenario_calculates_dcf_reports_and_replays(tmp_path):
    from valuationagent.application.agent_runtime import CalculateValuation
    from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
    from valuationagent.application.result_document import build_result_document, document_sections
    from valuationagent.application.workspace_artifacts import ReportWrite, write_report

    app, runtime = fixture(tmp_path)
    runtime.session.draft.methods = ["dcf"]
    runtime.session.draft.industry = "半导体"
    runtime.session.draft.valuation_date = date(2026, 10, 3)
    for year in range(2022, 2026):
        selections = []
        numbers = raw_values() | {"cash_and_non_operating_assets": Decimal(101), "interest_bearing_debt": Decimal(17)}
        numbers["revenue"] += Decimal(100 * (year - 2022))
        for metric, amount in numbers.items():
            message = app.state.store.add_message(runtime.session.session_id, "user", f"{year}年情景假设：{metric}为{amount}元。", "agent")
            selections.append({"metric": metric, "amount_text": f"{amount}元", "unit": "元", "message_id": message.message_id,
                "period_end": f"{year}-12-31", "period_quote": f"{year}年"})
        record_inputs(runtime, RecordInputs(user_values=selections))
    message = app.state.store.add_message(runtime.session.session_id, "user", "普通股10股，WACC 10%，永续增长2%。", "agent")
    record_inputs(runtime, RecordInputs(user_values=[
        {"metric": "common_shares", "amount_text": "10股", "unit": "股", "scope": "issuer", "message_id": message.message_id},
        {"metric": "wacc", "amount_text": "10%", "unit": "%", "scope": "assumption", "message_id": message.message_id},
        {"metric": "terminal_growth", "amount_text": "2%", "unit": "%", "scope": "assumption", "message_id": message.message_id},
    ]))
    request = prepare_dataset(runtime.session, ["dcf"])
    assert request.forecast_years == 10 and request.discount_policy == "year_end"
    assert len(request.historical_financials) == 3
    assert request.financials.ebit_margin == Decimal(200) / 1300
    assert request.financials.statement_items["cash_paid_for_ppe_intangibles"] == -80
    assert not any(row["metric"] == "capital_expenditure" for row in request.input_records)
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] in {"completed", "completed_with_warnings"}, outcome
    record = app.state.store.get_run(outcome["run_id"])
    assert record.result.dcf.per_share_value > 0
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    document = build_result_document(app.state.research, runtime.session)
    assert len(document["input_derivations"]) == 20
    assert "确定性输入推导" in str(document_sections(document))
    artifact = write_report(app.state.research, runtime.session, ReportWrite(format="md"))
    assert artifact["numeric_result_available"]
    content = app.state.store.get_artifact(runtime.session.session_id, artifact["artifact_id"])[1].decode()
    assert DERIVATION_VERSION in content and "income_tax_expense / profit_before_tax" in content
    assert not runtime.session.search_history
