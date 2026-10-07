from datetime import date
from decimal import Decimal

import pytest

from test_input_calculations import user_rows
from test_input_workspace import fixture
from valuationagent.application.input_calculations import apply_calculations
from valuationagent.application.input_derivations import derive_snapshot_inputs
from valuationagent.application.record_inputs import record_inputs
from valuationagent.schemas.inputs import RecordInputs


def declaration(metric, ids, *, period=None):
    return {"metric": metric, "period_end": period,
        "terms": [{"input_id": identifier, "operation": "add"} for identifier in ids],
        "rationale": "合成测试明确上述原始组成属于同主体同期间且互不重叠，声明映射仅用于测试计算；实际业务须由模型说明包含范围及遗漏风险。",
        "limitations": ["原始数值绑定不等于口径完整或独立审计；未列出的组成不得默认补零。"]}


def test_declared_depreciation_components_feed_ebitda_without_retyping_numbers(tmp_path):
    app, runtime = fixture(tmp_path)
    ids = user_rows(app, runtime, [("ebit", "200", "annual"),
        ("raw.depreciation", "30.125", "annual"), ("raw.amortization", "19.875", "annual")], 2025)
    draft = declaration("depreciation_amortization", [ids["raw.depreciation"], ids["raw.amortization"]], period="2025-12-31")
    record_inputs(runtime, RecordInputs(calculations=[draft]))
    values, evidence, _ = apply_calculations(runtime.session.input_dataset, date(2025, 12, 31),
        {"ebit": Decimal(200)}, {}, {"depreciation_amortization"})
    assert values["depreciation_amortization"] == 50
    assert derive_snapshot_inputs(values, evidence, {"ebitda"})[0]["ebitda"] == 250
    assert not record_inputs(runtime, RecordInputs(calculations=[draft]))["saved_calculation_ids"]


@pytest.mark.parametrize("metric", ["lease_liabilities", "minority_interest", "interest_bearing_debt"])
def test_single_source_declaration_does_not_need_fabricated_second_operand(tmp_path, metric):
    app, runtime = fixture(tmp_path)
    ids = user_rows(app, runtime, [("raw.disclosed_component", "12.345", "instant")], 2025)
    draft = declaration(metric, list(ids.values()), period="2025-12-31")
    if metric == "interest_bearing_debt":
        draft["debt_includes_leases"] = False
    record_inputs(runtime, RecordInputs(calculations=[draft]))
    values, _, _ = apply_calculations(runtime.session.input_dataset, date(2025, 12, 31), {}, {}, {metric})
    assert values[metric] == Decimal("12.345")
    assert runtime.session.input_dataset.active_records()[0].metric == "raw.disclosed_component"


@pytest.mark.parametrize("metric,kind,value,error", [
    ("depreciation_amortization", "instant", "20", "KIND"),
    ("lease_liabilities", "annual", "20", "KIND"),
    ("depreciation_amortization", "annual", "-20", "DOMAIN"),
    ("lease_liabilities", "instant", "-20", "DOMAIN"),
])
def test_new_declarations_retain_domain_and_period_guards(tmp_path, metric, kind, value, error):
    app, runtime = fixture(tmp_path)
    ids = user_rows(app, runtime, [("raw.component", value, kind)], 2025)
    with pytest.raises(ValueError, match="INPUT_CALCULATION_" + error):
        record_inputs(runtime, RecordInputs(calculations=[declaration(metric, list(ids.values()), period="2025-12-31")]))
    assert runtime.session.input_dataset.calculations == []


def test_empty_formula_and_numeric_literals_are_not_supported(tmp_path):
    with pytest.raises(ValueError):
        RecordInputs(calculations=[declaration("depreciation_amortization", [])])
    draft = declaration("depreciation_amortization", ["input_missing"])
    draft["terms"][0]["value"] = 0
    with pytest.raises(ValueError):
        RecordInputs(calculations=[draft])


def test_replay_source_manifest_binds_provider_interpretation_contracts():
    from valuationagent.application.reproducibility import model_files

    files = model_files()
    assert "application/provider_inputs.py" in files
    assert "market/tushare_contracts.py" in files


def test_raw_provider_lease_cannot_be_hidden_from_debt_coverage_policy(tmp_path):
    from test_provider_raw_inputs import raw_workspace
    from valuationagent.application.input_acquisition import acquire_financial_inputs

    _, runtime, args, _ = raw_workspace(tmp_path)
    acquire_financial_inputs(runtime, args)
    row = next(row for row in runtime.session.input_dataset.active_records()
        if row.role == "historical" and row.period_end == date(2025, 12, 31)
        and row.metric == "raw.tushare_balancesheet_lease_liab")
    draft = declaration("interest_bearing_debt", [row.input_id], period="2025-12-31")
    draft["debt_includes_leases"] = False
    with pytest.raises(ValueError, match="INPUT_CALCULATION_LEASE"):
        record_inputs(runtime, RecordInputs(calculations=[draft]))


def test_target_declared_lease_and_minority_are_applied_frozen_and_replayed(tmp_path):
    from valuationagent.application.agent_runtime import CalculateValuation
    from valuationagent.application.input_workspace import prepare_dataset
    from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle

    app, runtime = fixture(tmp_path)
    ids = user_rows(app, runtime, [("ebitda", "250", "annual"),
        ("cash_and_non_operating_assets", "100", "instant"), ("interest_bearing_debt", "0", "instant"),
        ("raw.lease", "20", "instant"), ("raw.minority", "10", "instant")])
    message = app.state.store.add_message(runtime.session.session_id, "user", "普通股100股，EV/EBITDA取10倍。", "agent")
    record_inputs(runtime, RecordInputs(user_values=[
        {"metric": "common_shares", "amount_text": "100股", "unit": "股", "scope": "issuer", "message_id": message.message_id},
        {"metric": "ev_ebitda_multiple", "amount_text": "10倍", "unit": "ratio", "scope": "assumption", "message_id": message.message_id}],
        calculations=[declaration("lease_liabilities", [ids["raw.lease"]]),
            declaration("minority_interest", [ids["raw.minority"]])]))
    runtime.session.draft.methods = ["ev_ebitda"]
    request = prepare_dataset(runtime.session, ["ev_ebitda"])
    assert request.financials.lease_liabilities == 20
    assert request.financials.minority_interest == 10
    assert {row["metric"] for row in request.input_calculations} == {"lease_liabilities", "minority_interest"}
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["method_completion"]["all_requested_methods_completed"], outcome
    record = app.state.store.get_run(outcome["run_id"])
    assert record.result.relative[0].per_share_value == Decimal("25.7")
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]


def test_source_raw_interpretations_complete_peer_ev_report_and_replay(tmp_path):
    from test_input_peers import peer_workspace
    from test_source_peer_bridge import source_row, BASELINE, BALANCE, BRIDGE
    from valuationagent.application.agent_runtime import CalculateValuation
    from valuationagent.application.input_peer_bridge import verify_source_peers
    from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle

    app, runtime, _ = peer_workspace(tmp_path, count=3)
    for ticker, choice in list(runtime.session.input_dataset.comparables.items()):
        record_inputs(runtime, RecordInputs(comparables=[choice.model_dump() | {"capital_bridge": BRIDGE}]))
        drafts = []
        for metric, amount in {"ebit": 800000, "depreciation_amortization": 200000,
                "cash_and_non_operating_assets": 300000, "interest_bearing_debt": 200000,
                "lease_liabilities": 50000, "minority_interest": 100000}.items():
            annual = metric in {"ebit", "depreciation_amortization"}
            period = BASELINE if annual else BALANCE
            row = source_row(runtime, ticker, "raw.disclosed_" + metric, amount,
                period=period.isoformat(), instant=not annual, raw=True)
            draft = declaration(metric, [row.input_id], period=period.isoformat()) | {"entity_ticker": ticker}
            if metric == "interest_bearing_debt":
                draft["debt_includes_leases"] = False
            drafts.append(draft)
        record_inputs(runtime, RecordInputs(calculations=drafts))
    for metric, amount in {"ebitda": 500000, "cash_and_non_operating_assets": 100000,
            "interest_bearing_debt": 0, "lease_liabilities": 0, "minority_interest": 0}.items():
        source_row(runtime, runtime.session.draft.ticker, metric, amount, instant=metric != "ebitda", role="historical")
    runtime.session.draft.methods = ["pe", "ps", "ev_ebitda"]
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["method_completion"]["all_requested_methods_completed"], outcome
    assert outcome["report_delivery"]["status"] == "saved"
    record = app.state.store.get_run(outcome["run_id"])
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    assert all(peer.ev_ebitda == peer.enterprise_value / Decimal(1000000) for peer in record.request.peers)
    tampered = record.request.model_copy(deep=True)
    tampered.peers[0].ev_ebitda += 1
    with pytest.raises(ValueError, match="INPUT_PEER_REPLAY"):
        verify_source_peers(tampered)
