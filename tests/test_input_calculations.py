from datetime import date
from decimal import Decimal

import pytest

from test_input_workspace import fixture
from valuationagent.application.input_calculations import apply_calculations, CALCULATION_VERSION
from valuationagent.application.input_views import InspectInputs, inspect_inputs
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.record_inputs import record_inputs
from valuationagent.schemas.inputs import RecordInputs


RATIONALE = "合成测试声明：营业收入减完整经营成本费用得到经营EBIT；原始科目不包含融资成本、所得税或非经营投资收益，完整性是假设而非独立审计。"


def user_rows(app, runtime, amounts, year=None):
    selections = []
    for metric, amount, kind in amounts:
        text = (f"{year}年" if year else "情景假设") + f"：{metric}为{amount}元。"
        message = app.state.store.add_message(runtime.session.session_id, "user", text, "agent")
        selections.append({"metric": metric, "amount_text": f"{amount}元", "unit": "元", "message_id": message.message_id,
            "period_kind": kind, **({"period_end": f"{year}-12-31", "period_quote": f"{year}年"} if year else {})})
    record_inputs(runtime, RecordInputs(user_values=selections))
    return {row.metric: row.input_id for row in runtime.session.input_dataset.active_records() if row.period_end == (date(year, 12, 31) if year else None)}


def formula(ids, metric="ebit", year=None, positive="revenue", negative="raw.operating_cost"):
    return {"metric": metric, "period_end": f"{year}-12-31" if year else None,
        "terms": [{"input_id": ids[positive], "operation": "add"}, {"input_id": ids[negative], "operation": "subtract"}],
        "rationale": RATIONALE, "limitations": ["合成测试假设各项互不重叠且覆盖完整，不代表实际发行人财务结论。"]}


def workspace(tmp_path):
    app, runtime = fixture(tmp_path)
    ids = user_rows(app, runtime, [("revenue", "1000", "annual"), ("raw.operating_cost", "800", "annual")])
    return app, runtime, ids


def test_declared_ebit_is_decimal_auditable_idempotent_and_readable(tmp_path):
    _, runtime, ids = workspace(tmp_path)
    args = RecordInputs(calculations=[formula(ids)])
    outcome = record_inputs(runtime, args)
    assert len(outcome["saved_calculation_ids"]) == 1
    assert not record_inputs(runtime, args)["saved_calculation_ids"]
    assert len(inspect_inputs(runtime.session, InspectInputs(section="calculations"))["items"]) == 1
    numbers, refs, expressions = apply_calculations(runtime.session.input_dataset, None, {}, {}, {"ebit"})
    assert numbers["ebit"] == 200
    assert {ref.evidence_id for ref in refs["ebit"]} == set(ids.values())
    assert expressions["declared_calculation_policy"] == CALCULATION_VERSION
    assert not any(row.metric == "ebit" for row in runtime.session.input_dataset.records)


@pytest.mark.parametrize("problem,code", [
    ("foreign", "STALE"), ("duplicate", "DUPLICATE"), ("period", "PERIOD"), ("kind", "KIND"),
    ("scope", "SCOPE"), ("currency", "SCOPE"), ("role", "SCOPE"), ("unit", "SCOPE"),
    ("lease", "LEASE"),
])
def test_invalid_declared_calculations_are_atomic(tmp_path, problem, code):
    _, runtime, ids = workspace(tmp_path)
    draft = formula(ids)
    row = runtime.session.input_dataset.records[-1]
    if problem == "foreign":
        draft["terms"][0]["input_id"] = "input_other_workspace"
    elif problem == "duplicate":
        draft["terms"][1]["input_id"] = draft["terms"][0]["input_id"]
    elif problem == "period":
        draft["period_end"] = "2025-12-31"
    elif problem == "kind":
        row.period_kind = "instant"
    elif problem == "lease":
        draft["debt_includes_leases"] = True
    else:
        setattr(row, problem, {"scope": "parent", "currency": "USD", "role": "comparable", "unit": "股"}[problem])
    before = runtime.session.input_dataset.model_dump()
    with pytest.raises(ValueError, match="INPUT_CALCULATION_" + code):
        record_inputs(runtime, RecordInputs(calculations=[draft]))
    assert runtime.session.input_dataset.model_dump() == before


def test_replacing_input_requires_explicit_formula_replacement(tmp_path):
    app, runtime, ids = workspace(tmp_path)
    original = record_inputs(runtime, RecordInputs(calculations=[formula(ids)]))["saved_calculation_ids"][0]
    message = app.state.store.add_message(runtime.session.session_id, "user", "更正经营成本为750元。", "agent")
    updated = record_inputs(runtime, RecordInputs(user_values=[{"metric": "raw.operating_cost", "amount_text": "750元", "unit": "元",
        "period_kind": "annual", "message_id": message.message_id, "replaces": [ids["raw.operating_cost"]]}]))["saved_input_ids"][0]
    with pytest.raises(ValueError, match="INPUT_CALCULATION_STALE"):
        apply_calculations(runtime.session.input_dataset, None, {}, {}, {"ebit"})
    replacement = formula(ids | {"raw.operating_cost": updated}) | {"replaces": [original]}
    record_inputs(runtime, RecordInputs(calculations=[replacement]))
    assert not record_inputs(runtime, RecordInputs(calculations=[replacement]))["saved_calculation_ids"]
    assert apply_calculations(runtime.session.input_dataset, None, {}, {}, {"ebit"})[0]["ebit"] == 250
    assert len(runtime.session.input_dataset.calculations) == 2


def test_unrelated_formula_does_not_block_pe_and_direct_conflict_is_not_overwritten(tmp_path):
    _, runtime, ids = workspace(tmp_path)
    record_inputs(runtime, RecordInputs(calculations=[formula(ids)]))
    with pytest.raises(ValueError, match="INPUT_CALCULATION_CONFLICT"):
        apply_calculations(runtime.session.input_dataset, None, {"ebit": Decimal(999)}, {}, {"ebit"})
    runtime.session.input_dataset.calculations[0].terms[0].input_id = "missing"
    assert apply_calculations(runtime.session.input_dataset, None, {}, {}, {"net_income_parent"})[0] == {}


def test_raw_document_money_has_no_fixed_dictionary_or_final_field_alias(tmp_path):
    from observation_fixtures import fixture_observations
    from test_observation_extraction import submit, review
    from test_selected_model_inputs import source_workspace

    _, runtime = source_workspace(tmp_path)
    args = fixture_observations(runtime, [
        {"metric": "营业收入", "standard_metric": "revenue", "raw_value": "1000"},
        {"metric": "合成口径完整经营成本费用", "standard_metric": "raw.operating_cost", "raw_value": "800"},
    ])
    assert submit(runtime, args)["saved_count"] == 2
    review(runtime)
    selected = runtime.session.facts[-2:]
    assert all(not fact.warnings for fact in selected)
    record_inputs(runtime, RecordInputs(source_values=[{"fact_id": selected[0].fact_id}, {"fact_id": selected[1].fact_id, "as_raw": True}]))
    rows = runtime.session.input_dataset.active_records()
    assert rows[-1].label == "合成口径完整经营成本费用" and rows[-1].period_kind == "annual"
    ids = {"revenue": rows[0].input_id, "raw.operating_cost": rows[1].input_id}
    record_inputs(runtime, RecordInputs(calculations=[formula(ids, year=2025)]))
    assert apply_calculations(runtime.session.input_dataset, date(2025, 12, 31), {}, {}, {"ebit"})[0]["ebit"] == 200
    selected[-1].verification["semantic_review"]["status"] = "pending"
    with pytest.raises(ValueError, match="INPUT_ADMISSION"):
        prepare_dataset(runtime.session, ["dcf"])


def test_four_year_declared_ebit_and_cash_bridge_to_dcf_report_replay(tmp_path):
    from test_input_derivations import raw_values
    from valuationagent.application.agent_runtime import CalculateValuation
    from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
    from valuationagent.application.result_document import build_result_document, document_sections
    from valuationagent.application.workspace_artifacts import ReportWrite, write_report

    app, runtime = fixture(tmp_path)
    runtime.session.draft.methods = ["dcf"]
    runtime.session.draft.industry = "半导体"
    runtime.session.draft.valuation_date = date(2026, 10, 3)
    for year in range(2022, 2026):
        numbers = raw_values()
        numbers.pop("ebit")
        numbers["revenue"] += Decimal(100 * (year - 2022))
        amounts = [(metric, str(value), "annual") for metric, value in numbers.items()]
        amounts += [("raw.operating_cost", str(numbers["revenue"] - 200), "annual"),
            ("raw.cash", "151", "instant"), ("raw.unavailable_cash", "50", "instant"),
            ("raw.short_debt", "11", "instant"), ("raw.long_debt", "6", "instant"), ("lease_liabilities", "0", "instant")]
        ids = user_rows(app, runtime, amounts, year)
        debt = formula(ids, "interest_bearing_debt", year, "raw.short_debt", "raw.long_debt")
        debt["terms"][1]["operation"] = "add"
        debt["debt_includes_leases"] = False
        debt["rationale"] = "合成用户情景明确只有上述两项借款，租赁单列且已显式给零；不代表来源披露穷尽或实际公司没有其他带息义务。"
        cash = formula(ids, "cash_and_non_operating_assets", year, "raw.cash", "raw.unavailable_cash")
        cash["rationale"] = "用户情景中从货币资金扣除不可用于桥接的现金，假设扣除项已完整涵盖受限及经营所需现金且无重叠；不代表独立财务审计。"
        record_inputs(runtime, RecordInputs(calculations=[formula(ids, year=year), cash, debt]))
    message = app.state.store.add_message(runtime.session.session_id, "user", "普通股10股，WACC 10%，永续增长2%。", "agent")
    record_inputs(runtime, RecordInputs(user_values=[
        {"metric": "common_shares", "amount_text": "10股", "unit": "股", "scope": "issuer", "message_id": message.message_id},
        {"metric": "wacc", "amount_text": "10%", "unit": "%", "scope": "assumption", "message_id": message.message_id},
        {"metric": "terminal_growth", "amount_text": "2%", "unit": "%", "scope": "assumption", "message_id": message.message_id},
    ]))
    request = prepare_dataset(runtime.session, ["dcf"])
    assert len(request.input_calculations) == 12
    assert request.financials.cash_and_non_operating_assets == 101
    assert request.financials.interest_bearing_debt == 17
    assert request.financials.interest_bearing_debt_includes_leases is False
    assert request.financials.ebit_margin == Decimal(200) / 1300
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] in {"completed", "completed_with_warnings"}, outcome
    record = app.state.store.get_run(outcome["run_id"])
    assert record.result.dcf.per_share_value > 0
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    from valuationagent.application.input_calculations import verify_frozen_calculations

    tampered = record.request.model_copy(deep=True)
    tampered.financials.statement_items["ebit"] += 1
    with pytest.raises(ValueError, match="INPUT_CALCULATION_REPLAY"):
        verify_frozen_calculations(tampered)
    changed_policy = record.request.model_copy(deep=True)
    changed_policy.financials.calculation_methods["declared_calculation_policy"] = "unknown-version"
    with pytest.raises(ValueError, match="INPUT_CALCULATION_REPLAY"):
        verify_frozen_calculations(changed_policy)
    from valuationagent.application.input_derivations import verify_frozen_derivations

    wrong_margin = record.request.model_copy(deep=True)
    wrong_margin.financials.ebit_margin = Decimal("0.9")
    with pytest.raises(ValueError, match="INPUT_DERIVATION_REPLAY"):
        verify_frozen_derivations(wrong_margin)
    document = build_result_document(app.state.research, runtime.session)
    assert len(document["input_calculations"]) == 12
    assert {row["value"] for row in document["input_calculations"] if row["metric"] == "ebit"} == {"200"}
    assert "未经独立审计" in str(document_sections(document))
    report = write_report(app.state.research, runtime.session, ReportWrite(format="md"))
    assert report["numeric_result_available"]
    assert not runtime.session.search_history


def test_negative_cash_and_unspecified_debt_lease_policy_are_not_silently_fixed(tmp_path):
    app, runtime = fixture(tmp_path)
    ids = user_rows(app, runtime, [("raw.cash", "50", "instant"), ("raw.unavailable", "80", "instant")])
    with pytest.raises(ValueError, match="INPUT_CALCULATION_DOMAIN"):
        record_inputs(runtime, RecordInputs(calculations=[formula(ids, "cash_and_non_operating_assets", positive="raw.cash", negative="raw.unavailable")]))
    with pytest.raises(ValueError, match="INPUT_CALCULATION_LEASE"):
        record_inputs(runtime, RecordInputs(calculations=[formula(ids, "interest_bearing_debt", positive="raw.unavailable", negative="raw.cash")]))
    assert not runtime.session.input_dataset.calculations


def test_explicit_lease_component_cannot_claim_debt_excludes_leases(tmp_path):
    app, runtime = fixture(tmp_path)
    ids = user_rows(app, runtime, [("raw.borrowings", "100", "instant"), ("lease_liabilities", "20", "instant")])
    draft = formula(ids, "interest_bearing_debt", positive="raw.borrowings", negative="lease_liabilities")
    draft["terms"][1]["operation"] = "add"
    draft["debt_includes_leases"] = False
    with pytest.raises(ValueError, match="INPUT_CALCULATION_LEASE"):
        record_inputs(runtime, RecordInputs(calculations=[draft]))
    draft["debt_includes_leases"] = True
    record_inputs(runtime, RecordInputs(calculations=[draft]))
    values = apply_calculations(runtime.session.input_dataset, None, {}, {}, {"interest_bearing_debt"})[0]
    assert values["interest_bearing_debt"] == 120 and values["interest_bearing_debt_includes_leases"] is True


@pytest.mark.parametrize("change", ["expression", "constant", "operation"])
def test_calculation_schema_has_no_executable_expression_or_numeric_literal(tmp_path, change):
    _, _, ids = workspace(tmp_path)
    draft = formula(ids)
    if change == "expression":
        draft["expression"] = "print('untrusted')"
    elif change == "constant":
        draft["terms"][0]["value"] = "100000"
    else:
        draft["terms"][0]["operation"] = "execute"
    with pytest.raises(ValueError):
        RecordInputs(calculations=[draft])
