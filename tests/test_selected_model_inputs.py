from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from observation_fixtures import fixture_observations
from test_input_workspace import fixture
from test_observation_extraction import review, submit
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.application.result_document import build_result_document, document_sections
from valuationagent.application.workspace_artifacts import ReportWrite, write_report
from valuationagent.application.workspace_sensitivity import SensitivityRequest, analyze_sensitivity
from valuationagent.schemas.inputs import RecordInputs


def source_workspace(tmp_path, *, provider=False):
    app, runtime = fixture(tmp_path, "只用已有文件数据，PE取20倍，不要联网。")
    runtime.session.draft.company = "Synthetic issuer"
    runtime.session.draft.ticker = "600123"
    runtime.session.draft.valuation_date = date(2026, 3, 1)
    runtime.session.draft.industry = "半导体"
    specification = [
        {"metric": "net_income_parent", "raw_value": "500000", "unit": "元", "period": "2025-12-31"},
        {"metric": "common_shares", "raw_value": "1000000", "unit": "股", "period": "2026-02-28", "semantic_role": "equity", "fcff_treatment": "exclude"},
    ]
    args = fixture_observations(runtime, specification)
    if provider:
        runtime.session.documents[0].provenance_type = "structured_provider"
        runtime.session.documents[0].provider = "synthetic-audited-provider"
        blocks = runtime.service.store.research_blocks(runtime.session.session_id, args["file_id"])
        for block in blocks:
            block["location"]["published_at"] = "2026-03-01"
        runtime.service.store.save_research_blocks(runtime.session.session_id, args["file_id"], blocks)
    result = submit(runtime, args)
    assert result["saved_count"] == 2, result
    review(runtime)
    assert all(not fact.warnings for fact in runtime.session.facts)
    app.state.store.save_research(runtime.session)
    return app, runtime


def choose(runtime, fact_ids=None):
    return record_inputs(runtime, RecordInputs(source_values=[{"fact_id": key} for key in
        (fact_ids or [fact.fact_id for fact in runtime.session.facts])]))


def multiple(runtime):
    return record_inputs(runtime, RecordInputs(user_values=[{
        "metric": "pe_multiple", "amount_text": "20倍", "unit": "ratio", "scope": "assumption"}]))


def test_reviewed_raw_document_fields_are_selected_without_renaming(tmp_path):
    from valuationagent.application.input_derivations import derive_snapshot_inputs
    from valuationagent.application.input_workspace import input_evidence

    _, runtime = source_workspace(tmp_path)
    args = fixture_observations(runtime, [
        {"metric": "profit_before_tax", "raw_value": "800000", "period": "2025"},
        {"metric": "income_tax_expense", "raw_value": "200000", "period": "2025"},
        {"metric": "cash_paid_for_ppe_intangibles", "raw_value": "300000", "period": "2025",
            "semantic_role": "investing", "ebit_treatment": "exclude", "fcff_treatment": "include"},
    ])
    result = submit(runtime, args)
    assert result["saved_count"] == 3, result
    review(runtime)
    choose(runtime)
    rows = runtime.session.input_dataset.active_records()
    values, evidence, formulas = derive_snapshot_inputs({row.metric: row.value for row in rows},
        {row.metric: [input_evidence(row)] for row in rows}, {"tax_rate", "capital_expenditure"})
    assert values["tax_rate"] == Decimal("0.25") and values["capital_expenditure"] == 300000
    assert len(evidence["tax_rate"]) == 2
    assert all(row.assertion == "reported" for row in rows)
    assert not any(row.metric in {"tax_rate", "capital_expenditure"} for row in rows)
    assert "capital_expenditure" in formulas


def test_raw_source_flow_cannot_be_rebound_as_an_instant(tmp_path):
    from valuationagent.application.input_sources import source_fact

    _, runtime = source_workspace(tmp_path)
    args = fixture_observations(runtime, [{"metric": "income_tax_expense", "raw_value": "200000", "period": "2025"}])
    submit(runtime, args)
    review(runtime)
    fact = runtime.session.facts[-1]
    fact.verification["reading_proof"]["row"]["period_kind"] = "instant"
    fact.period = "2025-12-31"
    with pytest.raises(ValueError, match="INPUT_PERIOD"):
        source_fact(runtime.session, fact.fact_id)


@pytest.mark.parametrize("provider", [False, True])
@pytest.mark.parametrize("assumption_first", [False, True])
def test_reviewed_source_and_user_assumption_share_calculation_report_and_replay(tmp_path, provider, assumption_first):
    app, runtime = source_workspace(tmp_path, provider=provider)
    if assumption_first:
        multiple(runtime)
    result = choose(runtime)
    if not assumption_first:
        multiple(runtime)
    class ForbiddenProvider:
        version = "do-not-refetch"

        def resolve(self, request, store):
            pytest.fail("Selected source inputs must not trigger another network acquisition")

    app.state.research._market_clients[runtime.session.session_id] = ForbiddenProvider()
    request = prepare_dataset(runtime.session, ["pe"])
    assert request.analysis_basis == "research"
    assert request.financials.period_end == date(2025, 12, 31)
    assert request.financials.common_shares_as_of == date(2026, 2, 28)
    assert {row["source"]["kind"] for row in request.input_records} == {"user", "provider" if provider else "document"}
    assert len(request.assumption_evidence["pe_multiple"]) == 1
    assert not choose(runtime)["saved_input_ids"]
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] in {"completed", "completed_with_warnings"}, outcome
    record = app.state.store.get_run(outcome["run_id"])
    assert record.result.relative[0].per_share_value == Decimal("10")
    assert record.request.input_records == request.input_records
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    artifact = write_report(app.state.research, runtime.session, ReportWrite(format="pdf"))
    assert artifact["numeric_result_available"] and artifact["download_url"]
    metadata, raw = app.state.store.get_artifact(runtime.session.session_id, artifact["artifact_id"])
    assert raw.startswith(b"%PDF") and metadata["valuation_run_id"] == record.run_id
    document = build_result_document(app.state.research, runtime.session)
    assert len(document["sources"]) == 1
    assert len(document["verified_facts"]) == 2
    assert document["explicit_multiples"] == {"pe": "20"}
    text = str(document_sections(document))
    assert "模型输入与来源性质" in text
    assert "来源解释已复核，非独立审计" in text
    assert "用户指定倍数，不使用可比样本统计" in text
    assert "用户输入与假设（未经外部核验）" not in text
    trial = analyze_sensitivity(runtime, SensitivityRequest(method="pe", parameter="multiple", values=[15, 25]))
    assert [Decimal(row["per_share_value"]) for row in trial["scenarios"]] == [Decimal("7.5"), Decimal("12.5")]
    assert app.state.store.get_run(record.run_id).result.relative[0].per_share_value == Decimal("10")
    assert len(result["saved_input_ids"]) == 2


def test_bad_selection_is_atomic_and_does_not_create_a_dataset(tmp_path):
    _, runtime = source_workspace(tmp_path)
    with pytest.raises(ValueError, match="INPUT_SOURCE"):
        choose(runtime, [runtime.session.facts[0].fact_id, "foreign-fact"])
    assert runtime.session.input_dataset is None


@pytest.mark.parametrize("problem", ["warning", "rejected", "pending_review", "parent", "other_company", "future", "unknown_currency"])
def test_source_selection_preserves_admission_boundaries(tmp_path, problem):
    _, runtime = source_workspace(tmp_path)
    fact = runtime.session.facts[0]
    if problem == "warning":
        fact.warnings.append("unit ambiguous")
    elif problem == "rejected":
        fact.status = "rejected"
    elif problem == "pending_review":
        fact.verification["semantic_review"]["status"] = "pending"
    elif problem == "parent":
        fact.scope = "parent"
    elif problem == "other_company":
        runtime.session.draft.ticker = "600999"
    elif problem == "future":
        fact.published_at = date(2026, 3, 2)
    else:
        fact.verification["reading_proof"]["basis"]["currency"] = "unknown"
    with pytest.raises(ValueError, match="INPUT_"):
        choose(runtime)
    assert runtime.session.input_dataset is None


def test_selection_checks_original_bytes(tmp_path):
    app, runtime = source_workspace(tmp_path)
    file_id = runtime.session.documents[0].file_id
    Path(app.state.store.get_file(file_id)["storage_path"]).write_bytes(b"replaced original")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        choose(runtime)
    assert runtime.session.input_dataset is None


@pytest.mark.parametrize("change", ["source", "input", "review"])
def test_changed_source_or_interpretation_cannot_reuse_selected_input(tmp_path, change):
    _, runtime = source_workspace(tmp_path)
    choose(runtime)
    multiple(runtime)
    if change == "source":
        runtime.session.facts[0].normalized_value = "600000"
    elif change == "input":
        runtime.session.input_dataset.records[0].value = Decimal("600000")
    else:
        runtime.session.facts[0].verification["semantic_review"]["status"] = "needs_evidence"
    with pytest.raises(ValueError, match="INPUT_SOURCE_CHANGED|INPUT_ADMISSION"):
        prepare_dataset(runtime.session, ["pe"])


def test_dated_shares_do_not_change_operating_baseline_and_staleness_is_visible(tmp_path):
    _, runtime = source_workspace(tmp_path)
    choose(runtime)
    multiple(runtime)
    runtime.session.draft.valuation_date = date(2026, 10, 3)
    with pytest.raises(ValueError, match="INPUT_SHARES_DATE"):
        prepare_dataset(runtime.session, ["pe"])


def test_share_counts_are_currency_neutral(tmp_path):
    _, runtime = source_workspace(tmp_path)
    shares = runtime.session.facts[1]
    shares.verification["reading_proof"]["basis"]["currency"] = None
    choose(runtime)
    multiple(runtime)
    request = prepare_dataset(runtime.session, ["pe"])
    share_record = next(row for row in request.input_records if row["metric"] == "common_shares")
    assert share_record["currency"] is None
    assert request.company.currency == "CNY"
    assert request.financials.common_shares == Decimal("1000000")


def test_user_financial_override_keeps_source_records_and_scenario_label(tmp_path):
    app, runtime = source_workspace(tmp_path)
    choose(runtime)
    multiple(runtime)
    old = runtime.session.input_dataset.records[0]
    message = app.state.store.add_message(runtime.session.session_id, "user", "假设归母净利润60万元。", "agent")
    record_inputs(runtime, RecordInputs(user_values=[{"metric": "net_income_parent", "amount_text": "60万元",
        "message_id": message.message_id, "unit": "万元", "replaces": [old.input_id]}]))
    request = prepare_dataset(runtime.session, ["pe"])
    assert request.analysis_basis == "user_scenario"
    assert request.financials.net_income_parent == Decimal("600000")
    assert runtime.session.facts[0].normalized_value == "500000"
    assert old in runtime.session.input_dataset.records


@pytest.mark.parametrize("amount,unit,metric,expected", [
    ("200亿元", "亿元", "revenue", "200亿元"),
    ("200", "亿元", "revenue", "200 亿元"),
    ("3倍", "ratio", "ps_multiple", "3倍"),
    ("0.15", "ratio", "ebit_margin", "0.15 （比率）"),
])
def test_report_preserves_units_without_duplicate_suffix(amount, unit, metric, expected):
    from valuationagent.application.result_document import input_amount

    assert input_amount({"original_amount": amount, "unit": unit, "metric": metric}) == expected


def test_live_narration_check_recognizes_eps_but_not_an_arbitrary_number():
    import importlib.util
    from types import SimpleNamespace
    from valuationagent.schemas.models import FinancialSnapshot

    spec = importlib.util.spec_from_file_location("live_input_workspace", Path(__file__).resolve().parents[1] / "scripts/live_input_workspace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = SimpleNamespace(request=SimpleNamespace(financials=FinancialSnapshot(net_income_parent=250000000, common_shares=500000000)))
    price = SimpleNamespace(method="pe", per_share_value=Decimal(6), equity_value=Decimal(3000000000))
    values = module.known_monetary_values(record, price)
    assert Decimal("0.50") in values
    assert Decimal("0.55") not in values
