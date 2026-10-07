import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from test_input_workspace import fixture
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.provider_inputs import InputCandidateQuery, list_input_candidates, provider_candidates, provider_record, validate_provider_input
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.application.result_document import build_result_document, document_sections
from valuationagent.application.workspace_artifacts import ReportWrite, write_report
from valuationagent.application.workspace_sensitivity import SensitivityRequest, analyze_sensitivity
from valuationagent.market.research_data import FinancialHistoryRequest, fetch_history
from valuationagent.market.tushare import TushareApiClient, TushareDataProvider
from valuationagent.schemas.inputs import RecordInputs


TEXT = "只用已有API数据，PE取20倍，不要联网。"


def provider_workspace(tmp_path, **changes):
    app, runtime = fixture(tmp_path, TEXT)
    session = runtime.session
    session.draft.company = "Synthetic issuer"
    session.draft.ticker = "600123"
    session.draft.industry = "半导体"
    session.draft.valuation_date = date(2026, 10, 3)
    session.information_cutoff_date = date(2026, 10, 3)
    row = {"ts_code": "600123.SH", "ann_date": "20260403", "f_ann_date": "20260403",
        "end_date": "20251231", "report_type": "1", "n_income_attr_p": "500000.123456789", "revenue": "6000000"}
    row.update(changes)
    shares = {"ts_code": "600123.SH", "trade_date": "20260930", "total_share": "100.25"}

    def handler(request):
        statement = json.loads(request.content)["api_name"]
        record = ({"ts_code": "600123.SH", "name": "Synthetic issuer", "fullname": "Synthetic issuer company",
            "industry": "半导体", "list_date": "20100101"} if statement == "stock_basic"
            else shares if statement == "daily_basic" else row)
        return httpx.Response(200, json={"code": 0, "data": {"fields": list(record), "items": [list(record.values())]}})

    provider = TushareDataProvider(TushareApiClient("synthetic-test-secret", transport=httpx.MockTransport(handler)))
    runtime.service.attach_market(session.session_id, provider)
    result = fetch_history(runtime, FinancialHistoryRequest(years=[2025], statements=["income", "statistics"]))
    app.state.store.save_research(session)
    source_ids = [item["file_id"] for item in result["documents"]]
    args = RecordInputs(
        user_values=[{"metric": "pe_multiple", "amount_text": "20倍", "unit": "ratio", "scope": "assumption"}],
        provider_values=[
            {"candidate_id": f"{source_ids[0]}@0:n_income_attr_p"},
            {"candidate_id": f"{source_ids[1]}@0:total_share"},
        ])
    return app, runtime, args


@pytest.mark.parametrize("years", [[], [2026], [2025]])
def test_market_date_statistics_are_not_blocked_by_annual_years(tmp_path, years):
    _, runtime, _ = provider_workspace(tmp_path)
    result = fetch_history(runtime, FinancialHistoryRequest(years=years, statements=["statistics"]))
    assert result["date_only_statements"] == ["statistics"]
    assert result["information_cutoff"] == "2026-10-03"
    assert result["documents"][0]["cached"]
    assert len(runtime.session.documents) == 3


def test_valid_fetch_without_saved_target_keeps_sources_and_defers_input_role(tmp_path):
    from test_input_acquisition import acquisition_workspace

    _, runtime, _, calls = acquisition_workspace(tmp_path)
    runtime.session.draft.ticker = ""
    result = fetch_history(runtime, FinancialHistoryRequest(ticker="600123.SH", years=[2025], statements=["income", "statistics"]))
    assert result["status"] == "sources_saved" and len(result["documents"]) == 2
    assert len(calls) == 3 and runtime.session.draft.ticker == ""
    first = result["documents"][0]
    assert first["input_candidates"]["excluded_by_reason"] == {"INPUT_TARGET_REQUIRED": 1}
    assert "update_task" in first["input_candidates"]["instruction"]
    assert runtime.session.input_dataset is None
    runtime.session.draft.ticker = "600123.SH"
    candidates = list_input_candidates(runtime, InputCandidateQuery(file_id=first["file_id"]))
    assert candidates["total"] > 0 and len(calls) == 3
    assert all(item["role"] == "historical" for item in candidates["items"])


@pytest.mark.parametrize("statements", [["income"], ["income", "statistics"]])
@pytest.mark.parametrize("years", [[], [2026]])
def test_current_year_financials_remain_rejected_before_acquisition(tmp_path, statements, years):
    _, runtime, _ = provider_workspace(tmp_path)
    with pytest.raises(ValueError, match="ANNUAL_PERIOD_REQUIRED"):
        fetch_history(runtime, FinancialHistoryRequest(years=years, statements=statements))
    assert len(runtime.session.documents) == 3


def test_cashflow_contract_preserves_original_capex_instead_of_inventing_a_final_field(tmp_path):
    _, runtime, _ = provider_workspace(tmp_path, c_pay_acq_const_fiolta="123456.78")
    result = fetch_history(runtime, FinancialHistoryRequest(years=[2025], statements=["cashflow"]))
    file_id = result["documents"][0]["file_id"]
    record_inputs(runtime, RecordInputs(provider_values=[{"candidate_id": file_id + "@0:c_pay_acq_const_fiolta"}]))
    row = runtime.session.input_dataset.active_records()[0]
    assert row.metric == "cash_paid_for_ppe_intangibles" and row.value == Decimal("123456.78")
    assert row.source.provider_binding["record"]["c_pay_acq_const_fiolta"] == "123456.78"
    assert row.unit == "元" and row.scope == "consolidated"
    assert row.period_end == date(2025, 12, 31) and row.assertion == "reported"
    validate_provider_input(runtime.session, row)


def test_direct_api_contract_inputs_calculate_report_replay_and_sensitivity(tmp_path):
    app, runtime, args = provider_workspace(tmp_path)
    saved = record_inputs(runtime, args)
    assert len(saved["saved_input_ids"]) == 3
    assert sorted(row["source_kind"] for row in saved["saved_records"]) == ["provider", "provider", "user"]
    assert all("source" not in row for row in saved["saved_records"])
    assert not record_inputs(runtime, args)["saved_input_ids"]
    request = prepare_dataset(runtime.session, ["pe"])
    assert request.analysis_basis == "research"
    assert request.financials.net_income_parent == Decimal("500000.123456789")
    assert request.financials.common_shares == Decimal("1002500")
    assert request.financials.common_shares_as_of == date(2026, 9, 30)
    assert request.financials.period_end == date(2025, 12, 31)
    assert len(request.input_records) == 3 and not runtime.session.facts
    source = request.input_records[1]["source"]
    assert source["provider_binding"]["record"]["n_income_attr_p"] == "500000.123456789"
    assert "供应商字段契约校验" in request.financials.evidence["net_income_parent"][0].note
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] in {"completed", "completed_with_warnings"}, outcome
    record = app.state.store.get_run(outcome["run_id"])
    price = record.result.relative[0].per_share_value
    assert price == (Decimal("500000.123456789") * 20 / Decimal("1002500")).quantize(Decimal("0.0001"))
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    artifact = write_report(app.state.research, runtime.session, ReportWrite(format="pdf"))
    assert artifact["numeric_result_available"]
    assert app.state.store.get_artifact(runtime.session.session_id, artifact["artifact_id"])[1].startswith(b"%PDF")
    text = str(document_sections(build_result_document(app.state.research, runtime.session)))
    assert "供应商字段契约校验" in text
    assert "来源解释已复核" not in text
    study = analyze_sensitivity(runtime, SensitivityRequest(method="pe", parameter="multiple", values=[15, 25]))
    assert Decimal(study["scenarios"][0]["per_share_value"]) == (Decimal("500000.123456789") * 15 / Decimal("1002500")).quantize(Decimal("0.0001"))
    assert app.state.store.get_run(record.run_id).result.relative[0].per_share_value == price


@pytest.mark.parametrize("changes,error", [
    ({"ts_code": "000999.SZ"}, "INPUT_PROVIDER_ENTITY"),
    ({"ann_date": "20261010", "f_ann_date": "20261010"}, "INPUT_PROVIDER_DATE"),
    ({"end_date": "20260630"}, "INPUT_PROVIDER_DATE"),
    ({"end_date": "20250930"}, "INPUT_PROVIDER_PERIOD"),
    ({"report_type": "6"}, "INPUT_PROVIDER_PERIOD"),
    ({"end_date": "not-a-date"}, "INPUT_PROVIDER_DATE"),
    ({"f_ann_date": None, "ann_date": None}, "INPUT_PROVIDER_DATE"),
    ({"n_income_attr_p": None}, "INPUT_PROVIDER_AMOUNT"),
    ({"n_income_attr_p": "NaN"}, "INPUT_PROVIDER_AMOUNT"),
    ({"n_income_attr_p": True}, "INPUT_PROVIDER_AMOUNT"),
])
def test_bad_provider_batch_does_not_persist_user_assumptions(tmp_path, changes, error):
    app, runtime, args = provider_workspace(tmp_path, **changes)
    with pytest.raises(ValueError, match=error):
        record_inputs(runtime, args)
    assert runtime.session.input_dataset is None
    assert app.state.store.get_research(runtime.session.session_id).input_dataset is None


@pytest.mark.parametrize("problem", ["field", "reference", "upload", "foreign", "replacement", "date"])
def test_wrong_selection_cannot_bypass_source_or_financial_semantics(tmp_path, problem):
    _, runtime, args = provider_workspace(tmp_path)
    if problem == "field":
        args.provider_values[0].candidate_id = args.provider_values[0].candidate_id.replace("n_income_attr_p", "operate_profit")
    elif problem == "reference":
        args.provider_values[0].candidate_id = "made-up-reference"
    elif problem == "upload":
        next(document for document in runtime.session.documents if ":income:" in document.provider).provenance_type = "uploaded"
    elif problem == "foreign":
        args.provider_values[0].candidate_id = "file_ffffffff@0:n_income_attr_p"
    elif problem == "replacement":
        args.provider_values[0].replaces = ["missing"]
    else:
        runtime.session.information_cutoff_date = date(2026, 1, 1)
    with pytest.raises(ValueError):
        record_inputs(runtime, args)
    assert runtime.session.input_dataset is None


@pytest.mark.parametrize("problem", ["input", "raw", "binding", "contract", "quote"])
def test_selected_api_values_and_frozen_bytes_cannot_drift(tmp_path, problem):
    app, runtime, args = provider_workspace(tmp_path)
    record_inputs(runtime, args)
    row = runtime.session.input_dataset.records[1]
    if problem == "input":
        row.value = Decimal(600000)
    elif problem == "raw":
        Path(app.state.store.get_file(row.source.file_id)["storage_path"]).write_bytes(b"changed")
        with pytest.raises(ValueError, match="SOURCE_CHANGED"):
            app.state.workspaces._freeze_prevaluation_request(runtime.session, {"methods": ["pe"]})
        return
    elif problem == "binding":
        row.source.provider_binding["record"]["n_income_attr_p"] = "600000"
    elif problem == "contract":
        row.source.provider_binding["contract_version"] = "other-version"
    else:
        row.source.quote = "faked quote"
    with pytest.raises(ValueError, match="INPUT_SOURCE_CHANGED|INPUT_PROVIDER_CONTRACT"):
        prepare_dataset(runtime.session, ["pe"])


def test_raw_record_recheck_rejects_changed_binding_even_with_updated_hash(tmp_path):
    _, runtime, args = provider_workspace(tmp_path)
    record_inputs(runtime, args)
    row = runtime.session.input_dataset.records[1]
    with pytest.raises(ValueError, match="INPUT_SOURCE_CHANGED"):
        validate_provider_input(runtime.session, row,
            json.dumps({"data": {"fields": list(row.source.provider_binding["record"]),
                "items": [list({**row.source.provider_binding["record"], "n_income_attr_p": "900000"}.values())]}}).encode())


def test_source_failure_after_valid_provider_selection_preserves_existing_dataset(tmp_path):
    app, runtime, args = provider_workspace(tmp_path)
    record_inputs(runtime, RecordInputs(user_values=args.user_values))
    before = runtime.session.input_dataset.model_dump()
    args.source_values = RecordInputs(source_values=[{"fact_id": "missing"}]).source_values
    with pytest.raises(ValueError, match="INPUT_SOURCE"):
        record_inputs(runtime, args)
    assert runtime.session.input_dataset.model_dump() == before
    assert app.state.store.get_research(runtime.session.session_id).input_dataset.model_dump() == before


@pytest.mark.parametrize("fields,values", [(["same", "same"], [1, 2]), (["one"], [1, 2]), ([5], [1])])
def test_raw_field_position_errors_are_not_truncated(fields, values):
    with pytest.raises(ValueError, match="INPUT_PROVIDER_SHAPE"):
        provider_record(json.dumps({"data": {"fields": fields, "items": [values]}}).encode(), "/data/items/0")


def test_fetched_candidates_show_exact_values_and_can_be_selected_without_mapping(tmp_path):
    app, runtime, _ = provider_workspace(tmp_path)
    response = fetch_history(runtime, FinancialHistoryRequest(years=[2025], statements=["income", "statistics"]))
    assert all(document["cached"] for document in response["documents"])
    candidates = [item for document in response["documents"] for item in document["input_candidates"]["items"]]
    profit = next(item for item in candidates if item["metric"] == "net_income_parent")
    assert profit["original_amount"] == "500000.123456789"
    assert profit["normalized_value"] == "500000.123456789"
    assert profit["period_end"] == "2025-12-31" and profit["published_at"] == "2026-04-03"
    assert profit["role"] == "historical" and profit["ticker"] == "600123.SH"
    shares = next(item for item in candidates if item["metric"] == "common_shares")
    assert shares["original_amount"] == "100.25" and shares["normalized_value"] == "1002500.00"
    assert shares["unit"] == "万股" and shares["as_of"] == "2026-09-30"
    assert runtime.session.input_dataset is None
    record_inputs(runtime, RecordInputs(provider_values=[{"candidate_id": profit["candidate_id"]}]))
    assert runtime.session.input_dataset.records[0].value == Decimal("500000.123456789")
    assert app.state.store.get_research(runtime.session.session_id).input_dataset.records[0].source.kind == "provider"


def test_candidates_exclude_invalid_values_without_silently_zero_filling(tmp_path):
    _, runtime, _ = provider_workspace(tmp_path, n_income_attr_p=None)
    response = fetch_history(runtime, FinancialHistoryRequest(years=[2025], statements=["income"]))
    candidates = response["documents"][0]["input_candidates"]
    assert [item["metric"] for item in candidates["items"]] == ["revenue"]
    assert candidates["excluded_by_reason"] == {"INPUT_PROVIDER_AMOUNT": 1}


def test_candidate_selection_rejects_mapping_overrides(tmp_path):
    _, _, args = provider_workspace(tmp_path)
    for key, value in {"metric": "ebitda", "field": "operate_profit", "role": "comparable", "value": 100}.items():
        with pytest.raises(ValueError, match="extra_forbidden"):
            RecordInputs(provider_values=[{**args.provider_values[0].model_dump(), key: value}])


def test_cached_candidates_recheck_source_integrity(tmp_path):
    app, runtime, _ = provider_workspace(tmp_path)
    document = next(document for document in runtime.session.documents if ":income:" in document.provider)
    Path(app.state.store.get_file(document.file_id)["storage_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        fetch_history(runtime, FinancialHistoryRequest(years=[2025], statements=["income"]))


def test_candidate_pages_are_bounded_and_keep_original_record_addresses(tmp_path):
    _, runtime, _ = provider_workspace(tmp_path)
    document = next(document for document in runtime.session.documents if ":income:" in document.provider)
    fields = ["ts_code", "ann_date", "end_date", "report_type", "revenue", "n_income_attr_p"]
    rows = [["600123.SH", f"{year + 1}0403", f"{year}1231", "1", str(year * 10), str(year)] for year in range(2016, 2026)]
    raw = json.dumps({"data": {"fields": fields, "items": rows}}).encode()
    first = provider_candidates(runtime.session, document, raw)
    assert len(first["items"]) == 6 and first["total"] == 20
    assert first["items"][0]["candidate_id"] == f"{document.file_id}@9:revenue"
    second = provider_candidates(runtime.session, document, raw, InputCandidateQuery(**first["next_action"]["arguments"]))
    assert not {item["candidate_id"] for item in first["items"]} & {item["candidate_id"] for item in second["items"]}
    selected = provider_candidates(runtime.session, document, raw,
        InputCandidateQuery(file_id=document.file_id, period_end="2020-12-31", metrics=["net_income_parent"]))
    assert selected["total"] == 1 and selected["next_action"] is None
    assert selected["items"][0]["candidate_id"] == f"{document.file_id}@4:n_income_attr_p"
    assert selected["items"][0]["original_amount"] == "2020"


def test_list_candidates_reads_saved_bytes_without_network_or_automatic_admission(tmp_path):
    _, runtime, args = provider_workspace(tmp_path)
    file_id = args.provider_values[0].candidate_id.split("@")[0]
    result = list_input_candidates(runtime, InputCandidateQuery(file_id=file_id, metrics=["revenue"]))
    assert result["total"] == 1 and result["items"][0]["original_amount"] == "6000000"
    assert runtime.session.input_dataset is None
    with pytest.raises(ValueError):
        list_input_candidates(runtime, InputCandidateQuery(file_id="file_ffffffff"))
