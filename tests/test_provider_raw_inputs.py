import json
from datetime import date
from decimal import Decimal

import httpx
import pytest

from test_input_workspace import fixture
from valuationagent.application.input_acquisition import AcquireFinancialInputs, acquire_financial_inputs
from valuationagent.application.provider_inputs import InputCandidateQuery, list_input_candidates, validate_provider_input
from valuationagent.application.record_inputs import record_inputs
from valuationagent.market.research_data import FinancialHistoryRequest, fetch_history
from valuationagent.market.tushare import TushareApiClient, TushareDataProvider
from valuationagent.market.tushare_contracts import CONTRACT_VERSION
from valuationagent.schemas.inputs import RecordInputs


def raw_workspace(tmp_path, changes=None):
    app, runtime = fixture(tmp_path, "只用测试夹具中的API快照，PE按20倍测试。不要联网。")
    session = runtime.session
    session.draft.company = "Synthetic issuer"
    session.draft.ticker = "600123.SH"
    session.draft.industry = "家用电器"
    session.draft.methods = ["pe", "ps", "ev_ebitda"]
    session.draft.valuation_date = date(2026, 10, 3)
    session.information_cutoff_date = date(2026, 10, 3)
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        ticker, statement = payload["params"]["ts_code"], payload["api_name"]
        calls.append(payload)
        if statement == "stock_basic":
            records = [{"ts_code": ticker, "name": "Synthetic issuer" if ticker == "600123.SH" else "Synthetic peer",
                "industry": "家用电器", "list_date": "20100101"}]
        elif statement == "daily_basic":
            records = [{"ts_code": ticker, "trade_date": "20260930", "total_share": "100",
                "close": "15", "total_mv": "1500"}]
        else:
            financial = {
                "income": {"revenue": "10000000", "n_income_attr_p": "1500000", "total_profit": "2000000",
                    "income_tax": "500000", "operate_profit": "1800000", "int_exp": None,
                    "fin_exp_int_exp": "100000.123456789", "fin_exp": "-200000"},
                "cashflow": {"c_pay_acq_const_fiolta": "500000", "depr_fa_coga_dpba": "200000",
                    "amort_intang_assets": "50000", "lt_amort_deferred_exp": "0", "use_right_asset_dep": "20000"},
                "balancesheet": {"money_cap": "1000000", "st_borr": "50000", "lt_borr": "0",
                    "bond_payable": None, "lease_liab": "40000", "minority_int": "10000",
                    "non_cur_liab_due_1y": "20000", "inventories": "700000", "accounts_receiv": "400000",
                    "acct_payable": "500000"},
            }[statement]
            records = [{"ts_code": ticker, "ann_date": f"{year + 1}0403", "f_ann_date": f"{year + 1}0403",
                "end_date": f"{year}1231", "report_type": "1", "comp_type": "1", **financial,
                **(changes or {}).get(statement, {})} for year in (2024, 2025)]
        fields = list(records[0])
        return httpx.Response(200, json={"code": 0, "data": {"fields": fields,
            "items": [[record[field] for field in fields] for record in records]}})

    runtime.service.attach_market(session.session_id,
        TushareDataProvider(TushareApiClient("synthetic-test-secret", transport=httpx.MockTransport(handler))))
    args = AcquireFinancialInputs(years=[2024, 2025], comparables=[{
        "ticker": "600124.SH", "name": "Synthetic peer",
        "rationale": "仅用于测试同类制造业样本取数，实际业务差异仍需分析。"}])
    return app, runtime, args, calls


def test_raw_fields_acquired_once_preserve_meaning_period_entity_and_null(tmp_path):
    _, runtime, args, calls = raw_workspace(tmp_path)
    outcome = acquire_financial_inputs(runtime, args)
    records = runtime.session.input_dataset.active_records()
    raw = [row for row in records if row.metric.startswith("raw.")]
    assert raw and outcome["raw_operand_count"] == len(raw)
    interest = next(row for row in raw if row.metric == "raw.tushare_income_fin_exp_int_exp")
    assert interest.value == Decimal("100000.123456789")
    assert interest.unit == "元" and interest.currency == "CNY" and interest.period_kind == "annual"
    assert interest.label == "财务费用：利息费用" and interest.as_of is None
    assert interest.source.provider_binding["admission_kind"] == "raw_operand"
    assert interest.source.provider_binding["contract_version"] == CONTRACT_VERSION
    assert not any(row.metric in {"ebit", "ebitda", "cash_and_non_operating_assets", "interest_bearing_debt",
        "lease_liabilities", "minority_interest", "depreciation_amortization"} for row in records)
    assert not any(row.metric.endswith(("_bond_payable", "_int_exp"))
        and not row.metric.endswith("fin_exp_int_exp") for row in raw)
    balances = [row for row in raw if "balancesheet" in row.metric]
    assert all(row.period_kind == "instant" and row.as_of is None and row.period_end for row in balances)
    zero = next(row for row in balances if row.metric.endswith("_lt_borr"))
    assert zero.value == 0 and zero.original_amount == "0"
    peer_raw = [row for row in raw if row.role == "comparable"]
    assert peer_raw and all(row.entity_ticker == "600124.SH" and row.period_end == date(2025, 12, 31) for row in peer_raw)
    assert {row.period_end.year for row in raw if row.role == "historical"} == {2024, 2025}
    assert not runtime.session.facts
    assert outcome["interpretation_targets"] and "record_inputs(calculations=" in outcome["instruction"]
    assert all(not metric.startswith("raw.") for row in outcome["coverage"] for metric in row["admitted"])
    before_calls = len(calls)
    repeat = acquire_financial_inputs(runtime, args)
    assert len(calls) == before_calls and repeat["new_input_count"] == 0
    assert repeat["raw_operand_count"] == outcome["raw_operand_count"]


def test_history_requests_correct_interest_and_lease_depreciation_fields_and_instant_labels(tmp_path):
    _, runtime, _, calls = raw_workspace(tmp_path)
    response = fetch_history(runtime, FinancialHistoryRequest(years=[2025], statements=["income", "cashflow", "balancesheet"]))
    requested = {item["api_name"]: item["fields"].split(",") for item in calls if item["api_name"] != "stock_basic"}
    assert "fin_exp_int_exp" in requested["income"] and "int_exp" in requested["income"]
    assert "use_right_asset_dep" in requested["cashflow"]
    assert all("comp_type" in fields for fields in requested.values())
    balance = next(item for item in response["documents"] if "balancesheet" in next(
        doc.provider for doc in runtime.session.documents if doc.file_id == item["file_id"]))
    document = next(doc for doc in runtime.session.documents if doc.file_id == balance["file_id"])
    assert document
    blocks = runtime.service.store.research_blocks(runtime.session.session_id, document.file_id)
    assert all("2025-12-31期末余额" in block["text"] for block in blocks)
    assert all(block["location"]["period_kind"] == "instant" for block in blocks)


@pytest.mark.parametrize("changes,error", [
    ({"fin_exp_int_exp": None}, "INPUT_PROVIDER_AMOUNT"),
    ({"fin_exp_int_exp": "NaN"}, "INPUT_PROVIDER_AMOUNT"),
    ({"fin_exp_int_exp": True}, "INPUT_PROVIDER_AMOUNT"),
    ({"report_type": "6"}, "INPUT_PROVIDER_PERIOD"),
    ({"comp_type": "2"}, "INPUT_PROVIDER_SCOPE"),
    ({"comp_type": None}, "INPUT_PROVIDER_SCOPE"),
    ({"comp_type": True}, "INPUT_PROVIDER_SCOPE"),
    ({"end_date": "20250930"}, "INPUT_PROVIDER_PERIOD"),
    ({"f_ann_date": "20261004"}, "INPUT_PROVIDER_DATE"),
    ({"ts_code": "600125.SH"}, "INPUT_PROVIDER_ENTITY"),
])
def test_raw_candidate_guards_do_not_relax_for_modeling_operands(tmp_path, changes, error):
    _, runtime, _, _ = raw_workspace(tmp_path, {"income": changes})
    response = fetch_history(runtime, FinancialHistoryRequest(years=[2025], statements=["income"]))
    source = response["documents"][0]
    assert source["input_candidates"]["excluded_by_reason"].get(error)
    with pytest.raises(ValueError, match=error):
        record_inputs(runtime, RecordInputs(provider_values=[{"candidate_id": source["file_id"] + "@1:fin_exp_int_exp"}]))
    assert runtime.session.input_dataset is None


def test_existing_pe_path_does_not_accumulate_unrequested_raw_inputs(tmp_path):
    _, runtime, args, _ = raw_workspace(tmp_path)
    runtime.session.draft.methods = ["pe", "ps"]
    acquire_financial_inputs(runtime, args)
    assert not any(row.metric.startswith("raw.") for row in runtime.session.input_dataset.active_records())
    args.extra_statements = ["balancesheet"]
    acquire_financial_inputs(runtime, args)
    raw = [row for row in runtime.session.input_dataset.active_records() if row.metric.startswith("raw.")]
    assert raw and all("balancesheet" in row.metric for row in raw)


@pytest.mark.parametrize("field,value", [("metric", "ebit"), ("value", Decimal("999")), ("period_kind", "instant"),
    ("entity_ticker", "600124.SH"), ("label", "EBIT")])
def test_raw_provider_inputs_are_not_semantically_relabelable(tmp_path, field, value):
    _, runtime, args, _ = raw_workspace(tmp_path)
    acquire_financial_inputs(runtime, args)
    row = next(row for row in runtime.session.input_dataset.records if row.metric == "raw.tushare_income_fin_exp_int_exp")
    validate_provider_input(runtime.session, row)
    setattr(row, field, value)
    with pytest.raises(ValueError, match="INPUT_SOURCE_CHANGED|INPUT_PROVIDER_FIELD"):
        validate_provider_input(runtime.session, row)


def test_saved_raw_operands_can_be_referenced_without_rereading_or_retyping_numbers(tmp_path):
    _, runtime, args, calls = raw_workspace(tmp_path)
    acquire_financial_inputs(runtime, args)
    source = next(document for document in runtime.session.documents if ":income:" in document.provider)
    candidates = list_input_candidates(runtime, InputCandidateQuery(file_id=source.file_id,
        period_end=date(2025, 12, 31), metrics=["raw.tushare_income_fin_exp_int_exp"]))
    assert len(candidates["items"]) == 1 and candidates["items"][0]["admission_kind"] == "raw_operand"
    records = [row for row in runtime.session.input_dataset.active_records()
        if row.role == "historical" and row.period_end == date(2025, 12, 31)]
    required = {"profit_before_tax", "raw.tushare_income_fin_exp_int_exp"}
    before = len(calls)
    outcome = record_inputs(runtime, RecordInputs(calculations=[{"metric": "ebit", "period_end": "2025-12-31",
        "terms": [{"input_id": row.input_id, "operation": "add"} for row in records if row.metric in required],
        "rationale": "本测试按利润总额加财务费用中的利息费用定义未调整EBIT，保留投资收益等未剔除的限制，不等同经营利润。",
        "limitations": ["仅验证原始科目可引用；投资收益等未调整，不代表经营利润已经审计。"]}]))
    assert outcome["saved_calculation_ids"] and len(calls) == before
    assert not runtime.session.facts
    assert runtime.session.input_dataset.active_calculations()[0].metric == "ebit"
