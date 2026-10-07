import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_input_acquisition import acquisition_workspace
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.input_acquisition import acquire_financial_inputs
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.issuer_identity import audit_request_identities, identity_record, require_identity


def test_wrong_peer_name_is_rejected_without_binding_foreign_company_inputs(tmp_path):
    _, runtime, args, calls = acquisition_workspace(tmp_path)
    args.comparables[0].name = "Another issuer"
    with pytest.raises(ValueError, match="ISSUER_IDENTITY_MISMATCH.*600101.SH.*Synthetic peer 0.*Another issuer"):
        acquire_financial_inputs(runtime, args)
    assert runtime.session.input_dataset is None
    assert require_identity(runtime.session, "600101.SH")["record"]["name"] == "Synthetic peer 0"
    assert ("600101.SH", "stock_basic") in calls
    args.comparables[0].name = "Synthetic peer 0"
    result = acquire_financial_inputs(runtime, args)
    assert result["new_input_count"] == 30
    assert calls.count(("600101.SH", "stock_basic")) == 1
    assert prepare_dataset(runtime.session, ["pe"]).peers[0].name == "Synthetic peer 0"


def test_wrong_target_name_cannot_be_justified_by_valid_ticker(tmp_path):
    _, runtime, args, _ = acquisition_workspace(tmp_path)
    runtime.session.draft.company = "Unrelated manufacturer"
    with pytest.raises(ValueError, match="ISSUER_IDENTITY_MISMATCH.*Synthetic issuer.*Unrelated"):
        acquire_financial_inputs(runtime, args)
    assert runtime.session.input_dataset is None


def test_peer_rename_after_acquisition_is_rechecked_before_calculation(tmp_path):
    _, runtime, args, _ = acquisition_workspace(tmp_path)
    acquire_financial_inputs(runtime, args)
    runtime.session.input_dataset.comparables["600101.SH"].name = "Unrelated issuer"
    with pytest.raises(ValueError, match="ISSUER_IDENTITY_MISMATCH"):
        prepare_dataset(runtime.session, ["pe"])
    assert runtime.session.valuation_run_id is None


def test_identity_raw_bytes_are_checked_before_frozen_calculation(tmp_path):
    app, runtime, args, _ = acquisition_workspace(tmp_path)
    acquire_financial_inputs(runtime, args)
    proof = require_identity(runtime.session, "600101.SH")
    Path(app.state.store.get_file(proof["file_id"])["storage_path"]).write_bytes(b"tampered fixture")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        app.state.workspaces._freeze_prevaluation_request(runtime.session, {"methods": ["pe"]})
    result = runtime.calculate(CalculateValuation())
    assert result["status"] == "inputs_missing"
    assert runtime.session.valuation_run_id is None


@pytest.mark.parametrize("changes,code", [
    ({"industry": "银行"}, "ISSUER_FINANCIAL_SCOPE"),
    ({"list_date": "20271001"}, "ISSUER_NOT_LISTED_AT_CUTOFF"),
])
def test_identity_missing_or_outside_scope_does_not_get_a_free_pass(tmp_path, changes, code):
    _, runtime, args, _ = acquisition_workspace(tmp_path, changes={("600123.SH", "stock_basic"): changes})
    with pytest.raises(ValueError, match=code):
        acquire_financial_inputs(runtime, args)
    assert runtime.session.input_dataset is None


def test_missing_identity_preserves_readable_raw_financials_but_not_admission(tmp_path):
    _, runtime, args, calls = acquisition_workspace(tmp_path, changes={("600123.SH", "stock_basic"): {"name": None}})
    result = acquire_financial_inputs(runtime, args.model_copy(update={"comparables": []}))
    assert result["status"] == "partial" and result["new_input_count"] == 0
    assert result["sources"] and not runtime.session.input_dataset.active_records()
    assert any("ISSUER_IDENTITY_MISSING" in issue.get("error", "") for issue in result["issues"])
    acquire_financial_inputs(runtime, args.model_copy(update={"comparables": []}))
    assert calls.count(("600123.SH", "stock_basic")) == 1


def test_identity_parser_rejects_duplicates_wrong_ticker_and_bad_shapes():
    fields = ["ts_code", "name"]
    for rows in ([], [["600124.SH", "Other"]], [["600123.SH", "Name"], ["600123.SH", "Name"]], [["600123.SH"]]):
        with pytest.raises(ValueError, match="ISSUER_IDENTITY"):
            identity_record(json.dumps({"data": {"fields": fields, "items": rows}}), "600123.SH")


def test_identity_matching_accepts_only_literal_normalization_and_named_aliases(tmp_path):
    _, runtime, args, _ = acquisition_workspace(tmp_path)
    acquire_financial_inputs(runtime, args)
    assert require_identity(runtime.session, "600123.SH", " SYNTHETIC ISSUER ")["record"]["name"] == "Synthetic issuer"
    assert require_identity(runtime.session, "600123.SH", "600123")
    with pytest.raises(ValueError, match="ISSUER_IDENTITY_MISMATCH"):
        require_identity(runtime.session, "600123.SH", "Synthetic")


def test_frozen_identity_audit_covers_ineligible_peers_retained_for_traceability(tmp_path):
    app, runtime, args, _ = acquisition_workspace(tmp_path,
        changes={("600101.SH", "income"): {"n_income_attr_p": "-1"}})
    acquire_financial_inputs(runtime, args)
    request = prepare_dataset(runtime.session, ["pe"])
    assert len(request.peers) == 4
    audit = audit_request_identities(app.state.store, runtime.session, request)
    assert audit["status"] == "verified" and len(audit["issuers"]) == 6
    request.peers[0].name = "Wrong issuer"
    with pytest.raises(ValueError, match="ISSUER_IDENTITY_MISMATCH"):
        audit_request_identities(app.state.store, runtime.session, request)


def test_user_scenarios_do_not_require_a_market_code_or_online_identity():
    request = SimpleNamespace(company=SimpleNamespace(ticker="synthetic-not-a-stock"),
        input_records=[{"source": {"kind": "user"}}])
    assert audit_request_identities(None, None, request)["status"] == "not_applicable"


def test_frozen_provider_inputs_cannot_drop_their_identity_proof(tmp_path):
    app, runtime, args, _ = acquisition_workspace(tmp_path)
    acquire_financial_inputs(runtime, args)
    request = prepare_dataset(runtime.session, ["pe"])
    request.input_records[0]["source"]["provider_binding"].pop("issuer_identity")
    with pytest.raises(ValueError, match="ISSUER_IDENTITY_REQUIRED"):
        audit_request_identities(app.state.store, runtime.session, request)
