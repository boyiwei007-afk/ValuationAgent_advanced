import json
from decimal import Decimal

import pytest

from test_user_peers import scenario
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.record_inputs import record_inputs
from valuationagent.schemas.inputs import RecordInputs


def conflicting_peers(tmp_path, metric="ebitda"):
    app, runtime = scenario(tmp_path)
    parent = next(row for row in runtime.session.input_dataset.active_records()
        if row.role == "comparable" and row.entity == "可比A" and row.metric == "revenue")
    locator = json.loads(parent.source.locator)
    record_inputs(runtime, RecordInputs(user_basis={"period_quote": "2025年度", "as_of_quote": "2025年12月31日"},
        user_values=[{"metric": metric, "amount_text": parent.original_amount, "unit": parent.unit,
            "role": "comparable", "entity": parent.entity, "period_end": "2025-12-31", "as_of": "2025-12-31",
            "amount_occurrence": locator["amount_occurrence"]}]))
    return app, runtime


@pytest.mark.parametrize("methods", [["pe"], ["ps"], ["pe", "ps"]])
def test_unused_peer_ebitda_conflict_does_not_block_equity_methods(tmp_path, methods):
    _, runtime = conflicting_peers(tmp_path)
    before = runtime.session.input_dataset.model_dump()
    request = prepare_dataset(runtime.session, methods)
    assert len(request.peers) == 3
    assert request.peers[0].pe == (Decimal(12) if "pe" in methods else None)
    assert request.peers[0].ps == (Decimal("1.2") if "ps" in methods else None)
    assert all(row["metric"] != "ebitda" for row in request.input_records if row["role"] == "comparable")
    assert runtime.session.input_dataset.model_dump() == before


def test_needed_peer_conflict_preserves_guard_and_names_actual_inputs(tmp_path):
    _, runtime = conflicting_peers(tmp_path)
    identifiers = {row.input_id for row in runtime.session.input_dataset.active_records()
        if row.entity == "可比A" and row.metric == "ebitda"}
    with pytest.raises(ValueError, match="INPUT_PEER_CONFLICT") as error:
        prepare_dataset(runtime.session, ["ev_ebitda"])
    assert all(identifier in str(error.value) for identifier in identifiers)
    assert "150万元" in str(error.value) and "1000万元" in str(error.value)


def test_unused_peer_earnings_conflict_does_not_block_ps(tmp_path):
    _, runtime = conflicting_peers(tmp_path, metric="net_income_parent")
    request = prepare_dataset(runtime.session, ["ps"])
    assert len(request.peers) == 3
    with pytest.raises(ValueError, match="INPUT_PEER_CONFLICT"):
        prepare_dataset(runtime.session, ["pe"])


def test_user_peer_policy_mismatch_requires_matching_frozen_version(tmp_path):
    from valuationagent.application.user_peers import verify_user_peers

    _, runtime = scenario(tmp_path)
    request = prepare_dataset(runtime.session, ["pe"])
    request.peers[0].calculation_methods["user_peer_policy"] = "different-frozen-policy"
    with pytest.raises(ValueError, match="INPUT_PEER_POLICY_VERSION"):
        verify_user_peers(request)
