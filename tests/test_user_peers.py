from datetime import date
from decimal import Decimal

import pytest

from test_input_workspace import fixture
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.schemas.inputs import RecordInputs


def scenario(tmp_path, missing=None):
    target = "目标：收入1000万元，净利润150万元，EBITDA250万元，股数100万股，现金100万元，债务0万元。"
    lines = [f"可比{name}：收入1000万元，净利润100万元，EBITDA150万元，股数100万股，股价{price}元；现金、债务、租赁、少数权益均为0万元。"
        for name, price in zip("ABC", (12, 15, 18))]
    text = "不要联网。人工情景，2025年度和2025年12月31日数据在当日已知。\n" + "\n".join([target, *lines])
    app, runtime = fixture(tmp_path, text)
    runtime.session.draft.valuation_date = date(2025, 12, 31)
    runtime.session.draft.methods = ["pe", "ps", "ev_ebitda"]
    values = []

    def add(metric, amount, unit, line, name=""):
        if name and metric == missing:
            return
        values.append({"metric": metric, "amount_text": amount, "unit": unit, "value_context": line,
            "role": "comparable" if name else "historical", "entity": name,
            "period_end": "2025-12-31", "as_of": "2025-12-31"})

    for metric, amount, unit in [("revenue", "1000万元", "万元"), ("net_income_parent", "150万元", "万元"),
        ("ebitda", "250万元", "万元"), ("common_shares", "100万股", "万股"),
        ("cash_and_non_operating_assets", "100万元", "万元"), ("interest_bearing_debt", "0万元", "万元")]:
        add(metric, amount, unit, target)
    for name, price, line in zip("ABC", (12, 15, 18), lines):
        for metric, amount, unit in [("revenue", "1000万元", "万元"), ("net_income_parent", "100万元", "万元"),
            ("ebitda", "150万元", "万元"), ("common_shares", "100万股", "万股"), ("market_price", f"{price}元", "元"),
            *[(metric, "0万元", "万元") for metric in ("cash_and_non_operating_assets", "interest_bearing_debt", "lease_liabilities", "minority_interest")]]:
            add(metric, amount, unit, line, "可比" + name)
    args = RecordInputs(user_basis={"period_quote": "2025年度", "as_of_quote": "2025年12月31日"}, user_values=values)
    record_inputs(runtime, args)
    return app, runtime


def test_unlisted_user_peers_compute_multiples_with_explicit_capital_bridge_and_replay(tmp_path):
    app, runtime = scenario(tmp_path)
    request = prepare_dataset(runtime.session, ["pe", "ps", "ev_ebitda"])
    assert request.company.ticker is None
    assert request.analysis_basis == "user_scenario"
    assert [peer.ticker for peer in request.peers] == ["user:可比A", "user:可比B", "user:可比C"]
    assert [peer.pe for peer in request.peers] == [12, 15, 18]
    assert [peer.ps for peer in request.peers] == [Decimal("1.2"), Decimal("1.5"), Decimal("1.8")]
    assert [peer.ev_ebitda for peer in request.peers] == [8, 10, 12]
    assert all(peer.evidence["ev_ebitda"] for peer in request.peers)
    runtime.calculate(CalculateValuation())
    record = app.state.store.get_run(runtime.session.valuation_run_id)
    assert {item.method: item.per_share_value for item in record.result.relative} == {
        "pe": Decimal("22.5"), "ps": Decimal(15), "ev_ebitda": Decimal(26)}
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    from valuationagent.application.user_peers import verify_user_peers

    request.peers[0].pe += 1
    with pytest.raises(ValueError, match="INPUT_PEER_REPLAY"):
        verify_user_peers(request)


def test_missing_peer_zero_blocks_ev_but_not_pe(tmp_path):
    _, runtime = scenario(tmp_path, missing="lease_liabilities")
    request = prepare_dataset(runtime.session, ["pe"])
    assert len(request.peers) == 3
    with pytest.raises(ValueError, match="lease_liabilities"):
        prepare_dataset(runtime.session, ["ev_ebitda"])
