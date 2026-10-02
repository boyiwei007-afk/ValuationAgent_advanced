from datetime import date
from decimal import Decimal

import pytest

from valuationagent.application.peer_inputs import assemble_peers
from valuationagent.application.research_valuation import ResearchValuationAssembler
from test_multisource_extraction import attach, runtime_at
from test_observation_extraction import review, submit


def peer_component(runtime, metric, value, *, ticker="600456", year="2025", unit="万元", **changes):
    market = metric == "market_cap"
    scope = "issuer" if market else "consolidated"
    period = "2026-09-30" if market else year + "-12-31"
    text = f"Synthetic peer {ticker}\nScope {scope}\nUnit {unit} CNY\n{period}\n{metric} {value}"
    block = attach(runtime, f"{ticker}-{metric}.txt", text, published="2026-09-30")[0]
    args = {"file_id": block["file_id"], "anchors": {
        key: {"block_id": block["block_id"], "start_line": number}
        for key, number in [("entity", 1), ("scope", 2), ("unit", 3), ("period", 4), ("value", 5)]},
        "basis": {"entity_name": "Synthetic peer", "entity_ticker": ticker, "entity_refs": ["entity"],
                  "scope": scope, "scope_refs": ["scope"], "unit": unit, "currency": "CNY", "unit_refs": ["unit"]},
        "rows": [{"metric": metric, "standard_metric": metric, "raw_value": value,
                  "value_ref": "value", "label_refs": ["value"], "period_refs": ["period"],
                  "period_kind": "instant" if market else "annual", "period_end": period,
                  "role": "comparable", "rationale": "Synthetic fixture: independent source explicitly discloses issuer, scope, period and CNY amount; not real model accuracy evidence."}]}
    if not market:
        args["rows"][0]["period_start"] = year + "-01-01"
    args["rows"][0].update(changes)
    submit(runtime, args)
    fact = runtime.session.facts[-1]
    review(runtime, [fact.fact_id])
    return fact


def peers(runtime, baseline=date(2025, 12, 31)):
    return ResearchValuationAssembler()._peers(runtime.session, baseline)


def test_source_backed_components_derive_pe_ps_with_units_and_replay_provenance(tmp_path):
    runtime = runtime_at(tmp_path)
    market = peer_component(runtime, "market_cap", "20", unit="亿元")
    income = peer_component(runtime, "net_income_parent", "10000")
    revenue = peer_component(runtime, "revenue", "50000")
    peer = peers(runtime)[0]
    assert (peer.pe, peer.ps, peer.market_cap) == (20, 4, 2000000000)
    assert peer.financial_period_end == date(2025, 12, 31)
    assert peer.calculation_methods == {"pe": "market_cap / net_income_parent", "ps": "market_cap / revenue"}
    assert {ref.evidence_id for ref in peer.evidence["pe"]} == {market.fact_id, income.fact_id}
    assert {ref.evidence_id for ref in peer.evidence["ps"]} == {market.fact_id, revenue.fact_id}
    assert all(ref.source_sha256 and ref.source_url and "deterministic_formula=" in ref.note for ref in peer.evidence["pe"])
    assert all(fact.standard_metric not in {"pe", "ps"} for fact in runtime.session.facts)


def test_explicit_recent_pricing_day_preserves_valuation_day_and_source_dates(tmp_path):
    from valuationagent.schemas.research import ResearchDraft

    runtime = runtime_at(tmp_path)
    peer_component(runtime, "market_cap", "200000")
    peer_component(runtime, "net_income_parent", "10000")
    runtime.session.draft.valuation_date = date(2026, 10, 2)
    with pytest.raises(ValueError, match="统一行情日"):
        peers(runtime)
    draft = runtime.session.draft.model_dump()
    draft.update(peer_pricing_date=date(2026, 9, 30),
                 peer_pricing_rationale="采用来源明确披露的9月30日统一行情；距估值日两天，期间价格变化风险保留。")
    runtime.session.draft = ResearchDraft.model_validate(draft)
    peer = peers(runtime)[0]
    assert peer.pe == Decimal(20) and peer.as_of_date == date(2026, 9, 30)
    assert runtime.session.draft.valuation_date == date(2026, 10, 2)
    assert "行情日2026-09-30" in peer.rationale and "价格变化风险" in peer.rationale
    assert runtime.session.facts[0].period == "2026-09-30"


@pytest.mark.parametrize("pricing,reason,cutoff", [
    ("2026-10-03", "不能用未来行情来估计过去价值。", None),
    ("2026-09-24", "日期已过期，不能静默使用该价格。", None),
    ("2026-09-30", "", None),
    ("2026-09-30", "统一日期但超出信息可得时点，应拒绝。", "2026-09-29"),
])
def test_explicit_pricing_date_must_be_bounded_and_explained(pricing, reason, cutoff):
    from pydantic import ValidationError
    from valuationagent.schemas.research import ResearchDraft

    with pytest.raises(ValidationError):
        ResearchDraft(valuation_date="2026-10-02", peer_pricing_date=pricing,
                      peer_pricing_rationale=reason, information_cutoff_date=cutoff)


@pytest.mark.parametrize("change", ["wrong_date", "future_disclosure", "currency", "parent", "ttm", "self", "conflict"])
def test_invalid_components_cannot_enter_peer_calculation(tmp_path, change):
    runtime = runtime_at(tmp_path)
    market = peer_component(runtime, "market_cap", "200000")
    income = peer_component(runtime, "net_income_parent", "10000")
    if change == "wrong_date":
        market.period = "2026-09-29"
    elif change == "future_disclosure":
        market.published_at = date(2026, 10, 1)
    elif change == "currency":
        market.verification["reading_proof"]["basis"]["currency"] = "USD"
    elif change == "parent":
        income.scope = "parent"
    elif change == "ttm":
        income.verification["reading_proof"]["row"]["period_kind"] = "ttm"
    elif change == "self":
        market.peer_ticker = runtime.session.draft.ticker
    else:
        runtime.session.facts.append(market.model_copy(update={"fact_id": "conflicting", "normalized_value": "2100000000"}))
    with pytest.raises(ValueError):
        peers(runtime)


def test_incomplete_or_loss_making_peers_do_not_inflate_sample_counts(tmp_path):
    runtime = runtime_at(tmp_path)
    peer_component(runtime, "market_cap", "200000")
    peer_component(runtime, "net_income_parent", "-10000")
    assert not peers(runtime)
    peer_component(runtime, "revenue", "50000")
    assert peers(runtime)[0].pe is None and peers(runtime)[0].ps == 4
    assert not peers(runtime, date(2024, 12, 31))


def test_duplicate_evidence_does_not_duplicate_samples_or_average_conflicts(tmp_path):
    runtime = runtime_at(tmp_path)
    peer_component(runtime, "market_cap", "200000")
    peer_component(runtime, "net_income_parent", "10000")
    duplicate = runtime.session.facts[-1].model_copy(deep=True, update={"fact_id": "second-source"})
    runtime.session.facts.append(duplicate)
    assert len(peers(runtime)) == 1 and peers(runtime)[0].pe == Decimal(20)
    assert len(peers(runtime)[0].evidence["pe"]) == 3


def test_unreviewed_peer_components_are_pending_dependencies(tmp_path):
    runtime = runtime_at(tmp_path)
    market = peer_component(runtime, "market_cap", "200000")
    market.status = "proposed"
    assert market in ResearchValuationAssembler.pending_blockers(runtime.session)
    assert not peers(runtime)


def test_raw_component_does_not_accept_quarter_as_annual(tmp_path):
    runtime = runtime_at(tmp_path)
    fact = peer_component(runtime, "net_income_parent", "10000", period_kind="interim", period_end="2025-06-30")
    assert fact.status == "proposed"
    assert any("MODEL_PEER_ANNUAL" in warning for warning in fact.warnings)


def test_direct_multiple_requires_disclosed_matching_denominator_year(tmp_path):
    runtime = runtime_at(tmp_path)
    market = peer_component(runtime, "market_cap", "200000")
    direct = market.model_copy(deep=True, update={"fact_id": "direct", "metric": "pe", "standard_metric": "pe",
        "normalized_value": "20", "unit": "ratio", "multiple_basis": "FY", "denominator_period_end": date(2024, 12, 31)})
    runtime.session.facts = [direct]
    with pytest.raises(ValueError, match="基期"):
        peers(runtime)
    direct.denominator_period_end = None
    with pytest.raises(ValueError, match="FY分母"):
        peers(runtime)


def test_direct_and_derived_conflict_requires_review_not_silent_override(tmp_path):
    runtime = runtime_at(tmp_path)
    market = peer_component(runtime, "market_cap", "200000")
    peer_component(runtime, "net_income_parent", "10000")
    direct = market.model_copy(deep=True, update={"fact_id": "direct", "metric": "pe", "standard_metric": "pe",
        "normalized_value": "30", "unit": "ratio", "multiple_basis": "FY", "denominator_period_end": date(2025, 12, 31)})
    runtime.session.facts.append(direct)
    with pytest.raises(ValueError, match="推导不一致"):
        peers(runtime)
