import copy
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from valuationagent.application.observation_extraction import (
    ExtractObservations, ObservationSelection, ReviewObservations, REVIEW_PENDING,
    extract_observations, prepare_reviews, review_observations,
)
from valuationagent.application.extraction_recovery import record_attempt, recovery_plan
from valuationagent.application.research_valuation import _period
from valuationagent.application.ledgers import sync_evidence_and_fact_ledgers
from valuationagent.schemas.workspace import EvidenceRecord, ValuationWorkspace
from valuationagent.schemas.research import FactCandidate
from test_multisource_extraction import attach, runtime_at


def example(runtime, *, public=False, name="report.txt", amount="1,200", year="2025", domain="one.example.test"):
    block = attach(runtime, name, f"Sample issuer 600123\nRevenue {amount}\nYear ended {year}\nConsolidated group\nAmounts are expressed in thousands of RMB", public=public, domain=domain)[0]
    def span(line, quote, **extra):
        return {"block_id": block["block_id"], "start_line": line, "quote": quote, **extra}
    return {"file_id": block["file_id"], "anchors": {
        "issuer": span(1, "Sample issuer 600123"), "value": span(2, amount), "label": span(2, "Revenue"),
        "period": span(3, "Year ended " + year), "scope": span(4, "Consolidated group"),
        "unit": span(5, "Amounts are expressed in thousands of RMB")},
        "basis": {"entity_name": runtime.session.draft.company, "entity_ticker": "600123", "entity_refs": ["issuer"],
                  "scope": "consolidated", "scope_refs": ["scope"], "unit": "千元", "currency": "CNY", "unit_refs": ["unit"]},
        "rows": [{"metric": "Revenue", "standard_metric": "revenue", "raw_value": amount, "value_ref": "value",
                  "label_refs": ["label"], "period_kind": "annual", "period_start": year + "-01-01", "period_end": year + "-12-31",
                  "period_refs": ["period"], "semantic_role": "operating",
                  "rationale": "结合集团范围及后文计量说明，将完整年度营业收入映射为revenue；不把上年比较列当成本年。"}]}


def submit(runtime, args):
    return extract_observations(runtime, ExtractObservations.model_validate(args))


def review(runtime, fact_ids=None, **changes):
    fact_ids = fact_ids or [fact.fact_id for fact in runtime.session.facts if fact.status != "rejected"]
    packets = prepare_reviews(runtime, ObservationSelection(fact_ids=fact_ids))["packets"]
    return review_observations(runtime, ReviewObservations(reviews=[{
        "fact_id": packet["fact_id"], "packet_id": packet["packet_id"],
        "checks": {key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")},
        **({"source_period_end": packet["interpretation"]["row"]["period_end"]} if packet.get("requires_period_readback") else {}),
        "rationale": "已重新对照原始上下文，核对当前主体、完整年度、数值边界、计量单位和报表口径，映射与原始科目一致。",
        **changes,
    } for packet in packets]))


def test_non_template_english_notes_after_rows_are_interpreted_then_reviewed(tmp_path):
    runtime = runtime_at(tmp_path)
    result = submit(runtime, example(runtime))
    assert result["saved_count"] == 1
    fact = runtime.session.facts[0]
    assert fact.normalized_value == "1200000"
    assert fact.status == "proposed" and REVIEW_PENDING in fact.warnings
    assert fact.mapping_confidence == 0
    review(runtime)
    assert fact.status == "confirmed" and not fact.warnings
    assert fact.verification["semantic_review"]["independent_audit"] is False


def test_peer_identity_error_guides_role_repair_without_rewriting_entity(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    runtime.session.draft.ticker = "600999"
    failed = submit(runtime, args)
    assert failed["saved_count"] == 0
    assert "ENTITY_TARGET_MISMATCH" in failed["rows"][0]["error"]
    assert "role=comparable" in failed["rows"][0]["repair"]
    assert "不得改写主体" in failed["rows"][0]["repair"]
    assert "value_ref须填" not in failed["rows"][0]["repair"]
    assert runtime.session.draft.ticker == "600999" and not runtime.session.facts
    args["rows"][0]["role"] = "comparable"
    assert submit(runtime, args)["saved_count"] == 1
    assert runtime.session.facts[0].peer_ticker == "600123"
    assert runtime.session.facts[0].role == "comparable"


def test_model_type_failure_directs_correction_not_review_or_reader_retry(tmp_path):
    from observation_fixtures import fixture_observations

    runtime = runtime_at(tmp_path)
    args = fixture_observations(runtime, [{"metric": "common_shares", "raw_value": "1000", "period": "2025-12-31", "unit": "股"}])
    args["rows"][0].update(period_kind="annual", period_start="2025-01-01")
    result = submit(runtime, args)
    fact = runtime.session.facts[0]
    assert result["rows"][0]["next_action"]["tool"] == "extract_observations"
    assert result["rows"][0]["next_action"]["replaces"] == [fact.fact_id]
    assert "period_kind=instant" in str(result["rows"][0]["warnings"])
    record_attempt(runtime.session, "extract_observations", args, result)
    assert runtime.session.reading_attempts[-1]["status"] == "failed"
    choices = recovery_plan(runtime.session)["files"][0]["next_choices"]
    assert choices[0]["tool"] == "extract_observations"
    assert not any(choice["tool"] == "read_file" for choice in choices)
    assert review(runtime)["reviews"][0]["next_action"]["tool"] == "extract_observations"
    assert fact.status == "proposed"
    args["rows"][0].update(period_kind="instant", period_start=None, replaces=[fact.fact_id])
    submit(runtime, args)
    review(runtime, [runtime.session.facts[-1].fact_id])
    assert runtime.session.facts[-1].status == "confirmed"
    assert fact.status == "rejected"


def test_reviewed_dated_shares_enter_timing_and_latest_denominator(tmp_path):
    from observation_fixtures import fixture_observations
    from valuationagent.application.research_valuation import ResearchValuationAssembler

    runtime = runtime_at(tmp_path)
    args = fixture_observations(runtime, [
        {"metric": "revenue", "raw_value": "100000", "period": "2025"},
        {"metric": "net_income_parent", "raw_value": "10000", "period": "2025"},
        {"metric": "common_shares", "raw_value": "1000", "unit": "股", "period": "2025"},
    ])
    submit(runtime, args)
    review(runtime)
    assembler = ResearchValuationAssembler()
    annual = runtime.session.facts[-1]
    assert assembler._verified_dated_issuer_shares(annual)
    assert assembler.capital_structure_timing_issue(runtime.session)["code"] == "STALE_POINT_IN_TIME_SHARES"
    later = fixture_observations(runtime, [{"metric": "common_shares", "raw_value": "1000", "unit": "股", "period": "2026"}])
    block = runtime.service.store.research_blocks(runtime.session.session_id, later["file_id"])[0]
    block["text"] = block["text"].replace("2026-12-31", "2026-06-30")
    runtime.service.store.save_research_blocks(runtime.session.session_id, later["file_id"], [block])
    later["anchors"]["row0_period"]["quote"] = "As of 2026-06-30"
    later["rows"][0]["period_end"] = "2026-06-30"
    submit(runtime, later)
    assert not assembler._verified_dated_issuer_shares(runtime.session.facts[-1])
    review(runtime)
    assert assembler.capital_structure_timing_issue(runtime.session) is None
    financials = assembler._structured_financials(runtime.session)[-1]
    assert financials.common_shares_as_of == date(2026, 6, 30)
    changed = runtime.session.facts[-1].model_copy(deep=True)
    changed.period = "2026-09-30"
    assert not assembler._verified_dated_issuer_shares(changed)


def test_comparable_raw_chinese_label_uses_reviewed_standard_mapping(tmp_path):
    from valuationagent.application.research_valuation import ResearchValuationAssembler

    runtime = runtime_at(tmp_path)
    block = attach(runtime, "peer.txt", "样本同行 600456\n发行人市盈率 20\n2026-09-30 FY2025\n单位：倍", published="2026-09-30")[0]
    anchors = {key: {"block_id": block["block_id"], "start_line": number}
               for key, number in [("issuer", 1), ("value", 2), ("date", 3), ("unit", 4)]}
    args = {"file_id": block["file_id"], "anchors": anchors,
            "basis": {"entity_name": "样本同行", "entity_ticker": "600456", "entity_refs": ["issuer"],
                      "scope": "issuer", "scope_refs": ["value"], "unit": "ratio", "unit_refs": ["unit"]},
            "rows": [{"metric": "市盈率", "standard_metric": "pe", "raw_value": "20", "value_ref": "value",
                      "label_refs": ["value"], "period_kind": "instant", "period_end": "2026-09-30", "period_refs": ["date"],
                      "role": "comparable", "multiple_basis": "FY", "denominator_period_end": "2025-12-31", "denominator_refs": ["date"],
                      "rationale": "合成测试：同定价日及完整年度分母已在原文明确，保留中文原始标签，标准映射为PE。"}]}
    submit(runtime, args)
    review(runtime)
    peers = ResearchValuationAssembler()._peers(runtime.session)
    assert len(peers) == 1 and peers[0].pe == 20
    assert peers[0].evidence["pe"]
    assert "不等于人工批准" in peers[0].rationale


@pytest.mark.parametrize("year", ["2019", "2021", "2024", "2025"])
@pytest.mark.parametrize("company", ["样本芯片公司", "样本食品公司", "样本汽车公司"])
def test_same_interpreter_accepts_multiple_companies_and_historical_years(tmp_path, year, company):
    runtime = runtime_at(tmp_path, company=company)
    submit(runtime, example(runtime, year=year))
    review(runtime)
    assert runtime.session.facts[0].period == year
    assert runtime.session.facts[0].status == "confirmed"


def test_repeated_year_and_restated_column_have_explicit_nonambiguous_locations(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    blocks = runtime.service.store.research_blocks(runtime.session.session_id, args["file_id"])
    blocks[0]["text"] = blocks[0]["text"].replace("Year ended 2025", "2025 as reported | 2025 restated")
    runtime.service.store.save_research_blocks(runtime.session.session_id, args["file_id"], blocks)
    args["anchors"]["period"].update(quote="2025", occurrence=1)
    args["anchors"]["restatement"] = {"block_id": blocks[0]["block_id"], "start_line": 3, "quote": "restated"}
    args["rows"][0].update(revision="restated", revision_refs=["restatement"])
    assert submit(runtime, args)["saved_count"] == 1
    review(runtime)
    assert runtime.session.facts[0].status == "confirmed"


def test_row_basis_supports_mixed_parent_and_consolidated_columns(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    blocks = runtime.service.store.research_blocks(runtime.session.session_id, args["file_id"])
    blocks[0]["text"] += "\nParent entity Revenue 900\nParent entity accounts in RMB units"
    runtime.service.store.save_research_blocks(runtime.session.session_id, args["file_id"], blocks)
    for key, line, quote in [("pvalue", 6, "900"), ("plabel", 6, "Revenue"), ("pscope", 7, "Parent entity"), ("punit", 7, "RMB units")]:
        args["anchors"][key] = {"block_id": blocks[0]["block_id"], "start_line": line, "quote": quote}
    parent = copy.deepcopy(args["rows"][0])
    parent.update(raw_value="900", value_ref="pvalue", label_refs=["plabel"],
                  basis={**args["basis"], "scope": "parent", "scope_refs": ["pscope"], "unit": "元", "unit_refs": ["punit"]})
    args["rows"].append(parent)
    assert submit(runtime, args)["saved_count"] == 2
    review(runtime)
    assert {fact.normalized_value for fact in runtime.session.facts} == {"900", "1200000"}
    assert all(fact.status == "confirmed" for fact in runtime.session.facts)


@pytest.mark.parametrize("failure", ["quote", "value", "cross_file", "empty", "ambiguous_occurrence"])
def test_mechanical_errors_never_become_reviewable_observations(tmp_path, failure):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    if failure == "quote":
        args["anchors"]["label"]["quote"] = "fabricated label"
    elif failure == "value":
        args["rows"][0]["raw_value"] = "12000"
    elif failure == "cross_file":
        other = attach(runtime, "another.txt", "1,200")[0]
        args["anchors"]["value"]["block_id"] = other["block_id"]
    elif failure == "empty":
        args["anchors"]["value"]["quote"] = "Revenue"
        args["rows"][0]["raw_value"] = "0"
    else:
        args["anchors"]["period"]["occurrence"] = 10
    assert submit(runtime, args)["ok"] is False
    assert not runtime.session.facts


def test_partial_batch_keeps_success_and_reports_only_failed_rows(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    wrong = copy.deepcopy(args["rows"][0])
    wrong["raw_value"] = "9999"
    args["rows"].append(wrong)
    result = submit(runtime, args)
    assert result["saved_count"] == 1 and "error" in result["rows"][1]
    assert submit(runtime, args)["rows"][0]["duplicate"]
    assert len(runtime.session.facts) == 1


@pytest.mark.parametrize("raw_value,expected", [("73,968,640,704.54", "73968640704.54"), ("66209053612.11", "66209053612.11")])
def test_full_financial_row_locates_llm_selected_column_without_requiring_number_only_quote(tmp_path, raw_value, expected):
    runtime = runtime_at(tmp_path)
    text = "73,968,640,704.54    66,209,053,612.11    11.72%    57,321,059,453.15"
    args = example(runtime, amount=text)
    args["anchors"]["value"]["quote"] = "Revenue " + text
    args["rows"][0]["raw_value"] = raw_value
    args["basis"]["unit"] = "元"
    assert submit(runtime, args)["saved_count"] == 1
    fact = runtime.session.facts[0]
    assert fact.normalized_value == expected
    proof = fact.verification["reading_proof"]
    assert proof["anchors"]["value"]["quote"].startswith("Revenue ")
    assert proof["resolved_value"]["quote"] == fact.quote
    assert proof["resolved_value"]["char_offset"] > 0
    packets = prepare_reviews(runtime, ObservationSelection(fact_ids=[fact.fact_id]))["packets"]
    assert packets[0]["resolved_value"] == proof["resolved_value"]
    assert fact.status == "proposed"
    review(runtime)
    assert fact.status == "confirmed"


def test_duplicate_equal_values_need_explicit_column_and_preserve_occurrence(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime, amount="1,200 | 1,200")
    args["rows"][0]["raw_value"] = "1200"
    result = submit(runtime, args)
    assert "AMOUNT_AMBIGUOUS" in result["rows"][0]["error"]
    assert result["rows"][0]["source_quote"] == "1,200 | 1,200"
    args["rows"][0]["value_occurrence"] = 1
    assert submit(runtime, args)["saved_count"] == 1
    proof = runtime.session.facts[0].verification["reading_proof"]
    assert proof["resolved_value"]["char_offset"] == len("Revenue 1,200 | ")
    assert proof["resolved_value"]["occurrence"] == 1
    args["rows"][0]["value_occurrence"] = 2
    assert "AMOUNT_OCCURRENCE" in submit(runtime, args)["rows"][0]["error"]


def test_line_references_need_not_retype_pdf_whitespace(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    for anchor in args["anchors"].values():
        anchor.pop("quote")
    assert submit(runtime, args)["saved_count"] == 1
    fact = runtime.session.facts[0]
    assert fact.verification["reading_proof"]["anchors"]["value"]["quote"] == "Revenue 1,200"
    review(runtime)
    assert fact.status == "confirmed"


def test_wrong_line_returns_actual_source_and_locations_not_automatic_correction(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    args["anchors"]["value"].update(start_line=3)
    args["anchors"]["value"].pop("quote")
    result = submit(runtime, args)
    assert not result["ok"]
    assert result["rows"][0]["source_quote"] == "Year ended 2025"
    assert result["rows"][0]["value_locations"][0]["start_line"] == 2
    assert not runtime.session.facts


def test_missing_reference_names_return_concrete_schema_repair(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    args["rows"][0]["value_ref"] = "1,200"
    result = submit(runtime, args)
    assert "ANCHOR_MISSING" in result["rows"][0]["error"]
    assert "有效ID" in result["rows"][0]["error"] and "value" in result["rows"][0]["error"]
    assert not runtime.session.facts


def test_plain_reported_metric_does_not_require_an_operating_financing_split(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    args["rows"][0]["semantic_role"] = "unknown"
    assert submit(runtime, args)["saved_count"] == 1
    review(runtime)
    assert runtime.session.facts[0].status == "confirmed"


def test_cash_bridge_still_requires_explicit_economic_treatment(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    args["rows"][0].update(standard_metric="cash_and_non_operating_assets", period_kind="instant", period_start=None, semantic_role="unknown")
    submit(runtime, args)
    review(runtime)
    assert runtime.session.facts[0].status == "proposed"
    assert any("MODEL_CASH" in warning for warning in runtime.session.facts[0].warnings)


def test_review_batch_shares_original_context_without_dropping_evidence(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    second = copy.deepcopy(args["rows"][0])
    second.update(period_start="2024-01-01", period_end="2024-12-31")
    args["rows"].append(second)
    submit(runtime, args)
    result = prepare_reviews(runtime, ObservationSelection(fact_ids=[fact.fact_id for fact in runtime.session.facts]))
    assert len(result["packets"]) == 2 and len(result["original_context"]) == 1
    assert result["packets"][0]["context_ids"] == result["packets"][1]["context_ids"]
    assert all(reference in result["original_context"] for packet in result["packets"] for reference in packet["context_ids"])


@pytest.mark.parametrize("text,raw_value", [
    ("−1,200 | 900", "1200"), ("(1,200) | 900", "1200"), ("12% | 900", "12"),
    ("1e4 | 900", "1"), ("1e4 | 900", "4"), ("123456 | 900", "2345"),
    ("-- | 900", "0"), ("1,200.00900.00", "1200.00"), ("12 % | 900", "12"),
])
def test_full_row_does_not_discard_signs_percent_exponents_or_invent_zero(tmp_path, text, raw_value):
    runtime = runtime_at(tmp_path)
    args = example(runtime, amount=text)
    args["rows"][0]["raw_value"] = raw_value
    assert submit(runtime, args)["saved_count"] == 0
    assert not runtime.session.facts


@pytest.mark.parametrize("text,raw_value,unit,expected", [
    ("−1,200 | 900", "-1200", "元", "-1200"),
    ("（1,200） | 900", "-1200", "元", "-1200"),
    ("12％ | 900", "12", "%", "0.12"),
])
def test_full_row_preserves_negative_and_percent_literals(tmp_path, text, raw_value, unit, expected):
    runtime = runtime_at(tmp_path)
    args = example(runtime, amount=text)
    args["rows"][0]["raw_value"] = raw_value
    args["basis"]["unit"] = unit
    assert submit(runtime, args)["saved_count"] == 1
    assert runtime.session.facts[0].normalized_value == expected


def test_semantic_uncertainty_is_not_cleared_by_self_confidence(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    args["rows"][0]["uncertainties"] = ["当前原文未说明是否为集团口径"]
    submit(runtime, args)
    with pytest.raises(ValueError, match="REVIEW_UNRESOLVED"):
        review(runtime)
    assert runtime.session.facts[0].status == "proposed"
    args["rows"][0]["confidence"] = 1
    with pytest.raises(ValidationError):
        ExtractObservations.model_validate(args)


def test_review_uses_fresh_packet_and_cannot_invent_reading_or_clear_cutoff(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    fact = runtime.session.facts[0]
    with pytest.raises(ValueError, match="REVIEW_PACKET_REQUIRED"):
        review_observations(runtime, ReviewObservations(reviews=[{"fact_id": fact.fact_id, "packet_id": "invented",
            "checks": {key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")},
            "rationale": "即使声称已经阅读了完整原文，也不能跳过取得工具复核包。"}]))
    runtime.session.draft.valuation_date = date(2025, 1, 1)
    review(runtime)
    assert fact.status == "proposed"
    assert any("截止日" in issue for issue in fact.warnings)


def test_wrong_period_semantics_can_be_explicitly_rejected_without_losing_raw_evidence(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    checks = {key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")}
    checks["period"] = "contradicted"
    review(runtime, checks=checks)
    fact = runtime.session.facts[0]
    assert fact.status == "proposed"
    assert fact.verification["observation"]["status"] == "verified"
    assert fact.verification["semantic_review"]["status"] == "needs_evidence"


@pytest.mark.parametrize("kind", ["interim", "ttm"])
def test_nonannual_flows_never_enter_annual_history_even_when_ending_december(tmp_path, kind):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    args["rows"][0].update(period_kind=kind, period_start="2025-10-01")
    submit(runtime, args)
    review(runtime)
    assert _period(runtime.session.facts[0].period) is None


def test_review_cannot_clear_public_source_or_dimension_warnings(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime, public=True)
    args["basis"]["unit"] = "万股"
    submit(runtime, args)
    review(runtime)
    assert any("MODEL_DIMENSION" in issue for issue in runtime.session.facts[0].warnings)
    assert any("PUBLIC_SOURCE" in issue for issue in runtime.session.facts[0].warnings)


def test_source_tampering_and_observation_tampering_invalidate_review(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    submit(runtime, args)
    fact = runtime.session.facts[0]
    fact.normalized_value = "99"
    with pytest.raises(ValueError, match="OBSERVATION_CHANGED"):
        prepare_reviews(runtime, ObservationSelection(fact_ids=[fact.fact_id]))
    fact.normalized_value = "1200000"
    metadata = runtime.service.store.get_file(args["file_id"])
    Path(metadata["storage_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        prepare_reviews(runtime, ObservationSelection(fact_ids=[fact.fact_id]))


def test_standard_metric_conflicts_are_detected_across_different_printed_labels(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    review(runtime)
    args = example(runtime, name="second.txt", amount="1,300")
    args["rows"][0]["metric"] = "营业收入"
    result = submit(runtime, args)
    review(runtime, [result["rows"][0]["fact_id"]])
    assert all(fact.status == "proposed" for fact in runtime.session.facts)
    assert all(any("数值冲突" in issue for issue in fact.warnings) for fact in runtime.session.facts)


def test_review_proof_and_semantic_status_are_audited_separately(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    workspace = ValuationWorkspace(workspace_id="workspace_proof", research_session_id=runtime.session.session_id)
    runtime.service.store.create_workspace(workspace)
    sync_evidence_and_fact_ledgers(runtime.service.store, workspace, runtime.session)
    records = runtime.service.store.list_workspace_records(workspace.workspace_id, "evidence", EvidenceRecord)
    assert records[-1].status == "candidate"
    review(runtime)
    sync_evidence_and_fact_ledgers(runtime.service.store, workspace, runtime.session)
    records = runtime.service.store.list_workspace_records(workspace.workspace_id, "evidence", EvidenceRecord)
    assert any(record.binding_proof.get("semantic_review", {}).get("status") == "supported" for record in records)


def test_recovery_is_source_scoped_persisted_and_does_not_expand_upload_permissions(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    runtime.session.data_source_preference = "upload"
    record_attempt(runtime.session, "extract_observations", args, {"ok": False, "error": {"message": "ANCHOR_TEXT"}})
    choices = recovery_plan(runtime.session)["files"][0]["next_choices"]
    assert any(choice["arguments"]["view"] == "raw_text" for choice in choices if "arguments" in choice)
    assert not any(choice["tool"] == "search_sources" for choice in choices)
    runtime.service.store.save_research(runtime.session)
    assert runtime.service.store.get_research(runtime.session.session_id).reading_attempts


def test_main_loop_uses_new_protocol_and_never_invokes_retired_binders(tmp_path, monkeypatch):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    def retired(*args, **kwargs):
        pytest.fail("retired extraction path must not run")
    monkeypatch.setattr(runtime.service, "_validate_candidates", retired)
    monkeypatch.setattr("valuationagent.core.table_interpretation.compile_table", retired)
    class Model:
        def __init__(self):
            self.calls = 0
        def chat(self, messages, **kwargs):
            tools = {tool["function"]["name"] for tool in kwargs["tools"]}
            assert not tools.intersection({"propose_facts", "repair_facts", "interpret_financial_table", "propose_financial_facts"})
            self.calls += 1
            if self.calls == 1:
                name, arguments = "extract_observations", args
            elif self.calls == 2:
                name, arguments = "prepare_observation_review", {"fact_ids": [runtime.session.facts[0].fact_id]}
            elif self.calls == 3:
                packet = json.loads(messages[-1]["content"])["packets"][0]
                name, arguments = "review_observations", {"reviews": [{"fact_id": packet["fact_id"], "packet_id": packet["packet_id"],
                    "checks": {key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")},
                    "rationale": "已对照集团范围、年度列与后文单位说明，完整核对金额和当前研究主体，不把模型置信度作为验证。"}]}
            else:
                name, arguments = "finish_response", {"answer": "已完成原文定位及LLM复核，尚未运行估值。"}
            return {"tool_calls": [{"id": str(self.calls), "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}
    runtime.run(Model())
    assert runtime.session.facts[0].status == "confirmed"


@pytest.mark.parametrize("amount,quote", [("-1,200", "1,200"), ("−1,200", "1,200"), ("(1,200)", "1,200"),
    ("1,200%", "1,200"), ("12,000", "2,000"), ("12000.99", "12000"), ("12000.99", "99"), ("120009000", "12000")])
def test_numeric_anchor_cannot_trim_sign_scale_or_digits(tmp_path, amount, quote):
    runtime = runtime_at(tmp_path)
    args = example(runtime, amount=amount)
    args["anchors"]["value"]["quote"] = quote
    args["rows"][0]["raw_value"] = quote
    result = submit(runtime, args)
    assert not result["ok"] and "AMOUNT_BOUNDARY" in result["rows"][0]["error"]
    assert not runtime.session.facts


@pytest.mark.parametrize("currency", [None, "USD", "HKD"])
def test_money_without_supported_currency_remains_outside_model(tmp_path, currency):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    args["basis"]["currency"] = currency
    submit(runtime, args)
    review(runtime)
    assert runtime.session.facts[0].status == "proposed"
    assert any("MODEL_CURRENCY" in warning for warning in runtime.session.facts[0].warnings)


def test_failed_semantic_review_is_counted_and_offers_source_local_recovery(tmp_path):
    from valuationagent.application.evidence_status import evidence_counts
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    submit(runtime, args)
    result = review(runtime, checks={key: "ambiguous" for key in ("entity", "amount", "period", "unit", "scope", "mapping")})
    record_attempt(runtime.session, "review_observations", {"reviews": [{"fact_id": runtime.session.facts[0].fact_id}]}, result)
    assert evidence_counts(runtime.session.facts)["semantic_review_pending"] == 1
    assert recovery_plan(runtime.session)["files"][0]["file_id"] == args["file_id"]
    assert "SEMANTIC_REVIEW_NEEDS_EVIDENCE" in result["reviews"][0]["warnings"][0]


def test_reviewed_cross_source_labels_use_canonical_metric_without_promoting_grade(tmp_path):
    from valuationagent.application.financial_evidence import FactSelection, corroborate
    runtime = runtime_at(tmp_path)
    first = example(runtime, public=True)
    submit(runtime, first)
    second = example(runtime, public=True, name="second.txt", amount="1,200.0", domain="two.other.test")
    second["rows"][0]["metric"] = "营业收入"
    submit(runtime, second)
    review(runtime)
    selected = [fact.fact_id for fact in runtime.session.facts]
    result = corroborate(runtime, FactSelection(fact_ids=selected, reason="不同站点及原件，标准科目和换算金额相同；不声称上游独立。"))
    assert result["source_grade_upgraded"] is False
    assert all(fact.status == "confirmed" for fact in runtime.session.facts)
    assert all(fact.verification["source_assessment"]["source_tier"] == "C" for fact in runtime.session.facts)


def test_comparable_financial_components_can_be_corroborated_without_becoming_target_facts(tmp_path):
    from valuationagent.application.financial_evidence import FactSelection, corroborate

    runtime = runtime_at(tmp_path)
    for domain, name, amount in [("one.example.test", "one.txt", "1,200"), ("two.other.test", "two.txt", "1,200.0")]:
        args = example(runtime, public=True, name=name, amount=amount, domain=domain)
        args["rows"][0]["role"] = "comparable"
        submit(runtime, args)
    review(runtime)
    result = corroborate(runtime, FactSelection(fact_ids=[fact.fact_id for fact in runtime.session.facts],
        reason="两个不同原文快照的同行合并收入一致，不代表上游独立，也不提升来源等级。"))
    assert not result["source_grade_upgraded"]
    assert all(fact.status == "confirmed" and fact.role == "comparable" for fact in runtime.session.facts)


def test_peer_corroboration_key_distinguishes_denominator_years_and_issuers():
    from valuationagent.application.financial_evidence import _corroboration_identity

    fact = FactCandidate(metric="pe", standard_metric="pe", raw_value="20", normalized_value="20", unit="ratio",
        role="comparable", peer_ticker="600456", peer_name="Synthetic peer", multiple_basis="FY", denominator_period_end=date(2025, 12, 31),
        period="2026-09-30", block_id="synthetic:1", quote="20")
    assert _corroboration_identity(fact) != _corroboration_identity(fact.model_copy(update={"denominator_period_end": date(2024, 12, 31)}))
    assert _corroboration_identity(fact) != _corroboration_identity(fact.model_copy(update={"peer_ticker": "600457"}))


def test_semantic_treatment_mutation_cannot_reuse_review_packet(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    runtime.session.facts[0].semantic_role = "financial_subsidiary"
    with pytest.raises(ValueError, match="OBSERVATION_CHANGED"):
        review(runtime)


def segmented_example(runtime):
    args = example(runtime, amount="1,300,029,680.192,071,807,133.32")
    args["anchors"]["value"]["quote"] = "1,300,029,680.19"
    args["anchors"]["previous"] = {**args["anchors"]["value"], "quote": "2,071,807,133.32"}
    args["rows"][0].update(raw_value="1,300,029,680.19", value_segments=["value", "previous"])
    return args


def test_llm_segments_joined_columns_without_a_static_table_parser(tmp_path):
    runtime = runtime_at(tmp_path)
    args = segmented_example(runtime)
    assert submit(runtime, args)["saved_count"] == 1
    assert runtime.session.facts[0].status == "proposed"
    review(runtime)
    assert runtime.session.facts[0].status == "confirmed"
    proof = runtime.session.facts[0].verification["reading_proof"]
    assert proof["row"]["value_segments"] == ["value", "previous"]
    assert proof["anchors"]["previous"]["quote"] == "2,071,807,133.32"


@pytest.mark.parametrize("failure", ["gap", "overlap", "missing", "duplicate", "sign"])
def test_llm_segmentation_cannot_omit_overlap_or_discard_numeric_characters(tmp_path, failure):
    runtime = runtime_at(tmp_path)
    args = segmented_example(runtime)
    if failure == "gap":
        args["anchors"]["previous"]["quote"] = "071,807,133.32"
    elif failure == "overlap":
        args["anchors"]["value"]["quote"] = "1,300,029,680.192"
    elif failure == "missing":
        args["rows"][0]["value_segments"] = ["previous"]
    elif failure == "duplicate":
        args["rows"][0]["value_segments"] = ["value", "value", "previous"]
    else:
        args = example(runtime, amount="-100200")
        args["anchors"]["value"]["quote"] = "100"
        args["anchors"]["previous"] = {**args["anchors"]["value"], "quote": "200"}
        args["rows"][0].update(raw_value="100", value_segments=["value", "previous"])
    assert not submit(runtime, args)["ok"]
    assert not runtime.session.facts


def test_changed_source_grade_invalidates_saved_review(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    runtime.session.documents[0].authority_tier = "D"
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        review(runtime)


@pytest.mark.parametrize("mistake", ["period", "unit", "scope", "metric_label"])
def test_same_source_anchor_can_correct_a_previously_wrong_interpretation(tmp_path, mistake):
    runtime = runtime_at(tmp_path)
    correct = example(runtime)
    wrong = copy.deepcopy(correct)
    if mistake == "period":
        wrong["rows"][0].update(period_start="2024-01-01", period_end="2024-12-31")
    elif mistake == "unit":
        wrong["basis"]["unit"] = "元"
    elif mistake == "metric_label":
        wrong["rows"][0].update(metric="wrong original label", standard_metric="unknown_mapping")
    else:
        wrong["basis"]["scope"] = "parent"
    submit(runtime, wrong)
    previous = runtime.session.facts[0]
    previous.status, previous.warnings = "confirmed", []
    correct["rows"][0]["replaces"] = [previous.fact_id]
    result = submit(runtime, correct)
    assert result["saved_count"] == 1 and previous.status == "confirmed"
    review(runtime, [result["rows"][0]["fact_id"]])
    assert previous.status == "rejected"
    assert runtime.session.facts[-1].status == "confirmed"


def test_changed_mapping_cannot_replace_a_different_source_value(tmp_path):
    runtime = runtime_at(tmp_path)
    previous_args = example(runtime)
    previous_args["rows"][0].update(standard_metric="unknown_mapping")
    submit(runtime, previous_args)
    previous = runtime.session.facts[0]
    other_args = example(runtime, amount="2,000")
    other_args["rows"][0]["replaces"] = [previous.fact_id]
    result = submit(runtime, other_args)
    assert result["saved_count"] == 0
    assert "REPLACEMENT_SCOPE" in result["rows"][0]["error"]
    assert previous.status == "proposed"


def test_audit_does_not_invent_mapping_confidence_or_currency(tmp_path):
    from valuationagent.schemas.workspace import WorkspaceFact
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    args["basis"]["currency"] = "USD"
    submit(runtime, args)
    review(runtime)
    workspace = ValuationWorkspace(workspace_id="workspace_currency", research_session_id=runtime.session.session_id)
    runtime.service.store.create_workspace(workspace)
    assert workspace.currency == "CNY"
    sync_evidence_and_fact_ledgers(runtime.service.store, workspace, runtime.session)
    facts = runtime.service.store.list_workspace_records(workspace.workspace_id, "fact", WorkspaceFact)
    assert facts[-1].currency == "USD" and facts[-1].confidence is None
    evidence = runtime.service.valuation_assembler._evidence(runtime.session, runtime.session.facts[0])
    assert "mapping_confidence" not in evidence.note
    assert "currency=USD" in evidence.note and "independent_audit=False" in evidence.note
