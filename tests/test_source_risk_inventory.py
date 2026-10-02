"""Synthetic omission guards; source-row detection never supplies a valuation amount."""
from datetime import date

import pytest

from valuationagent.application.source_risks import source_risk_inventory
from valuationagent.schemas.research import DocumentSummary, FactCandidate, ResearchSession


BASELINE = date(2024, 12, 31)
HASH = "a" * 64


def fixture(*, text="少数股东权益   1,000.00  900.00", name="合成公司2024年年度报告.pdf",
            role="historical_financials", location=None, block_count=1):
    session = ResearchSession(session_id="source_risk_test", documents=[DocumentSummary(
        file_id="file_a", name=name, role=role, block_count=block_count, sha256=HASH,
    )])
    block = {"file_id": "file_a", "block_id": "file_a:1", "text": text,
             "location": {"page": 6, **(location or {})}}
    return session, [block]


def reviewed(*, metric="少数股东权益", value="1000", period="2024年度", quote=None,
             block_id="file_a:1", scope="consolidated", status="confirmed", **changes):
    row = quote or "少数股东权益   1,000.00  900.00"
    fields = dict(
        fact_id="fact_risk", metric=metric, raw_value=value, normalized_value=value,
        unit="元", period=period, scope=scope, status=status, block_id=block_id,
        quote=row, source_sha256=HASH,
        verification={"scope": "consolidated", "period": "2024", "year_column": 2024,
                      "unit": "元", "source_row": row},
    )
    fields.update(changes)
    return FactCandidate(**fields)


@pytest.mark.parametrize("line,metric", [
    ("少数股东权益 1,000.00 900.00", "minority_interest"),
    ("受限货币资金 100 90", "restricted_cash"),
    ("存放中央银行法定存款准备金 六、30 100.00 90.00", "restricted_cash"),
    ("吸收存款及同业存放 0.00 900.00", "financial_institution_deposits"),
    ("拆出资金 10,000 9,000", "interbank_lending"),
    ("不能随时支取的同业存款 100.00", "restricted_interbank_deposits"),
    ("交易性金融资产 七、2 7,617,576,114.87 5,841,004,849.56", "trading_financial_assets"),
    ("其中：少 数 股 东 权 益 100.00", "minority_interest"),
    ("（一）少数股东权益（元） -100.00 90.00", "minority_interest"),
])
def test_numeric_risk_rows_are_inventoried_with_provenance_not_calculated(line, metric):
    session, blocks = fixture(text=line)
    result = source_risk_inventory(session, blocks, BASELINE)
    assert len(result["unresolved"]) == 1
    risk = result["unresolved"][0]
    assert risk["metric"] == metric and risk["quote"] == line
    assert risk["page"] == 6 and risk["source_sha256"] == HASH
    assert risk["block_id"] == "file_a:1" and not risk["resolved"]
    assert "value" not in risk and "normalized_value" not in risk
    assert "不能视作零" in risk["message"]


@pytest.mark.parametrize("line", [
    "少数股东权益较上年增加100万元。", "报告第3节讨论少数股东权益100元",
    "少数股东权益占比 10%", "少数股东权益 10%", "少数股东权益及相关说明（2024年）",
    "少数股东权益", "少数股东权益 - -", "少数股东权益 未来2025年预计100万元",
])
def test_prose_headings_percentages_and_blank_rows_do_not_become_amounts(line):
    session, blocks = fixture(text=line)
    assert not source_risk_inventory(session, blocks, BASELINE)["matches"]


@pytest.mark.parametrize("role", ["comparables", "comparable", "peer", "peers", "assumptions"])
def test_non_target_document_roles_are_excluded(role):
    session, blocks = fixture(role=role)
    result = source_risk_inventory(session, blocks, BASELINE)
    assert result["scanned_block_count"] == 0 and not result["matches"]


def test_search_snippet_is_not_scanned_as_a_source_document():
    session, blocks = fixture(location={"source_type": "web_search"})
    assert not source_risk_inventory(session, blocks, BASELINE)["matches"]


@pytest.mark.parametrize("name,location", [
    ("合成公司2023年年度报告.pdf", {}), ("annual-2023.pdf", {}),
    ("report.pdf", {"report_year": 2023}), ("report.pdf", {"period_end": "2023-12-31"}),
])
def test_explicit_other_fiscal_year_is_not_a_current_baseline_omission(name, location):
    session, blocks = fixture(name=name, location=location)
    assert not source_risk_inventory(session, blocks, BASELINE)["matches"]


def test_publication_year_is_not_confused_with_fiscal_year():
    session, blocks = fixture(name="2025-03-28发布的2024年年度报告.pdf", location={"published_at": "2025-03-28"})
    assert source_risk_inventory(session, blocks, BASELINE)["unresolved"]


def test_unknown_source_year_retains_cautious_review_message():
    session, blocks = fixture(name="annual-report.pdf")
    risk = source_risk_inventory(session, blocks, BASELINE)["unresolved"][0]
    assert risk["period_basis"] == "unknown_loaded_source" and "年度未能" in risk["message"]


def test_filename_containing_only_publication_date_is_not_a_fiscal_year_claim():
    session, blocks = fixture(name="download-2025-03-28.pdf")
    assert source_risk_inventory(session, blocks, BASELINE)["unresolved"]


def test_confirmed_verified_same_source_nonzero_is_handed_to_financial_gate():
    session, blocks = fixture()
    session.facts = [reviewed()]
    result = source_risk_inventory(session, blocks, BASELINE)
    assert not result["unresolved"] and result["matches"][0]["fact_ids"] == ["fact_risk"]
    assert "非零风险仍受估值桥接门禁" in result["matches"][0]["message"]


def test_identical_official_file_copy_reuses_verified_fact_by_content_hash():
    session, _ = fixture()
    session.documents.append(DocumentSummary(
        file_id="file_b", name="official-copy.pdf", role="evidence",
        block_count=1, sha256=HASH,
    ))
    copied = [{
        "file_id": "file_b", "block_id": "file_b:1",
        "text": "少数股东权益   1,000.00  900.00", "location": {"page": 6},
    }]
    session.facts = [reviewed()]
    result = source_risk_inventory(session, copied, BASELINE)
    assert not result["unresolved"]
    assert result["matches"][0]["fact_ids"] == ["fact_risk"]


@pytest.mark.parametrize("period", ["2024-12-31", "2024年12月31日", "2024年1-12月", "2024-01-01至2024-12-31"])
def test_equivalent_unambiguous_annual_periods_do_not_create_review_deadlocks(period):
    session, blocks = fixture()
    session.facts = [reviewed(period=period)]
    assert not source_risk_inventory(session, blocks, BASELINE)["unresolved"]


@pytest.mark.parametrize("changes", [
    {"status": "proposed"}, {"status": "rejected"}, {"warnings": ["年份列不明确"]},
    {"scope": "parent"}, {"scope": "issuer"}, {"period": "2023年度"},
    {"period": "2024上半年"}, {"period": "2024-06-30"}, {"block_id": "file_b:1"},
    {"role": "comparable"}, {"metric": "货币资金"}, {"source_sha256": "b" * 64},
    {"source_type": "user_note"},
    {"normalized_value": None}, {"normalized_value": "NaN"}, {"verification": {}},
])
def test_unreviewed_wrong_scope_period_source_or_metric_cannot_clear_omission(changes):
    session, blocks = fixture()
    session.facts = [reviewed(**changes)]
    assert source_risk_inventory(session, blocks, BASELINE)["unresolved"]


def test_explicit_verified_zero_only_clears_its_own_exact_source_row():
    row = "少数股东权益 0.00 900.00"
    session, blocks = fixture(text=row)
    session.facts = [reviewed(value="0", quote=row)]
    assert not source_risk_inventory(session, blocks, BASELINE)["unresolved"]
    blocks.append({"block_id": "file_a:2", "text": "少数股东权益 500.00 900.00", "location": {"page": 7}})
    unresolved = source_risk_inventory(session, blocks, BASELINE)["unresolved"]
    assert len(unresolved) == 1 and unresolved[0]["block_id"] == "file_a:2"


def test_zero_from_another_metric_or_source_cannot_clear_risk():
    session, blocks = fixture()
    session.facts = [reviewed(value="0", metric="受限货币资金"), reviewed(value="0", block_id="file_b:1")]
    assert source_risk_inventory(session, blocks, BASELINE)["unresolved"]


def test_partial_loaded_blocks_never_claim_full_file_coverage():
    session, blocks = fixture(text="营业收入 1000.00", block_count=12)
    result = source_risk_inventory(session, blocks, BASELINE)
    assert not result["matches"] and result["scanned_block_count"] == 1
    assert result["scan_scope"] == "loaded_source_blocks_only"
    assert any("不代表全文件" in note for note in result["limitations"])
    assert "未加载" in result["limitations"][0]


def test_duplicate_rows_do_not_duplicate_risk_prompts():
    session, blocks = fixture()
    result = source_risk_inventory(session, blocks * 2, BASELINE)
    assert len(result["matches"]) == 1 and result["scanned_block_count"] == 1
