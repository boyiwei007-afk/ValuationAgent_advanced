"""Offline regressions found while inspecting actual annual-report extraction.

Fixtures are minimal, anonymized statement layouts. They are not investment
inputs and tests do not fetch data or replace a blank accounting cell with zero.
"""
import hashlib
from datetime import date

import pytest

from valuationagent.application.research import CandidateInput
from valuationagent.core.documents import _line_chunks, parse_document
from valuationagent.core.evidence import bind_evidence, evidence_context
from valuationagent.schemas.research import ResearchDraft


def _bind(text, metric, value, *, unit="万元", period="2024", aliases=()):
    block = {"block_id": "statement:1", "file_id": "statement", "location": {"page": 59}, "text": text}
    candidate = CandidateInput(
        metric=metric, raw_value=value, unit=unit, period=period,
        scope="consolidated", block_id=block["block_id"], quote=text,
    )
    draft = ResearchDraft(company="测试股份", ticker="600001", valuation_date=date(2025, 6, 30))
    return bind_evidence(candidate, block, evidence_context(block, [block]), draft, aliases)


@pytest.mark.parametrize("blank_label,following", [
    ("短期借款", "长期借款 100 90"),
    ("应付债券", "租赁负债 100 90"),
    ("一年内到期的非流动负债", "其他流动负债 100 90"),
])
def test_blank_debt_row_cannot_borrow_neighbor_amount(blank_label, following):
    text = "测试股份600001\n合并资产负债表\n单位：万元\n项目 2024年 2023年\n" + blank_label + "\n" + following
    warnings, checks = _bind(text, blank_label, "100")
    assert any("不能借用相邻科目" in warning for warning in warnings)
    assert checks["source_row"] == blank_label
    assert "year_column" not in checks


def test_true_values_only_continuation_still_binds():
    text = "测试股份600001\n合并资产负债表\n单位：万元\n项目 2024年 2023年\n短期借款\n100 90"
    warnings, checks = _bind(text, "短期借款", "100")
    assert not warnings
    assert checks["year_column"] == 2024


def test_existing_wrong_amount_does_not_fall_through_to_neighbor():
    text = "测试股份600001\n合并资产负债表\n单位：万元\n项目 2024年 2023年\n短期借款 5 4\n长期借款 100 90"
    assert _bind(text, "短期借款", "100")[0]


def _note_table(*, missing_current=False):
    # Same PDF layout as a real 2024 annual report: note 10 is not an amount,
    # and the absent adjacent year must remain absent, never move left or zero.
    def row(label, note, current, prior):
        return f"{label:<20}{note:>4}{current:>24}{prior:>24}"
    rows = [
        "测试股份600001\n合并资产负债表\n单位：元\n项目 附注 2024年 2023年",
        row("应收账款", "5", "18,974,192.75", "60,373,410.41"),
        row("存货", "9", "54,343,285,157.47", "46,435,185,061.53"),
        row("一年内到期的非流动资产", "10", "" if missing_current else "1,210,959,803.42", "1,210,959,803.42" if missing_current else ""),
    ]
    return "\n".join(rows)


@pytest.mark.parametrize("missing_current,year,missing", [(False, "2024", 2023), (True, "2023", 2024)])
def test_note_plus_amount_does_not_shift_year_when_adjacent_cell_blank(missing_current, year, missing):
    text = _note_table(missing_current=missing_current)
    warnings, checks = _bind(text, "一年内到期的非流动资产", "1,210,959,803.42", unit="元", period=year)
    assert not warnings, warnings
    assert checks["year_column"] == int(year)
    assert checks["missing_year_columns"] == [missing]
    assert checks["note_column_excluded"]
    wrong, wrong_checks = _bind(text, "一年内到期的非流动资产", "1,210,959,803.42", unit="元", period=str(missing))
    assert any("年度列冲突" in warning for warning in wrong)
    assert "year_column" not in wrong_checks


def test_note_identifier_cannot_become_a_financial_amount():
    warnings, checks = _bind(_note_table(), "一年内到期的非流动资产", "10", unit="元")
    assert warnings
    assert "year_column" not in checks


def test_unstable_column_alignment_is_not_claimed_as_evidence():
    text = _note_table().replace("18,974,192.75", "   18,974,192.75")
    _, checks = _bind(text, "一年内到期的非流动资产", "1,210,959,803.42", unit="元")
    assert "column_alignment" not in checks


def test_chunk_boundaries_never_split_normal_statement_lines(tmp_path):
    # The previous fixed 2,000-character cut split a tax-asset row label in a
    # real issuer's PDF. Keep both its label and amounts in a single citation.
    padding = "说明" * 990 + "\n"
    row = "递延所得税资产 5,520,006,868.83 4,645,887,425.10"
    text = padding + row + "\n其他非流动资产 232,395,817.46 109,563,497.23"
    path = tmp_path / "annual-report-excerpt.txt"
    raw = text.encode("utf-8")
    path.write_bytes(raw)
    blocks, warnings = parse_document({"file_id": "statement", "storage_path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    assert not warnings
    assert any(row in block["text"] for block in blocks)
    assert "".join(block["text"] for block in blocks) == text
    assert max(map(lambda block: len(block["text"]), blocks)) <= 2000


@pytest.mark.parametrize("text", ["x" * 6001, "x\n" * 2001, "", "a\r\nb\r\nc", "x" * 2100 + "\nline"])
def test_line_chunks_remain_lossless_and_bounded(text):
    chunks = list(_line_chunks(text))
    assert "".join(chunks) == text
    assert all(0 < len(chunk) <= 2000 for chunk in chunks)


def _issuer_bind(quote, *, value="125,619.78", unit="万股", period="2024年12月31日",
                 identity="测试股份600001", block_text=None, metric="普通股股数",
                 aliases=("common_shares", "总股本", "普通股股数"), published="2025-04-03"):
    candidate = CandidateInput(
        metric=metric, raw_value=value, unit=unit, period=period,
        block_id="issuer:1", quote=quote, scope="issuer",
    )
    block = {"block_id": "issuer:1", "file_id": "issuer", "location": {"page": 74, "published_at": published},
             "text": quote if block_text is None else block_text}
    draft = ResearchDraft(company="测试股份", ticker="600001", valuation_date=date(2025, 6, 30))
    return bind_evidence(candidate, block, [block], draft, aliases, identity_text=identity)


@pytest.mark.parametrize("quote,period", [
    ("截至   2024年\n12月31日，公司总股本为   125,619.78万股。", "2024年12月31日"),
    ("截至2024年12月31日，本公司普通股股份总数为125,619.78万股。", "2024-12-31"),
    ("2024年末，测试股份的股份总数为125,619.78万股。", "2024"),
    ("截止至2024年度末，本公司普通股股数：125,619.78万股。", "2024年度"),
])
def test_explicit_issuer_shares_need_no_consolidated_statement_title(quote, period):
    warnings, checks = _issuer_bind(quote, period=period)
    assert not warnings, warnings
    assert checks["scope"] == "issuer"
    assert checks["binding"] == "issuer_common_shares"
    assert checks["period_end"] == "2024-12-31"
    assert checks["unit"] == "万股"
    assert checks["issuer"] == "600001"


def test_hkex_final_allotment_binds_listing_date_total_without_deriving_tranches():
    quote = (
        "佛山市海天调味食品股份有限公司\n"
        "开始买卖日 2025年6月19日\n"
        "发售股份及股本\n"
        "于上市时已发行的股份数目（于超额配股权获行使前） 5,839,632,244"
    )
    block = {
        "block_id": "hkex:3",
        "file_id": "hkex",
        "location": {
            "page": 3,
            "published_at": "2025-06-18",
            "url": "https://www.hkexnews.hk/listedco/listconews/sehk/2025/0618/final.pdf",
        },
        "text": quote,
    }
    candidate = CandidateInput(
        metric="于上市时已发行的股份数目（于超额配股权获行使前）",
        standard_metric="common_shares",
        raw_value="5,839,632,244",
        unit="股",
        period="2025-06-19",
        scope="issuer",
        block_id=block["block_id"],
        quote=quote,
    )
    draft = ResearchDraft(
        company="佛山市海天调味食品股份有限公司",
        ticker="603288.SH",
        valuation_date=date(2025, 6, 30),
    )

    warnings, checks = bind_evidence(
        candidate,
        block,
        [block],
        draft,
        ("common_shares", "普通股股数", "股份总数"),
        identity_text=quote,
    )

    assert not warnings
    assert checks["binding"] == "issuer_listing_issued_shares"
    assert checks["period_end"] == "2025-06-19"
    assert checks["publication_date"] == "2025-06-18"
    assert checks["official_exchange"] == "HKEX"
    assert checks["scope"] == "issuer"


@pytest.mark.parametrize("change", ["host", "period", "published"])
def test_hkex_listing_total_does_not_relax_host_date_or_cutoff(change):
    quote = (
        "佛山市海天调味食品股份有限公司\n"
        "开始买卖日 2025年6月19日\n"
        "发售股份及股本\n"
        "于上市时已发行的股份数目（于超额配股权获行使前） 5,839,632,244"
    )
    location = {
        "published_at": "2025-06-18",
        "url": "https://www1.hkexnews.hk/listedco/listconews/sehk/final.pdf",
    }
    period = "2025-06-19"
    if change == "host":
        location["url"] = "https://example.com/final.pdf"
    elif change == "period":
        period = "2025-06-18"
    else:
        location["published_at"] = "2025-07-01"
    block = {"block_id": "hkex:3", "file_id": "hkex", "location": location, "text": quote}
    candidate = CandidateInput(
        metric="于上市时已发行的股份数目（于超额配股权获行使前）",
        standard_metric="common_shares",
        raw_value="5,839,632,244",
        unit="股",
        period=period,
        scope="issuer",
        block_id=block["block_id"],
        quote=quote,
    )
    warnings, checks = bind_evidence(
        candidate,
        block,
        [block],
        ResearchDraft(
            company="佛山市海天调味食品股份有限公司",
            ticker="603288.SH",
            valuation_date=date(2025, 6, 30),
        ),
        ("common_shares",),
        identity_text=quote,
    )
    assert warnings
    assert "scope" not in checks


@pytest.mark.parametrize("quote,changes", [
    ("截至2023年12月31日，公司总股本为125,619.78万股。", {}),
    ("截至2024年6月30日，公司总股本为125,619.78万股。", {"period": "2024"}),
    ("截至2024年12月31日，公司总股本为125,619.78万元。", {}),
    ("截至2024年12月31日，公司总股本为125,619.78万股。", {"unit": "股"}),
    ("截至2024年12月31日，公司总股本为125,619.78万股。", {"value": "125619.79"}),
    ("截至2024年12月31日，公司总股本为125,619.78万股。", {"identity": "另一股份600002"}),
    ("截至2024年12月31日，另一股份总股本为125,619.78万股。", {}),
    ("截至2024年12月31日，子公司总股本为125,619.78万股。", {}),
    ("公司以截至2024年12月31日公司总股本为125,619.78万股作为分红基数。", {}),
    ("截至2024年12月31日，公司流通股股份总数为125,619.78万股。", {}),
    ("截至2024年12月31日，公司总股本为125,619.78万股，其中含优先股。", {}),
    ("截至2024年12月31日，公司总股本为125,619.78万股。", {"aliases": ("revenue",), "metric": "营业收入"}),
    ("截至2024年12月31日，公司总股本为125,619.78万股。", {"published": "2025-07-01"}),
    ("截至2024年12月31日，公司总股本为125,619.78万股。", {"published": "invalid"}),
    ("截至2026年12月31日，公司总股本为125,619.78万股。", {"period": "2026"}),
    ("截至2024年12月31日，公司总股本为125,619.78万股。", {"block_text": "截至2024年12月31日，公司总股本为999,999.99万股。"}),
    ("2024年度公司总股本为125,619.78万股。", {}),
    ("截至2024年2月31日，公司总股本为125,619.78万股。", {}),
])
def test_issuer_shares_do_not_relax_identity_date_unit_or_source_binding(quote, changes):
    warnings, checks = _issuer_bind(quote, **changes)
    assert warnings
    assert "scope" not in checks


def test_issuer_disclosure_must_be_single_unambiguous_continuous_statement():
    quote = "截至2024年12月31日，公司总股本为125,619.78万股。\n截至2023年12月31日，公司总股本为120,000万股。"
    warnings, _ = _issuer_bind(quote)
    assert any("单一明确" in warning for warning in warnings)


def test_issuer_path_does_not_relax_regular_statement_scope():
    quote = "测试股份600001\n截至2024年12月31日，公司总股本为125,619.78万股。"
    warnings, checks = _bind(quote, "总股本", "125,619.78", unit="万股", aliases=("common_shares", "总股本"))
    assert any("口径缺少" in warning for warning in warnings)
    assert checks.get("scope") != "issuer"
