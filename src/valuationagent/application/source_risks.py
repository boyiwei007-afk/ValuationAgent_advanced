"""Inventory risk-bearing rows in loaded source text, without extracting money.

This is a fail-closed omission guard, not a source-verification replacement or
an assertion that an entire filing has been audited.  The ordinary evidence
binder must still validate every fact (including an explicit zero).
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation

from valuationagent.finance.integrity import EQUITY_BRIDGE_REVIEW_LABELS


RISK_ALIASES = {
    "minority_interest": ("少数股东权益", "noncontrolling_interest"),
    "restricted_cash": (
        "受限货币资金", "使用受到限制的货币资金", "存放中央银行法定存款准备金",
        "法定存款准备金",
    ),
    "financial_institution_deposits": ("吸收存款及同业存放",),
    "interbank_lending": ("拆出资金",),
    "restricted_interbank_deposits": ("不能随时支取的同业存款", "受限拆出资金"),
}
_PEER_ROLES = {"comparables", "comparable", "peer", "peers", "assumptions"}
_NUMBER = re.compile(r"(?<![\d.])[-+−－]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?(?![\d.])")


def _get(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _compact(value):
    return re.sub(r"[\s_\-]+", "", str(value)).casefold()


def _metric(value):
    key = _compact(value)
    for canonical, aliases in RISK_ALIASES.items():
        if key in {_compact(canonical), *map(_compact, aliases)}:
            return canonical
    return None


def _file_id(block):
    return str(block.get("file_id") or str(block.get("block_id", "")).split(":", 1)[0])


def _period_end(value):
    raw = str(value or "").strip()
    if re.search(r"Q[1-4]|H1|第?[一二三四1234]季度|季报|半年度?|半年报", raw, re.I):
        return None
    annual_range = re.fullmatch(r"(20\d{2})年?\s*1\s*[-—至到~～]\s*12\s*月", raw)
    if annual_range:
        return date(int(annual_range[1]), 12, 31)
    date_range = re.fullmatch(
        r"(20\d{2})[-年/.]0?1[-月/.]0?1日?\s*(?:至|到|—|~|～)\s*"
        r"(20\d{2})[-年/.]12[-月/.]31日?", raw,
    )
    if date_range and date_range[1] == date_range[2]:
        return date(int(date_range[2]), 12, 31)
    exact = re.fullmatch(r"(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})日?", raw)
    annual = re.fullmatch(r"(20\d{2})(?:年(?:度|末)?|年度末)?", raw)
    try:
        return date(*map(int, exact.groups())) if exact else date(int(annual[1]), 12, 31) if annual else None
    except ValueError:
        return None


def _explicit_report_year(name, location):
    # Publication/download dates are deliberately not fiscal periods.
    for key in ("report_year", "fiscal_year", "period_end", "report_period"):
        value = location.get(key)
        if value is not None:
            match = re.search(r"(?<!\d)(20\d{2})(?!\d)", str(value))
            if match:
                return int(match[1]), f"location.{key}"
    annual = re.search(r"(?<!\d)(20\d{2})\s*年?\s*(?:年度?报告|年报|annual)", name, re.I)
    if not annual:
        annual = re.search(r"annual[-_ ]*(20\d{2})(?!\d)", name, re.I)
    if annual:
        return int(annual[1]), "document_name"
    # An otherwise unlabelled year may be the download/publication year.
    # Do not drop a 2024 filing simply because it was saved as 2025-03-28.pdf.
    return None, "unknown_loaded_source"


def _risk_row(line):
    # Permit common financial-row numbering, not a label embedded in prose.
    raw = re.sub(r"^\s*(?:(?:[（(]?[一二三四五六七八九十百\d]+[）)．.、])\s*)?(?:其中\s*[:：]\s*)?", "", line)
    for metric, aliases in RISK_ALIASES.items():
        for label in sorted((metric, *aliases), key=len, reverse=True):
            pattern = r"\s*".join(map(re.escape, label))
            match = re.match(pattern, raw, re.I)
            if not match:
                continue
            tail = raw[match.end():].strip()
            # Exact label + numeric cells.  Narrative comparisons, percentages,
            # future years in sentences and table headings do not qualify.
            tail = re.sub(r"^[（(](?:人民币)?(?:亿|百万|万|千)?元[）)]", "", tail).strip()
            if not tail or re.search(r"[^\d\s,，.．()（）+−－\-—、:：|一二三四五六七八九十百附注]", tail):
                continue
            if _NUMBER.search(tail):
                return metric
    return None


def _verified_fact(fact, metric, file_id, baseline, quote, block_id, source_sha256):
    if (_get(fact, "status") != "confirmed" or _get(fact, "warnings", [])
            or _get(fact, "role") != "historical" or _get(fact, "scope") != "consolidated"
            or _get(fact, "source_type", "document") != "document"
            or _metric(_get(fact, "metric")) != metric
            or _period_end(_get(fact, "period")) != baseline
            or str(_get(fact, "block_id", "")).split(":", 1)[0] != file_id):
        return False
    verification = _get(fact, "verification", {}) or {}
    if (verification.get("scope") != "consolidated"
            or str(verification.get("period")) != str(baseline.year)
            or not verification.get("source_row")
            or verification.get("unit") not in {"元", "千元", "万元", "百万元", "亿元"}):
        return False
    try:
        amount = Decimal(str(_get(fact, "normalized_value")))
    except InvalidOperation:
        return False
    if not amount.is_finite():
        return False
    fact_hash = _get(fact, "source_sha256", "")
    if source_sha256 and fact_hash and source_sha256 != fact_hash:
        return False
    # A reviewed non-zero already invokes the separate financial model gate.
    # Zero must bind this exact source row: a zero somewhere else in the same
    # filing cannot hide another non-zero/blank-year exposure.
    if amount == 0 and (_get(fact, "block_id") != block_id
                        or _compact(verification["source_row"]) != _compact(quote)):
        return False
    return True


def source_risk_inventory(session, blocks, baseline_period_end: date):
    """Return source rows requiring extraction/review before DCF or EV/EBITDA.

    ``blocks`` is the caller's loaded raw-text sequence, not search summaries.
    Returned rows preserve provenance but intentionally contain no parsed value.
    Empty ``unresolved`` is NOT a completeness certificate for unloaded pages.
    """
    documents = {_get(doc, "file_id"): doc for doc in _get(session, "documents", [])}
    facts = _get(session, "facts", [])
    scanned, loaded_counts, matches, seen, seen_blocks = 0, Counter(), [], set(), set()
    limitations = ["仅扫描传入的已加载原文块；未加载、截断、图片/OCR遗漏部分不在覆盖范围。"]
    for block in blocks:
        file_id, location = _file_id(block), block.get("location") or {}
        doc = documents.get(file_id)
        if (location.get("source_type") in {"web_search", "user_note"}
                or _get(doc, "role", "") in _PEER_ROLES
                or location.get("role") in _PEER_ROLES
                or block.get("role") in _PEER_ROLES):
            continue
        name = _get(doc, "name", "")
        year, period_basis = _explicit_report_year(name, location)
        if year is not None and year != baseline_period_end.year:
            continue
        block_identity = (file_id, block.get("block_id"))
        if block_identity in seen_blocks:
            continue
        seen_blocks.add(block_identity)
        scanned += 1
        loaded_counts[file_id] += 1
        sha256 = location.get("source_sha256") or _get(doc, "sha256", "")
        for line in str(block.get("text", "")).splitlines():
            metric = _risk_row(line)
            if metric is None:
                continue
            quote = line.strip()
            identity = (block.get("block_id"), metric, quote)
            if identity in seen:
                continue
            seen.add(identity)
            accepted = [fact for fact in facts if _verified_fact(
                fact, metric, file_id, baseline_period_end, quote,
                block.get("block_id"), sha256,
            )]
            matches.append({
                "metric": metric, "label": EQUITY_BRIDGE_REVIEW_LABELS[metric],
                "file_id": file_id, "file_name": name, "block_id": block.get("block_id"),
                "page": location.get("page"), "location": dict(location), "quote": quote,
                "source_sha256": sha256, "source_url": location.get("source_url") or location.get("url", ""),
                "period_basis": period_basis, "resolved": bool(accepted),
                "fact_ids": [_get(fact, "fact_id") for fact in accepted],
                "message": ("已绑定当前基期已确认财务事实；非零风险仍受估值桥接门禁约束。" if accepted
                            else "原文存在风险科目数字行，尚未绑定当前基期、合并口径且通过来源核验的已确认事实；须提取复核，不能视作零。")
                    + (" 来源所属年度未能从文件名/位置确定，保守保留此提示。" if year is None else ""),
            })
    for file_id, count in loaded_counts.items():
        doc = documents.get(file_id)
        if doc and (count < _get(doc, "block_count", count)
                    or _get(doc, "parse_status", "parsed") != "parsed"
                    or _get(doc, "warnings", [])):
            limitations.append(f"{_get(doc, 'name', file_id)}：原文加载/解析存在覆盖限制，不代表全文件排查完成。")
    return {
        "version": "source-risk-inventory-v1", "period_end": baseline_period_end.isoformat(),
        "scanned_block_count": scanned, "scan_scope": "loaded_source_blocks_only",
        "matches": matches, "unresolved": [row for row in matches if not row["resolved"]],
        "limitations": list(dict.fromkeys(limitations)),
    }
