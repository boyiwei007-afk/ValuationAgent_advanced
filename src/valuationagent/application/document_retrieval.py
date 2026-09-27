"""Rank verbatim source blocks; relevance never substitutes for evidence checks."""
import re

from valuationagent.core.evidence import STATEMENT_TITLE


def rank_document_blocks(blocks, query):
    terms = [re.sub(r"\s+", "", term).casefold()
             for term in re.split(r"[\s,，;；|、]+", query) if term]
    issuer_share_query = any(term in query for term in ("股份总数", "普通股股数", "总股本"))
    if issuer_share_query and "总股本" not in terms:
        terms.append("总股本")
    amount = re.compile(r"\d{1,3}(?:[,，]\d{3})+(?:\.\d+)?|\d+\.\d{2}|(?<!\d)\d{3,}(?!\d)")
    ranked = []
    for index, block in enumerate(blocks):
        compact = re.sub(r"\s+", "", block["text"]).casefold()
        matches = sum(term in compact for term in terms)
        if not matches:
            continue
        numeric_rows = sum(
            bool(amount.search(line)) and any(term in re.sub(r"\s+", "", line).casefold() for term in terms)
            for line in block["text"].splitlines()
        )
        page = block.get("location", {}).get("page")
        context = [block]
        if page:
            context += [b for b in blocks[max(0, index - 8):index]
                        if page - 1 <= b.get("location", {}).get("page", -1) <= page]
        statement = any(re.search(STATEMENT_TITLE, b["text"]) for b in context)
        dated_issuer_total = bool(
            issuer_share_query
            and re.search(r"截至(?:19|20)\d{2}年\d{1,2}月\d{1,2}日.{0,20}公司总股本", compact)
            and amount.search(compact)
        )
        # Exact numeric rows in a nearby primary statement precede overview
        # summaries and contents pages, including integer-only thousand-CNY
        # tables. This ordering is only a retrieval hint, never scope approval.
        score = (dated_issuer_total, bool(numeric_rows), statement, matches, numeric_rows, "单位" in compact)
        ranked.append((score, block))
    return [block for _, block in sorted(ranked, key=lambda item: item[0], reverse=True)]
