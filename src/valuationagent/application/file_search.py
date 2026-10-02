"""Bounded whole-file discovery; interpretation belongs to the model."""
from pathlib import Path
from math import log1p, sqrt
from typing import Literal

from pydantic import Field

from valuationagent.application.file_workspace import (
    FileReference, UNTRUSTED, _pdf, _publish_view, file_operation, source_bytes,
)


class FileSearch(FileReference):
    queries: list[str] = Field(min_length=1, max_length=8, description="自行选择的原文关键词，任一命中；不是自然语言问题。不解释财务含义。")
    start_page: int = Field(default=1, ge=1, le=20000)
    page_limit: int = Field(default=300, ge=1, le=600)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=12, ge=1, le=30)
    order: Literal["relevance", "document"] = Field(default="relevance", description="relevance按关键词稀有度/覆盖排序，避免通用短词淹没具体命中；document按原文位置顺序。排序不解释科目或数值。")
    match_mode: Literal["any", "all"] = Field(default="any", description="any为任一关键词命中；all要求同一行包含所有关键词。")


def _compact(value):
    return "".join(value.casefold().split())


@file_operation
def search_file(store, session, args, check_cancel=lambda: None):
    meta, raw = source_bytes(store, session, args.file_id)
    queries = list(dict.fromkeys(_compact(query) for query in args.queries))
    if any(not query or len(query) > 100 for query in queries):
        raise ValueError("SEARCH_QUERY_INVALID: 每个关键词去空白后须为1至100字符。")
    check_cancel()
    blocks = store.research_blocks(session.session_id, args.file_id)
    coverage = {"kind": "stored_text", "complete_file": False}
    if Path(meta["storage_path"]).suffix.lower() == ".pdf":
        reader = _pdf(raw)
        page_count = len(reader.pages)
        if args.start_page > page_count:
            raise ValueError("PAGE_NOT_FOUND: 请求页码超过全文页数。")
        end_page = min(page_count, args.start_page + args.page_limit - 1)
        cached_pages = {block["location"].get("page") for block in blocks
                        if block["location"].get("view") == "pdf_search"}
        rows, empty_pages, failed_pages, text_size = [], [], [], 0
        last_page = args.start_page - 1
        for page_number in range(args.start_page, end_page + 1):
            check_cancel()
            if page_number in cached_pages:
                last_page = page_number
                continue
            try:
                page = reader.pages[page_number - 1]
                content = page.extract_text() or "" if "/Contents" in page else ""
            except (ValueError, TypeError, NotImplementedError):
                failed_pages.append(page_number)
                last_page = page_number
                continue
            if text_size + len(content) > 1500000:
                break
            if content.strip():
                rows.append((content, {"page": page_number, "decoder": "pypdf", "purpose": "full_text_discovery"}))
                text_size += len(content)
            else:
                empty_pages.append(page_number)
            last_page = page_number
        if rows:
            _publish_view(store, session, args.file_id, rows, "pdf_search", check_cancel,
                          max_chars=1500000, max_blocks=2000)
        blocks = [block for block in store.research_blocks(session.session_id, args.file_id)
                  if block["location"].get("view") == "pdf_search"
                  and args.start_page <= block["location"].get("page", 0) <= last_page]
        blocks.sort(key=lambda block: block["location"].get("page", 0))
        coverage = {"kind": "pdf_text_layer", "pages": page_count, "start_page": args.start_page,
                    "end_page": last_page, "next_page": last_page + 1 if last_page < page_count else None,
                    "complete_file": args.start_page == 1 and last_page == page_count and not failed_pages and not empty_pages,
                    "pages_without_text": empty_pages, "failed_pages": failed_pages}
    matches = []
    for block in blocks:
        check_cancel()
        lines = block["text"].splitlines()
        for index, line in enumerate(lines):
            matched = [query for query in queries if query in _compact(line)]
            if not matched or args.match_mode == "all" and len(matched) != len(queries):
                continue
            first, last = max(0, index - 2), min(len(lines), index + 4)
            matches.append({"block_id": block["block_id"], "location": block["location"],
                            "matched_line": index + 1, "queries": matched,
                            "lines": [{"line": number + 1, "text": lines[number][:600]}
                                      for number in range(first, last)],
                            "excerpt_truncated": any(len(lines[number]) > 600 for number in range(first, last))})
    if args.order == "relevance":
        frequencies = {query: len({match["block_id"] for match in matches if query in match["queries"]}) for query in queries}
        weights = {query: log1p(len(blocks) / max(1, frequencies[query])) * sqrt(len(query)) for query in queries}
        matches.sort(key=lambda match: sum(weights[query] for query in match["queries"]), reverse=True)
    selected = matches[args.offset:args.offset + args.limit]
    next_offset = args.offset + len(selected)
    page_hits = {}
    for match in matches:
        if page_number := match["location"].get("page"):
            group = page_hits.setdefault(page_number, {"page": page_number, "matches": 0, "queries": []})
            group["matches"] += 1
            group["queries"] = list(dict.fromkeys([*group["queries"], *match["queries"]]))
    return {"file_id": args.file_id, "source_sha256": meta["sha256"], "coverage": coverage,
            "matches": selected, "total_matches": len(matches), "order": args.order, "match_mode": args.match_mode,
            "page_hits": list(page_hits.values())[:100], "page_hits_omitted": max(0, len(page_hits) - 100),
            "next_offset": next_offset if next_offset < len(matches) else None, "trust": UNTRUSTED,
            "instruction": "这是关键词位置，不是财务解释或核验。按location.page或block_id读取原文，不以目录印刷页号或块后缀当PDF页码。默认按稀有关键词相关性排序，不按金融规则筛选；通用短词命中过多时改用具体词或match_mode=all，order=document可查看页序。可用offset看其余命中；coverage.next_page表示尚未扫描的页面。未命中不证明数据不存在，跨行词、扫描页和不可解码页需其他视图或来源。非PDF仅检索已存文本，覆盖未声明为全文。"}
