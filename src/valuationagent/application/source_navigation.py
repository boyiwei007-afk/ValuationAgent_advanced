"""Bounded navigation of immutable source links, without executing source scripts."""
import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

from pydantic import Field

from valuationagent.application.file_workspace import FileReference, _pdf, file_operation, source_bytes, source_document
from valuationagent.schemas.models import ApiModel


class SourceLinks(FileReference):
    query: str = Field(default="", max_length=200)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=40)
    start_page: int = Field(default=1, ge=1, le=20000)
    page_count: int = Field(default=20, ge=1, le=25)


class FollowSourceLink(ApiModel):
    file_id: str = Field(min_length=1, max_length=100)
    link_id: str = Field(pattern="^link_[0-9a-f]{24}$")
    page: int = Field(default=1, ge=1, le=20000, description="PDF链接的原文物理页码，来自list_source_links返回值；非PDF使用1。")


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.current = None
        self.base = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "base" and not self.base:
            self.base = attrs.get("href", "")
        if tag == "a" and attrs.get("href") and len(self.links) < 2000:
            self.current = {"url": attrs["href"], "text": attrs.get("title", "")[:300], "line": self.getpos()[0]}
            self.links.append(self.current)

    def handle_data(self, text):
        if self.current:
            self.current["text"] = (self.current["text"] + " " + text.strip())[:300]

    def handle_endtag(self, tag):
        if tag == "a":
            self.current = None


def public_link(base, raw):
    if not isinstance(raw, str) or not raw.strip() or raw.lstrip().startswith("#"):
        return None
    from valuationagent.application.research import ResearchService

    try:
        url = urljoin(base, raw.strip())
        parts = urlsplit(url)
        if parts.scheme.lower() == "http" and (parts.hostname or "").lower() in ResearchService._DISCLOSURE_HOSTS:
            url = urlunsplit(("https", parts.netloc, parts.path, parts.query, ""))
        return ResearchService._safe_public_url(url) if len(url) <= 2000 else None
    except ValueError:
        return None


@file_operation
def list_source_links(store, session, args, check_cancel=None, *, target_link_id=None):
    meta, raw = source_bytes(store, session, args.file_id)
    document = source_document(session, args.file_id)
    suffix = Path(meta["storage_path"]).suffix.lower()
    rows, next_page, truncated = [], None, False
    base = document.source_url
    if suffix == ".pdf":
        reader = _pdf(raw)
        total_pages = len(reader.pages)
        if args.start_page > total_pages:
            raise ValueError("LINK_PAGE_RANGE: 请求页码超过原始PDF范围")
        end = min(total_pages, args.start_page - 1 + args.page_count)
        for index in range(args.start_page - 1, end):
            if check_cancel:
                check_cancel()
            for reference in reader.pages[index].get("/Annots", []):
                annotation = reference.get_object()
                action = annotation.get("/A")
                if action:
                    action = action.get_object()
                    if action.get("/S") == "/URI":
                        rows.append({"url": str(action.get("/URI", "")), "text": str(annotation.get("/Contents", ""))[:300], "page": index + 1})
                if len(rows) >= 2000:
                    truncated = True
                    break
            if truncated:
                break
        next_page = end + 1 if end < total_pages else None
    elif suffix in {".html", ".htm", ".txt", ".md", ".json", ".csv", ".tsv"}:
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("LINK_TEXT_LIMIT: 链接枚举目前仅支持不超过2MB的文本；不截断原文，不执行脚本，可使用其他定位方式")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("gb18030")
        if suffix in {".html", ".htm"}:
            parser = LinkParser()
            parser.feed(text)
            parser.close()
            rows = parser.links
            base = public_link(base, parser.base) or base
        else:
            for number, line in enumerate(text.splitlines(), 1):
                if check_cancel:
                    check_cancel()
                for match in re.finditer(r"https?://[^\s<>\"'\]\)]+", line):
                    rows.append({"url": match.group(), "text": line[:300], "line": number})
                if len(rows) >= 2000:
                    break
        truncated = len(rows) >= 2000
    else:
        raise ValueError("LINK_FORMAT_UNSUPPORTED: 当前链接枚举支持PDF注释、HTML和纯文本，不执行Office宏或网页脚本")
    links = {}
    for row in rows[:2000]:
        url = public_link(base, row["url"])
        if not url:
            continue
        link_id = "link_" + hashlib.sha256((meta["sha256"] + "\n" + url).encode()).hexdigest()[:24]
        links.setdefault(link_id, {**row, "url": url, "link_id": link_id,
                                  "follow_arguments": {"file_id": args.file_id, "link_id": link_id, "page": row.get("page", 1)}})
    matched = [link for link in links.values() if (link["link_id"] == target_link_id if target_link_id else
               args.query.casefold() in (link["text"] + " " + link["url"]).casefold())]
    selected = matched[args.offset:args.offset + args.limit]
    end = args.offset + len(selected)
    return {"file_id": args.file_id, "source_sha256": meta["sha256"], "links": selected, "total_matches": len(matched),
            "next_offset": end if end < len(matched) else None, "next_page": next_page, "truncated": truncated,
            "untrusted_source_data": True,
            "instruction": "链接仅为原文中的线索，不继承父页披露日、公司或来源等级。按真实follow_arguments选择follow_source_link；仅下载公开原文，不执行脚本或登录。"}


def follow_source_link(runtime, args):
    if runtime.session.data_source_preference == "upload":
        raise ValueError("NETWORK_OUT_OF_SCOPE: 当前仅允许上传资料；可以查看链接文字，但不联网打开")
    packet = list_source_links(runtime.service.store, runtime.session,
                               SourceLinks(file_id=args.file_id, start_page=args.page, page_count=1, limit=40), runtime.service._check_execution,
                               target_link_id=args.link_id)
    selected = next((row for row in packet["links"] if row["link_id"] == args.link_id), None)
    if selected is None:
        raise ValueError("SOURCE_LINK_NOT_FOUND: 当前原件/页面不存在该link_id；先list_source_links，不编造链接或追加查询参数")
    for document in runtime.session.documents:
        if document.acquisition_ref == args.link_id or document.source_url == selected["url"] and not document.file_id.startswith("web_"):
            source_bytes(runtime.service.store, runtime.session, document.file_id)
            return {"status": "already_fetched", "file_id": document.file_id, "source_url": document.source_url,
                    "instruction": "此来源已保存，直接读取；不重新下载。"}
    if len(runtime.followed_links) >= 8:
        raise ValueError("SOURCE_LINK_BUDGET: 本轮最多打开8个原文链接；继续读取已有文件或保存具体检查点，不无限爬取")
    if args.link_id in runtime.followed_links:
        raise ValueError("SOURCE_LINK_REPEATED: 本轮该链接已尝试；检查现有原文或其他来源，不重复同一失败下载")
    runtime.followed_links.add(args.link_id)
    location = {"url": selected["url"], "source_type": "source_link", "provider": "document-links",
                "parent_source_file_id": args.file_id, "source_link_id": args.link_id}
    result = runtime.service._fetch_registered_source(runtime.session, args.link_id, location)
    return {**result, "parent_source_file_id": args.file_id, "source_link_id": args.link_id}
