"""Provider-neutral search adapters.

The mock provider is intentionally deterministic and performs no network I/O.
Real search services can implement the same protocol without changing Agent,
CLI, Web, or evidence contracts.
"""

from __future__ import annotations

import hashlib
import html
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Protocol, Sequence
from urllib.parse import urlsplit

import httpx

from valuationagent.schemas.agent import SearchHit, SearchQuery, SearchResult


class SearchProvider(Protocol):
    provider_id: str
    version: str

    def search(self, query: SearchQuery) -> SearchResult: ...


class UnavailableSearchProvider:
    """Explicit placeholder used until a network provider is configured."""

    provider_id = "unavailable"
    version = "0.1"

    def search(self, query: SearchQuery) -> SearchResult:
        return SearchResult(
            query=query,
            provider=self.provider_id,
            provider_version=self.version,
            status="not_configured",
            warnings=["未配置联网搜索 provider；没有发出网络请求。"],
        )


class MockSearchProvider:
    """Deterministic fixture-backed provider for workflow and UI acceptance."""

    provider_id = "mock-search"
    version = "0.1"

    def __init__(self, hits: Sequence[SearchHit | dict] = ()):
        self._hits = [item if isinstance(item, SearchHit) else SearchHit.model_validate(item) for item in hits]

    @staticmethod
    def _query_id(query: SearchQuery) -> str:
        digest = hashlib.sha256(query.model_dump_json().encode("utf-8")).hexdigest()[:20]
        return f"search_{digest}"

    def search(self, query: SearchQuery) -> SearchResult:
        terms = [term.lower() for term in query.query.split() if term.strip()]
        ranked = []
        for hit in self._hits:
            haystack = f"{hit.title} {hit.domain} {hit.snippet}".lower()
            score = sum(term in haystack for term in terms) / max(len(terms), 1)
            if score or not terms:
                ranked.append(hit.model_copy(update={"relevance": round(score, 4)}))
        ranked.sort(key=lambda item: (-item.relevance, item.source_id))
        ranked = ranked[: query.candidate_limit]
        warnings = [
            f"mock provider；结果仅用于联调，不代表已完成公开资料核验。",
            f"query_id={self._query_id(query)}",
        ]
        return SearchResult(
            query=query,
            provider=self.provider_id,
            provider_version=self.version,
            status="completed" if ranked else "no_results",
            hits=ranked,
            warnings=warnings,
        )


class TavilySearchProvider:
    """Real-time search adapter with source URLs and an explicit cutoff date."""

    provider_id = "tavily"
    version = "2026-09-search-v2"

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = "https://api.tavily.com/search",
        timeout_seconds: float = 30,
        transport=None,
    ):
        if not api_key.strip():
            raise ValueError("Tavily API Key不能为空。")
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("搜索接口必须是无账号信息的HTTPS地址。")
        self._api_key = api_key.strip()
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    @staticmethod
    def _date(value: date | None) -> str | None:
        return value.isoformat() if value else None

    def search(self, query: SearchQuery) -> SearchResult:
        payload = {
            "query": query.query,
            "topic": "general",
            "search_depth": "advanced",
            "max_results": query.candidate_limit,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_domains": query.allowed_domains,
        }
        cutoff = query.information_cutoff or query.as_of_date
        if cutoff:
            payload["end_date"] = self._date(cutoff)
        request_attempts = 0
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                transport=self._transport,
                follow_redirects=False,
            ) as client:
                response = None
                for attempt in range(3):
                    request_attempts = attempt + 1
                    try:
                        response = client.post(
                            self.endpoint,
                            headers={
                                "Authorization": f"Bearer {self._api_key}",
                                "Content-Type": "application/json",
                            },
                            json=payload,
                        )
                    except (httpx.TimeoutException, httpx.RequestError):
                        if attempt < 2:
                            continue
                        raise
                    if response.status_code in {429, 502, 503, 504} and attempt < 2:
                        continue
                    break
                if response is None:
                    raise httpx.RequestError("search request did not start")
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code in {401, 403}:
                error_code = "SEARCH_AUTH_FAILED"
                message = "Tavily API Key 无效或无权访问，请重新配置 Tavily Key。"
            elif status_code == 429:
                error_code = "SEARCH_RATE_LIMITED"
                message = "Tavily 请求达到额度或频率上限，请稍后重试或检查账户额度。"
            else:
                error_code = f"SEARCH_HTTP_{status_code}"
                message = f"Tavily 服务返回 HTTP {status_code}，本次检索未完成。"
            return SearchResult(
                query=query,
                provider=self.provider_id,
                provider_version=self.version,
                status="failed",
                error_code=error_code,
                error_message=message,
                warnings=[message],
            )
        except httpx.TimeoutException:
            message = "Tavily 检索超时，本次没有取得搜索结果。"
            return SearchResult(
                query=query, provider=self.provider_id,
                provider_version=self.version, status="failed",
                error_code="SEARCH_TIMEOUT", error_message=message,
                warnings=[message],
            )
        except httpx.RequestError:
            message = "无法连接 Tavily，请检查网络、代理或防火墙设置。"
            return SearchResult(
                query=query, provider=self.provider_id,
                provider_version=self.version, status="failed",
                error_code="SEARCH_CONNECTION_FAILED", error_message=message,
                warnings=[message],
            )
        except (ValueError, TypeError):
            message = "Tavily 返回了无法识别的响应，本次没有采用任何搜索结果。"
            return SearchResult(
                query=query, provider=self.provider_id,
                provider_version=self.version, status="failed",
                error_code="SEARCH_RESPONSE_INVALID", error_message=message,
                warnings=[message],
            )
        hits = []
        for index, item in enumerate(body.get("results") or []):
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            if not url or not title:
                continue
            domain = urlsplit(url).netloc.lower()
            published = item.get("published_date")
            try:
                published_at = date.fromisoformat(str(published)[:10]) if published else None
            except ValueError:
                published_at = None
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
            hits.append(
                SearchHit(
                    source_id=f"tavily_{digest}",
                    title=title,
                    url=url,
                    domain=domain,
                    snippet=str(item.get("content") or "")[:2000],
                    published_at=published_at,
                    relevance=max(0, min(1, float(item.get("score") or 0))),
                )
            )
        warnings = []
        if request_attempts > 1:
            warnings.append(f"搜索服务出现瞬时故障，系统自动重试 {request_attempts - 1} 次后成功。")
        if cutoff:
            warnings.append(f"检索截止日：{cutoff.isoformat()}；结果仍需核对网页发布日期。")
        return SearchResult(
            query=query,
            provider=self.provider_id,
            provider_version=self.version,
            status="completed" if hits else "no_results",
            hits=hits,
            warnings=warnings,
        )


class CninfoAnnouncementProvider:
    """Search CNINFO's official disclosure catalogue by code and report year."""

    provider_id = "cninfo-announcements"
    version = "2026-09-official-catalogue-v2"

    def __init__(
        self,
        *,
        endpoint: str = "https://www.cninfo.com.cn/new/hisAnnouncement/query",
        timeout_seconds: float = 30,
        transport=None,
    ):
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or parsed.hostname != "www.cninfo.com.cn":
            raise ValueError("巨潮公告目录必须使用官方 HTTPS 地址。")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    @staticmethod
    def _market(ticker: str) -> tuple[str, str]:
        code = ticker.split(".", 1)[0].strip()
        if not re.fullmatch(r"[036]\d{5}", code):
            raise ValueError("官方年报检索目前需要沪深 A 股六位代码。")
        return code, ("sse" if code.startswith("6") else "szse")

    @staticmethod
    def _clean_title(value: str) -> str:
        return html.unescape(re.sub(r"<[^>]+>", "", str(value or ""))).strip()

    def search_annual_reports(
        self, ticker: str, years: Sequence[int], *, cutoff: date | None = None,
        company_name: str | None = None,
    ) -> SearchResult:
        return self.search_reports(ticker, years, cutoff=cutoff, company_name=company_name)

    def search_reports(
        self,
        ticker: str,
        years: Sequence[int],
        *,
        cutoff: date | None = None,
        company_name: str | None = None,
        report_type: str = "annual",
    ) -> SearchResult:
        labels = {"annual": "年度报告", "semiannual": "半年度报告", "q1": "第一季度报告", "q3": "第三季度报告"}
        if report_type not in labels:
            raise ValueError("不支持的定期报告类型。")
        report_label = labels[report_type]
        def matches_year(title, year):
            return bool(re.search(str(year) + r"年?" + report_label, title))
        code, column = self._market(ticker)
        selected_years = sorted({int(year) for year in years if 1990 <= int(year) <= 2100})
        if not selected_years:
            raise ValueError("请至少指定一个年报年份。")
        if len(selected_years) > 10:
            raise ValueError("单次最多检索十个年报年份。")
        cutoff = cutoff or date.today()
        start = date(min(selected_years) + (report_type == "annual"), 1, 1)
        end = cutoff
        query = SearchQuery(
            query=f"{code} {' '.join(str(year) for year in selected_years)} {report_label}",
            ticker=ticker,
            company_name=company_name,
            purpose="financials",
            as_of_date=cutoff,
            information_cutoff=cutoff,
            candidate_limit=min(50, len(selected_years) * 3),
            allowed_domains=["static.cninfo.com.cn"],
        )
        if start > end:
            return SearchResult(
                query=query,
                provider=self.provider_id,
                provider_version=self.version,
                status="no_results",
                warnings=["所选估值截止日早于这些年度报告的通常披露时间。"],
            )
        payload = {
            "pageNum": "1",
            "pageSize": "50",
            "column": column,
            "tabName": "fulltext",
            "plate": "sh" if column == "sse" else "sz",
            "stock": "",
            "searchkey": code,
            "secid": "",
            "category": "category_ndbg_szsh;" if report_type == "annual" else "",
            "trade": "",
            "seDate": f"{start.isoformat()}~{end.isoformat()}",
            "sortName": "time",
            "sortType": "desc",
            "isHLtitle": "true",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ValuationAgent/1.0; public-disclosure-research)",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://www.cninfo.com.cn",
            "Referer": f"https://www.cninfo.com.cn/new/fulltextSearch?keyWord={code}",
            "X-Requested-With": "XMLHttpRequest",
        }
        transient_retries = 0
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                transport=self._transport,
                follow_redirects=False,
                headers=headers,
            ) as client:
                def post_with_retry(data):
                    nonlocal transient_retries
                    response = None
                    for attempt in range(3):
                        try:
                            response = client.post(self.endpoint, data=data)
                        except (httpx.TimeoutException, httpx.RequestError):
                            if attempt < 2:
                                transient_retries += 1
                                continue
                            raise
                        if response.status_code in {429, 502, 503, 504} and attempt < 2:
                            transient_retries += 1
                            continue
                        response.raise_for_status()
                        return response
                    if response is None:
                        raise httpx.RequestError("official catalogue request did not start")
                    response.raise_for_status()
                    return response

                response = post_with_retry(payload)
                body = response.json()
                announcements = body.get("announcements") or []
                located_years = set()
                for item in announcements:
                    title = re.sub(r"\s+", "", self._clean_title(item.get("announcementTitle")))
                    if str(item.get("secCode") or "").strip() != code or "摘要" in title or "英文版" in title:
                        continue
                    located_years.update(
                        year for year in selected_years if matches_year(title, year)
                    )
                # Full-text code search can omit isolated historical years,
                # and some older Shenzhen main-board codes return no hits at
                # all. Re-query with the exact issuer ID whenever the first
                # response does not cover every requested year. Prefer the ID
                # returned by CNINFO; only use the legacy deterministic ID as
                # a fallback when the first response is empty.
                if not set(selected_years).issubset(located_years):
                    org_id = next((
                        str(item.get("orgId") or "").strip()
                        for item in announcements
                        if str(item.get("secCode") or "").strip() == code and item.get("orgId")
                    ), "")
                    if not org_id:
                        issuer_prefix = "gssh" if column == "sse" else "gssz"
                        org_id = f"{issuer_prefix}0{code}"
                    exact_payload = {
                        **payload,
                        "searchkey": "",
                        "stock": f"{code},{org_id}",
                    }
                    response = post_with_retry(exact_payload)
                    exact_body = response.json()
                    combined = {}
                    for item in [*announcements, *(exact_body.get("announcements") or [])]:
                        identity = str(item.get("announcementId") or item.get("adjunctUrl") or "")
                        if identity:
                            combined[identity] = item
                    body = {**body, "announcements": list(combined.values())}
        except httpx.HTTPStatusError as exc:
            message = f"巨潮资讯公告目录返回 HTTP {exc.response.status_code}，本次未取得年报目录。"
            return SearchResult(
                query=query, provider=self.provider_id, provider_version=self.version,
                status="failed", error_code=f"CNINFO_HTTP_{exc.response.status_code}",
                error_message=message, warnings=[message],
            )
        except httpx.TimeoutException:
            message = "巨潮资讯公告目录查询超时，本次未取得年报目录。"
            return SearchResult(
                query=query, provider=self.provider_id, provider_version=self.version,
                status="failed", error_code="CNINFO_TIMEOUT", error_message=message,
                warnings=[message],
            )
        except httpx.RequestError:
            message = "无法连接巨潮资讯公告目录，请检查网络后重试。"
            return SearchResult(
                query=query, provider=self.provider_id, provider_version=self.version,
                status="failed", error_code="CNINFO_CONNECTION_FAILED", error_message=message,
                warnings=[message],
            )
        except (ValueError, TypeError):
            message = "巨潮资讯公告目录返回了无法识别的数据，本次未采用任何链接。"
            return SearchResult(
                query=query, provider=self.provider_id, provider_version=self.version,
                status="failed", error_code="CNINFO_RESPONSE_INVALID", error_message=message,
                warnings=[message],
            )

        by_year: dict[int, SearchHit] = {}
        for item in body.get("announcements") or []:
            if str(item.get("secCode") or "").strip() != code:
                continue
            title = self._clean_title(item.get("announcementTitle"))
            normalized = re.sub(r"\s+", "", title)
            if "摘要" in normalized or "英文版" in normalized:
                continue
            report_year = next(
                (year for year in selected_years if matches_year(normalized, year)),
                None,
            )
            if report_year is None:
                continue
            path = str(item.get("adjunctUrl") or "").lstrip("/")
            if not path.casefold().endswith(".pdf"):
                continue
            published_at = None
            try:
                published_at = datetime.fromtimestamp(
                    int(item.get("announcementTime")) / 1000,
                    tz=timezone(timedelta(hours=8)),
                ).date()
            except (TypeError, ValueError, OSError):
                pass
            if published_at and published_at > cutoff:
                continue
            source_id = str(item.get("announcementId") or "").strip() or hashlib.sha256(
                path.encode("utf-8")
            ).hexdigest()[:24]
            hit = SearchHit(
                source_id=f"cninfo_{source_id}",
                title=title or f"{company_name or code} {report_year} 年{report_label}",
                url=f"https://static.cninfo.com.cn/{path}",
                domain="static.cninfo.com.cn",
                snippet=(
                    f"巨潮资讯官方公告目录；证券代码 {code}；报告年度 {report_year}；"
                    f"公告日期 {published_at.isoformat() if published_at else '待从PDF核对'}。"
                ),
                published_at=published_at,
                relevance=1,
            )
            current = by_year.get(report_year)
            # Prefer a later corrected full report when more than one official
            # version exists for the same year.
            if current is None or (hit.published_at or date.min) > (current.published_at or date.min):
                by_year[report_year] = hit
        hits = [by_year[year] for year in sorted(by_year, reverse=True)]
        missing = [year for year in selected_years if year not in by_year]
        warnings = [
            "结果来自巨潮资讯官方公告目录；财务数字仍须下载 PDF 原文并核对页码、单位和口径。"
        ]
        if transient_retries:
            warnings.append(f"官方目录出现瞬时故障，系统自动重试 {transient_retries} 次后成功。")
        if missing:
            warnings.append("未定位到完整" + report_label + "：" + "、".join(map(str, missing)))
        return SearchResult(
            query=query,
            provider=self.provider_id,
            provider_version=self.version,
            status="completed" if hits else "no_results",
            hits=hits,
            warnings=warnings,
        )


def create_search_provider():
    """Build the configured real provider without persisting its credential."""

    selected = os.getenv("VALUATION_SEARCH_PROVIDER", "auto").strip().lower()
    key = os.getenv("TAVILY_API_KEY") or os.getenv("VALUATION_SEARCH_API_KEY")
    if selected in {"none", "off", "unavailable"}:
        return UnavailableSearchProvider()
    if selected in {"auto", "tavily"} and key:
        return TavilySearchProvider(key)
    if selected not in {"auto", "tavily"}:
        raise ValueError("VALUATION_SEARCH_PROVIDER当前仅支持tavily或none。")
    return UnavailableSearchProvider()
