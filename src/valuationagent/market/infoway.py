from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from valuationagent.market.history import HistorySnapshot
from valuationagent.market.tushare import normalize_a_share_ticker


BASE_URL = "https://data.infoway.io"
SCHEMA_URL = "https://blog.infoway.io/en/fundamental-data-api-guide/"
ENDPOINTS = {"income": "income_statement", "balancesheet": "balance_sheet", "cashflow": "cash_flow", "statistics": "statistics"}
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class InfowayApiError(ValueError):
    pass


class InfowayApiClient:
    def __init__(self, api_key, *, transport=None, min_interval=3.0, cache_seconds=600.0,
                 clock=time.monotonic, sleep=time.sleep):
        if not api_key.strip():
            raise ValueError("INFOWAY_KEY: API Key不能为空")
        self._api_key = api_key.strip()
        self._transport = transport
        self._min_interval = min_interval
        self._cache_seconds = cache_seconds
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._next_request = 0.0
        self._cache = OrderedDict()
        self.connection_verified = False

    def _wait(self, check_cancel):
        while (remaining := self._next_request - self._clock()) > 0:
            check_cancel()
            self._sleep(min(remaining, 0.25))
        check_cancel()

    def _retry_delay(self, value):
        try:
            return max(0.0, float(value))
        except (ValueError, TypeError):
            try:
                return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                return 5.0

    def query(self, ticker, statement, *, check_cancel=lambda: None):
        if statement not in ENDPOINTS:
            raise InfowayApiError("PROVIDER_CAPABILITY: 未支持的Infoway报表类型")
        symbol = normalize_a_share_ticker(ticker)
        key = (symbol, statement)
        with self._lock:
            check_cancel()
            cached = self._cache.get(key)
            if cached and self._clock() - cached[0] < self._cache_seconds:
                self._cache.move_to_end(key)
                return cached[1], cached[2]
            endpoint = f"{BASE_URL}/common/basic/financial/{ENDPOINTS[statement]}"
            for attempt in range(2):
                if self._next_request - self._clock() > 30:
                    raise InfowayApiError("PROVIDER_RATE_LIMIT: Infoway仍处于限流冷却期，请改用已有资料，不要轮询")
                self._wait(check_cancel)
                self._next_request = self._clock() + self._min_interval
                try:
                    with httpx.Client(transport=self._transport, timeout=25, follow_redirects=False, trust_env=False) as client:
                        with client.stream("GET", endpoint, headers={"apiKey": self._api_key},
                                           params={"symbol": symbol, "type": "STOCK_CN", "period_type": "fy"}) as response:
                            status = response.status_code
                            retry_after = response.headers.get("retry-after")
                            body = bytearray()
                            for chunk in response.iter_bytes():
                                check_cancel()
                                body.extend(chunk)
                                if len(body) > MAX_RESPONSE_BYTES:
                                    raise InfowayApiError("PROVIDER_RESPONSE_SIZE: Infoway响应超出读取上限")
                    raw = bytes(body)
                except httpx.HTTPError:
                    raise InfowayApiError("PROVIDER_CONNECTION: Infoway网络请求失败；没有更改已有输入") from None
                if self._api_key.encode() in raw:
                    raise InfowayApiError("PROVIDER_SECRET_ECHO: 响应包含凭证，拒绝保存")
                if status == 429:
                    self._next_request = max(self._next_request, self._clock() + self._retry_delay(retry_after))
                    if attempt == 0:
                        continue
                    raise InfowayApiError("PROVIDER_RATE_LIMIT: Infoway限流，已完成一次有界重试；请使用缓存或其他来源")
                if status != 200:
                    raise InfowayApiError(f"PROVIDER_HTTP_{status}: Infoway请求未成功，响应正文不进入日志")
                try:
                    payload = json.loads(raw, parse_float=str)
                except (ValueError, UnicodeError):
                    raise InfowayApiError("PROVIDER_FORMAT: Infoway未返回有效JSON") from None
                if not isinstance(payload, dict):
                    raise InfowayApiError("PROVIDER_FORMAT: Infoway响应必须为对象")
                if payload.get("ret") in (429, "429"):
                    self._next_request = max(self._next_request, self._clock() + 5)
                    if attempt == 0:
                        continue
                    raise InfowayApiError("PROVIDER_RATE_LIMIT: Infoway业务限流，已完成一次有界重试")
                if payload.get("ret") != 200 or not isinstance(payload.get("data"), list):
                    raise InfowayApiError("PROVIDER_RESPONSE: Infoway返回业务错误或不支持的数据结构")
                retrieved_at = datetime.now(timezone.utc).isoformat()
                self.connection_verified = True
                self._cache[key] = (self._clock(), raw, retrieved_at)
                self._cache.move_to_end(key)
                while len(self._cache) > 64:
                    self._cache.popitem(last=False)
                return raw, retrieved_at


class InfowayDataProvider:
    provider_id = "infoway"
    version = "infoway-history-2026-10-03.2"
    history_statements = tuple(ENDPOINTS)

    def __init__(self, client):
        self.client = client

    def fetch_history(self, ticker, statement, years, cutoff, check_cancel=lambda: None):
        symbol = normalize_a_share_ticker(ticker)
        raw, retrieved_at = self.client.query(symbol, statement, check_cancel=check_cancel)
        rows = json.loads(raw, parse_float=str)["data"]
        endpoint = f"{BASE_URL}/common/basic/financial/{ENDPOINTS[statement]}"
        blocks = []
        fields = {}
        accepted = 0
        for index, row in enumerate(rows):
            check_cancel()
            if not isinstance(row, dict):
                continue
            try:
                period = date.fromisoformat(str(row.get("periodDate", "")))
            except ValueError:
                continue
            if (row.get("symbol") != symbol or row.get("periodType") != "fy" or period.year not in years
                    or (period.month, period.day) != (12, 31) or period > cutoff):
                continue
            accepted += 1
            if isinstance(row.get("itemId"), str):
                fields[row["itemId"]] = row.get("itemName", "")
            blocks.append({"text": json.dumps(row, ensure_ascii=False, indent=2), "location": {
                "source_type": "structured_provider", "source_url": endpoint,
                "schema_reference": SCHEMA_URL, "json_pointer": f"/data/{index}",
                "record_index": index, "statement": statement, "provider_field": row.get("itemId"),
                "period_end": period.isoformat(), "retrieved_at": retrieved_at,
                "missing_metadata": ["currency", "unit_scale", "consolidation_scope", "published_at"],
                "value_semantics": "itemValue为该报告期值；ttm为滚动值；currentValue无独立日期，不得当作periodDate的股数或市值。"}})
        return HistorySnapshot(raw=raw, source_url=endpoint, blocks=blocks,
            accepted_records=accepted, excluded_records=len(rows) - accepted,
            catalog={"record_path": "/data", "fields": fields,
                "period_dates": sorted({block["location"]["period_end"] for block in blocks}),
                "reading_hint": "用read_file(view=records,record_path=/data,record_filters={itemId:[需要的原始字段ID],periodDate:[所需年度]},limit=12)一次阅读同指标跨年度记录。不要顺序翻完所有报表行；元数据缺失仍须如实说明。"},
            warnings=["Infoway是供应商转录，不是发行人原件。该接口未声明完整币种、单位倍率、合并口径及披露日，需要补充依据；抓取日不是披露日。",
                      "itemId与itemName交给LLM结合原始记录解释，不把net_income自动映射为归母净利润，不把平均股数或单一股份类别映射为发行人总股数。",
                      "periodDate仅为供应商声明。报告年度、收入定义与股份范围仍需核对；不得凭相同金额自动平移年份或将单一股份类别扩大为发行人全部股份。",
                      "筛选报告期不等于历史时点可得性验证；过去信息截止日的研究仍须补披露时间与重述版本依据。"])
