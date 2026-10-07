"""Shared, finance-independent research conversations for CLI and Web."""
import ipaddress
import re
import socket
import threading
import time
import uuid
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import ClassVar, Literal
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpx
from pydantic import Field, ValidationError, field_validator, model_validator

from valuationagent.application.research_valuation import DEBT_COMPONENTS, METRIC_ALIASES, ResearchValuationAssembler, financial_mapping_issue, mapped_financial_metric, normalize_financial_metric
from valuationagent.application.valuation_plan import scope_key, valuation_progress
from valuationagent.core.data import LocalDataProvider
from valuationagent.core.documents import parse_document
from valuationagent.core.evidence import bind_evidence, evidence_context, numeric_tokens
from valuationagent.core.tools import canonical
from valuationagent.llm.client import LlmError
from valuationagent.llm.context import AGENT_PROMPT_VERSION
from valuationagent.schemas.models import ApiModel
from valuationagent.schemas.research import DocumentSummary, FactCandidate, ForecastInputs, ForecastProposal, ResearchIssue, ResearchMemoryItem, ResearchSession, ResearchTurn, SemanticMappingAlternative
from valuationagent.search.providers import (
    CninfoAnnouncementProvider,
    UnavailableSearchProvider,
)


_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?|prompts?)"),
    re.compile(r"(?i)(?:system|developer)\s+(?:prompt|message)|reveal\s+(?:your\s+)?prompt"),
    re.compile(r"(?i)(?:send|post|upload|exfiltrate).{0,80}(?:api[_ -]?key|token|secret|password)"),
    re.compile(r"忽略.{0,16}(?:之前|以上|此前).{0,16}(?:指令|规则|提示)"),
    re.compile(r"(?:泄露|发送|上传|提供).{0,40}(?:密钥|口令|系统提示|内部提示词)"),
)


def _annotate_untrusted_source(block):
    """Label external text as data and flag common prompt-injection phrases."""

    text = str(block.get("text") or "")
    flags = [
        f"prompt_injection_pattern_{index}"
        for index, pattern in enumerate(_PROMPT_INJECTION_PATTERNS, 1)
        if pattern.search(text)
    ]
    location = {
        **(block.get("location") or {}),
        "untrusted_source_data": True,
    }
    if flags:
        location["security_flags"] = flags
    return {**block, "location": location}


def _information_cutoff(session):
    return session.information_cutoff_date or session.draft.valuation_date


class ReadDocument(ApiModel):
    file_id: str
    start_page: int | None = Field(default=None, ge=1, le=2000, description="长PDF按页补读的起始页；每次最多25页，原文位置保持不变。其他文件勿填。")
    query: str = Field(default="", max_length=200)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=6, ge=1, le=8)


class InspectContext(ApiModel):
    section: Literal["overview", "facts", "memory", "documents", "user_notes"] = "overview"
    query: str = Field(default="", max_length=200, description="按关键词或 ID 查找历史字段、记忆或用户原话。")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=8, ge=1, le=20)


class ProposeForecast(ApiModel):
    inputs: ForecastInputs
    rationale: str = Field(min_length=20, max_length=2400, description="分清来源事实与预测判断，解释增长、利润率、WACC和永续增长依据；不得声称假设是历史事实。")
    risks: list[str] = Field(min_length=1, max_length=8)
    evidence_ids: list[str] = Field(min_length=1, max_length=30, description="关联已保存且有效的input_id、已核验财务候选ID或已读取原文块ID；用户情景可引用用户输入，不要求联网核验。不能编造ID或引用搜索摘要。")


class CandidateInput(ApiModel):
    metric: str = Field(
        min_length=1,
        max_length=120,
        description="来源中的原始科目名，必须按原文保留，不能为了适配模型而改名。",
    )
    standard_metric: str = Field(
        default="",
        max_length=120,
        json_schema_extra={"enum": ["", *sorted(METRIC_ALIASES)]},
        description=(
            "基于表名、附注、上下级标题、相邻行及业务主体判断的建议标准字段。"
            "优先使用系统规范英文名；上下文不足时留空并先继续读取关联附注。"
        ),
    )
    semantic_role: Literal[
        "operating", "financing", "financial_subsidiary", "investing",
        "tax", "equity", "non_operating", "unknown",
    ] = Field(
        default="unknown",
        description="该原始科目在当前上下文中的经济角色，而不是仅由科目文字猜测。",
    )
    ebit_treatment: Literal["include", "exclude", "review"] = Field(
        default="review",
        description="是否作为EBIT直接值或确定性推导输入；不确定用review。",
    )
    fcff_treatment: Literal["include", "exclude", "review"] = Field(
        default="review",
        description="是否作为FCFF直接值或确定性推导输入；不确定用review。",
    )
    equity_bridge_treatment: Literal["include", "exclude", "review"] = Field(
        default="review",
        description="是否进入企业价值到股权价值桥接；不确定用review。",
    )
    mapping_confidence: float = Field(
        default=0,
        ge=0,
        le=1,
        description="上下文语义映射置信度0到1；不是来源真实性评分。",
    )
    mapping_rationale: str = Field(
        default="",
        max_length=1200,
        description="引用表/附注/业务主体及相邻科目，解释建议映射和模型处理。",
    )
    alternative_interpretations: list[SemanticMappingAlternative] = Field(
        default_factory=list,
        max_length=4,
        description="仍然合理的替代解释；差异重大且补读附注后仍无法消歧才交用户判断。",
    )
    raw_value: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "原文中实际出现的数值，不得插值、推算或移到其他年份。唯一例外是完整合并资产负债表中，"
            "经明确年度表头和稳定列对齐证明目标年度金额格为空的债务科目，可提交0；系统会再次严格校验。"
        ),
    )
    unit: Literal["元", "千元", "万元", "百万元", "亿元", "股", "千股", "万股", "百万股", "亿股", "%", "ratio", "unknown"] = Field(
        default="unknown",
        description="只能使用枚举中的标准值；来源未说明或单位冲突时使用 unknown。",
    )
    period: str = Field(
        default="unknown",
        description="原文明确对应的报告期；不得把披露日期或相邻列年份当作报告期。",
    )
    scope: Literal["consolidated", "parent", "issuer", "unknown"] = Field(
        default="unknown",
        description="consolidated（合并）、parent（母公司）、issuer（仅发行人总股数，须有明确主体/截止日/股数单位的正文，或完整年报股份变动表总数行与表头）、unknown（无法判断）。其他财务科目不得使用issuer。",
    )
    role: Literal["historical", "assumption", "policy", "comparable"] = "historical"
    peer_ticker: str = Field(default="", max_length=24, description="可比公司代码，仅role=comparable时填写")
    peer_name: str = Field(default="", max_length=200, description="可比公司全称，必须由原文验证")
    multiple_basis: Literal["FY", "TTM", "forward", "unknown"] = Field(default="unknown", description="可比倍数分母口径；不得将TTM或预测倍数当作年度FY")
    block_id: str = Field(description="包含该候选值及其字段、期间或单位依据的来源块 ID。")
    table_id: str = Field(default="", description="interpret_financial_table返回的共享表格结构ID；不填写则仅使用本地原文绑定。")
    context_block_ids: list[str] = Field(default_factory=list, max_length=8,
        description="同一文件中补充公司名称、报表口径、年度列及单位表头的原文块ID；不得以模型自述替代表头。")
    quote: str = Field(
        min_length=1,
        max_length=2400,
        description="来源中的连续原文，须覆盖数值，并尽量同时覆盖字段名、年份、单位和口径。",
    )

    @field_validator("unit", mode="before")
    @classmethod
    def normalize_unit(cls, value):
        text = str(value or "unknown").strip().lower()
        aliases = {
            "人民币元": "元", "rmb": "元", "cny": "元",
            "人民币万元": "万元", "人民币亿元": "亿元", "人民币千元": "千元", "人民币百万元": "百万元",
            "百分比": "%", "percent": "%", "比例": "ratio", "倍": "ratio",
            "未知": "unknown", "不明": "unknown", "": "unknown",
        }
        allowed = {"元", "千元", "万元", "百万元", "亿元", "股", "千股", "万股", "百万股", "亿股", "%", "ratio", "unknown"}
        return aliases.get(text, text if text in allowed else "unknown")

    @field_validator("scope", mode="before")
    @classmethod
    def normalize_scope(cls, value):
        text = str(value or "unknown").strip().lower()
        if text in {"consolidated", "合并", "合并口径", "合并报表"} or text.startswith("合并（"):
            return "consolidated"
        if text in {"parent", "母公司", "母公司口径", "母公司报表"} or text.startswith("母公司（"):
            return "parent"
        if text in {"issuer", "发行人", "发行人口径"}:
            return "issuer"
        return "unknown"

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, value):
        aliases = {
            "历史": "historical", "历史数据": "historical",
            "假设": "assumption", "预测假设": "assumption",
            "政策": "policy", "政策数据": "policy",
        }
        text = str(value or "historical").strip().lower()
        return aliases.get(text, text)

    @model_validator(mode="after")
    def normalize_comparable_metric(self):
        if self.role == "comparable":
            aliases = {"市盈率": "pe", "p/e": "pe", "pe": "pe", "市销率": "ps", "p/s": "ps", "ps": "ps",
                       "ev/ebitda": "ev_ebitda", "ev_ebitda": "ev_ebitda", "企业价值倍数": "ev_ebitda"}
            self.metric = aliases.get(self.metric.strip().lower(), self.metric)
        return self


class ProposeFacts(ApiModel):
    candidates: list[CandidateInput] = Field(
        min_length=1, max_length=80,
        description="语义字段较丰富，建议每批4至8项；模型输出接近上限时最多4项。最多80项，每项均须绑定原文、期间、单位和口径。",
    )
    missing: list[str] = Field(default_factory=list, max_length=30)
    replaces: list[str] = Field(default_factory=list, max_length=80)


class MemoryUpdate(ApiModel):
    key: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_.:-]+$",
        description="稳定、可复用的记忆键；同一事项变更时沿用原 key。",
    )
    kind: Literal["goal", "preference", "constraint", "decision", "definition"]
    content: str = Field(
        min_length=1,
        max_length=600,
        description="用户明确表达的长期上下文；不得写入财务事实、推断、临时结果或秘密。",
    )


class UpdateMemory(ApiModel):
    updates: list[MemoryUpdate] = Field(default_factory=list, max_length=8)
    remove_keys: list[str] = Field(default_factory=list, max_length=8)


class SearchSources(ApiModel):
    query: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=500)
    purpose: Literal["company_profile", "financials", "comparables", "policy", "other"] = "other"
    as_of_date: date | None = None
    allowed_domains: list[str] = Field(default_factory=list, max_length=12)
    target_ticker: str = Field(default="", max_length=20, description="检索另一家公司的正式披露时，明确填写它的代码；不改变估值对象。不填且query含一个不同的六位代码时使用该代码。")
    report_years: list[int] = Field(default_factory=list, max_length=10, description="明确年度清单，例如[2022,2023,2024,2025]；DCF目标近十年，不是只取最新年报。")
    report_type: Literal["annual", "semiannual", "q1", "q3"] | None = None
    source_route: Literal["auto", "web", "official_catalogue"] = Field(default="auto", description="web直接搜索公开网页/财务表，不强制先搜PDF目录；官方定期报告用official_catalogue。")


class FetchSearchSource(ApiModel):
    file_id: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "search_sources 返回的 web_* 来源 ID；官方披露可下载 PDF，"
            "其他公开 HTTPS 来源可读取 PDF、HTML 或纯文本正文。"
        ),
    )


def _id(prefix):
    return prefix + uuid.uuid4().hex


def _text(session, zh, en):
    return en if session.language == "en-US" else zh


class ResearchService:
    _DISCLOSURE_HOSTS: ClassVar[set[str]] = {
        "static.cninfo.com.cn",
        "www.cninfo.com.cn",
        "disc.static.szse.cn",
        "www.szse.cn",
        "www.sse.com.cn",
        "static.sse.com.cn",
        "www.bse.cn",
        "static.bse.cn",
        # Hong Kong listed-company filings are first-party exchange
        # disclosures too.  Treating them as an arbitrary public webpage used
        # the smaller generic download limit and incorrectly labelled the
        # source as non-official, which made A+H capital actions hard to verify.
        "hkexnews.hk",
        "www.hkexnews.hk",
        "www1.hkexnews.hk",
    }

    def __init__(
        self,
        store,
        *,
        search_provider=None,
        official_search_provider=None,
        tool_providers=(),
        remote_transport=None,
        history_provider=None,
    ):
        self.store = store
        self._clients = {}
        self._search_clients = {}
        # Data-service credentials are deliberately process-local.  They may
        # be connected while an agent turn is already running, so keep the
        # bindings and their cache-busting epoch behind a small lock instead
        # of rewriting the persisted ResearchSession mid-turn (which would
        # race with the turn's optimistic session save).
        self._data_client_lock = threading.RLock()
        self._search_client_epochs = {}
        self._market_clients = {}
        self.history_provider = history_provider
        self.search_provider = search_provider or UnavailableSearchProvider()
        self.official_search_provider = official_search_provider or CninfoAnnouncementProvider()
        self.tool_providers = tuple(tool_providers)
        self.remote_transport = remote_transport
        self.valuation_assembler = ResearchValuationAssembler(block_loader=self._blocks)
        self._execution = threading.local()

    def execution_state(self, session_id):
        job = self.store.research_job(session_id)
        if not job:
            return {"status": "idle", "active": self.store.active(session_id), "stage": ""}
        if job["status"] in {"queued", "running"} and time.time() - job["updated"] > 35 and not self.store.active(session_id):
            job["status"] = "interrupted"
        return {"request_id": job["request_id"], "status": job["status"], "stage": job["stage"],
                "active": job["status"] in {"queued", "running"}, "cancel_requested": bool(job["cancel_requested"]),
                "started_at": job["created"], "elapsed_seconds": round(time.time() - job["created"] if job["status"] in {"queued", "running"} else job["updated"] - job["created"], 1)}

    def _owns_turn(self, session_id):
        current = getattr(self._execution, "current", None)
        return bool(current and current["session_id"] == session_id)

    def model_connected(self, session_id: str) -> bool:
        """Return process-local model availability without exposing credentials."""
        self.store.get_research(session_id)
        return session_id in self._clients

    def reserve_turn(self, session_id, turn):
        self.store.get_research(session_id)
        state = self.execution_state(session_id)
        if state["status"] == "interrupted":
            self.store.update_research_job(session_id, state["request_id"], status="interrupted")
        request_id = turn.request_id or _id("request_")
        body = turn.model_dump(mode="json", exclude={"request_id"})
        body["content"] = self._redact_text(body["content"])
        return request_id, self.store.reserve_research_job(session_id, request_id, body)

    def cancel_turn(self, session_id):
        state = self.execution_state(session_id)
        if state["active"] and state.get("request_id"):
            self.store.update_research_job(session_id, state["request_id"], cancel=True)
        return self.execution_state(session_id)

    def _check_execution(self):
        state = getattr(self._execution, "current", None)
        if not state:
            return
        now = time.monotonic()
        if now - state.get("last_check", 0) < .15:
            return
        state["last_check"] = now
        job = self.store.research_job(state["session_id"], state["request_id"])
        if job and job["cancel_requested"]:
            raise LlmError("EXECUTION_CANCELLED: 已按你的要求停止，已完成资料和确认结果已保存。")
        if now > state["deadline"]:
            raise LlmError("EXECUTION_TIME_LIMIT: 本次执行已达到时间预算，已保存进度，可继续未完成步骤。")

    @staticmethod
    def _model_identity(llm):
        config = getattr(llm, "config", None)
        return (
            str(getattr(config, "provider", "") or llm.__class__.__name__),
            str(getattr(config, "model", "") or "attached-model"),
        )

    def create(self, language="zh-CN", llm=None, *, data_source_preference=""):
        provider, model = self._model_identity(llm) if llm is not None else ("", "")
        session = ResearchSession(
            session_id=_id("research_"),
            language=language,
            prompt_version=AGENT_PROMPT_VERSION,
            model_provider=provider,
            model_name=model,
            data_source_preference=data_source_preference,
        )
        self.store.create_research(session)
        if llm is not None:
            self._clients[session.session_id] = llm
            self.store.append_event(
                session.session_id, type="model.attached", stage="configuration",
                status="completed", summary=f"{provider} / {model}",
                payload={"provider": provider, "model": model, "prompt_version": AGENT_PROMPT_VERSION},
            )
        self._say(session, _text(session,
            "欢迎来到 ValuationAgent。可以自由讨论方法，也可以提供公司与估值目标。我会使用检索、原文核验和确定性计算工具推进任务；自动模式生成估值草案，审阅模式先等你批准。资料不足会明确说明，不编造数值。",
            "Welcome to ValuationAgent. Name the company and valuation goal; files are optional. I will read sources, gather required inputs and prepare a model for one combined review. If essential data remains unavailable, bounded research ends with an explanatory report."))
        return session

    def attach(self, session_id, llm):
        if self.store.active(session_id):
            raise ValueError("当前会话正在处理请求，请稍后连接。")
        session = self.store.get_research(session_id)
        self._clients[session_id] = llm
        session.model_provider, session.model_name = self._model_identity(llm)
        session.prompt_version = AGENT_PROMPT_VERSION
        self.store.append_event(
            session.session_id, type="model.attached", stage="configuration",
            status="completed", summary=f"{session.model_provider} / {session.model_name}",
            payload={"provider": session.model_provider, "model": session.model_name,
                     "prompt_version": AGENT_PROMPT_VERSION},
        )
        if session.last_issue and session.last_issue.code in {
            "LLM_HTTP_401", "LLM_HTTP_403", "MODEL_SESSION_REVOKED", "MODEL_CONNECTION_REQUIRED",
            "LLM_TIMEOUT", "LLM_CONNECTION_FAILED", "LLM_NETWORK_FAILED", "LLM_RESPONSE_INVALID_JSON",
        }:
            self._resolve_issue(session)
        self.store.save_research(session)

    def attach_search(self, session_id, provider):
        """Hot-attach a session-only search provider without persisting its secret."""
        self.store.get_research(session_id)
        provider_id = str(getattr(provider, "provider_id", "") or "unknown")
        if provider_id == "unavailable" or not callable(getattr(provider, "search", None)):
            raise ValueError("联网搜索服务配置无效。")
        with self._data_client_lock:
            self._search_clients[session_id] = provider
            self._search_client_epochs[session_id] = (
                self._search_client_epochs.get(session_id, 0) + 1
            )
        self.store.append_event(
            session_id,
            type="search.attached",
            stage="configuration",
            status="completed",
            summary=f"联网搜索凭证已加载 · {provider_id}",
            payload={
                "provider": provider_id,
                "provider_version": str(getattr(provider, "version", "")),
            },
        )

    def attach_market(self, session_id, provider):
        """Attach a session-only A-share data provider without storing its token."""
        self.store.get_research(session_id)
        if not any(callable(getattr(provider, capability, None)) for capability in ("resolve", "fetch_history")):
            raise TypeError("A股取数服务配置无效。")
        with self._data_client_lock:
            self._market_clients[session_id] = provider
        self.store.append_event(
            session_id,
            type="market.attached",
            stage="configuration",
            status="completed",
            summary=f"A股取数凭证已加载 · {getattr(provider, 'version', 'unknown')}",
            payload={"provider_version": str(getattr(provider, "version", ""))},
        )

    def data_service_status(self, session_id, *, default_market=None):
        """Return public availability metadata; never return credentials."""
        self.store.get_research(session_id)
        with self._data_client_lock:
            search = self._search_clients.get(session_id, self.search_provider)
            market = self._market_clients.get(session_id, self.history_provider or default_market)
        search_id = str(getattr(search, "provider_id", "unavailable") or "unavailable")
        market_version = str(getattr(market, "version", "") or "")
        return {
            "official_disclosure": {
                "available": True,
                "provider": str(getattr(
                    self.official_search_provider,
                    "provider_id",
                    "cninfo-announcements",
                )),
                "requires_key": False,
            },
            "search": {
                "available": search_id != "unavailable",
                "provider": search_id,
                "connection_status": "verified" if getattr(search, "connection_verified", False) else "configured" if search_id != "unavailable" else "unconfigured",
            },
            "market": {
                "available": bool(market and any(callable(getattr(market, capability, None)) for capability in ("resolve", "fetch_history")))
                and not isinstance(market, LocalDataProvider),
                "provider": market_version or "unavailable",
                "statements": list(getattr(market, "history_statements", ())),
                "connection_status": "verified" if getattr(getattr(market, "client", None), "connection_verified", False) else "configured" if market and not isinstance(market, LocalDataProvider) else "unconfigured",
            },
        }

    @classmethod
    def _safe_disclosure_url(cls, raw_url):
        """Accept direct URLs on supported mainland/HK exchange hosts."""
        parts = urlsplit(str(raw_url or "").strip())
        host = (parts.hostname or "").rstrip(".").casefold()
        if parts.username or parts.password or parts.port not in {None, 443}:
            raise ValueError("原始资料链接包含不安全的认证信息或端口。")
        if host not in cls._DISCLOSURE_HOSTS:
            raise ValueError("当前只允许下载巨潮资讯、交易所等官方披露站点的原始文件。")
        if parts.scheme.casefold() not in {"http", "https"}:
            raise ValueError("原始资料链接必须使用 HTTP 或 HTTPS。")
        # Official legacy result pages occasionally expose an http link. Use
        # the equivalent encrypted endpoint and never send credentials.
        return urlunsplit(("https", host, parts.path, parts.query, ""))

    @staticmethod
    def _safe_public_url(raw_url):
        """Reject local, credential-bearing and non-HTTPS web targets."""
        parts = urlsplit(str(raw_url or "").strip())
        host = (parts.hostname or "").rstrip(".").casefold()
        if parts.scheme.casefold() != "https" or not host:
            raise ValueError("公开网页原文必须使用完整的 HTTPS 地址。")
        if parts.username or parts.password or parts.port not in {None, 443}:
            raise ValueError("公开网页链接包含不安全的认证信息或端口。")
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise ValueError("公开网页链接不能指向本机或内部网络。")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("公开网页链接不能指向私有、回环或保留地址。")
        return urlunsplit(("https", host, parts.path or "/", parts.query, ""))

    def _verify_public_host(self, host):
        """Resolve real network targets before fetching to reduce SSRF risk."""
        if self.remote_transport is not None:
            # A custom transport does not use the machine network. Literal and
            # special host checks above still apply to integration tests.
            return
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            }
        except OSError:
            raise ValueError("公开网页域名无法解析，请选择另一个搜索结果。") from None
        if not addresses:
            raise ValueError("公开网页域名没有可用地址，请选择另一个搜索结果。")
        for raw_address in addresses:
            try:
                address = ipaddress.ip_address(raw_address.split("%", 1)[0])
            except ValueError:
                raise ValueError("公开网页域名解析结果无效。") from None
            if not address.is_global:
                raise ValueError("公开网页域名解析到了非公网地址，已停止下载。")

    def _download_public_source(self, raw_url):
        """Download a bounded public PDF/HTML/text source returned by search."""
        url = self._safe_public_url(raw_url)
        headers = {
            "User-Agent": "ValuationAgent/1.0 (+research; public-source-reader)",
            "Accept": "application/pdf,text/html,application/xhtml+xml,text/plain;q=0.9",
        }
        try:
            with httpx.Client(
                timeout=httpx.Timeout(30, connect=10),
                follow_redirects=False,
                transport=self.remote_transport,
                headers=headers,
            ) as client:
                for _ in range(4):
                    host = urlsplit(url).hostname or ""
                    self._verify_public_host(host)
                    with client.stream("GET", url) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise ValueError("公开网页发生了无目标重定向。")
                            url = self._safe_public_url(urljoin(url, location))
                            continue
                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError as exc:
                            raise ValueError(
                                f"公开网页下载失败（HTTP {exc.response.status_code}）。"
                            ) from None
                        try:
                            declared = int(response.headers.get("content-length", "0") or 0)
                        except ValueError:
                            declared = 0
                        if declared > 15 * 1024 * 1024:
                            raise ValueError("公开网页原文超过 15 MB 下载上限。")
                        chunks, total = [], 0
                        for chunk in response.iter_bytes():
                            total += len(chunk)
                            if total > 15 * 1024 * 1024:
                                raise ValueError("公开网页原文超过 15 MB 下载上限。")
                            chunks.append(chunk)
                        payload = b"".join(chunks)
                        content_type = response.headers.get(
                            "content-type", ""
                        ).split(";", 1)[0].strip().casefold()
                        prefix = payload[:512].lstrip().lower()
                        if payload.startswith(b"%PDF"):
                            return url, payload, "application/pdf", ".pdf"
                        if (
                            content_type in {"text/html", "application/xhtml+xml"}
                            or prefix.startswith((b"<!doctype html", b"<html"))
                        ):
                            return url, payload, "text/html", ".html"
                        if content_type == "text/plain":
                            return url, payload, "text/plain", ".txt"
                        if content_type in {"application/json", "text/csv", "text/tab-separated-values"}:
                            return url, payload, content_type, {"application/json": ".json", "text/csv": ".csv", "text/tab-separated-values": ".tsv"}[content_type]
                        raise ValueError(
                            "该搜索结果不是可解析的 PDF、HTML 或纯文本原文。"
                        )
                raise ValueError("公开网页重定向次数过多。")
        except httpx.TimeoutException:
            raise ValueError("公开网页下载超时。请重试或选择另一个来源。") from None
        except httpx.RequestError:
            raise ValueError("无法连接公开网页。请检查网络或选择另一个来源。") from None

    def _download_disclosure_pdf(self, raw_url):
        url = self._safe_disclosure_url(raw_url)
        headers = {
            "User-Agent": "ValuationAgent/1.0 (+research; public-disclosure-reader)",
            "Accept": "application/pdf,application/octet-stream;q=0.8",
        }
        try:
            with httpx.Client(
                timeout=httpx.Timeout(35, connect=12),
                follow_redirects=False,
                transport=self.remote_transport,
                headers=headers,
            ) as client:
                for _ in range(4):
                    with client.stream("GET", url) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise ValueError("原始资料链接发生了无目标重定向。")
                            url = self._safe_disclosure_url(urljoin(url, location))
                            continue
                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError as exc:
                            raise ValueError(
                                f"原始资料下载失败（HTTP {exc.response.status_code}）。"
                            ) from None
                        try:
                            declared = int(response.headers.get("content-length", "0") or 0)
                        except ValueError:
                            declared = 0
                        if declared > 50 * 1024 * 1024:
                            raise ValueError("原始资料超过 50 MB 下载上限。")
                        chunks, total = [], 0
                        for chunk in response.iter_bytes():
                            total += len(chunk)
                            if total > 50 * 1024 * 1024:
                                raise ValueError("原始资料超过 50 MB 下载上限。")
                            chunks.append(chunk)
                        payload = b"".join(chunks)
                        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
                        if not payload.startswith(b"%PDF"):
                            raise ValueError(
                                "该搜索结果不是可解析的 PDF 原文。请继续检索巨潮资讯或交易所的 PDF 公告链接。"
                            )
                        return url, payload, content_type or "application/pdf"
                raise ValueError("原始资料链接重定向次数过多。")
        except httpx.TimeoutException:
            raise ValueError("原始资料下载超时。请重试或选择另一个官方公告链接。") from None
        except httpx.RequestError:
            raise ValueError("无法连接原始资料站点。请检查网络，或选择另一个官方公告链接。") from None

    def _fetch_search_source(self, session, file_id):
        """Download, parse and register a source discovered by search."""
        existing_document = next((document for document in session.documents if document.file_id == file_id), None)
        if existing_document and existing_document.acquisition_ref and not file_id.startswith("web_"):
            from valuationagent.application.file_workspace import source_bytes

            source_bytes(self.store, session, file_id)
            return {"status": "already_fetched", "file_id": file_id, "name": existing_document.name,
                    "block_count": existing_document.block_count, "warnings": existing_document.warnings,
                    "instruction": "这就是已下载的原文ID，不再请求网络；直接inspect_file/read_file读取。"}
        if file_id not in {doc.file_id for doc in session.documents} or not file_id.startswith("web_"):
            raise ValueError("只能打开本次研究中 search_sources 返回的候选来源。")
        lead_blocks = self.store.research_blocks(session.session_id, file_id)
        if not lead_blocks:
            raise ValueError("搜索候选缺少可追溯链接。")
        location = lead_blocks[0].get("location") or {}
        source_url = location.get("url")
        if location.get("source_type") != "web_search" or not source_url:
            raise ValueError("该来源不是可下载的联网搜索候选。")
        for document in session.documents:
            if document.file_id.startswith("web_"):
                continue
            existing = self.store.research_blocks(session.session_id, document.file_id)
            if document.acquisition_ref == file_id or existing and (existing[0].get("location") or {}).get("parent_search_file_id") == file_id:
                return {
                    "status": "already_fetched",
                    "file_id": document.file_id,
                    "name": document.name,
                    "block_count": document.block_count,
                    "warnings": document.warnings,
                }

        if session.data_source_preference == "upload":
            raise ValueError("NETWORK_OUT_OF_SCOPE: 当前任务仅允许上传数据，不能下载新来源")
        return self._fetch_registered_source(session, file_id, location)

    def _fetch_registered_source(self, session, acquisition_ref, location):
        if session.data_source_preference == "upload":
            raise ValueError("NETWORK_OUT_OF_SCOPE: 当前任务仅允许上传数据，不能下载新来源")
        self._check_execution()
        source_url = location["url"]
        source_host = (urlsplit(source_url).hostname or "").rstrip(".").casefold()
        official = source_host in self._DISCLOSURE_HOSTS and (
            location.get("source_type") == "web_search" or urlsplit(source_url).path.lower().endswith(".pdf"))
        if official:
            resolved_url, payload, content_type = self._download_disclosure_pdf(source_url)
            suffix = ".pdf"
        else:
            resolved_url, payload, content_type, suffix = self._download_public_source(source_url)
        remote_name = unquote(PurePosixPath(urlsplit(resolved_url).path).name) or (
            "公开披露原文.pdf" if official else "公开网页原文" + suffix
        )
        if not remote_name.casefold().endswith(suffix):
            remote_name = str(PurePosixPath(remote_name).with_suffix(suffix))
        meta = self.store.save_upload(remote_name, "evidence", content_type, payload)
        try:
            blocks, warnings = parse_document(self.store.get_file(meta["file_id"]), check_cancel=self._check_execution, pdf_page_limit=25)
        except LlmError:
            raise
        except Exception:
            blocks, warnings = [], ["原文已保存，但初始文本解码失败。可用inspect_file检查结构、read_file切换视图或view_pdf_page查看页图；未形成财务事实。"]
        published_at = location.get("published_at")
        if not official and not published_at:
            dates = set(re.findall(r"(?:发布日期|发布时间|公告日期|披露日期)\s*[:：]?\s*(20\d{2}[-/]\d{1,2}[-/]\d{1,2})", "\n".join(block["text"] for block in blocks[:8])))
            if len(dates) == 1:
                parts = re.split(r"[-/]", dates.pop())
                try:
                    published_at = date(*map(int, parts)).isoformat()
                except ValueError:
                    pass
        if not official:
            warnings = list(dict.fromkeys([
                (
                    "来源为公开网页原文；政策、业务与行业信息须结合发布日期和来源主体核验，"
                    "历史字段可在完成绑定与跨来源核对后进入标注来源限制的草案，不得伪装成官方披露。"
                ),
                *warnings,
            ]))[:30]
        for block in blocks:
            block["location"] = {
                **(block.get("location") or {}),
                "source_type": "remote_document" if official else "remote_web_document",
                "source_url": resolved_url,
                "source_domain": urlsplit(resolved_url).hostname,
                "parent_search_file_id": acquisition_ref if location.get("source_type") == "web_search" else None,
                "parent_source_file_id": location.get("parent_source_file_id"),
                "source_link_id": location.get("source_link_id"),
                "acquisition_type": location.get("source_type"),
                "search_provider": location.get("provider"),
                "search_query": location.get("search_query"),
                "search_target_ticker": location.get("target_ticker"),
                "published_at": published_at,
            }
        self.store.save_research_blocks(session.session_id, meta["file_id"], blocks)
        document = DocumentSummary(
            file_id=meta["file_id"],
            name=remote_name,
            role="evidence",
            block_count=len(blocks),
            sha256=meta["sha256"],
            size_bytes=meta["size_bytes"],
            warnings=warnings,
            parse_status="unreadable" if not blocks else "partial" if warnings else "parsed",
            provenance_type="official_filing" if official else "public_web",
            authority_tier="A" if official else "C",
            source_confidence=0.99 if official else 0.55,
            provider=location.get("provider") or "",
            source_url=resolved_url,
            acquisition_ref=acquisition_ref,
        )
        session.documents.append(document)
        return {
            "status": "fetched",
            "file_id": document.file_id,
            "name": document.name,
            "block_count": document.block_count,
            "warnings": document.warnings,
            "source_url": resolved_url,
            "instruction": (
                "原文已取得。先begin_file_task明确主体、角色与重点字段，再用search_file定位和read_file精读；财务事实只引用返回的原文block_id。完成本文件后end_file_task继续其他缺口。"
                if official else
                "下一步用read_file阅读正文；批量提取前可begin_file_task隔离其他来源。历史字段须绑定主体/期间/单位/口径，"
                "并用corroborate_facts交叉核对；非官方来源保留等级与限制。"
            ),
        }

    def snapshot(self, session_id, *, compact=False, after=None):
        session = self.store.get_research(session_id)
        report = self.store.research_report(session_id, summary=True)
        if compact:
            events = self.store.event_page(session_id, after)
            messages = self.store.message_page(session_id)
            return {"session": session.model_dump(mode="json"), "result_document": report, "execution": self.execution_state(session_id),
                    **events, "messages": messages["messages"], "message_before": messages["before"], "older_messages": messages["has_more"]}
        return {"session": session.model_dump(mode="json"), "result_document": report,
                "execution": self.execution_state(session_id),
                "messages": [m.model_dump(mode="json") for m in self.store.list_messages(session_id)],
                "events": [e.model_dump(mode="json") for e in self.store.list_events(session_id)]}

    def submit_valuation(self, session_id, runner, *, request_override=None):
        """Create one auditable valuation run from the confirmed research state."""
        if self.store.active(session_id) and not self._owns_turn(session_id):
            raise ValueError("当前工作区正在处理请求，请稍后提交估值。")
        session = self.store.get_research(session_id)
        request = (
            request_override
            if request_override is not None
            else self.valuation_assembler.build(session)
        )
        parent_id = None
        if session.valuation_run_id:
            try:
                record = runner.store.get_run(session.valuation_run_id)
                if record.request == request:
                    if request.data_source == "ticker" and callable(getattr(self._market_clients.get(session_id), "resolve", None)):
                        runner.attach_data(record.run_id, self._market_clients[session_id])
                    session.pending_action = ""
                    session.status = "submitted"
                    self.store.save_research(session)
                    return record
                if str(record.status) in {"created", "running"}:
                    raise ValueError("上一项正式估值仍在执行，请等待完成后再提交修改后的研究范围。")
                parent_id = record.run_id
            except KeyError:
                session.valuation_run_id = None
        record = runner.create_run(
            request,
            parent_id=parent_id,
            reason="研究会话输入更新后重新提交" if parent_id else None,
        )
        if request.data_source == "ticker" and callable(getattr(self._market_clients.get(session_id), "resolve", None)):
            runner.attach_data(record.run_id, self._market_clients[session_id])
        session.valuation_run_id = record.run_id
        session.pending_action = ""
        session.status = "submitted"
        self.store.save_research(session)
        self.store.append_event(
            session_id,
            type="valuation.submitted",
            stage="handoff",
            status="completed",
            summary=f"研究会话已提交估值任务 {record.run_id}",
            payload={
                "run_id": record.run_id,
                "parent_run_id": parent_id,
                "data_source": request.data_source,
                "confirmed_fact_ids": [
                    fact.fact_id for fact in session.facts if fact.status == "confirmed"
                ],
            },
        )
        runner.store.append_event(
            record.run_id,
            type="research.handoff",
            stage="data_intake",
            status="completed",
            summary=f"来自研究会话 {session_id}",
            payload={"research_session_id": session_id, "research_revision": session.revision},
        )
        return record

    def approve_valuation_plan(self, session_id, expected_progress, *, automatic=False):
        """Apply the workspace's single, state-hashed approval without an LLM turn."""

        if self.store.active(session_id) and not self._owns_turn(session_id):
            raise ValueError("当前工作区正在处理请求，请稍后批准。")
        session = self.store.get_research(session_id)
        current = valuation_progress(session, self.valuation_assembler)
        if not current.get("ready_for_review"):
            raise ValueError("估值方案已变化或不再可执行，请重新生成集中复核包。")
        compared_keys = tuple(expected_progress)
        if canonical({key: current.get(key) for key in compared_keys}) != canonical(
            expected_progress
        ):
            raise ValueError("估值方案或证据已变化，请重新生成集中复核包。")
        staged = set(current.get("staged_fact_ids") or [])
        for fact in session.facts:
            if fact.fact_id in staged and not fact.warnings:
                fact.status = "confirmed"
        superseded = {
            old for fact_id in staged
            for old in session.staged_supersessions.get(fact_id, [])
        }
        for fact in session.facts:
            if fact.fact_id in superseded:
                fact.status = "rejected"
        if session.forecast_proposal:
            session.forecast_proposal.status = "confirmed"
        session.valuation_methods_override = list(current["methods"])
        session.valuation_method_exclusions = dict(
            current.get("excluded_methods") or {}
        )
        session.status = "ready_for_valuation"
        session.gaps = []
        self.store.append_event(
            session_id,
            type="valuation.plan_confirmed",
            stage="planning",
            status="completed",
            summary="自动策略接受估值输入及预测假设以生成草案；非人工批准" if automatic else "用户通过工作区集中检查点确认整套估值输入及预测假设",
            payload={
                "approved_by": "automatic_policy" if automatic else "user",
                "forecast_proposal_id": current.get("forecast_proposal_id"),
                "fact_ids": sorted(staged),
                "requested_methods": current.get("requested_methods", current["methods"]),
                "effective_methods": current["methods"],
                "excluded_methods": current.get("excluded_methods", {}),
            },
        )
        self.store.save_research(session)
        return session

    def _say(self, session, content):
        content = self._redact_text(content)
        self.store.add_message(session.session_id, "assistant", content, "research")
        self.store.append_event(session.session_id, type="conversation.message", stage="research", summary=content, status="completed")

    def _tool(self, session, name, args, fn):
        self._check_execution()
        state = getattr(self._execution, "current", None)
        if state:
            self.store.update_research_job(session.session_id, state["request_id"], stage=name)
        call_id = _id("call_")
        started = time.monotonic()
        self.store.append_event(session.session_id, type="tool.started", stage="research", tool=name,
                                tool_call_id=call_id, status="running", summary=name,
                                payload={"arguments": self._redact_value(args)})
        try:
            result = fn()
        except Exception as exc:
            if isinstance(exc, ValidationError):
                fields = [".".join(str(part) for part in item["loc"])
                          for item in exc.errors(include_input=False)]
                message = "工具参数不符合约束" + ("：" + "、".join(fields[:12]) if fields else "。")
            elif isinstance(exc, (ValueError, LlmError)):
                message = self._redact_text(str(exc))[:1200]
            else:
                message = "文件或工具处理失败，请检查资料后重试。"
            self.store.append_event(session.session_id, type="tool.failed", stage="research", tool=name,
                tool_call_id=call_id, status="failed", summary=message,
                duration_ms=int((time.monotonic() - started) * 1000),
                payload={"error_type": exc.__class__.__name__})
            # The same safe error also goes back to the model for correction.
            if isinstance(exc, LlmError):
                raise LlmError(message) from None
            if isinstance(exc, ValueError) and not isinstance(exc, ValidationError):
                raise ValueError(message) from None  # noqa: TRY004 - preserve tool error contract
            raise
        self.store.append_event(session.session_id, type="tool.completed", stage="research", tool=name,
            tool_call_id=call_id, status="completed", summary=name,
            duration_ms=int((time.monotonic() - started) * 1000),
            payload={"output": self._redact_value(result)})
        self.store.save_research(session)
        return self._redact_value(result)


    @staticmethod
    def _contains_secret(text):
        return bool(re.search(r"(?i)\b(?:sk|key|tvly)-[a-z0-9_-]{12,}\b|\bbearer\s+[a-z0-9._-]{12,}|\b[a-z0-9_-]{16,}-infoway\b", text))

    @staticmethod
    def _redact_text(text):
        return re.sub(
            r"(?i)\b(?:sk|key|tvly)-[a-z0-9_-]{12,}\b|\bbearer\s+[a-z0-9._-]{12,}|\b[a-z0-9_-]{16,}-infoway\b",
            "[REDACTED_CREDENTIAL]",
            text,
        )

    @classmethod
    def _redact_value(cls, value):
        if isinstance(value, str):
            return cls._redact_text(value)
        if isinstance(value, dict):
            return {key: cls._redact_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._redact_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._redact_value(item) for item in value)
        return value

    def _apply_memory(self, session, updates, remove_keys):
        current = {item.key: item for item in session.memory}
        removed, changed, rejected = [], [], []
        for key in dict.fromkeys(remove_keys):
            if key in current:
                current.pop(key)
                removed.append(key)
        user_messages = [m for m in self.store.list_messages(session.session_id) if m.role == "user"]
        source_message_id = user_messages[-1].message_id if user_messages else ""
        for update in updates:
            if self._contains_secret(update.content):
                rejected.append(update.key)
                continue
            # Reinsert replacements at the end so the bounded list keeps recent decisions.
            current.pop(update.key, None)
            current[update.key] = ResearchMemoryItem(
                **update.model_dump(), source_message_id=source_message_id
            )
            changed.append(update.key)
        session.memory = list(current.values())[-80:]
        if changed or removed or rejected:
            self.store.append_event(
                session.session_id,
                type="memory.updated",
                stage="context",
                status="completed" if not rejected else "warning",
                summary=f"长期记忆更新 {len(changed)}，移除 {len(removed)}，拒绝 {len(rejected)}",
                payload={"updated_keys": changed, "removed_keys": removed, "rejected_keys": rejected},
            )
        return {"updated": changed, "removed": removed, "rejected": rejected}


    def _resolve_issue(self, session, status="resolved"):
        if session.last_issue and session.last_issue.status in {"open", "retrying"}:
            session.last_issue.status = status
            self.store.append_event(
                session.session_id,
                type="agent.recovery_resolved",
                stage=session.last_issue.stage,
                status="completed",
                summary=session.last_issue.code,
                payload={"issue_id": session.last_issue.issue_id, "resolution": status},
            )


    def _apply_task_draft(self, session, draft, *, automatic=False):
        """Apply an explicit/unambiguous scope without manufacturing a gate."""

        previous_company = (session.draft.company, session.draft.ticker)
        session.draft = draft.model_copy(deep=True)
        if draft.information_cutoff_date is not None:
            session.information_cutoff_date = draft.information_cutoff_date
        if automatic:
            session.pending_action = "valuation"
        session.outcome_status = ""
        session.outcome_reason = ""
        session.valuation_methods_override = []
        session.valuation_method_exclusions = {}
        if any(previous_company) and previous_company != (
            session.draft.company, session.draft.ticker
        ):
            for fact in session.facts:
                if fact.status == "confirmed":
                    fact.status = "proposed"
                    fact.warnings.append("研究公司已变更，请重新核对该字段的归属")
        if (
            session.forecast_proposal
            and session.forecast_proposal.scope_key != scope_key(session)
        ):
            self.store.append_event(
                session.session_id,
                type="valuation.forecast_invalidated",
                stage="planning",
                status="completed",
                summary="任务范围变化，旧预测保留在审计记录但不沿用",
                payload=session.forecast_proposal.model_dump(mode="json"),
            )
            session.forecast_proposal = None
        if automatic:
            self.store.append_event(
                session.session_id,
                type="scope.auto_confirmed",
                stage="scope",
                status="completed",
                summary="自动模式采用当前任务范围；其中模型选择不等于用户明确指定或独立核验",
                payload={
                    "ticker": session.draft.ticker,
                    "valuation_date": str(session.draft.valuation_date or ""),
                    "methods": list(session.draft.methods),
                },
            )
        return (
            not session.data_source_preference
            and not session.documents
            and not any(fact.status == "confirmed" for fact in session.facts)
        )

    def _blocks(self, session):
        blocks = {
            b["block_id"]: _annotate_untrusted_source(b)
            for doc in session.documents
            for b in self.store.research_blocks(session.session_id, doc.file_id)
        }
        for message in self.store.list_messages(session.session_id):
            if message.role == "user":
                block_id = "message:" + message.message_id
                blocks[block_id] = {"block_id": block_id, "text": message.content,
                                    "location": {"message_id": message.message_id, "source_type": "user_note"}}
        return blocks

    def _deterministic_report_disclosure_shares(
        self, session, blocks, existing_facts
    ):
        """Extract an exact issuer total stated as of an official report date.

        Annual reports often put the year-end share table hundreds of pages
        away from a page-three sentence stating the newer issuer total as of
        disclosure.  LLM extraction can legitimately miss that second line.
        This narrow parser does not infer or add issuance tranches: it accepts
        only the report's explicit *total share capital*, binds the relative
        date to the official publication date, and sends the candidate through
        the same evidence validator as every model-proposed fact.
        """
        valuation_date = session.draft.valuation_date
        if not valuation_date:
            return []
        cutoff = session.information_cutoff_date or valuation_date
        documents = {document.file_id: document for document in session.documents}
        pattern = re.compile(
            r"(?:以)?截至本(?:年度)?报告披露之日[，,]?"
            r"(?:本公司|公司)(?:的)?总股本(?:为|是|共计|[:：])?"
            r"(?P<amount>\d{1,3}(?:[,，]\d{3})+|\d+)(?P<unit>股|万股|亿股)"
        )
        validated = []
        for block in blocks.values():
            location = block.get("location") or {}
            if location.get("source_type") == "web_search":
                continue
            file_id = str(
                block.get("file_id")
                or str(block.get("block_id") or "").split(":", 1)[0]
            )
            document = documents.get(file_id)
            source_url = str(
                location.get("source_url") or location.get("url") or ""
            )
            source_host = (urlsplit(source_url).hostname or "").rstrip(".").casefold()
            if (
                document is None
                or document.authority_tier != "A"
                or source_host not in self._DISCLOSURE_HOSTS
            ):
                continue
            try:
                published = date.fromisoformat(str(location.get("published_at"))[:10])
            except (TypeError, ValueError):
                continue
            if published > cutoff or published > valuation_date:
                continue
            compact_text = re.sub(r"\s+", "", str(block.get("text") or ""))
            for match in pattern.finditer(compact_text):
                item = CandidateInput(
                    metric="公司总股本",
                    standard_metric="common_shares",
                    semantic_role="equity",
                    ebit_treatment="exclude",
                    fcff_treatment="exclude",
                    equity_bridge_treatment="include",
                    mapping_confidence=1,
                    mapping_rationale=(
                        "交易所正式年报明确披露报告日发行人总股本；"
                        "系统仅提取总数，不采用扣除回购后的分红基数。"
                    ),
                    raw_value=match.group("amount"),
                    unit=match.group("unit"),
                    period=published.isoformat(),
                    scope="issuer",
                    role="historical",
                    block_id=str(block.get("block_id") or ""),
                    quote=match.group(0),
                )
                facts, _ = self._validate_candidates(session, [item], blocks)
                validated.extend(
                    (published, fact)
                    for fact in facts
                    if not fact.warnings
                )
        if not validated:
            return []
        latest_date = max(item[0] for item in validated)
        latest = [fact for published, fact in validated if published == latest_date]
        values = {fact.normalized_value for fact in latest}
        if len(values) != 1:
            self.store.append_event(
                session.session_id,
                type="facts.deterministic_conflict",
                stage="evidence",
                status="warning",
                summary="同一正式披露日存在多个发行人总股本数值；未自动选择。",
                payload={
                    "period": latest_date.isoformat(),
                    "block_ids": [fact.block_id for fact in latest],
                },
            )
            return []
        selected = latest[0]
        if any(
            mapped_financial_metric(fact) == "common_shares"
            and fact.scope == "issuer"
            and fact.period == selected.period
            and fact.normalized_value == selected.normalized_value
            and fact.status != "rejected"
            for fact in existing_facts
        ):
            return []
        return [selected]


    def _validate_candidates(self, session, items, blocks=None):
        from valuationagent.application.financial_evidence import PUBLIC_REVIEW, source_assessment

        blocks = blocks if blocks is not None else self._blocks(session)
        by_file = {}
        for block in blocks.values():
            by_file.setdefault(block["block_id"].rsplit(":", 1)[0], []).append(block)
        candidates = []
        rejected = []
        for item in items:
            supplied_block_id = item.block_id
            block = blocks.get(item.block_id)
            compact = lambda text: re.sub(r"\s+", "", text)
            if block and compact(item.quote) not in compact(block["text"]):
                # Models sometimes cite the page-continuation block while the
                # exact row sits in one explicitly supplied same-file context
                # block. Relocate only on a unique verbatim match; never fuzzy
                # search the whole filing or silently join separated excerpts.
                source_file = item.block_id.rsplit(":", 1)[0]
                relocated = [
                    blocks[key] for key in item.context_block_ids
                    if key in blocks
                    and key.rsplit(":", 1)[0] == source_file
                    and compact(item.quote) in compact(blocks[key]["text"])
                ]
                if len(relocated) == 1:
                    block = relocated[0]
                    item = item.model_copy(update={"block_id": block["block_id"]})
            if not block or compact(item.quote) not in compact(block["text"]):
                rejected.append(f"{item.metric}：引用不是当前资料中的连续原文；quote请仅复制科目所在原文行，表头用context_block_ids补充，不要拼接相隔的行")
                continue
            raw_value = item.raw_value.strip().replace("−", "-")
            # Validate the value before removing separators. Otherwise adjacent
            # report columns such as ``100 90`` collapse to ``10090`` and look
            # like one legitimate amount. Spaces/commas are accepted only when
            # they form conventional three-digit thousands groups.
            scalar_core = r"(?:\d+(?:\.\d+)?|\d{1,3}(?:[,，\s]+\d{3})+(?:\.\d+)?)"
            scalar_pattern = rf"[+-]?{scalar_core}"
            parenthetical = re.fullmatch(rf"\(\s*({scalar_core})\s*\)", raw_value)
            if not parenthetical and not re.fullmatch(scalar_pattern, raw_value):
                rejected.append(f"{item.metric}：原始值包含多个数字或不是单一有效数值")
                continue
            numeric_text = parenthetical.group(1) if parenthetical else raw_value
            clean_value = re.sub(r"[,，\s]", "", numeric_text)
            try:
                amount = Decimal(clean_value)
                if parenthetical:
                    amount = -amount
                if not amount.is_finite():
                    raise InvalidOperation()
            except InvalidOperation:
                rejected.append(f"{item.metric}：原始值包含多个数字或不是单一有效数值")
                continue
            # Preserve whitespace between adjacent report columns. Removing it
            # joined values such as ``1,286... 1,550...`` into one giant number
            # and falsely rejected the first, correctly quoted value.
            quoted = item.quote.replace("−", "-")
            # A positive candidate must not match the numeric suffix of a
            # negative source value (e.g. 12000 inside -12000).
            source_parts = re.split(r"([,，\s]+)", raw_value)
            source_pattern = "".join(
                r"[,，\s]+" if re.fullmatch(r"[,，\s]+", part) else re.escape(part)
                for part in source_parts
            )
            raw_metric = normalize_financial_metric(item.metric)
            proposed_standard = item.standard_metric.strip()
            metric = (
                normalize_financial_metric(proposed_standard)
                if proposed_standard and item.role == "historical"
                else raw_metric
            )
            if proposed_standard and item.role == "historical" and not metric:
                rejected.append(
                    f"{item.metric}：standard_metric不是系统支持的规范字段；"
                    "请保留原始科目名并改用可识别规范字段，或留空继续补读上下文"
                )
                continue
            value_appears_in_quote = bool(re.search(
                r"(?<![\d.+\-(])" + source_pattern + r"(?![\d.)])",
                quoted,
            ) or amount in numeric_tokens(quoted))
            blank_debt_zero_claim = (
                amount == 0 and metric in DEBT_COMPONENTS
                and item.role == "historical" and item.scope == "consolidated"
            )
            if not value_appears_in_quote and not blank_debt_zero_claim:
                rejected.append(f"{item.metric}：候选数值未出现在引用原文中")
                continue
            fact = FactCandidate(**item.model_dump(), fact_id=_id("fact_"))
            if issue := financial_mapping_issue(fact):
                fact.warnings.append(issue)
            if (
                item.role == "historical"
                and not proposed_standard
                and (
                    re.search(r"利息", item.metric)
                    or raw_metric in {
                        "cash_and_non_operating_assets",
                        "trading_financial_assets",
                    }
                    and item.metric.strip() != raw_metric
                )
            ):
                fact.warnings.append(
                    "上下文敏感科目缺少LLM语义映射：请补读关联附注并填写standard_metric、"
                    "semantic_role、模型处理、置信度和理由"
                )
            if proposed_standard:
                if item.semantic_role == "unknown":
                    fact.warnings.append("语义映射缺少经营/融资/金融子公司等经济角色")
                if item.mapping_confidence < 0.65:
                    fact.warnings.append("语义映射置信度低于模型准入阈值0.65，请补读关联附注")
                if len(item.mapping_rationale.strip()) < 12:
                    fact.warnings.append("语义映射理由不足，须说明表/附注、业务主体与相邻科目依据")
                if metric == "interest_expense" and (
                    item.semantic_role != "financing"
                    or item.ebit_treatment != "include"
                ):
                    fact.warnings.append(
                        "融资利息准入冲突：只有明确属于融资并作为EBIT加回输入的利息才可映射interest_expense"
                    )
                if metric == "financial_subsidiary_interest_expense" and (
                    item.semantic_role != "financial_subsidiary"
                    or "include" in {
                        item.ebit_treatment,
                        item.fcff_treatment,
                        item.equity_bridge_treatment,
                    }
                ):
                    fact.warnings.append(
                        "金融子公司经营利息只能作为模型外审计事实保留，不得混入普通工业企业EBIT/FCFF/权益桥接"
                    )
                if metric in DEBT_COMPONENTS and (
                    item.ebit_treatment != "exclude"
                    or item.fcff_treatment != "exclude"
                    or item.equity_bridge_treatment != "include"
                ):
                    fact.warnings.append(
                        "债务模型处理冲突：债务组成不进入EBIT或FCFF本体，只进入企业价值到股权价值桥接"
                    )
                if metric == "trading_financial_assets" and (
                    item.semantic_role != "non_operating"
                    or item.ebit_treatment != "exclude"
                    or item.fcff_treatment != "exclude"
                    or item.equity_bridge_treatment != "include"
                ):
                    fact.warnings.append(
                        "交易性金融资产模型处理冲突：须由关联附注证明其为非经营性资产，"
                        "不进入EBIT/FCFF本体，只进入企业价值到股权价值桥接"
                    )
                if metric == "cash_paid_for_ppe_intangibles" and (
                    item.semantic_role != "investing"
                    or item.ebit_treatment != "exclude"
                    or item.fcff_treatment != "include"
                    or item.equity_bridge_treatment != "exclude"
                ):
                    fact.warnings.append(
                        "资本开支基础科目模型处理冲突：投资活动现金支出应标记investing，"
                        "作为FCFF确定性推导输入，不进入EBIT或权益桥接"
                    )
                if (
                    raw_metric == "cash_paid_for_ppe_intangibles"
                    and metric != "cash_paid_for_ppe_intangibles"
                ):
                    fact.warnings.append(
                        "资本开支映射越级：原始现金流量表科目应保留为cash_paid_for_ppe_intangibles，"
                        "再由确定性组装器推导capital_expenditure并记录公式"
                    )
            if item.block_id.startswith("message:"):
                fact.source_type = "user_note"
            location = block.get("location") or {}
            file_id = item.block_id.rsplit(":", 1)[0]
            extra = [blocks.get(key) for key in item.context_block_ids]
            if any(b is None or b["block_id"].rsplit(":", 1)[0] != file_id for b in extra):
                rejected.append(f"{item.metric}：表头片段必须来自同一文件且存在于本次研究")
                continue
            file_blocks = by_file[file_id]
            context = evidence_context(block, file_blocks, item.context_block_ids)
            if item.scope == "issuer" and metric != "common_shares":
                fact.warnings.append("发行人口径只适用于可核验的普通股股份总数，其他财务科目需合并/母公司报表依据")
            binding_warnings, verification = bind_evidence(
                item, block, context, session.draft, METRIC_ALIASES.get(metric, ()),
                identity_text="\n".join(b["text"] for b in file_blocks[:8]),
                table_binding=self._table_binding(session, item, blocks),
            )
            block_lines = block["text"].splitlines()
            raw_key = compact(item.metric)
            row_indexes = [
                index for index, line in enumerate(block_lines)
                if raw_key and raw_key in compact(line)
            ]
            source_index = row_indexes[0] if len(row_indexes) == 1 else None
            adjacent_rows = (
                block_lines[max(0, source_index - 3):source_index + 4]
                if source_index is not None else []
            )
            verification["source_context"] = {
                "raw_metric": item.metric,
                "block_id": item.block_id,
                "supplied_block_id": supplied_block_id,
                "block_relocated_from_context": supplied_block_id != item.block_id,
                "context_block_ids": [b["block_id"] for b in context],
                "adjacent_rows": adjacent_rows,
                "location": location,
            }
            verification["semantic_mapping"] = {
                "standard_metric": proposed_standard,
                "resolved_metric": metric or "",
                "resolution_method": (
                    "llm_context_mapping" if proposed_standard else "exact_accounting_alias"
                ),
                "semantic_role": item.semantic_role,
                "ebit_treatment": item.ebit_treatment,
                "fcff_treatment": item.fcff_treatment,
                "equity_bridge_treatment": item.equity_bridge_treatment,
                "confidence": item.mapping_confidence,
                "rationale": item.mapping_rationale,
                "alternative_interpretations": [
                    alternative.model_dump(mode="json")
                    for alternative in item.alternative_interpretations
                ],
            }
            fact.verification = verification
            if fact.period == "unknown" and verification.get("period"):
                fact.period = str(verification["period"])
            if (not value_appears_in_quote
                    and not verification.get("source_blank_as_zero")):
                rejected.append(
                    f"{item.metric}：候选0未由完整资产负债表的目标年度空白金额格验证"
                )
                continue
            if fact.role == "historical" and metric and fact.unit != "unknown":
                allowed_units = ({"股", "千股", "万股", "百万股", "亿股"} if metric == "common_shares"
                                 else {"%", "ratio"} if metric in {"ebit_margin", "tax_rate"}
                                 else {"元", "千元", "万元", "百万元", "亿元"})
                if fact.unit not in allowed_units:
                    fact.warnings.append("字段计量维度不符：股数、金额和比率不能互相替代，请核对科目及单位")
            fact.context_block_ids = list(dict.fromkeys(b["block_id"] for b in context))[-8:]
            fact.source_location = location
            fact.source_url = location.get("source_url") or location.get("url") or ""
            document = next((d for d in session.documents if d.file_id == file_id), None)
            fact.source_sha256 = document.sha256 if document else ""
            if location.get("published_at"):
                try:
                    fact.published_at = date.fromisoformat(str(location["published_at"])[:10])
                except ValueError:
                    pass
            if location.get("source_type") == "web_search":
                fact.warnings.append("来源仅为联网搜索摘要，尚未读取并核对原始公告")
            if fact.role == "historical" and any(entry.get("location", {}).get("merged_cells") for entry in [block, *context]):
                fact.warnings.append("表格含未展开的合并或嵌套单元格，不能直接绑定扁平年度列；请换可明确对齐的来源")
            if (
                location.get("source_type") == "remote_web_document"
                and fact.role == "historical"
            ):
                fact.warnings.append(PUBLIC_REVIEW)
                if not fact.published_at:
                    fact.warnings.append("网页披露日期尚未核验；不能用抓取日冒充发布日期")
            if fact.role == "historical" and not verification.get("year_column") and re.search(
                r"上年度末|上年末|上年同期|去年同期|期初余额|比较期",
                item.quote,
            ):
                fact.warnings.append("数值来自比较列，需核对目标期间原始报表的表头和口径")
            factors = {"元": "1", "千元": "1000", "万元": "10000", "百万元": "1000000", "亿元": "100000000", "股": "1", "千股": "1000", "万股": "10000", "百万股": "1000000", "亿股": "100000000", "%": "0.01", "ratio": "1"}
            if fact.unit in factors:
                fact.normalized_value = str(amount * Decimal(factors[fact.unit]))
            if fact.unit == "unknown":
                fact.warnings.append("单位待确认")
            if fact.period == "unknown":
                fact.warnings.append("期间待确认")
            if fact.scope == "unknown" and fact.role == "historical":
                fact.warnings.append("合并/母公司口径待确认")
            fact.warnings = list(dict.fromkeys([*fact.warnings, *binding_warnings]))
            fact.verification["observation"] = {
                "status": "needs_repair" if binding_warnings else "verified",
                "issues": binding_warnings,
                "meaning": "原文数值、期间、单位、主体和口径绑定；不代表估值处理已通过。",
            }
            cutoff = _information_cutoff(session)
            if fact.published_at and cutoff and fact.published_at > cutoff:
                fact.warnings.append("来源披露日期晚于信息截止日")
            fact.verification["source_assessment"] = source_assessment(fact, document, cutoff)
            candidates.append(fact)
        return candidates, rejected

    @staticmethod
    def _table_binding(session, item, blocks):
        if not item.table_id:
            return None
        from valuationagent.core.table_interpretation import get_table_binding

        if item.role != "historical" or item.scope == "issuer":
            raise ValueError("TABLE_ROLE_UNSUPPORTED: 年度表格契约不能替代发行人股数截止日或可比倍数取证。")
        return get_table_binding(session, item.table_id, item.block_id, blocks)

    @staticmethod
    def _fact_identity(fact):
        # Do not equate different raw statement concepts merely because both
        # map to the same model input (e.g. revenue and total revenue).
        return (fact.metric, fact.period, fact.scope, fact.role,
                fact.peer_ticker or fact.peer_name, fact.multiple_basis, fact.unit)

    def _propose_forecast(self, session, args):
        from valuationagent.application.forecast_inputs import forecast_input_evidence

        if session.pending_action != "valuation" or "dcf" not in session.draft.methods:
            raise ValueError("只有已请求DCF估值时才能提出预测方案；不能自动改变估值方法")
        blocks = self._blocks(session)
        facts = {f.fact_id: f for f in session.facts if f.status != "rejected" and not f.warnings}
        input_ids = {ref.evidence_id for ref in forecast_input_evidence(session, args.evidence_ids)}
        for key in args.evidence_ids:
            if key in input_ids:
                continue
            if key in facts:
                cutoff = _information_cutoff(session)
                if facts[key].published_at and cutoff and facts[key].published_at > cutoff:
                    raise ValueError("预测依据的披露日晚于信息截止日，不得使用未来信息")
                continue
            source = blocks.get(key)
            if not source or source.get("location", {}).get("source_type") == "web_search":
                raise ValueError("预测依据必须关联已核验字段或原文，不能使用不存在的引用或搜索摘要")
            published = source.get("location", {}).get("published_at")
            cutoff = _information_cutoff(session)
            if published and cutoff and date.fromisoformat(str(published)[:10]) > cutoff:
                raise ValueError("预测依据的披露日晚于信息截止日，不得使用未来信息")
        session.forecast_proposal = ForecastProposal(proposal_id=_id("forecast_"), scope_key=scope_key(session),
            inputs=args.inputs, rationale=self._redact_text(args.rationale),
            risks=[self._redact_text(risk) for risk in args.risks], evidence_ids=args.evidence_ids)
        session.status = "collecting"
        self.store.append_event(session.session_id, type="valuation.forecast_proposed", stage="planning",
            status="completed", summary="已保存十年预测方案；自动模式可按已授权流程生成草案，审阅模式需用户批准；未修改历史事实",
            payload=session.forecast_proposal.model_dump(mode="json"))
        progress = valuation_progress(session, self.valuation_assembler)
        session.gaps = [progress["blocking_reason"]] if progress["blocking_reason"] else []
        return {"status": "forecast_staged", "progress": progress, "instruction": progress["instruction"]}

    def _register_attachments(self, session, file_ids):
        for file_id in dict.fromkeys(file_ids):
            if any(doc.file_id == file_id for doc in session.documents):
                continue
            try:
                meta = self.store.get_file(file_id)
            except (KeyError, ValueError):
                message = f"附件 {self._redact_text(file_id)} 的引用已失效或不存在；未读取该文件，已继续处理其他附件。需要时请重新上传。"
                if message not in session.gaps:
                    session.gaps.append(message)
                self.store.append_event(session.session_id, type="document.unavailable", stage="document", status="warning",
                    summary=message, payload={"file_id": self._redact_text(file_id)})
                continue
            self.store.save_research_blocks(session.session_id, meta["file_id"], [])
            session.documents.append(DocumentSummary(file_id=meta["file_id"], name=meta["original_name"],
                role=meta["role"], block_count=0, sha256=meta["sha256"], size_bytes=meta["size_bytes"],
                parse_status="pending", provenance_type="user_upload", authority_tier="B",
                source_confidence=0, provider="user_upload"))
        self.store.save_research(session)

    def _llm_turn(self, session, llm):
        from valuationagent.application.agent_runtime import WorkspaceAgentRuntime

        return WorkspaceAgentRuntime(
            self, session, getattr(self, "workspace_service", None)
        ).run(llm)

    def turn(self, session_id, turn: ResearchTurn, *, reserved=False, compact_result=False):
        request_id = turn.request_id
        if not reserved:
            request_id, created = self.reserve_turn(session_id, turn)
            if not created:
                return self.snapshot(session_id, compact=compact_result)
        owner = _id("lease_")
        if not self.store.acquire(session_id, owner):
            self.store.update_research_job(session_id, request_id, status="failed")
            raise ValueError("当前会话正在处理请求，请稍后重试。")
        self.store.update_research_job(session_id, request_id, status="running", stage="planning")
        self._execution.current = {"session_id": session_id, "request_id": request_id,
                                   "deadline": time.monotonic() + turn.time_budget_seconds}
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(5):
                self.store.heartbeat(session_id, owner)
                self.store.update_research_job(session_id, request_id)

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            session = self.store.get_research(session_id)
            if turn.language is not None:
                session.language = turn.language
            content = self._redact_text(turn.content.strip() or "请读取这些附件")
            self.store.add_message(session_id, "user", content, "agent")
            session.last_issue = None
            session.outcome_status = ""
            session.outcome_reason = ""
            self.store.append_event(session_id, type="turn.started", stage="agent",
                                    status="running", summary=content[:300])
            status = "completed"
            try:
                self._check_execution()
                from valuationagent.application.turn_control import prepare_turn

                self._register_attachments(session, turn.file_ids)
                llm = self._clients.get(session_id)
                if llm is not None:
                    prepare_turn(self, session, llm)
                else:
                    session.turn_control = None
                may_read = bool(session.turn_control and "files" in session.turn_control.effects)
                pending = [doc for doc in session.documents if may_read and (doc.parse_status == "pending"
                    or doc.parse_status == "unreadable" and doc.file_id in turn.file_ids)]
                for existing_doc in pending:
                    file_id = existing_doc.file_id
                    meta = self.store.get_file(file_id)
                    session.documents.remove(existing_doc)

                    def parse(meta=meta):
                        blocks, warnings = parse_document(meta, check_cancel=self._check_execution, pdf_page_limit=25)
                        self.store.save_research_blocks(session_id, meta["file_id"], blocks)
                        doc = DocumentSummary(file_id=meta["file_id"], name=meta["original_name"], role=meta["role"],
                                              block_count=len(blocks), sha256=meta["sha256"],
                                              size_bytes=meta["size_bytes"], warnings=warnings,
                                              parse_status="unreadable" if not blocks else "partial" if warnings else "parsed",
                                              provenance_type="user_upload", authority_tier="B",
                                              source_confidence=0.85, provider="user_upload")
                        session.documents.append(doc)
                        return doc.model_dump(mode="json")

                    try:
                        self._tool(session, "parse_document", {"file_id": file_id, "name": meta["original_name"]}, parse)
                    except LlmError:
                        raise  # User cancellation and execution budgets must still stop promptly.
                    except Exception as exc:  # One bad file must not discard the rest of a batch.
                        message = self._redact_text(str(exc))[:500]
                        warning = f"解析失败（{type(exc).__name__}）：{message}。此文件未进入财务计算。"
                        self.store.save_research_blocks(session_id, meta["file_id"], [])
                        session.documents.append(DocumentSummary(file_id=meta["file_id"], name=meta["original_name"], role=meta["role"],
                            block_count=0, sha256=meta["sha256"], size_bytes=meta["size_bytes"],
                            warnings=[warning], parse_status="unreadable",
                            provenance_type="user_upload", authority_tier="B",
                            source_confidence=0.2, provider="user_upload"))
                        self.store.append_event(session_id, type="document.unreadable", stage="document", status="warning",
                            summary=f"无法读取 {meta['original_name']}；继续处理其他来源", payload={"file_id": file_id, "warning": warning})
                if llm is None:
                    raise LlmError("MODEL_CONNECTION_REQUIRED: 工作区和附件已保存，请连接模型后继续。")
                result = self._llm_turn(session, llm)
                session.resume_context = result.get("_resume_context", {})
                self._say(session, result["answer"])
                self.store.append_event(session_id, type="turn.completed", stage="agent",
                    status="completed", summary="本轮任务已保存",
                    payload={"trace": result.get("_agent_trace", {}),
                             "evidence_ids": result.get("evidence_ids", [])})
            except Exception as exc:
                status = "cancelled" if str(exc).startswith("EXECUTION_CANCELLED") else "failed"
                visible = self._redact_text(str(exc)) if isinstance(exc, (LlmError, ValueError)) else "本轮执行失败，已保存进度；请重试或调整输入。"
                issue_code = visible.split(":", 1)[0][:100]
                protocol_issue = issue_code in {"TOOL_JSON_TRUNCATED", "TOOL_JSON_INVALID", "LLM_CONTEXT_LIMIT", "CONTEXT_BUDGET"}
                connection_issue = issue_code.startswith("LLM_") or issue_code in {"MODEL_CONNECTION_REQUIRED", "MODEL_SESSION_REVOKED"}
                if connection_issue or protocol_issue or issue_code in {"EXECUTION_TIME_LIMIT", "AGENT_STEP_LIMIT", "AGENT_NO_PROGRESS", "LIVE_TEST_BUDGET"}:
                    recent = [event for event in self.store.list_events(session_id) if event.type in {"tool.completed", "tool.failed"}][-5:]
                    session.resume_context = {
                        "reason": issue_code,
                        "request_id": request_id,
                        "last_tools": [{"tool": event.tool, "status": event.status, "summary": event.summary} for event in recent],
                        "next_steps": [step["title"] for step in session.plan if step.get("status") != "completed"][:5],
                        "instruction": ("模型思考耗尽了已配置的输出上限，尚未生成有效工具指令。先在模型连接中调大输出预算/上限并重新连接；保留原请求的联网限制，不重复取数，不改模型或思考模式。" if issue_code == "LLM_REASONING_LIMIT" else
                                        "模型输出/上下文协议失败，不是资料缺失。检查模型参数，缩小读取和提交批次，先处理已下载文件，不重放已经成功的工具。" if protocol_issue else
                                        "模型连接不可用或响应失败。先核对接口健康、真实模型ID和连接配置；恢复后继续已有资料，不把连接问题归为数据缺失，不自动切换模型。" if connection_issue else
                                        "保留原请求与联网/计算约束，先检查最近工具的具体失败原因和已保存输入；动作误判用revise_turn_plan修正，参数错误按原消息引用修复。不把执行停止误当资料缺失，不重复成功录入或下载。"),
                    }
                    visible = visible + " 已保存当前续做检查点；没有后台任务继续运行。" if connection_issue or protocol_issue else (
                        f"{issue_code}: 本轮已停止并保存续做检查点。"
                        f"已核验{sum(fact.status == 'confirmed' and not fact.warnings for fact in session.facts)}项事实，"
                        f"待修复{sum(fact.status == 'proposed' and bool(fact.warnings) for fact in session.facts)}项。"
                        f"另有{len(session.input_dataset.active_records()) if session.input_dataset else 0}项有效统一输入（用户输入不等于外部核验）。"
                        f"本次停止原因：{visible} "
                        "先处理具体失败，不重新搜集已提供数据；没有后台任务继续运行。"
                    )
                session.last_issue = ResearchIssue(
                    issue_id=_id("issue_"), code=issue_code,
                    stage="agent", message=visible[:1200],
                    retryable=not visible.startswith(("LLM_HTTP_401", "LLM_HTTP_403", "LLM_HTTP_402")),
                )
                self._say(session, visible)
                self.store.append_event(session_id, type="turn.interrupted", stage="agent",
                    status="failed", summary=visible[:1200],
                    payload={"error_type": type(exc).__name__})
            self.store.save_research(session)
            from valuationagent.application.result_document import ensure_result_document
            document = ensure_result_document(self, session)
            if status == "failed" or status == "completed" and session.resume_context:
                from valuationagent.application.workspace_artifacts import save_interruption_report

                try:
                    save_interruption_report(self, session, document, request_id)
                except (ValueError, OSError) as exc:
                    self.store.append_event(session_id, type="report.interruption_failed", stage="reporting", status="failed",
                        summary="执行状态与检查点已保存，但中断报告文件未生成",
                        payload={"request_id": request_id, "error": self._redact_text(str(exc))[:500]})
            self.store.update_research_job(session_id, request_id, status=status, stage="saved")
            return self.snapshot(session_id, compact=compact_result)
        finally:
            stop.set()
            worker.join(timeout=1)
            self.store.release(session_id, owner)
            job = self.store.research_job(session_id, request_id)
            if job and job["status"] == "running":
                self.store.update_research_job(session_id, request_id, status="failed")
            self._execution.current = None
