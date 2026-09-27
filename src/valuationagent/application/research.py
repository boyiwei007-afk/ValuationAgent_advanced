"""Shared, finance-independent research conversations for CLI and Web."""
import hashlib
import ipaddress
import json
import re
import socket
import threading
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import ClassVar, Literal
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpx
from pydantic import Field, ValidationError, field_validator, model_validator

from valuationagent.application.research_valuation import (
    METRIC_ALIASES,
    METRIC_LABELS,
    ResearchValuationAssembler,
    normalize_financial_metric,
    _period as financial_period,
)
from valuationagent.application.valuation_plan import scope_key, valuation_progress
from valuationagent.application.document_retrieval import rank_document_blocks
from valuationagent.core.data import LocalDataProvider
from valuationagent.core.documents import parse_document
from valuationagent.core.evidence import bind_evidence, evidence_context, numeric_tokens
from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec, canonical
from valuationagent.llm.agent import run_tool_loop
from valuationagent.llm.client import LlmError
from valuationagent.llm.context import (
    RESEARCH_PROMPT_VERSION,
    research_context,
    research_snapshot,
)
from valuationagent.llm.intent import interpret_intent, rejects_tushare
from valuationagent.schemas.agent import SearchQuery
from valuationagent.schemas.models import ApiModel
from valuationagent.schemas.research import (
    DocumentSummary,
    FactCandidate,
    ForecastInputs,
    ForecastProposal,
    ResearchChoice,
    ResearchDraft,
    ResearchIssue,
    ResearchMemoryItem,
    ResearchQuestion,
    ResearchSession,
    ResearchTurn,
)
from valuationagent.search.providers import (
    CninfoAnnouncementProvider,
    UnavailableSearchProvider,
)


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


class ProposeTask(ApiModel):
    draft: ResearchDraft


class ProposeForecast(ApiModel):
    inputs: ForecastInputs
    rationale: str = Field(min_length=20, max_length=2400, description="分清来源事实与预测判断，解释增长、利润率、WACC和永续增长依据；不得声称假设是历史事实。")
    risks: list[str] = Field(min_length=1, max_length=8)
    evidence_ids: list[str] = Field(min_length=1, max_length=30, description="关联已核验财务候选ID或已读取原文块ID；不能引用搜索摘要。")


class CandidateInput(ApiModel):
    metric: str = Field(
        min_length=1,
        max_length=120,
        description="来源中的字段名；只有映射证据明确时才使用标准字段名，歧义映射须先询问用户。",
    )
    raw_value: str = Field(
        min_length=1,
        max_length=100,
        description="原文中实际出现的数值，不得插值、推算或移到其他年份。",
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
        description="建议每批10至20项，最多80项。每项均须绑定原文、期间、单位和口径；不会截断或跳过证据校验。",
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


class FinishResponse(ApiModel):
    deliver_outcome: bool = Field(default=False, description="已实际读取或检索资料、必要数据仍不可得时设为true：交付控制器生成的缺口说明报告并结束本轮，不要求用户反复确认继续搜。若已有可执行方法，仍会优先集中确认并计算。")
    answer: str = Field(
        min_length=1,
        max_length=7000,
        description="面向用户的真实结论；资料异常时说明发现位置、不确定点、影响和已安全完成的部分。",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="支持回答的来源块 ID；不能引用未读取或不存在的来源。",
    )
    question: str = Field(
        default="",
        max_length=600,
        description="仅填写当前最小的阻塞问题；不得暗示系统已确定仍有歧义的字段或年份。",
    )
    options: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="可选的 2—3 个互斥方案，说明采用的口径或影响；没有合理候选时留空，让用户自由输入。",
    )
    memory_updates: list[MemoryUpdate] = Field(
        default_factory=list,
        max_length=8,
        description="只保存用户明确表达、且未来回合仍有用的目标、偏好、约束、决定或术语定义。",
    )
    memory_remove_keys: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="仅在用户明确撤回既有长期要求时填写对应 memory key。",
    )

    @classmethod
    def _choice_error(cls):
        return "需要用户决定时必须提供 2—3 个明确、互斥且可执行的选项。"

    @model_validator(mode="after")
    def validate_question_choices(self):
        if self.question and not 2 <= len(self.options) <= 3:
            raise ValueError(self._choice_error())
        if not self.question and self.options:
            raise ValueError("没有问题时不能单独提供选项。")
        return self


class SearchSources(ApiModel):
    query: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=500)
    purpose: Literal["company_profile", "financials", "comparables", "policy", "other"] = "other"
    as_of_date: date | None = None
    allowed_domains: list[str] = Field(default_factory=list, max_length=12)
    target_ticker: str = Field(default="", max_length=20, description="检索另一家公司的正式披露时，明确填写它的代码；不改变估值对象。不填且query含一个不同的六位代码时使用该代码。")


class FetchSearchSource(ApiModel):
    file_id: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "search_sources 返回的 web_* 来源 ID；官方披露可下载 PDF，"
            "其他公开 HTTPS 来源可读取 PDF、HTML 或纯文本正文。"
        ),
    )


class UpdateGaps(ApiModel):
    missing: list[str] = Field(default_factory=list, max_length=30)
    reason: str = Field(
        min_length=1,
        max_length=1000,
        description="说明实际缺失或冲突、来源位置及其影响；不得用默认值掩盖资料问题。",
    )


class ProposeDataSourceChange(ApiModel):
    source: Literal["online", "web", "upload"]
    reason: str = Field(min_length=1, max_length=500)


def _id(prefix):
    return prefix + uuid.uuid4().hex


def _text(session, zh, en):
    return en if session.language == "en-US" else zh


def _explicit_metric_requests(text):
    """Recognize a small, explicit list of valuation fields in a user turn."""
    compact = re.sub(r"[\s/／、·（）()_-]+", "", str(text or "")).casefold()
    targets = set(METRIC_LABELS)
    requested = []
    for metric_name in sorted(targets):
        aliases = METRIC_ALIASES.get(metric_name, {metric_name})
        if any(
            re.sub(r"[\s/／、·（）()_-]+", "", alias).casefold() in compact
            for alias in aliases
        ):
            requested.append(metric_name)
    return requested


def _explicit_fact_acceptance(text, *, valuation_review=False):
    """Only a complete affirmative command can accept an existing clean card."""
    if valuation_review and re.fullmatch(r"(?:请)?确认(?:整套)?估值方案(?:并(?:开始)?(?:计算|估值))?[。！!]*", re.sub(r"\s+", "", text)):
        return True
    return bool(re.fullmatch(
        r"(?:请)?(?:直接)?(?:全部|都)?(?:确认|同意采纳)"
        r"(?:(?:本批|这批|当前|上述|所有|全部)?(?:无警告|可确认|已核验)?"
        r"(?:候选字段|候选|字段|数据|提取结果))?"
        r"(?:[，,]*(?:并|然后)?(?:开始|运行|提交)(?:正式)?估值)?[。！!]*",
        re.sub(r"\s+", "", text),
    ))


def _valuation_finish_violation(answer, question="", options=(), *, public_research_authorized=False):
    """Reject model-authored UI that impersonates controller readiness.

    The language model may explain evidence and ask for a genuinely missing
    input, but only ``valuation_progress`` can declare a plan complete and only
    ``_request_valuation`` can create the combined approval card.
    """
    narrative = re.sub(r"\s+", "", str(answer or "")).casefold()
    decision = re.sub(
        r"\s+", "", "\n".join([str(question or ""), *(str(item) for item in options)])
    ).casefold()
    combined = narrative + "\n" + decision
    if decision and re.search(r"[\d,，]+(?:\.\d+)?(?:亿元|百万元|万元|千元|元|亿股|万股|千股|股|倍|%|％)", decision):
        return "普通澄清卡不能建议或选择具体财务数字；历史数值必须绑定原文，预测数字必须进入预测方案，最终统一确认。不能用选项点击绕过当前未通过的来源校验"
    if decision and re.search(
        r"(?:解析|绑定|校验).{0,12}(?:未通过|失败|不通过).{0,40}(?:裁定|核定|确认.{0,6}数值)|"
        r"我按你确认的数值提交|裁定并提交|"
        r"(?:营运资本变动|历史收入|历史债务).{0,110}(?:近似处理|假设替代|估计替代)", decision
    ):
        return "历史数值未通过来源校验时，不能让用户裁定模型建议的数字或用近似值绕过；请补正式证据，无法取得则交付说明报告"
    if public_research_authorized and decision and re.search(
        r"(?:允许|授权|同意|是否|要不要|请选择|继续).{0,50}"
        r"(?:联网|公开检索|巨潮|官方披露|搜索|检索|下载|补读|联网核验)|"
        r"(?:联网|公开检索|官方披露).{0,35}(?:许可|授权|允许|同意)", decision
    ):
        return "公开资料取证已获授权，不能再让用户批准联网核验、下载或继续补搜；应执行剩余取证，失败则交付说明报告"
    invalid_candidate_approval = (
        r"confirmed_fact_count",
        r"字段级确认",
        r"(?:界面|候选卡(?:片)?).{0,30}(?:确认|批准|采纳)",
        r"(?:确认|批准|采纳).{0,30}(?:待确认候选|候选字段|候选卡(?:片)?)",
    )
    if any(re.search(pattern, combined) for pattern in invalid_candidate_approval):
        return "候选确认必须由控制器生成可执行确认卡，不能要求用户到界面替模型改变确认计数"
    internal_continuation = (
        r"(?:读取|检索|工具调用).{0,16}(?:预算|配额).{0,16}(?:用尽|耗尽|不足)",
        r"(?:下一轮|下一步).{0,30}(?:继续|补读|修正|清理|收口|处理)",
        r"(?:是否(?:需要)?|要不要|需不需要).{0,30}(?:补取|下载|检索).{0,30}(?:年报|年度报告|历史资料|更多年度)",
    )
    if any(re.search(pattern, combined) for pattern in internal_continuation):
        return "读取配额和继续补证属于内部执行步骤，不能包装成需要用户选择的下一轮任务"
    false_ready = (
        r"(?:基期|输入|资料|模型).{0,18}(?:已|已经|均已|都已|现已)(?:提取|核验|补充)?(?:齐备|补全|完整|准备完成)",
        r"(?:已|已经|现已)(?:完成|具备).{0,12}(?:建模准备|估值准备|完整基期)",
        r"(?:ready|complete).{0,20}(?:valuation|model inputs|baseline)",
    )
    if any(re.search(pattern, combined) for pattern in false_ready):
        return "模型声称估值输入已齐备，但确定性准备检查尚未通过"
    if re.search(r"(?:确认|提交).{0,40}(?:整套)?估值方案", decision) or re.search(
        r"确认.{0,24}(?:进入|开始|直接).{0,16}(?:计算|估值|生成报告)", decision
    ):
        return "普通澄清卡不能冒充正式估值方案确认卡"
    if re.search(r"(?:有限|部分|带缺口|缺失项).{0,8}dcf|dcf.{0,12}(?:有限|部分|带缺口)", decision):
        return "DCF必需输入缺失时不能承诺输出有限或部分DCF"
    if re.search(
        r"(?:剔除|省略|跳过|不需|无需|不含|不依赖).{0,16}股数.{0,90}(?:dcf|估值|计算)|"
        r"(?:dcf|估值|计算).{0,50}(?:省略|跳过|不需|无需|不含|不依赖).{0,16}股数",
        decision,
    ):
        return "当前估值模型必需股数，不能通过澄清卡承诺省略必需输入后计算"
    if re.search(r"(?:采信|认定|确认|指定).{0,16}(?:普通股)?股数.{0,10}[\d,，.]+股", decision):
        return "历史股数必须绑定可核验来源；不能让用户点击模型建议的数字来替代证据核验"
    if re.search(
        r"(?:pe|市盈率).{0,30}(?:目标公司)?(?:收盘价|市价|总市值)|"
        r"(?:目标公司)?(?:收盘价|市价|总市值).{0,30}(?:pe|市盈率)",
        decision,
    ):
        return "目标公司市价不是PE相对估值的必要输入；应获取同期同口径可比公司倍数"
    return ""


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
    }

    def __init__(
        self,
        store,
        *,
        search_provider=None,
        official_search_provider=None,
        tool_providers=(),
        remote_transport=None,
    ):
        self.store = store
        self._clients = {}
        self._search_clients = {}
        self._market_clients = {}
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
            self.store.update_research_job(session_id, job["request_id"], status="interrupted")
            job["status"] = "interrupted"
        return {"request_id": job["request_id"], "status": job["status"], "stage": job["stage"],
                "active": job["status"] in {"queued", "running"}, "cancel_requested": bool(job["cancel_requested"]),
                "started_at": job["created"], "elapsed_seconds": round(time.time() - job["created"] if job["status"] in {"queued", "running"} else job["updated"] - job["created"], 1)}

    def reserve_turn(self, session_id, turn):
        self.store.get_research(session_id)
        self.execution_state(session_id)  # Reconcile an expired process lease.
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
            requires_model=llm is not None,
            prompt_version=RESEARCH_PROMPT_VERSION,
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
                payload={"provider": provider, "model": model, "prompt_version": RESEARCH_PROMPT_VERSION},
            )
        self._say(session, _text(session,
            "欢迎来到 ValuationAgent。输入公司和估值目标即可开始，也可以附上年报或财务表。我会依据资料策略读取文件、补齐数据并建模；关键输入和假设集中确认后计算。关键数据不可得时停止无效补搜，生成说明缺口与依据的报告。",
            "Welcome to ValuationAgent. Name the company and valuation goal; files are optional. I will read sources, gather required inputs and prepare a model for one combined review. If essential data remains unavailable, bounded research ends with an explanatory report."))
        return session

    def attach(self, session_id, llm):
        if self.store.active(session_id):
            raise ValueError("当前会话正在处理请求，请稍后连接。")
        session = self.store.get_research(session_id)
        self._clients[session_id] = llm
        session.requires_model = True
        session.model_provider, session.model_name = self._model_identity(llm)
        session.prompt_version = RESEARCH_PROMPT_VERSION
        self.store.append_event(
            session.session_id, type="model.attached", stage="configuration",
            status="completed", summary=f"{session.model_provider} / {session.model_name}",
            payload={"provider": session.model_provider, "model": session.model_name,
                     "prompt_version": RESEARCH_PROMPT_VERSION},
        )
        if session.last_issue and session.last_issue.code in {
            "LLM_HTTP_401", "LLM_HTTP_403", "MODEL_SESSION_REVOKED", "MODEL_CONNECTION_REQUIRED",
            "LLM_TIMEOUT", "LLM_CONNECTION_FAILED", "LLM_NETWORK_FAILED", "LLM_RESPONSE_INVALID_JSON",
        }:
            self._resolve_issue(session)
        self.store.save_research(session)

    def attach_search(self, session_id, provider):
        """Attach a session-only search provider without persisting its secret."""
        if self.store.active(session_id):
            raise ValueError("当前会话正在处理请求，请稍后连接搜索服务。")
        session = self.store.get_research(session_id)
        provider_id = str(getattr(provider, "provider_id", "") or "unknown")
        if provider_id == "unavailable" or not callable(getattr(provider, "search", None)):
            raise ValueError("联网搜索服务配置无效。")
        self._search_clients[session_id] = provider
        session.search_retry_epoch += 1
        self.store.save_research(session)
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
        if self.store.active(session_id):
            raise ValueError("当前会话正在处理请求，请稍后连接A股取数服务。")
        self.store.get_research(session_id)
        if not callable(getattr(provider, "resolve", None)):
            raise TypeError("A股取数服务配置无效。")
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
        search = self._search_clients.get(session_id, self.search_provider)
        market = self._market_clients.get(session_id, default_market)
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
                "available": bool(market and callable(getattr(market, "resolve", None)))
                and not isinstance(market, LocalDataProvider),
                "provider": market_version or "unavailable",
            },
        }

    @classmethod
    def _safe_disclosure_url(cls, raw_url):
        """Accept only direct HTTPS URLs on official A-share disclosure hosts."""
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
        """Download, parse and register an official PDF discovered by search."""
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
            if existing and (existing[0].get("location") or {}).get("parent_search_file_id") == file_id:
                return {
                    "status": "already_fetched",
                    "file_id": document.file_id,
                    "name": document.name,
                    "block_count": document.block_count,
                    "warnings": document.warnings,
                }

        source_host = (urlsplit(source_url).hostname or "").rstrip(".").casefold()
        official = source_host in self._DISCLOSURE_HOSTS
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
        blocks, warnings = parse_document(self.store.get_file(meta["file_id"]), check_cancel=self._check_execution)
        if not official:
            warnings = list(dict.fromkeys([
                (
                    "来源为公开网页原文；政策、业务与行业信息须结合发布日期和来源主体核验，"
                    "核心历史财务仍应以官方披露为准。"
                ),
                *warnings,
            ]))[:30]
        for block in blocks:
            block["location"] = {
                **(block.get("location") or {}),
                "source_type": "remote_document" if official else "remote_web_document",
                "source_url": resolved_url,
                "source_domain": urlsplit(resolved_url).hostname,
                "parent_search_file_id": file_id,
                "search_provider": location.get("provider"),
                "search_query": location.get("search_query"),
                "published_at": location.get("published_at"),
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
                "下一步调用 read_document 读取该 file_id；财务事实只能引用返回的原文 block_id。"
                if official else
                "下一步调用 read_document 阅读网页正文；用它分析政策、业务或行业，"
                "不要用非官方网页替代核心历史财务披露。"
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

    def submit_valuation(self, session_id, runner):
        """Create one auditable valuation run from the confirmed research state."""
        if self.store.active(session_id):
            raise ValueError("当前研究会话正在处理请求，请稍后提交估值。")
        session = self.store.get_research(session_id)
        request = self.valuation_assembler.build(session)
        parent_id = None
        if session.valuation_run_id:
            try:
                record = runner.store.get_run(session.valuation_run_id)
                if record.request == request:
                    if session_id in self._market_clients:
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
            self._clients.get(session_id),
            parent_id=parent_id,
            reason="研究会话输入更新后重新提交" if parent_id else None,
        )
        if session_id in self._market_clients:
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

    def _question(self, session, kind, title, choices, **kwargs):
        title = self._redact_text(title)
        session.question = ResearchQuestion(question_id=_id("question_"), kind=kind, title=title,
            options=[ResearchChoice(id=key, label=self._redact_text(label)) for key, label in choices], **kwargs)
        session.status = "waiting_confirmation"
        self.store.append_event(session.session_id, type="review.required", stage="research", status="waiting_confirmation",
                                summary=title, payload={"question_id": session.question.question_id})

    def _ask_data_source(self, session, title=None):
        """Require an explicit acquisition route before external data work."""
        self._question(
            session,
            "data_source",
            title or _text(
                session,
                "这次研究需要外部数据，你希望怎样提供？",
                "How would you like to provide the external data for this study?",
            ),
            [
                (
                    "online",
                    _text(
                        session,
                        "联网获取（Tushare 财务数据与 Tavily 公开资料）",
                        "Retrieve online (Tushare financials and Tavily public sources)",
                    ),
                ),
                (
                    "web",
                    _text(
                        session,
                        "不使用 Tushare；联网检索公开年报并逐项确认",
                        "Do not use Tushare; search public annual reports and confirm extracted facts",
                    ),
                ),
                (
                    "upload",
                    _text(
                        session,
                        "我来上传年报或财务数据文件",
                        "I will upload annual reports or financial data files",
                    ),
                ),
            ],
        )

    def _propose_data_source_change(self, session, source, reason):
        labels = {
            "online": _text(
                session,
                "恢复 Tushare 结构化取数，并在提交前检查权限",
                "Restore structured Tushare retrieval and check access before submission",
            ),
            "web": _text(
                session,
                "不使用 Tushare；联网检索公开年报并逐项确认",
                "Do not use Tushare; search public annual reports and confirm extracted facts",
            ),
            "upload": _text(
                session,
                "不使用 Tushare；改为上传年报或财务 Excel",
                "Do not use Tushare; upload annual reports or financial spreadsheets",
            ),
        }
        alternatives = [item for item in ("web", "upload", "online") if item != source]
        self._question(
            session,
            "data_source",
            _text(
                session,
                f"检测到你要更改数据来源：{reason}。正式估值所需的十年财务数据不能由网页搜索静默替代，请确认下一步。",
                f"You asked to change the data source: {reason}. Web search cannot silently replace the ten-year structured financial history required for valuation. Confirm the next step.",
            ),
            [(item, labels[item]) for item in (source, *alternatives)],
        )
        return {
            "_terminal": True,
            "answer": _text(
                session,
                "已停止自动提交，并暂停沿用上一轮的数据来源。请选择新的取数方式；也可以在 Chat 中说明其他要求。",
                "Automatic submission is paused and the previous data source will not be reused. Choose a new source or describe another requirement in Chat.",
            ),
        }

    @staticmethod
    def _contains_secret(text):
        return bool(re.search(r"(?i)\b(?:sk|key)-[a-z0-9_-]{12,}\b|\bbearer\s+[a-z0-9._-]{12,}", text))

    @staticmethod
    def _redact_text(text):
        return re.sub(
            r"(?i)\b(?:sk|key)-[a-z0-9_-]{12,}\b|\bbearer\s+[a-z0-9._-]{12,}",
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

    def _recover(self, session, exc, stage="unknown", context=None):
        raw = self._redact_text(str(exc)).strip()
        match = re.match(r"([A-Z][A-Z0-9_]+):", raw)
        code = match.group(1) if match else (
            "DOCUMENT_PROCESSING_FAILED" if stage == "document" else
            "INPUT_INVALID" if isinstance(exc, ValueError) else
            "AGENT_UNEXPECTED_ERROR"
        )
        if isinstance(exc, (ValueError, LlmError)):
            message = raw[:1200] or "当前步骤无法可靠完成。"
        elif isinstance(exc, OSError):
            message = "文件读取失败，请检查格式、权限或文件是否完整。"
        else:
            message = "处理过程中出现未预期问题，已停止当前步骤。"
        retryable = code not in {"LLM_HTTP_401", "LLM_HTTP_402", "LLM_HTTP_403", "MODEL_SESSION_REVOKED", "MODEL_CONNECTION_REQUIRED"}
        recovery_context = dict(context or {})
        if session.question and session.question.kind == "facts":
            recovery_context["pending_fact_review"] = session.question.model_dump(mode="json")
        elif session.last_issue and session.last_issue.context.get("pending_fact_review"):
            recovery_context["pending_fact_review"] = session.last_issue.context["pending_fact_review"]
        issue = ResearchIssue(
            issue_id=_id("issue_"),
            code=code,
            stage=stage,
            message=message,
            retryable=retryable,
            context=self._redact_value(recovery_context),
        )
        session.last_issue = issue
        if code in {"LLM_HTTP_401", "LLM_HTTP_402", "LLM_HTTP_403", "MODEL_SESSION_REVOKED", "MODEL_CONNECTION_REQUIRED"}:
            choices = [
                ("reconnect", _text(session, "重新配置模型后继续", "Reconnect a model and continue")),
                ("offline", _text(session, "切换为离线资料整理", "Switch to local preparation")),
                ("defer", _text(session, "保留进度，稍后处理", "Keep progress and decide later")),
            ]
        elif code in {"LLM_TIMEOUT", "LLM_CONNECTION_FAILED", "LLM_NETWORK_FAILED", "LLM_RESPONSE_INVALID_JSON"}:
            choices = [
                ("retry", _text(session, "重试模型当前步骤", "Retry the current model step")),
                ("reconnect", _text(session, "重新配置模型连接", "Reconnect the model")),
                ("defer", _text(session, "保留进度，稍后继续", "Keep progress and continue later")),
            ]
        elif code == "EVIDENCE_REPAIR_EXHAUSTED":
            choices = [
                ("retry", _text(session, "拆成更小步骤后重试", "Retry in smaller steps")),
                ("revise", _text(session, "我来补充资料或说明口径", "I will add material or clarify the scope")),
                ("defer", _text(
                    session,
                    "隔离失败候选，继续寻找替代证据",
                    "Quarantine failed candidates and continue with alternative evidence",
                )),
            ]
        elif code == "AGENT_NO_PROGRESS":
            choices = [
                ("retry", _text(session, "拆成更小步骤后重试", "Retry in smaller steps")),
                ("revise", _text(session, "我来补充资料或说明口径", "I will add material or clarify the scope")),
                ("defer", _text(session, "保留有效结果，跳过重复步骤", "Keep valid results and skip the repeated step")),
            ]
        else:
            choices = [
                ("retry", _text(session, "按现有资料重试当前步骤", "Retry with the current material")),
                ("revise", _text(session, "我来补充或修改要求", "I will clarify or revise the request")),
                ("defer", _text(session, "保留进度，跳过这一步", "Keep progress and skip this step")),
            ]
        if session.pending_action == "valuation":
            choices.insert(0, ("report", _text(session, "查看已保存的结果说明报告", "View the saved outcome report")))
        short_message = message[:360]
        title = _text(
            session,
            f"当前步骤未完成：{short_message} 已完成的资料和确认结果均已保留。" + ("可直接下载结果说明；补充资料或重试是可选操作。" if session.pending_action == "valuation" else "你希望怎样继续？"),
            f"This step did not complete: {short_message} Existing material and confirmed results were preserved. How should we continue?",
        )
        self._question(session, "recovery", title, choices)
        self.store.append_event(
            session.session_id,
            type="agent.recovery_required",
            stage=stage,
            status="waiting_confirmation",
            summary=message,
            payload={"issue_id": issue.issue_id, "code": code, "retryable": retryable},
        )
        if code == "EXECUTION_CANCELLED":
            return _text(session, "已停止本次执行。资料、候选字段和待确认事项均已保存，点击“继续上次任务”即可接着处理。", "Execution stopped. Sources, candidates and pending reviews are saved; resume whenever you are ready.")
        if code == "EXECUTION_TIME_LIMIT":
            return _text(session, "本轮已达到设置的时间预算，进度已保存。可以继续未完成步骤，不必重新上传资料。", "This turn reached its time budget. Progress is saved; continue the unfinished steps without uploading again.")
        if code == "AGENT_NO_PROGRESS":
            detail = message.partition(":")[2].strip() or "模型重复了同一项无效工具调用。"
            return _text(
                session,
                f"模型重复提交了未通过校验的工具调用，我已停止循环。最近失败原因：{detail} 当前有效资料和确认结果均已保留。",
                f"The model repeated a tool call that failed validation, so I stopped the loop. Latest failure: {detail} Valid material and confirmations are preserved.",
            )
        return _text(
            session,
            f"当前步骤已停止，未编造估值数字。问题代码：{code}。已有资料和进度已保存。" + ("结果说明报告可直接下载，包含缺口和失败原因。" if session.pending_action == "valuation" else "可选择下一步或补充说明。"),
            f"This step stopped without inventing valuation figures. Issue code: {code}. Progress is saved; an outcome report is available for valuation tasks.",
        )

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

    def _quarantine_candidates(self, session, candidate_ids, reason):
        """Remove unusable candidates from live blockers without erasing audit data."""
        ids = set(candidate_ids or [])
        quarantined = []
        for fact in session.facts:
            if fact.fact_id in ids and fact.status == "proposed" and fact.warnings:
                fact.status = "rejected"
                quarantined.append(fact)
        if not quarantined:
            return []
        labels = list(dict.fromkeys(
            METRIC_LABELS.get(normalize_financial_metric(fact.metric), fact.metric)
            for fact in quarantined
        ))
        gap = (
            "已隔离无法通过来源校验的候选：" + "、".join(labels)
            + "；若属于所选估值方法的必要输入，仍须从其他原文取得或由用户明确提供，不能按零处理。"
        )
        session.gaps = list(dict.fromkeys([*session.gaps, gap]))[:50]
        self.store.append_event(
            session.session_id,
            type="facts.quarantined",
            stage="research",
            status="completed",
            summary=f"隔离 {len(quarantined)} 个无法可靠绑定来源的候选",
            payload={
                "candidate_ids": [fact.fact_id for fact in quarantined],
                "metrics": [fact.metric for fact in quarantined],
                "reason": reason,
            },
        )
        return quarantined

    def _restore_pending_fact_review(self, session):
        if session.question or not session.last_issue or session.last_issue.status not in {"resolved", "deferred"}:
            return
        previous = session.last_issue.context.get("pending_fact_review") or {}
        facts = [f for f in session.facts if f.fact_id in previous.get("fact_ids", []) and f.status == "proposed"]
        if not facts:
            return
        eligible = sum(not f.warnings for f in facts)
        choices = [("defer", _text(session, "暂不确认，继续补充来源", "Keep pending and add sources")),
                   ("reject", _text(session, "拒绝本批候选，重新整理", "Reject this batch"))]
        if eligible:
            choices.insert(0, ("accept", _text(session, f"确认 {eligible} 个无警告字段", f"Confirm {eligible} clean candidate(s)")))
        self._question(session, "facts", _text(session, "继续核对暂停前的候选字段；已有确认结果不受影响。", "Review the candidates saved before the interruption; previous confirmations are unchanged."),
                       choices, fact_ids=[f.fact_id for f in facts], superseded_fact_ids=previous.get("superseded_fact_ids", []))

    def _propose_task(self, session, draft):
        draft = ResearchDraft.model_validate({**session.draft.model_dump(), **draft.model_dump(exclude_unset=True)})
        self._question(session, "task", _text(session, "请确认本次研究范围", "Confirm the research scope"),
            [("accept", _text(session, "采用这些设置", "Use these settings")),
             ("revise", _text(session, "我想修改", "I want to change this")),
             ("defer", _text(session, "暂不确定，继续整理资料", "Decide later; keep gathering material"))], proposed_draft=draft)
        return {"_terminal": True, "answer": _text(session,
            "已整理研究范围。请核对下方的公司、估值日和方法后选择，确认前不会覆盖当前设置。",
            "Review the proposed company, date and methods below. Current settings remain until you confirm.")}

    def _blocks(self, session):
        blocks = {b["block_id"]: b for doc in session.documents
                  for b in self.store.research_blocks(session.session_id, doc.file_id)}
        for message in self.store.list_messages(session.session_id):
            if message.role == "user":
                block_id = "message:" + message.message_id
                blocks[block_id] = {"block_id": block_id, "text": message.content,
                                    "location": {"message_id": message.message_id, "source_type": "user_note"}}
        return blocks

    def _metric_evidence_hints(self, session, metrics):
        """Return bounded source blocks that may contain an uncovered metric."""
        hints, used, seen = [], 0, set()
        documents = {document.file_id: document for document in session.documents}
        ranked = []
        for file_id, document in documents.items():
            for block in self.store.research_blocks(session.session_id, file_id):
                location = block.get("location") or {}
                if location.get("source_type") == "web_search":
                    continue
                text = str(block.get("text") or "")
                folded = text.casefold()
                for metric in metrics:
                    aliases = METRIC_ALIASES.get(metric, {metric})
                    matched = [alias for alias in aliases if alias.casefold() in folded]
                    if not matched:
                        continue
                    score = (
                        len(matched) * 10
                        + int(bool(re.search(r"\d", text))) * 4
                        + int(any(term in text for term in ("合并利润表", "合并财务报表", "单位：", "单位:"))) * 3
                        + int(location.get("source_type") == "remote_document") * 2
                    )
                    ranked.append((
                        -score,
                        metric,
                        block.get("block_id", ""),
                        document.name,
                        text,
                        location,
                    ))
        for _, metric, block_id, name, text, location in sorted(
            ranked, key=lambda row: row[:5]
        ):
            identity = (metric, block_id)
            if identity in seen:
                continue
            row = {
                "metric": metric,
                "label": METRIC_LABELS.get(metric, metric),
                "block_id": block_id,
                "document": name,
                "location": location,
                "text": text[:1800],
            }
            size = len(canonical(row))
            if hints and (len(hints) >= 8 or used + size > 14000):
                break
            hints.append(row)
            used += size
            seen.add(identity)
        return hints

    def _validate_candidates(self, session, items, blocks=None):
        blocks = blocks if blocks is not None else self._blocks(session)
        by_file = {}
        for block in blocks.values():
            by_file.setdefault(block["block_id"].rsplit(":", 1)[0], []).append(block)
        candidates = []
        rejected = []
        for item in items:
            block = blocks.get(item.block_id)
            compact = lambda text: re.sub(r"\s+", "", text)
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
            if not re.search(
                r"(?<![\d.+\-(])" + source_pattern + r"(?![\d.)])",
                quoted,
            ) and amount not in numeric_tokens(quoted):
                rejected.append(f"{item.metric}：候选数值未出现在引用原文中")
                continue
            fact = FactCandidate(**item.model_dump(), fact_id=_id("fact_"))
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
            metric = normalize_financial_metric(item.metric)
            if item.scope == "issuer" and metric != "common_shares":
                fact.warnings.append("发行人口径只适用于可核验的普通股股份总数，其他财务科目需合并/母公司报表依据")
            binding_warnings, verification = bind_evidence(
                item, block, context, session.draft, METRIC_ALIASES.get(metric, ()),
                identity_text="\n".join(b["text"] for b in file_blocks[:8]),
            )
            fact.verification = verification
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
            if (
                location.get("source_type") == "remote_web_document"
                and fact.role == "historical"
            ):
                fact.warnings.append(
                    "核心历史财务来自非官方网页；需改用交易所、巨潮或公司正式披露核验"
                )
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
            candidates.append(fact)
        return candidates, rejected

    @staticmethod
    def _fact_identity(fact):
        # Do not equate different raw statement concepts merely because both
        # map to the same model input (e.g. revenue and total revenue).
        return fact.metric, fact.period, fact.scope, fact.role, fact.peer_ticker

    def _facts(self, session, args, required_metrics=()):
        if session.pending_action == "valuation" and session.question and session.question.kind != "facts":
            raise ValueError("请先处理当前范围或授权问题，不能用财务候选覆盖该确认")
        old_facts = {f.fact_id: f for f in session.facts}
        candidates, rejected = self._validate_candidates(session, args.candidates)
        if not candidates:
            detail = "；".join(rejected[:5]) or "没有可核验的候选字段"
            raise ValueError("CANDIDATE_EVIDENCE_INVALID: " + detail)
        covered = {
            normalize_financial_metric(fact.metric)
            for fact in candidates
            if normalize_financial_metric(fact.metric)
        }
        explicitly_missing = set(_explicit_metric_requests(" ".join(args.missing)))
        uncovered = set(required_metrics) - covered - explicitly_missing
        if uncovered:
            return {
                "status": "incomplete_candidate_batch",
                "rejected_candidates": rejected,
                "validated_candidates": [
                    fact.model_dump(
                        mode="json",
                        include={
                            "metric", "raw_value", "unit", "period", "scope",
                            "role", "block_id", "quote",
                            "peer_ticker", "peer_name", "multiple_basis", "context_block_ids",
                        },
                    )
                    for fact in candidates
                ],
                "uncovered_metrics": [
                    {
                        "metric": metric,
                        "label": METRIC_LABELS.get(metric, metric),
                    }
                    for metric in sorted(uncovered)
                ],
                "suggested_source_blocks": self._metric_evidence_hints(
                    session, sorted(uncovered)
                ),
                "instruction": (
                    "候选尚未写入会话。请利用 suggested_source_blocks 核对未覆盖字段，"
                    "然后把 validated_candidates 与新增候选一起重新调用 propose_facts；"
                    "若仍没有可靠原文，在 missing 逐项说明已核对范围和原因。"
                ),
            }
        # A conflict is visible, and cannot be accepted by a blanket confirmation.
        identity = self._fact_identity
        correction_identity = lambda f: (f.metric, financial_period(f.period) or f.period, f.role, f.peer_ticker)
        unmatched = [key for key in args.replaces if key not in old_facts or not any(
            correction_identity(new) == correction_identity(old_facts[key]) for new in candidates)]
        if unmatched:
            # Never discard valid new evidence because another member of a
            # correction batch was rejected. Equally, never delete the old
            # field unless a matching valid replacement reaches final review.
            rejected += [f"更正未应用：{key} 没有对应字段、期间的有效新候选，旧值保持不变" for key in unmatched]
            args = args.model_copy(update={"replaces": [key for key in args.replaces if key not in unmatched]})
            self.store.append_event(session.session_id, type="facts.correction_rejected", stage="research", status="warning",
                summary="部分更正未匹配，保留旧字段并继续暂存其他候选", payload={"unmatched_replacement_ids": unmatched})
        for fact in candidates:
            for other in [*session.facts, *candidates]:
                if other.fact_id not in args.replaces and other.fact_id != fact.fact_id and other.status != "rejected" and (
                    other.metric, other.period, other.scope, other.role, other.peer_ticker
                ) == (fact.metric, fact.period, fact.scope, fact.role, fact.peer_ticker):
                    different = (Decimal(other.normalized_value) != Decimal(fact.normalized_value)) if other.normalized_value is not None and fact.normalized_value is not None else (other.raw_value, other.unit) != (fact.raw_value, fact.unit)
                    if different:
                        fact.warnings.append("同字段同期间存在冲突，需先更正候选")
                        break
        # Retrying the same extraction updates a pending record instead of
        # accumulating permanently blocking duplicates. Confirmed facts are
        # immutable here; changing their value still requires explicit review.
        persisted = []
        superseded = set(args.replaces)
        for fact in candidates:
            duplicate = next((old for old in session.facts if old.status == "proposed"
                              and identity(old) == identity(fact)
                              and old.raw_value == fact.raw_value and old.unit == fact.unit
                              and old.block_id == fact.block_id and old.quote == fact.quote), None)
            if duplicate:
                fact.fact_id = duplicate.fact_id
                session.facts[session.facts.index(duplicate)] = fact
            else:
                session.facts.append(fact)
            if fact.fact_id not in {f.fact_id for f in persisted}:
                persisted.append(fact)
            if not fact.warnings:
                superseded.update(old.fact_id for old in session.facts
                                  if old.status == "proposed" and old.fact_id != fact.fact_id
                                  and identity(old) == identity(fact)
                                  and old.normalized_value is not None and fact.normalized_value is not None
                                  and Decimal(old.normalized_value) == Decimal(fact.normalized_value))
        candidates = persisted
        superseded.difference_update(f.fact_id for f in candidates)
        rejected_gaps = ["候选未采纳：" + item for item in rejected]
        session.gaps = list(dict.fromkeys([*session.gaps, *args.missing, *rejected_gaps]))[:50]
        eligible = sum(not fact.warnings for fact in candidates)
        warned = len(candidates) - eligible
        if not eligible:
            # Feedback is for the reasoning engine, not an impossible user
            # confirmation card. The bounded controller decides when to ask
            # for help if improved evidence cannot be found.
            if session.question and session.question.kind == "facts":
                session.question = None
            session.status = "collecting"
            return {
                "status": "candidate_evidence_needs_repair",
                "candidates": [f.model_dump(mode="json") for f in candidates],
                "rejected_candidates": rejected,
                "instruction": (
                    "没有可确认字段，不要要求用户确认或重新说开始。请按warnings重新读取科目、"
                    "单位和年度表头（跨页时读取前页），或检索另一份正式披露。"
                    "context_block_ids只引用同文件真实表头；更正字段用replaces。"
                    "可先处理其他必要字段，无法取得证据时说明已核对范围并提出最小澄清。"
                ),
            }
        if session.pending_action == "valuation":
            for fact in candidates:
                if not fact.warnings:
                    replaced = [old.fact_id for old in session.facts if old.fact_id in superseded
                                and correction_identity(old) == correction_identity(fact)]
                    session.staged_supersessions[fact.fact_id] = sorted(set(
                        session.staged_supersessions.get(fact.fact_id, []) + replaced))
            # Evidence approval stays pending. Only a COPY is used to test model
            # readiness, so batch acquisition can continue without UI pauses.
            session.question = None
            session.status = "collecting"
            progress = valuation_progress(session, self.valuation_assembler)
            session.gaps = [progress["blocking_reason"]] if progress["blocking_reason"] else []
            return {"status": "valuation_inputs_staged", "eligible": eligible, "warned": warned,
                    "progress": progress, "candidates": [f.model_dump(mode="json") for f in candidates],
                    "rejected_candidates": rejected,
                    "instruction": progress["instruction"]}
        choices = [
            ("accept", _text(
                session,
                f"确认 {eligible} 个无警告字段" + (f"，保留 {warned} 个继续核对" if warned else ""),
                f"Confirm {eligible} clean candidate(s)" + (f"; keep {warned} for review" if warned else ""),
            )),
            ("reject", _text(session, "拒绝本批候选，重新整理", "Reject this batch")),
            ("defer", _text(session, "暂不确认，继续补充来源", "Keep all pending and add sources")),
        ]
        self._question(
            session,
            "facts",
            _text(
                session,
                f"本批 {len(candidates)} 个候选中，{eligible} 个可确认，{warned} 个仍需核对。",
                f"This batch contains {eligible} confirmable and {warned} pending candidate(s).",
            ),
            choices,
            fact_ids=[f.fact_id for f in candidates],
            superseded_fact_ids=sorted(superseded),
        )
        return {"_terminal": True, "answer": _text(session,
            f"已提取 {len(candidates)} 个候选字段，并保留原文引用。" +
            (f"另有 {len(rejected)} 个字段因证据不完整而隔离，未影响本批其他字段。" if rejected else "") +
            "带警告的字段不会被批量确认；确认仅代表采纳提取值，不代表财务审核通过。",
            f"Extracted {len(candidates)} candidates with source quotes. " +
            (f"{len(rejected)} unsupported candidates were isolated without discarding the valid batch. " if rejected else "") +
            "Warning-marked fields cannot be batch-confirmed. Confirmation accepts extraction; it is not a financial audit.")}

    def _refresh_pending_candidates(self, session, *, refresh_question=True):
        """Upgrade saved evidence checks without accepting or deleting facts."""
        pending = [f for f in session.facts if f.status == "proposed"]
        if not pending:
            return
        blocks = self._blocks(session)
        superseded = set(session.question.superseded_fact_ids) if session.question and session.question.kind == "facts" else set()
        superseded.update(old for new, ids in session.staged_supersessions.items()
                          if any(f.fact_id == new and f.status == "proposed" and not f.warnings for f in pending)
                          for old in ids)
        changes, repaired = [], []
        for old in pending:
            item = CandidateInput.model_validate(old.model_dump(include=set(CandidateInput.model_fields)))
            validated, rejected = self._validate_candidates(session, [item], blocks)
            if not validated:
                warnings = list(dict.fromkeys([*old.warnings, *rejected]))
                if old.warnings != warnings:
                    changes.append({"fact_id": old.fact_id, "before": old.warnings, "after": warnings})
                    old.warnings = warnings
                continue
            current = validated[0]
            current.fact_id = old.fact_id
            # A newly parsed quotation must not hide an existing conflicting
            # accepted or pending value for the same raw financial concept.
            if any(self._fact_identity(other) == self._fact_identity(current)
                   and other.fact_id != old.fact_id and other.status != "rejected"
                   and other.fact_id not in superseded
                   and other.normalized_value is not None and current.normalized_value is not None
                   and Decimal(other.normalized_value) != Decimal(current.normalized_value)
                   for other in session.facts):
                current.warnings.append("同字段同期间存在冲突，需先更正候选")
            if old.warnings != current.warnings or old.verification != current.verification:
                changes.append({"fact_id": old.fact_id, "before": old.warnings, "after": current.warnings,
                                "verification": current.verification})
                if old.warnings and not current.warnings:
                    repaired.append(current.fact_id)
            session.facts[session.facts.index(old)] = current
        if changes:
            self.store.append_event(session.session_id, type="facts.revalidated", stage="research",
                status="completed", summary=f"重新核验 {len(changes)} 个候选；没有自动确认任何数值",
                payload={"changes": changes})
        if not refresh_question:
            return
        question = session.question
        if session.pending_action == "valuation":
            if question and question.kind == "facts" and not question.valuation_review:
                session.question = None
                session.status = "collecting"
                self.store.append_event(session.session_id, type="valuation.batch_staged", stage="planning",
                    status="completed", summary="旧字段确认转入持续建模，保留候选且没有自动确认")
            return
        if question and question.kind != "facts":
            return
        if question and any(f.fact_id in question.fact_ids and not f.warnings
                            for f in session.facts):
            if any(option.id == "accept" for option in question.options):
                return
        elif question:
            session.question = None
            session.status = "collecting"
            self.store.append_event(session.session_id, type="review.repair_resumed", stage="research",
                status="completed", summary="不可确认的旧候选转回自动补证流程",
                payload={"question_id": question.question_id, "fact_ids": question.fact_ids})
        if repaired or question:
            needed = (self.valuation_assembler.pending_blockers(session)
                      if session.pending_action == "valuation" else session.facts)
            clean = [f for f in needed if f.status == "proposed" and not f.warnings]
            if clean:
                self._question(session, "facts", f"重新核验后，{len(clean)} 个字段可以确认；其余继续补证。",
                    [("accept", f"确认 {len(clean)} 个已核验字段并继续"),
                     ("defer", "继续查找资料，暂不确认"), ("reject", "拒绝这批候选")],
                    fact_ids=[f.fact_id for f in clean])

    def _answer_question(self, session, turn):
        question = session.question
        if question is None or turn.question_id != question.question_id:
            raise ValueError("这个确认问题已失效，请刷新后选择当前问题。")
        if turn.option_id not in {o.id for o in question.options}:
            raise ValueError("未知选项，请使用当前问题提供的选项。")
        ask_data_source_after = False
        if turn.option_id == "retry" and question.kind in {"search_failed", "clarification"}:
            session.search_retry_epoch += 1
        if question.kind == "task" and turn.option_id == "accept":
            previous_company = (session.draft.company, session.draft.ticker)
            session.draft = question.proposed_draft.model_copy(deep=True)
            session.valuation_methods_override = []
            session.valuation_method_exclusions = {}
            if any(previous_company) and previous_company != (session.draft.company, session.draft.ticker):
                for fact in session.facts:
                    if fact.status == "confirmed":
                        fact.status = "proposed"
                        fact.warnings.append("研究公司已变更，请重新核对该字段的归属")
            ask_data_source_after = (
                not session.data_source_preference
                and not session.documents
                and not any(fact.status == "confirmed" for fact in session.facts)
            )
        if question.kind == "data_source":
            session.data_source_preference = turn.option_id
            session.valuation_methods_override = []
            session.valuation_method_exclusions = {}
        if question.kind in {"search_unavailable", "search_failed"} and turn.option_id == "upload":
            session.data_source_preference = "upload"
        if question.kind != "facts" and session.forecast_proposal and session.forecast_proposal.scope_key != scope_key(session):
            self.store.append_event(session.session_id, type="valuation.forecast_invalidated", stage="planning",
                status="completed", summary="用户改变估值范围或来源，旧预测留在审计记录，不沿用到新模型",
                payload=session.forecast_proposal.model_dump(mode="json"))
            session.forecast_proposal = None
        fact_resolution = None
        if question.kind == "facts":
            if question.valuation_review and turn.option_id == "accept":
                current = valuation_progress(session, self.valuation_assembler)
                if not current["ready_for_review"] or current != question.valuation_review:
                    raise ValueError("估值方案或证据已变化，请重新生成集中确认卡；本次没有确认或计算")
                session.valuation_methods_override = list(current["methods"])
                session.valuation_method_exclusions = dict(current.get("excluded_methods", {}))
                proposal = session.forecast_proposal
                if proposal:
                    proposal.status = "confirmed"
                self.store.append_event(session.session_id, type="valuation.plan_confirmed", stage="planning",
                    status="completed", summary="用户确认整套估值输入及预测假设",
                    payload={"forecast_proposal_id": current["forecast_proposal_id"], "fact_ids": question.fact_ids,
                             "requested_methods": current.get("requested_methods", current["methods"]),
                             "effective_methods": current["methods"],
                             "excluded_methods": current.get("excluded_methods", {})})
            elif question.valuation_review and turn.option_id == "reject":
                session.forecast_proposal = None
                session.valuation_methods_override = []
                session.valuation_method_exclusions = {}
            batch = [fact for fact in session.facts if fact.fact_id in question.fact_ids]
            for fact in session.facts:
                if fact.fact_id in question.fact_ids:
                    if turn.option_id == "accept" and not fact.warnings:
                        fact.status = "confirmed"
                    elif turn.option_id == "reject":
                        fact.status = "rejected"
            accepted_metrics = {(f.metric, f.period, f.role, f.peer_ticker) for f in session.facts if f.fact_id in question.fact_ids and f.status == "confirmed"}
            if turn.option_id == "accept":
                for fact in session.facts:
                    if fact.fact_id in question.superseded_fact_ids and (fact.metric, fact.period, fact.role, fact.peer_ticker) in accepted_metrics:
                        fact.status = "rejected"
            fact_resolution = {
                "confirmed": sum(fact.status == "confirmed" for fact in batch),
                "pending": sum(fact.status == "proposed" for fact in batch),
                "rejected": sum(fact.status == "rejected" for fact in batch),
            }
        quarantined = []
        if question.kind == "recovery":
            if turn.option_id == "retry" and session.last_issue:
                session.last_issue.status = "retrying"
            elif turn.option_id == "offline":
                session.requires_model = False
                self._clients.pop(session.session_id, None)
                self._resolve_issue(session)
            elif (
                turn.option_id == "defer"
                and session.last_issue
                and session.last_issue.code == "EVIDENCE_REPAIR_EXHAUSTED"
            ):
                quarantined = self._quarantine_candidates(
                    session,
                    session.last_issue.context.get("candidate_ids", []),
                    "用户选择隔离证据修复失败的候选",
                )
                self._resolve_issue(session)
            elif turn.option_id in {"defer", "revise"}:
                self._resolve_issue(session, "deferred")
        label = next(o.label for o in question.options if o.id == turn.option_id)
        session.question = None
        session.status = "collecting"
        self.store.append_event(session.session_id, type="review.resolved", stage="research", status="completed", summary=label,
            payload={"question_id": question.question_id, "option_id": turn.option_id,
                     "fact_ids": question.fact_ids, "superseded_fact_ids": question.superseded_fact_ids})
        if turn.option_id == "revise" and question.kind != "recovery":
            return _text(session, "请直接输入希望修改的公司、日期或研究方法。当前设置尚未更改。", "Describe the company, date or method changes. Current settings have not changed.")
        if question.kind == "recovery":
            if turn.option_id == "retry":
                return _text(session, "正在按已保存的上下文重试；不会重复采纳已确认字段。", "Retrying from saved context without re-accepting confirmed facts.")
            if turn.option_id == "reconnect":
                return _text(session, "请重新连接模型后继续。CLI 使用 /connect；网页使用“连接模型”。当前进度不会丢失。", "Reconnect the model, then continue. Current progress is preserved.")
            if turn.option_id == "offline":
                return _text(session, "已切换为离线资料整理模式。仍可上传、预览、确认范围和导出记录。", "Switched to local preparation. Uploads, previews, scope confirmation and exports remain available.")
            if turn.option_id == "revise":
                return _text(session, "请直接输入补充信息或修改要求，我会从已保存的进度继续。", "Type the additional information or revised request, and I will continue from saved progress.")
            if quarantined:
                return _text(
                    session,
                    f"已隔离 {len(quarantined)} 个无法通过来源校验的候选；它们不会进入估值，也不会再阻塞其他有效字段。"
                    "若该指标是必要输入，Agent 将继续寻找替代原文，找不到时再请你提供最小必要信息。",
                    f"Quarantined {len(quarantined)} candidate(s) that failed source validation. "
                    "They cannot enter the valuation or block other valid inputs. The Agent will seek alternative evidence for any still-required metric.",
                )
            return _text(session, "已保留当前进度。你可以继续提出其他问题或补充资料。", "Progress is preserved. You can continue with another question or add material.")
        if question.kind == "facts" and fact_resolution is not None:
            if turn.option_id == "accept":
                if question.valuation_review:
                    return _text(session, "整套估值方案已确认，正在进入计算、敏感性分析和报告生成。", "Valuation inputs and assumptions confirmed. Continuing to calculation, sensitivity analysis and reporting.")
                return _text(
                    session,
                    f"本批实际确认 {fact_resolution['confirmed']} 个字段；"
                    f"{fact_resolution['pending']} 个带警告字段仍未进入估值。可继续补充来源、让 Agent 更正，或拒绝这些候选。",
                    f"Confirmed {fact_resolution['confirmed']} field(s). "
                    f"{fact_resolution['pending']} warning-marked field(s) remain excluded from valuation. Add sources, ask the Agent to revise them, or reject them.",
                )
            if turn.option_id == "reject":
                return _text(
                    session,
                    f"已拒绝本批 {fact_resolution['rejected']} 个候选，它们不会进入估值。",
                    f"Rejected {fact_resolution['rejected']} candidate(s); they will not enter valuation.",
                )
            return _text(
                session,
                f"已保留 {fact_resolution['pending']} 个待核对候选，均不会在确认前进入估值。",
                f"Kept {fact_resolution['pending']} candidate(s) pending; none enter valuation before confirmation.",
            )
        if ask_data_source_after:
            self._ask_data_source(session)
            return _text(
                session,
                "研究范围已确认。开始整理外部数据前，请先选择资料来源。",
                "The research scope is confirmed. Choose a data source before gathering external data.",
            )
        if question.kind == "data_source":
            if turn.option_id == "online":
                return _text(
                    session,
                    "已选择联网获取。Agent 将按需调用 Tushare 与 Tavily，并保留来源；服务未配置或鉴权失败时会立即说明。",
                    "Online retrieval selected. The Agent may use Tushare and Tavily with source records, and will report missing configuration or authentication failures.",
                )
            if turn.option_id == "web":
                return _text(
                    session,
                    "已选择公开资料自动检索。A股年报优先使用无需密钥的官方公告目录，Tavily 补充检索；可靠字段与预测假设集中确认后计算。若关键数据不可得，仍会生成说明报告。",
                    "Web research selected without Tushare. The Agent will use Tavily to find public reports and reliable sources. Extracted values remain candidates until each period, field, unit and scope is evidenced and confirmed; valuation starts only after complete financial snapshots are assembled.",
                )
            return _text(
                session,
                "已选择自行上传。请上传年报、财务 Excel 或其他原始资料；在你明确切换前，Agent 不会用联网结果替代这些资料。",
                "Upload selected. Add annual reports, financial spreadsheets or other source material; the Agent will not replace them with online results unless you explicitly switch.",
            )
        if turn.option_id == "report":
            return _text(session, "已结束本轮检索并生成结果说明。可下载 PDF、HTML 或 JSON；以后补充资料仍可继续。", "Retrieval stopped and an outcome report is available as PDF, HTML or JSON. Add data later to continue.")
        return _text(session, "已记录你的选择。可以继续上传资料、补充要求，或输入 /prepare 检查资料缺口。", "Your choice is saved. Add material or requirements, or use /prepare to check gaps.")

    def _prepare(self, session):
        blocking = []
        if not (session.draft.company or session.draft.ticker):
            blocking.append("研究公司尚未确认")
        if session.draft.valuation_date is None:
            blocking.append("估值基准日尚未确认")
        if not session.draft.methods:
            blocking.append("估值方法尚未确认")
        if not session.data_source_preference:
            blocking.append("尚未确认资料来源（联网获取或自行上传）")
        has_ticker = bool(session.draft.ticker)
        if session.data_source_preference in {"upload", "web"} and not any(
            f.status == "confirmed" for f in session.facts
        ):
            blocking.append(
                "已选择自行上传，但尚无可用于估值的已确认财务字段"
                if session.data_source_preference == "upload"
                else "已选择联网检索，但尚无可用于估值的已确认财务字段"
            )
        elif session.data_source_preference == "online" and not has_ticker and not any(
            f.status == "confirmed" for f in session.facts
        ):
            blocking.append("尚无已确认财务字段")
        online_ticker = session.data_source_preference == "online" and has_ticker
        if self.valuation_assembler.pending_blockers(session) and not online_ticker:
            blocking.append("仍有待确认的候选字段")
        if session.data_source_preference in {"upload", "web"} and any(
            f.status == "confirmed" for f in session.facts
        ):
            readiness_error = self.valuation_assembler.structured_readiness_error(session)
            if readiness_error:
                blocking.append(readiness_error)
        generated = {
            "研究公司尚未确认", "估值基准日尚未确认", "估值方法尚未确认",
            "尚未确认资料来源（联网获取或自行上传）",
            "已选择自行上传，但尚无可用于估值的已确认财务字段",
            "已选择联网检索，但尚无可用于估值的已确认财务字段",
            "尚无已确认财务字段", "仍有待确认的候选字段",
        }
        if not blocking:
            try:
                self.valuation_assembler.build(session)
            except ValueError as exc:
                blocking.append(str(exc))
        session.status = (
            "waiting_confirmation" if session.question is not None
            else "collecting" if blocking
            else "ready_for_valuation"
        )
        research_gaps = [
            gap for gap in session.gaps
            if gap not in generated
            and not gap.startswith("已确认字段尚不能组成完整年度快照")
            and not gap.startswith("已确认字段中没有可识别的完整年度财务快照")
        ]
        session.gaps = list(dict.fromkeys([*blocking, *research_gaps]))
        if not blocking and online_ticker:
            pending = sum(fact.status == "proposed" for fact in session.facts)
            return {"_terminal": True, "answer": _text(
                session,
                "资料准备检查：已具备正式在线取数条件。\n"
                "• 正式估值将按 A 股代码通过 Tushare 获取点时结构化财务数据和可比公司。\n"
                + (f"• 当前 {pending} 个未确认的搜索候选只保留为研究记录，不会进入正式计算。\n" if pending else "")
                + ("• 仍有研究资料缺口，但不会覆盖或替代 Tushare 数据。\n" if research_gaps else "")
                + "已经可以提交正式估值。你可以直接说“开始正式估值”；若 Tushare 尚未配置，系统会在取数前提示连接。",
                "Preparation check: ready for formal online data retrieval. The valuation will use point-in-time Tushare data and peers. "
                "Unconfirmed search candidates remain research notes and are excluded from calculations. You can now say “run the formal valuation”.")}
        return {"_terminal": True, "answer": _text(session,
            "资料准备检查：\n" + ("\n".join("• " + g for g in blocking) or "已确认研究范围及字段。") +
            ("\n可以提交正式估值；提交后会执行财务审核、十年预测、DCF、相对估值和敏感性分析。" if not blocking else "\n请先解决上述阻塞项目，再提交正式估值。"),
            "Preparation check: " + ("; ".join(blocking) or "Scope and facts confirmed.") +
            ("\nReady to submit the formal valuation workflow." if not blocking else "\nResolve these blocking items before formal valuation."))}

    def _request_valuation(self, session):
        """Validate readiness and request a controller-owned valuation handoff."""
        session.pending_action = "valuation"
        if session.question:
            return {"_terminal": True, "answer": session.question.title}
        progress = valuation_progress(session, self.valuation_assembler)
        proposal = session.forecast_proposal
        needs_review = bool(
            progress["staged_fact_ids"]
            or (proposal and proposal.status == "proposed")
            or (
                progress.get("degraded")
                and not session.valuation_methods_override
            )
        )
        if progress["ready_for_review"] and needs_review:
            ids = progress["staged_fact_ids"]
            superseded = sorted({old for key in ids for old in session.staged_supersessions.get(key, [])})
            self._question(session, "facts", _text(session,
                "估值模型输入已齐备。确认整套财务输入与预测假设后，直接计算并生成报告。",
                "Model inputs are ready. Review the inputs and assumptions, then calculate and generate the report."),
                [("accept", _text(session, "确认估值方案并计算", "Confirm plan and calculate")),
                 ("defer", _text(session, "先修改方案", "Revise the plan first"))],
                fact_ids=ids, superseded_fact_ids=superseded, valuation_review=progress)
            session.gaps = []
            return {"_terminal": True, "answer": _text(session,
                f"已完成建模准备：{' / '.join(progress['methods']).upper()}，基期 {progress['baseline_period']}。"
                + (
                    "以下原选方法因可靠数据不足不进入本次计算："
                    + "；".join(f"{method.upper()}（{reason}）" for method, reason in progress.get("excluded_methods", {}).items())
                    + "。\n"
                    if progress.get("excluded_methods") else ""
                )
                + f"本次集中复核 {len(ids)} 个财务输入。\n预测依据：{progress['forecast_rationale']}\n"
                + ("主要风险：" + "；".join(progress['risks']) + "\n" if progress['risks'] else "")
                + "预测属于假设，不是历史事实。确认后自动执行财务校验、估值区间、敏感性分析及报告生成。",
                "Model preparation is complete. Review the financial inputs and explicit forecast assumptions below; confirmation starts validation, valuation, sensitivity analysis and reporting.")}
        preparation = self._prepare(session)
        if session.status != "ready_for_valuation":
            return {
                "status": "valuation_inputs_missing",
                "progress": progress,
                "answer": preparation["answer"] + _text(
                    session,
                    "\n尚未提交计算，但估值目标已经记录。连接模型后 Agent 会从当前缺口继续检索、读取和提取；需要你确认时会停下，确认后自动续接，无需再次输入“开始估值”。",
                    "\nNo calculation was submitted, but the valuation goal is saved. With a model connected, the Agent will continue research from these gaps, pause only for required confirmation, and resume automatically afterward; you do not need to ask again.",
                ),
            }
        return {
            "_terminal": True,
            "_action": {"type": "submit_valuation"},
            "answer": _text(
                session,
                "已确认开始正式估值，正在交给确定性流水线执行在线取数、财务校验和模型计算。",
                "Formal valuation confirmed. Handing off to the deterministic pipeline for data retrieval, validation and model calculation.",
            ),
        }

    def _propose_forecast(self, session, args):
        if session.pending_action != "valuation" or "dcf" not in session.draft.methods:
            raise ValueError("只有已请求DCF估值时才能提出预测方案；不能自动改变估值方法")
        if session.question and session.question.kind != "facts":
            raise ValueError("请先确认研究范围或授权选择，不能用预测方案覆盖当前问题")
        blocks = self._blocks(session)
        facts = {f.fact_id: f for f in session.facts if f.status != "rejected" and not f.warnings}
        for key in args.evidence_ids:
            if key in facts:
                if facts[key].published_at and session.draft.valuation_date and facts[key].published_at > session.draft.valuation_date:
                    raise ValueError("预测依据的披露日晚于估值日，不得使用未来信息")
                continue
            source = blocks.get(key)
            if not source or source.get("location", {}).get("source_type") == "web_search":
                raise ValueError("预测依据必须关联已核验字段或原文，不能使用不存在的引用或搜索摘要")
            published = source.get("location", {}).get("published_at")
            if published and session.draft.valuation_date and date.fromisoformat(str(published)[:10]) > session.draft.valuation_date:
                raise ValueError("预测依据的披露日晚于估值日，不得使用未来信息")
        session.forecast_proposal = ForecastProposal(proposal_id=_id("forecast_"), scope_key=scope_key(session),
            inputs=args.inputs, rationale=self._redact_text(args.rationale),
            risks=[self._redact_text(risk) for risk in args.risks], evidence_ids=args.evidence_ids)
        session.question = None
        session.status = "collecting"
        self.store.append_event(session.session_id, type="valuation.forecast_proposed", stage="planning",
            status="completed", summary="模型提出十年三情景预测，等待整套方案确认；未修改历史事实",
            payload=session.forecast_proposal.model_dump(mode="json"))
        progress = valuation_progress(session, self.valuation_assembler)
        session.gaps = [progress["blocking_reason"]] if progress["blocking_reason"] else []
        return {"status": "forecast_staged", "progress": progress, "instruction": progress["instruction"]}

    def _advance_pending_valuation(self, session, llm, planning_hint, acknowledgement=""):
        """Keep a user-requested valuation moving across confirmation turns."""
        self._prepare(session)
        progress = valuation_progress(session, self.valuation_assembler)
        if session.question is not None:
            result = {
                "_terminal": True,
                "answer": _text(
                    session,
                    "估值目标已经记录。请先处理当前这项确认；提交选择后我会自动继续检索、补齐资料或进入计算，不需要再次发出开始指令。",
                    "The valuation goal is saved. Resolve the current confirmation and I will automatically continue research or submit the calculation; no second start command is needed.",
                ),
            }
        elif progress["ready_for_review"] or session.status == "ready_for_valuation":
            result = self._tool(
                session,
                "request_formal_valuation",
                {},
                lambda: self._request_valuation(session),
            )
        elif llm is not None:
            hint = dict(planning_hint or {})
            hint.update({
                "pending_action": "valuation",
                "preparation_status": session.status,
                "preparation_gaps": list(session.gaps),
                "controller_instruction": (
                    "Continue evidence acquisition and extraction now. Do not stop at a gap checklist "
                    "or ask the user to repeat the valuation request."
                ),
            })
            result = self._llm_turn(session, llm, hint)
        else:
            result = self._tool(
                session,
                "request_formal_valuation",
                {},
                lambda: self._request_valuation(session),
            )
        if acknowledgement:
            result["answer"] = acknowledgement + "\n\n" + result["answer"]
        return result

    def _llm_turn(self, session, llm, intent=None):
        session.prompt_version = RESEARCH_PROMPT_VERSION
        if session.pending_action == "valuation":
            self._prepare(session)
        pending_valuation = session.pending_action == "valuation"
        recent_user_messages = [
            message.content
            for message in self.store.list_messages(session.session_id)
            if message.role == "user"
        ]
        # During a persistent valuation, the deterministic assembler—not the
        # wording of a prior option label—defines the remaining inputs. Requiring
        # every metric mentioned in that label to succeed atomically discarded
        # otherwise valid partial batches and forced artificial "next turn"
        # confirmations. Atomic finite-field extraction remains useful for
        # ordinary research requests outside the valuation workflow.
        explicit_requested_metrics = (
            []
            if pending_valuation
            else _explicit_metric_requests(
                recent_user_messages[-1] if recent_user_messages else ""
            )
        )
        clean_fact_snapshot = {
            fact.fact_id for fact in session.facts
            if fact.status == "proposed" and not fact.warnings
        }
        document_reads: dict[str, dict[str, int]] = {}
        search_attempts: dict[str, int] = {}
        search_signatures: set[tuple[str, str]] = set()
        finish_attempts = {"invalid": 0}

        def inspect(args):
            messages = self.store.list_messages(session.session_id)
            if args.section == "overview":
                snapshot = research_snapshot(session, messages)
                return {
                    **snapshot.task_state,
                    "session_id": session.session_id,
                    "revision": session.revision,
                    "summary": session.summary,
                    "user_notes": [
                        {"block_id": "message:" + message.message_id,
                         "text": message.content,
                         "location": {"message_id": message.message_id, "source_type": "user_note"}}
                        for message in messages if message.role == "user"
                    ][-2:],
                    "retrieval": "使用 section、query、offset、limit 检索完整历史；按 next_offset 继续分页。",
                }
            if args.section == "user_notes":
                rows = [
                    {"block_id": "message:" + message.message_id,
                     "text": message.content,
                     "location": {"message_id": message.message_id, "source_type": "user_note"}}
                    for message in messages if message.role == "user"
                ]
            else:
                rows = [item.model_dump(mode="json") for item in getattr(session, args.section)]
            if args.query:
                needle = args.query.casefold()
                rows = [row for row in rows if needle in canonical(row).casefold()]
            page, used = [], 0
            for row in rows[args.offset:args.offset + args.limit]:
                size = len(canonical(row))
                if page and used + size > 24000:
                    break
                page.append(row)
                used += size
            end = args.offset + len(page)
            return {"section": args.section, "total": len(rows), "offset": args.offset,
                    "items": page, "next_offset": end if end < len(rows) else None}

        def gaps(args):
            session.gaps = list(dict.fromkeys(args.missing))
            return {"missing": session.gaps, "reason": args.reason}

        def read(args):
            blocks = self.store.research_blocks(session.session_id, args.file_id)
            if args.start_page is not None:
                meta = self.store.get_file(args.file_id)
                if not meta["storage_path"].lower().endswith(".pdf"):
                    raise ValueError("start_page只支持PDF资料")
                document = next(d for d in session.documents if d.file_id == args.file_id)
                loaded_pages = {b["location"].get("page") for b in blocks}
                requested_pages = set(range(args.start_page, args.start_page + 25))
                fully_read = document.parse_status == "parsed" and not document.warnings
                if fully_read or requested_pages <= loaded_pages:
                    # A page-window query of an already parsed filing is a
                    # cache lookup. Re-extraction both wastes time and can add
                    # a misleading "only pages X-Y read" coverage warning.
                    parsed, warnings = [], []
                else:
                    parsed, warnings = parse_document(meta, check_cancel=self._check_execution,
                        pdf_start_page=args.start_page, pdf_page_limit=25, block_offset=len(blocks))
                # No duplicate blocks when a model retries the same page range.
                known = {(b["text"], canonical(b["location"])) for b in blocks}
                original_location = next((b["location"] for b in blocks if b["location"].get("source_url")), {})
                for block in parsed:
                    block["location"].update({k: v for k, v in original_location.items() if k != "page"})
                    if (block["text"], canonical(block["location"])) not in known:
                        block["block_id"] = f"{args.file_id}:{len(blocks) + 1}"
                        blocks.append(block)
                self.store.save_research_blocks(session.session_id, args.file_id, blocks)
                document.block_count = len(blocks)
                document.warnings = list(dict.fromkeys([*document.warnings, *warnings]))
                blocks = [b for b in blocks if args.start_page <= b["location"].get("page", 0) < args.start_page + 25]
            counters = document_reads.setdefault(
                args.file_id, {"unfiltered": 0, "filtered": 0}
            )
            counter = "filtered" if args.query else "unfiltered"
            counters[counter] += 1
            # A formal valuation goal often needs several independent sections
            # from one long annual report.  Keep ordinary Q&A tight, while
            # allowing bounded, targeted extraction to finish the requested job.
            limit = (6 if pending_valuation else 2) if args.query else 1
            if counters[counter] > limit:
                return {
                    "total": len(blocks),
                    "offset": args.offset,
                    "blocks": [],
                    "next_offset": None,
                    "retrieval_budget_exhausted": True,
                    "guidance": (
                        "本轮对该文件的同类读取预算已用完。不要继续改变 offset 逐页扫描；"
                        "请先把已读证据形成的可靠候选提交给propose_facts。成功暂存新的无警告候选后，"
                        "该文件的定向读取预算会自动刷新，再按更具体关键词继续。"
                        "读取预算是内部控制，禁止要求用户选择‘下一轮继续读取’。"
                    ),
                }
            all_blocks = blocks
            if args.query:
                blocks = rank_document_blocks(blocks, args.query)
            page = blocks[args.offset:args.offset + args.limit]
            end = args.offset + len(page)
            result = {
                "total": len(blocks),
                "offset": args.offset,
                "blocks": page,
                "next_offset": end if end < len(blocks) else None,
                "retrieval_budget_remaining": limit - counters[counter],
            }
            if args.query:
                # Surface verbatim local headers along with matching rows.
                # IDs remain those of the stored original, not generated text.
                context = {}
                page_ids = {b["block_id"] for b in page}
                for block in page:
                    for candidate in evidence_context(block, all_blocks):
                        if candidate["block_id"] not in page_ids and re.search(r"单位|合并|母公司|项目.*20\d{2}|本期金额|本年金额", candidate["text"]):
                            context[candidate["block_id"]] = candidate
                result["context_blocks"] = list(context.values())[-6:]
                result["guidance"] = "检索按关键词覆盖与财务数值行排序，非全文页序。context_blocks为原文表头上下文，提取时引用这些ID并逐项核验，不得把目录或管理层摘要当作合并报表。"
            if not args.query and len(blocks) > args.limit:
                result["guidance"] = (
                    "这是长文档预览。不要沿 next_offset 顺序遍历全文；下一次请设置 query，"
                    "用与用户问题直接相关的关键词检索，随后综合回答并披露覆盖边界。"
                )
            return result

        def close_outcome(progress):
            reason = progress.get("blocking_detail", progress["blocking_reason"])
            session.question = None
            session.status = "collecting"
            session.gaps = list(dict.fromkeys([reason, *session.gaps]))[:50]
            session.summary = _text(session,
                "已停止本轮自动取证，结果说明报告已保存，可直接下载。当前没有可通过审核的数值估值；缺口：" + reason + "。已有原文和候选保留，补充有效数据后可以继续。无需确认继续多搜。",
                "Automatic research ended with a saved, downloadable outcome report. No defensible numeric valuation is available. Missing evidence: " + reason + ". Existing sources and candidates are preserved; continuing research is optional.")
            self.store.append_event(session.session_id, type="valuation.outcome_closed", stage="reporting", status="completed",
                summary="取证已收敛为说明报告，未要求继续补搜", payload={"blocking_reason": reason})
            return {"_terminal": True, "answer": session.summary}

        def finish(args):
            blocks = self._blocks(session)
            if any(key not in blocks for key in args.evidence_ids):
                raise ValueError("回答包含不存在的来源引用。")
            if session.pending_action == "valuation" and session.question is None:
                clean_now = {
                    fact.fact_id for fact in session.facts
                    if fact.status == "proposed" and not fact.warnings
                }
                claims_new_staging = re.search(
                    r"本轮[\s\S]{0,300}(?:已完成来源校验|已核验并暂存|"
                    r"(?:字段|科目|数据).{0,24}(?:暂存|并入|写入))",
                    args.answer,
                )
                if claims_new_staging and not (clean_now - clean_fact_snapshot):
                    raise ValueError(
                        "VALUATION_STATE_MISMATCH: 本轮没有新的无警告候选写入会话，"
                        "不能声称字段已完成来源校验、暂存或并入。请先调用propose_facts并根据工具返回如实说明。"
                    )
                warned_count = sum(
                    fact.status == "proposed" and bool(fact.warnings)
                    for fact in session.facts
                )
                claimed_warned_counts = {
                    int(match.group(1))
                    for match in re.finditer(
                        r"(\d+)\s*条(?:候选)?(?:处于)?\s*(?:needs_repair|需返工|需修复|带警告)",
                        args.answer,
                        flags=re.IGNORECASE,
                    )
                }
                if claimed_warned_counts and claimed_warned_counts != {warned_count}:
                    raise ValueError(
                        "VALUATION_STATE_MISMATCH: 模型陈述的需修复候选数量与控制器状态不一致。"
                        f"当前实际为 {warned_count} 条；请使用工具返回的candidate_counts或省略计数。"
                    )
                progress = valuation_progress(session, self.valuation_assembler)
                if progress["ready_for_review"]:
                    # The controller owns completion; a model's narrative must
                    # not strand a calculable task in research mode.
                    return self._request_valuation(session)
                if args.deliver_outcome:
                    if not any(doc.block_count for doc in session.documents) and not session.search_history:
                        raise ValueError("EVIDENCE_NOT_ATTEMPTED: 尚未读取或检索资料；请先实际取证，不能直接宣称数据不可得。")
                    return close_outcome(progress)
                violation = _valuation_finish_violation(
                    args.answer, args.question, args.options,
                    public_research_authorized=session.data_source_preference == "web",
                )
                if not violation and re.search(r"唯一.{0,8}(?:阻塞|缺项)|只(?:差|剩).{0,12}(?:一项|这项)", args.question):
                    violation = "确定性组装尚未通过，不能把模型推测的缺项称为唯一阻塞，也不能让用户确认代替证据核验"
                if violation:
                    finish_attempts["invalid"] += 1
                    if finish_attempts["invalid"] >= 2:
                        return close_outcome(progress)
                    raise ValueError(
                        "VALUATION_STATE_MISMATCH: " + violation + "。"
                        "当前确定性阻塞：" + progress["blocking_reason"] + "。"
                        "继续检索或提取真正缺失的输入；若必须询问用户，只能询问一个与该阻塞直接相关、"
                        "且不会承诺缺项计算的问题。"
                    )
            if args.question and session.question is None:
                labels = args.options
                self._question(session, "clarification", args.question,
                               [(f"choice_{i}", label) for i, label in enumerate(labels)])
            if session.pending_action == "valuation":
                self._prepare(session)
                if session.status != "ready_for_valuation" and session.question is None:
                    raise ValueError(
                        "VALUATION_GOAL_STILL_ACTIONABLE: 用户已要求完成估值，当前仍有准备缺口。"
                        "不能只返回缺口清单或要求用户再次开始；请继续调用检索/下载/读取/候选工具，"
                        "或在确有歧义、授权需求或资料不可得时提出一个最小确认问题。"
                    )
            answer = self._redact_text(args.answer)
            if args.evidence_ids:
                answer += "\n\n" + _text(session, "来源：", "Sources: ") + "\n".join(
                    key + " · " + canonical(blocks[key]["location"]) for key in args.evidence_ids)
            memory_result = self._apply_memory(session, args.memory_updates, args.memory_remove_keys)
            if memory_result["rejected"]:
                answer += "\n\n" + _text(
                    session,
                    "出于安全考虑，疑似包含密钥的内容未写入长期记忆。",
                    "For safety, content that appeared to contain a credential was not stored in long-term memory.",
                )
            session.summary = self._redact_text(args.answer)[:2000]
            return {"_terminal": True, "answer": answer}

        def remember(args):
            result = self._apply_memory(session, args.updates, args.remove_keys)
            return {"memory_update": result, "active_memory_count": len(session.memory)}

        def search(args):
            if not session.data_source_preference:
                self._ask_data_source(session)
                return {
                    "_terminal": True,
                    "answer": _text(
                        session,
                        "尚未确定资料来源，因此没有发出网络请求。请先选择联网获取或自行上传。",
                        "No data source has been selected, so no network request was made. Choose online retrieval or upload first.",
                    ),
                }
            if session.data_source_preference == "upload":
                self._ask_data_source(session, _text(
                    session,
                    "当前选择为自行上传。是否改用联网获取？",
                    "The current source is upload. Would you like to switch to online retrieval?",
                ))
                return {
                    "_terminal": True,
                    "answer": _text(
                        session,
                        "当前资料来源设置为自行上传，因此没有发出网络请求。",
                        "The source is currently set to upload, so no network request was made.",
                    ),
                }
            signature = (args.purpose, re.sub(r"\s+", " ", args.query).strip().casefold())
            evidence_key = hashlib.sha256(canonical({
                "scope": scope_key(session), "retry": session.search_retry_epoch,
                "facts": sorted((f.metric, f.period, f.normalized_value or "", f.block_id)
                                for f in session.facts if f.status != "rejected" and not f.warnings),
            }).encode()).hexdigest()
            prior = [item for item in session.search_history if item.get("evidence_key") == evidence_key]
            if signature in search_signatures or any((item.get("purpose"), item.get("normalized_query")) == signature for item in prior):
                return {
                    "ok": False,
                    "error": {
                        "code": "DUPLICATE_SEARCH",
                        "message": (
                            "相同数据状态下已经执行过该检索（含之前轮次）。请使用已有结果、改用不同正式来源，"
                            "或检查当前估值准备状态；不得原样轮询。"
                        ),
                        "recoverable": True,
                    },
                }
            search_signatures.add(signature)
            search_attempts[args.purpose] = search_attempts.get(args.purpose, 0) + 1
            if (
                pending_valuation
                and (len(prior) >= 6 or sum(item.get("purpose") == args.purpose for item in prior) >= 3
                     or search_attempts[args.purpose] > 3)
            ):
                progress = valuation_progress(session, self.valuation_assembler)
                if progress["ready_for_review"]:
                    return self._request_valuation(session)
                reason = progress.get("blocking_detail", progress["blocking_reason"])[:360]
                self._question(
                    session,
                    "clarification",
                    _text(
                        session,
                        f"已达到当前数据状态的检索上限，不再自动重复搜索。当前阻塞：{reason}",
                        f"Three distinct searches did not resolve the required data, so automatic searching has stopped. Current blocker: {reason}",
                    ),
                    [
                        ("report", _text(session, "查看结果说明报告", "View the outcome report")),
                        ("upload", _text(session, "上传或提供缺失的正式数据后继续", "Upload or provide the missing official data")),
                        ("change_methods", _text(session, "调整为现有资料能够支持的估值方法", "Use valuation methods supported by current data")),
                    ],
                )
                return {
                    "_terminal": True,
                    "answer": _text(
                        session,
                        "自动检索已经停止，结果说明报告已准备好，包含方法适用性、关键缺口和检索记录。可直接下载，也可以补充资料后继续。",
                        "Automatic retrieval stopped at its bounded limit. Existing sources and verified fields were preserved; the system will neither keep polling nor invent historical facts.",
                    ),
                }
            attempt = {"query": args.query, "normalized_query": signature[1], "purpose": args.purpose,
                       "evidence_key": evidence_key, "status": "attempted", "attempted_at": datetime.now(timezone.utc).isoformat()}
            session.search_history = [*session.search_history[-119:], attempt]
            explicit_codes = set(re.findall(r"(?<!\d)([036]\d{5})(?:\.(?:SH|SZ))?(?!\d)", args.query, re.I))
            query_code = next(iter(explicit_codes)) if len(explicit_codes) == 1 else ""
            target_ticker = args.target_ticker or (query_code if query_code and query_code != session.draft.ticker.split(".")[0] else session.draft.ticker)
            same_issuer = target_ticker.split(".")[0] == session.draft.ticker.split(".")[0]
            query = SearchQuery(
                query=args.query,
                ticker=target_ticker or None,
                company_name=(session.draft.company or None) if same_issuer else None,
                purpose=args.purpose,
                as_of_date=args.as_of_date or session.draft.valuation_date,
                information_cutoff=session.draft.valuation_date,
                allowed_domains=args.allowed_domains,
            )
            provider = self._search_clients.get(session.session_id, self.search_provider)
            cutoff = args.as_of_date or session.draft.valuation_date or date.today()  # noqa: DTZ011 - local user date
            years = sorted({int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", args.query)})
            wants_annual_reports = bool(
                target_ticker and len(explicit_codes) <= 1
                and args.purpose == "financials"
                and not re.search(r"季报|季度报告|半年报|半年度报告", args.query)
                and (
                    re.search(r"年报|年度报告|历史财务|三大报表|财务报表", args.query)
                    or not years
                )
            )
            official_result = None
            if wants_annual_reports:
                if not years:
                    years = list(range(cutoff.year - 10, cutoff.year))
                official_result = self.official_search_provider.search_annual_reports(
                    target_ticker,
                    years,
                    cutoff=cutoff,
                    company_name=(session.draft.company or None) if same_issuer else None,
                )
            if official_result is not None and official_result.status == "completed":
                result = official_result
            else:
                result = provider.search(query)
                if result.status == "not_configured" and official_result is not None:
                    result = official_result
                elif official_result is not None and official_result.warnings:
                    result.warnings = list(dict.fromkeys([
                        *official_result.warnings,
                        *result.warnings,
                    ]))[:30]
            attempt.update(status=result.status, provider=result.provider, source_ids=[hit.source_id for hit in result.hits])
            if result.status == "not_configured":
                session.gaps = list(dict.fromkeys([*session.gaps, args.reason]))[:50]
                self._question(session, "search_unavailable", _text(session,
                    "当前会话未配置联网搜索服务，怎样继续？", "Web search is not configured for this session. How would you like to continue?"),
                    [("report", _text(session, "先下载结果说明报告", "Download an outcome report")),
                     ("upload", _text(session, "我来上传资料", "I will upload sources")),
                     ("defer", _text(session, "保留缺口，继续研究", "Keep the gap and continue"))])
                return {"_terminal": True, "answer": _text(session,
                    "已记录检索需求：" + args.query + "。当前进程未配置 Tavily API Key，因此本次没有发出网络请求，也没有补入未经核验的数据。",
                    "Search need saved: " + args.query + ". This process has no Tavily API key, so no network request was made and no missing values were filled.")}
            if result.status == "failed":
                session.gaps = list(dict.fromkeys([*session.gaps, args.reason]))[:50]
                detail = result.error_message or _text(
                    session,
                    "联网搜索失败，本次没有取得任何资料。",
                    "Web search failed and returned no material.",
                )
                self._question(
                    session,
                    "search_failed",
                    _text(
                        session,
                        f"联网搜索未完成：{detail} 怎样继续？",
                        f"Web search did not complete: {detail} How would you like to continue?",
                    ),
                    [
                        ("report", _text(session, "先下载结果说明报告", "Download an outcome report")),
                        ("retry", _text(session, "重试本次联网检索", "Retry this web search")),
                        ("upload", _text(session, "改为上传资料", "Upload sources instead")),
                        ("defer", _text(session, "保留资料缺口", "Keep the data gap")),
                    ],
                )
                return {
                    "_terminal": True,
                    "answer": _text(
                        session,
                        f"{detail} 已保留检索需求和当前进度；没有把失败结果交给模型继续猜测。",
                        f"{detail} The search need and current progress were preserved; the failed result was not passed to the model as evidence.",
                    ),
                }
            output = result.model_dump(mode="json")
            for index, hit in enumerate(result.hits, 1):
                file_id = "web_" + hit.source_id
                block_id = f"{file_id}:1"
                text = "\n".join(part for part in [
                    hit.title,
                    f"URL: {hit.url}",
                    f"Published: {hit.published_at}" if hit.published_at else "",
                    hit.snippet,
                ] if part)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if file_id not in {document.file_id for document in session.documents}:
                    self.store.save_research_blocks(session.session_id, file_id, [{
                        "block_id": block_id,
                        "text": text,
                        "location": {
                            "source_type": "web_search",
                            "provider": result.provider,
                            "url": hit.url,
                            "domain": hit.domain,
                            "published_at": hit.published_at.isoformat() if hit.published_at else None,
                            "search_query": args.query,
                        },
                    }])
                    session.documents.append(DocumentSummary(
                        file_id=file_id,
                        name=hit.title,
                        role="evidence",
                        block_count=1,
                        sha256=digest,
                        size_bytes=len(text.encode("utf-8")),
                        warnings=[
                            "官方公告目录条目；形成财务事实前必须下载并读取 PDF 原文。"
                            if result.provider == "cninfo-announcements"
                            else "联网搜索摘要；形成关键事实前应打开原始URL核对全文。"
                        ],
                    ))
                output["hits"][index - 1]["block_id"] = block_id
                output["hits"][index - 1]["file_id"] = file_id
            if result.provider == "cninfo-announcements" and result.hits:
                output["instruction"] = (
                    "这些是按证券代码、报告年度和截止日筛选的巨潮官方年报目录。"
                    "下一步逐个调用 fetch_search_source 下载 PDF，再用 read_document 提取；"
                    "不要继续用 Tavily 猜链接，也不要从目录摘要提取财务数字。"
                )
            return output

        def preparation(_):
            # Inside an LLM turn this is evidence for the final response, not a
            # substitute for answering the user's actual question. Direct
            # `/prepare` still returns the terminal checklist in turn().
            if session.pending_action == "valuation":
                return valuation_progress(session, self.valuation_assembler)
            result = self._prepare(session)
            return {
                "status": session.status,
                "gaps": list(session.gaps),
                "checklist": result.get("answer", ""),
                "instruction": (
                    "Use this checklist only as planning evidence. If pending_action is valuation "
                    "and gaps remain, continue using acquisition and document tools instead of "
                    "ending with the checklist."
                ),
            }

        confirmed_source = None
        if isinstance(intent, dict):
            answered = intent.get("answered_question") or {}
            selected = intent.get("selected_option_id")
            if answered.get("kind") == "data_source" and selected in {"online", "web", "upload"}:
                confirmed_source = selected

        source_change_specs = [] if confirmed_source else [
            ToolSpec("propose_data_source_change", "用户要求停用Tushare、改用Tavily网页检索、改为上传，或恢复Tushare取数时调用。创建明确的数据来源确认选项并停止本轮提交；网页提取必须逐项确认后才能组装结构化财务数据。", ProposeDataSourceChange,
                     lambda args: self._propose_data_source_change(session, args.source, args.reason)),
        ]
        # A source-selection click authorizes acquisition, not extraction. Keep
        # fact proposals out of that same tool loop so the model cannot search,
        # read and place many numbers into a confirmation dialog before the
        # user has explicitly asked for extraction/review.
        partial_fact_attempts = {"count": 0}
        repair_attempts = {}

        def propose_fact_batch(args):
            clean_before = {
                fact.fact_id for fact in session.facts
                if fact.status == "proposed" and not fact.warnings
            }
            result = self._facts(
                session, args, required_metrics=explicit_requested_metrics
            )
            if pending_valuation:
                progress = valuation_progress(session, self.valuation_assembler)
                if progress["status"] == "unsupported_model_scope":
                    return close_outcome(progress)
            clean_after = {
                fact.fact_id for fact in session.facts
                if fact.status == "proposed" and not fact.warnings
            }
            new_clean = clean_after - clean_before
            if new_clean:
                # A bounded read budget prevents blind document scans. Once a
                # batch creates auditable model progress, allow another targeted
                # pass in the same user turn instead of asking the user to press
                # "continue" merely to reset an internal counter.
                for read_counts in document_reads.values():
                    read_counts["filtered"] = 0
                result["retrieval_budget_refreshed"] = True
                result["new_clean_candidate_count"] = len(new_clean)
            if result.get("status") == "candidate_evidence_needs_repair":
                def repair_key(candidate):
                    return (
                        normalize_financial_metric(candidate.get("metric", "")) or candidate.get("metric", ""),
                        candidate.get("period", ""),
                        candidate.get("role", ""),
                    )

                for candidate in result["candidates"]:
                    key = repair_key(candidate)
                    repair_attempts[key] = repair_attempts.get(key, 0) + 1
                result["repair_attempts_remaining"] = min(
                    max(0, 3 - repair_attempts[repair_key(candidate)])
                    for candidate in result["candidates"]
                )
                exhausted = [
                    candidate for candidate in result["candidates"]
                    if repair_attempts[repair_key(candidate)] >= 3
                ]
                if exhausted:
                    reasons = list(dict.fromkeys(
                        f"{candidate['metric']}：{reason}"
                        for candidate in exhausted for reason in candidate["warnings"]
                    ))
                    if session.pending_action != "valuation":
                        return {"_terminal": True, "answer": self._recover(
                            session,
                            ValueError(
                                "EVIDENCE_REPAIR_EXHAUSTED: 已尝试3轮取证，仍无法可靠确认："
                                + "；".join(reasons[:4])
                                + "。未确认数字不会进入研究结论，已保存来源和候选。"
                            ),
                            "document",
                            context={
                                "candidate_ids": [
                                    candidate["fact_id"] for candidate in exhausted
                                ]
                            },
                        )}
                    quarantined = self._quarantine_candidates(
                        session,
                        [candidate["fact_id"] for candidate in exhausted],
                        "同一字段连续3轮无法通过来源校验：" + "；".join(reasons[:4]),
                    )
                    exhausted_ids = {fact.fact_id for fact in quarantined}
                    remaining = [
                        candidate for candidate in result["candidates"]
                        if candidate["fact_id"] not in exhausted_ids
                    ]
                    result.update({
                        "status": (
                            "candidate_evidence_needs_repair"
                            if remaining else "candidate_evidence_quarantined"
                        ),
                        "candidates": remaining,
                        "quarantined_candidates": [
                            fact.model_dump(mode="json") for fact in quarantined
                        ],
                        "repair_attempts_remaining": (
                            min(
                                max(0, 3 - repair_attempts[repair_key(candidate)])
                                for candidate in remaining
                            ) if remaining else 0
                        ),
                        "instruction": (
                            "已隔离连续3轮无法可靠绑定来源的候选。不要再提交同一字段与同一引文；"
                            "请改用另一处正式披露、重建跨行表格，或处理其他必要输入。"
                            "只有替代证据确实不可得时，才向用户询问该字段的最小必要信息。"
                        ),
                    })
                    if session.pending_action == "valuation":
                        result["progress"] = valuation_progress(session, self.valuation_assembler)
                return result
            if result.get("status") != "incomplete_candidate_batch":
                return result
            partial_fact_attempts["count"] += 1
            if partial_fact_attempts["count"] < 3:
                return result
            labels = "、".join(
                item["label"] for item in result["uncovered_metrics"]
            )
            args.missing = list(dict.fromkeys([
                *args.missing,
                f"模型在本轮多次核对后仍未形成可靠候选：{labels}",
            ]))
            # Preserve useful, evidence-backed candidates instead of turning a
            # weaker model's repetition into a blocking recovery loop.
            return self._facts(session, args)

        fact_specs = [] if confirmed_source in {"online", "web"} and not pending_valuation else [
            ToolSpec("propose_facts", "从原文或用户消息提出无歧义的候选，并保留原始值、期间、单位和口径。scope使用consolidated、parent、unknown；仅发行人普通股股份总数可用issuer，可引用明确主体/截止日/单位的正文，或同一正式年报中带单位、变动前后数量表头及完整总数行的股份变动表；不能引用分红基数、流通股或金额股本。字段或年份有多种合理对应时先补原文，不得静默映射。年报未直接披露 EBIT、EBITDA、税率、折旧摊销、资本开支、营运资本变动或有息负债合计时，应提出原始基础科目（如profit_before_tax、income_tax_expense、interest_expense、独立折旧摊销组成含使用权资产、cash_paid_for_ppe_intangibles、三项现金流营运资本调整及五项债务组成），让确定性组装器计算并留痕；不得自行计算或把未找到项当零。更正用replaces指定字段ID，确认后才替换。", ProposeFacts,
                     propose_fact_batch),
        ]
        valuation_specs = [
            ToolSpec("request_formal_valuation", "用户请求估值后随时检查并推进：输入不齐时返回当前阻塞供继续补齐，不终止；已齐备时集中确认整套财务与预测方案，确认后直接进入确定性计算和报告。不要等待用户重复开始。", NoArguments,
                     lambda _: self._request_valuation(session)),
        ]
        base_specs = [
            ToolSpec("inspect_context", "读取有界任务摘要，或按 section、query、offset、limit 分页检索事实、长期记忆、文件清单及早期用户原话；返回 next_offset 时可继续读取。", InspectContext, inspect),
            ToolSpec("read_document", "读取当前会话已上传或已下载文件的原文块；支持关键词和分页，返回可引用 block_id。遇到表格时应同时读取标题、表头、单位、相邻行和注释，不能凭单个单元格判断字段或年份。读取预算属于内部控制；先用propose_facts暂存新证据即可刷新估值任务的定向读取预算，不得要求用户选择下一轮继续。", ReadDocument, read),
            ToolSpec("propose_task", "根据用户明确意图提出完整研究设置，等待用户确认；不是直接修改。", ProposeTask,
                     lambda args: self._propose_task(session, args.draft)),
            *fact_specs,
            ToolSpec("propose_forecast", "DCF经营预测入口，不是事实提取。基期财务完整但多年历史不可得时，依据已读取材料提出10年悲观/基准/乐观收入增长路径、可选利润率路径、WACC和永续增长率，并列明依据与风险；小数比例0.08代表8%。不写入历史事实，不自动批准，不替代基期缺失值；随整套估值方案集中确认。", ProposeForecast,
                     lambda args: self._propose_forecast(session, args)),
            ToolSpec("search_sources", "检索公开资料或同业。A股历年年报应设 purpose=financials 并在 query 写明年份；相对估值应设 purpose=comparables，优先查找至少3家可比公司的同期FY口径PE/PS/EV-EBITDA，目标公司自身收盘价不是PE必要输入。系统优先查询无需密钥的巨潮官方公告目录，再回退到已配置搜索 provider。结果只是候选来源，财务数字不得直接引用摘要；取得原文后调用 fetch_search_source 下载，再读取、核验并保留来源。同一查询不得重复；单轮同一目的最多尝试3组不同检索，达到上限后必须使用可执行方法进入最终方案，或一次性询问用户补数据/调整方法，不能继续轮询。", SearchSources, search),
            ToolSpec("fetch_search_source", "下载并解析 search_sources 找到的原文。巨潮或交易所结果读取官方 PDF；其他公开 HTTPS 结果可读取 PDF、HTML 或纯文本，用于政策、业务、行业和可比研究。财务事实不得直接引用搜索摘要，核心历史财务不得由非官方网页替代；先传入 web_* file_id，成功后再用 read_document 读取返回的新 file_id。", FetchSearchSource,
                     lambda args: self._fetch_search_source(session, args.file_id)),
            ToolSpec("update_data_gaps", "核对新资料后更新仍未解决的缺失、字段/年份/单位/口径冲突，写明位置和影响；不能用默认值掩盖问题，也不能代替正式财务校验。", UpdateGaps, gaps),
            ToolSpec("update_memory", "保存用户明确表达且跨回合仍有效的目标、偏好、约束、决定或术语定义；不保存财务数值、推断、临时结果或秘密。需要继续调用其他终止工具时先使用本工具。", UpdateMemory, remember),
            ToolSpec("check_preparation", "检查当前研究是否具备提交正式估值的条件。pending_action=valuation 且仍有缺口时，检查结果只是下一步检索与提取计划，不能作为终止答复。A股代码可走在线取数，确认字段可组装为结构化输入。", NoArguments,
                     preparation),
            *valuation_specs,
            *source_change_specs,
            ToolSpec("finish_response", "综合已有工具结果回答本轮。遇到资料异常时如实说明发现、位置、不确定点和影响；需要用户决定时只问与确定性 blocking_reason 直接相关且确实需要用户授权、补充资料或裁定冲突的最小问题，并提供 2—3 个互斥、可执行的选项。读取配额、下一轮继续检索、修正候选或补读年报都是内部执行步骤，不得包装成用户问题。不得自行声称估值输入齐备，不得创建整套方案确认或承诺缺项DCF；正式确认卡只能由 request_formal_valuation 生成。界面会自动追加 Chat 自由输入入口。", FinishResponse, finish),
        ]
        extension_specs, extensions = [], []
        for provider in self.tool_providers:
            specs = list(provider.tool_specs(session))
            extension_specs.extend(specs)
            extensions.append({
                "provider_id": str(getattr(provider, "provider_id", provider.__class__.__name__)),
                "version": str(getattr(provider, "version", "")),
                "tools": [spec.name for spec in specs],
            })
        registry = ToolRegistry([*base_specs, *extension_specs])
        messages = research_context(
            session,
            self.store.list_messages(session.session_id),
            intent.model_dump(mode="json") if hasattr(intent, "model_dump") else intent,
        )
        if confirmed_source:
            next_step = (
                (
                    "数据来源 web 已经由用户确认，且用户此前已经要求完成估值。本轮直接调用 search_sources "
                    "定位官方年报，随后 fetch_search_source、read_document、propose_facts 连续构建模型。"
                    "不要每批暂停；只有模型齐备时集中确认。不要再次询问数据来源或只报告下一步可以下载。"
                )
                if confirmed_source == "web" and pending_valuation
                else "数据来源 web 已经由用户确认。本轮直接调用 search_sources 开始检索公司公开年报或公告；"
                     "不要再次建议、询问或调用数据来源变更。本轮只做资料定位、下载和说明，"
                     "不得提出财务事实；待用户下一轮明确要求提取或核验后再逐项提出候选。"
                if confirmed_source == "web"
                else "该数据来源已经由用户确认；不要再次建议或询问同一项数据来源变更。"
                     "本轮只完成数据获取，不提出财务事实。"
            )
            messages.append({"role": "system", "content": next_step})
        if pending_valuation:
            messages.append({
                "role": "system",
                "content": (
                    "当前存在持续目标 pending_action=valuation。优先复用已下载的官方年报；资料不足时检索估值日前最近年度。"
                    "先补齐最近年度所选方法必要字段。DCF可按历史自动预测；历史不足时用propose_forecast提出明确且有依据的三情景路径，不得补造历史。"
                    "选择PE/PS/EV-EBITDA时，实际进入计算的方法必须收集至少3家同期FY可比倍数；不要把目标公司收盘价当作PE计算输入。"
                    "若至少一种已选方法已经可算，按check_preparation返回的降级方案进入最终集中确认；明确排除缺数据方法，不要无限检索拖住已有估值。"
                    "不要以准备缺口清单结束，也不要要求用户再次说开始估值。"
                    "propose_facts只是暂存已核验输入，继续推进；confirmed_fact_count=0本身不是要求用户去界面逐项确认的理由。"
                    "不得让用户点击不存在的候选卡来解除阻塞。check_preparation显示ready_for_review后立即request_formal_valuation，"
                    "由控制器集中生成唯一可执行确认卡，不再为非必要资料延迟估值。"
                    "文件读取配额是内部防扫描机制：每形成一批新的无警告候选即可刷新定向读取额度；"
                    "不得询问用户是否下一轮继续补读、清理needs_repair或补取更多历史年报。"
                    f"当前准备缺口：{canonical(session.gaps[:12])}"
                ),
            })
        if 1 < len(explicit_requested_metrics) <= 10:
            messages.append({
                "role": "system",
                "content": (
                    "用户本轮明确要求一组有限字段："
                    + canonical(explicit_requested_metrics)
                    + "。在调用 propose_facts 前逐项定向读取；有证据的字段合并为一批，"
                    "找不到的字段在 missing 中写明已核对范围和原因，避免让用户逐字段重复确认。"
                ),
            })
        context_hash = hashlib.sha256(canonical(messages).encode("utf-8")).hexdigest()
        self.store.append_event(
            session.session_id,
            type="agent.started",
            stage="planning",
            status="running",
            summary="LLM 正在理解意图并选择注册工具",
            payload={
                "prompt_version": RESEARCH_PROMPT_VERSION,
                "context_sha256": context_hash,
                "context_messages": len(messages),
                "model_provider": session.model_provider,
                "model_name": session.model_name,
                "registered_tools": list(registry.specs),
                "tool_extensions": extensions,
            },
        )
        def invoke(name, arguments, fn):
            try:
                visible_arguments = json.loads(arguments)
            except (TypeError, ValueError):
                visible_arguments = {
                    "invalid_json": True,
                    "raw_sha256": hashlib.sha256(str(arguments).encode("utf-8")).hexdigest(),
                }
            return self._tool(session, name, visible_arguments, fn)

        try:
            result = run_tool_loop(
                llm,
                messages,
                registry,
                invoke,
                max_rounds=40 if pending_valuation else 20,
                max_tokens=4500,
                check_cancel=self._check_execution,
            )
        except Exception as exc:
            self.store.append_event(
                session.session_id,
                type="agent.failed",
                stage="planning",
                status="failed",
                summary=self._redact_text(str(exc))[:500] if isinstance(exc, (ValueError, LlmError)) else "Agent 执行失败",
                payload={"prompt_version": RESEARCH_PROMPT_VERSION, "context_sha256": context_hash},
            )
            raise
        trace = result.pop("_agent_trace", {})
        self.store.append_event(
            session.session_id,
            type="agent.completed",
            stage="planning",
            status="completed",
            summary="Agent 已完成本轮决策",
            payload={"prompt_version": RESEARCH_PROMPT_VERSION, "context_sha256": context_hash, **trace},
        )
        return result

    def _offline(self, session, text):
        if text.strip() == "/prepare":
            return self._prepare(session)
        if text.startswith("/company "):
            draft = session.draft.model_copy(update={"company": text[9:].strip()})
            return self._propose_task(session, draft)
        if text.startswith("/date "):
            draft = session.draft.model_copy(update={"valuation_date": date.fromisoformat(text[6:].strip())})
            return self._propose_task(session, draft)
        if text.startswith("/industry "):
            draft = session.draft.model_copy(update={"industry": text[10:].strip()})
            return self._propose_task(session, draft)
        if text.startswith("/methods "):
            draft = ResearchDraft.model_validate({**session.draft.model_dump(), "methods": text[9:].strip().split(",")})
            return self._propose_task(session, draft)
        ticker = re.search(
            r"(?<!\d)([036]\d{5})(?:\.(SH|SZ|BJ))?(?!\d)",
            text,
            re.IGNORECASE,
        )
        if ticker:
            slots = interpret_intent(text).slots
            draft = session.draft.model_copy(update={
                "ticker": ticker[0].upper(),
                "company": slots.get("company", session.draft.company),
                "valuation_date": (
                    date.fromisoformat(slots["valuation_date"])
                    if slots.get("valuation_date")
                    else session.draft.valuation_date or date.today()  # noqa: DTZ011 - local user date
                ),
                "methods": slots.get("methods") or session.draft.methods or ["dcf", "pe", "ev_ebitda"],
                "objective": text[:2000],
            })
            return self._propose_task(session, draft)
        return {"answer": _text(session,
            "需求已保存。当前为资料整理模式，可以上传文件、查看原文和确认研究范围。连接模型后可理解自然语言、抽取字段并分析政策。\n"
            "可输入：/company 公司名称、/industry 行业、/date 2026-09-14、/methods dcf,pe、/prepare；连接模型后输入“提取已上传文件中的财务字段”。",
            "Your request is saved. Local preparation supports files, source previews and scope confirmation. Connect a model for natural-language understanding, extraction and policy analysis.\n"
            "Commands: /company Name, /industry Industry, /date 2026-09-14, /methods dcf,pe, /prepare.")}

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
        action = None
        try:
            session = self.store.get_research(session_id)
            if turn.language is not None:
                session.language = turn.language
            # Reject stale selections before logging or changing any state.
            if turn.option_id and (session.question is None or session.question.question_id != turn.question_id or
                                   turn.option_id not in {o.id for o in session.question.options}):
                raise ValueError("确认问题或选项已失效，请刷新后重试。")
            if turn.question_id and not turn.option_id and (
                session.question is None or session.question.question_id != turn.question_id
            ):
                raise ValueError("补充说明对应的确认问题已失效，请刷新后重试。")
            if turn.option_id and turn.content.strip() and session.question.kind in {"task", "facts"}:
                raise ValueError("修改要求请作为文字单独发送，不能同时确认旧候选。")
            confirmation_content = ""
            if (not turn.option_id and not turn.file_ids and session.question
                    and session.question.kind == "facts"
                    and any(o.id == "accept" for o in session.question.options)
                    and _explicit_fact_acceptance(turn.content, valuation_review=bool(session.question.valuation_review))):
                # Equivalent to clicking this already visible card's Accept;
                # never creates approval for newly discovered facts or warnings.
                confirmation_content = turn.content.strip()
                turn = turn.model_copy(update={"question_id": session.question.question_id,
                                               "option_id": "accept", "content": ""})
            pending_question = session.question.model_copy(deep=True) if session.question else None
            question_kind = pending_question.kind if pending_question else None
            label = next((o.label for o in pending_question.options if o.id == turn.option_id), "") if pending_question else ""
            recovery_context = dict(session.last_issue.context) if question_kind == "recovery" and session.last_issue else {}
            raw_content = confirmation_content or turn.content.strip() or label or "上传资料"
            content = self._redact_text(raw_content)
            user_message = self.store.add_message(session_id, "user", content, "research")
            if content != raw_content:
                self.store.append_event(
                    session_id, type="security.credential_redacted", stage="input",
                    status="completed", summary="疑似凭证已从对话内容中移除",
                )
            answered_question = pending_question.model_dump(mode="json") if turn.option_id else None
            if (
                turn.content.strip()
                and pending_question
                and not turn.option_id
                and (
                    turn.question_id == pending_question.question_id
                    or pending_question.kind in {"clarification", "recovery"}
                )
            ):
                # Free text is a first-class answer to a pending question. The
                # previous proposal remains in history, but no stale card can
                # accidentally be accepted after the Agent processes the text.
                answered_question = pending_question.model_dump(mode="json")
                session.question = None
                session.status = "collecting"
                if pending_question.kind == "recovery" and session.last_issue:
                    session.last_issue.status = "retrying"
                self.store.append_event(
                    session_id,
                    type="review.resolved",
                    stage="research",
                    status="completed",
                    summary="用户以文字补充或修改",
                    payload={"question_id": pending_question.question_id, "option_id": "free_text"},
                )
            intent = interpret_intent(content, {
                "company": session.draft.company,
                "ticker": session.draft.ticker,
                "revision": session.revision,
            })
            planning_hint = {
                "classification": intent.model_dump(mode="json"),
                "answered_question": answered_question,
                "selected_option_id": turn.option_id,
            }
            self.store.append_event(
                session_id,
                type="intent.classified",
                stage="planning",
                status="completed",
                summary=intent.intent,
                payload=intent.model_dump(mode="json"),
            )
            self.store.append_event(session_id, type="turn.started", stage="research", status="running", summary=content[:300])
            current_stage = "input"
            try:
                self._check_execution()
                if question_kind == "facts" and turn.option_id == "accept":
                    self._refresh_pending_candidates(session, refresh_question=False)
                acknowledgement = self._answer_question(session, turn) if turn.option_id else None
                retry_requested = question_kind == "recovery" and turn.option_id == "retry"
                retry_files = recovery_context.get("file_ids", []) if retry_requested else []
                retry_content = ""
                if retry_requested and recovery_context.get("message_id"):
                    retry_content = next((
                        message.content for message in self.store.list_messages(session_id)
                        if message.message_id == recovery_context["message_id"]
                    ), "")
                # Record the deliverable goal before parsing: even an entirely
                # unreadable upload must leave a downloadable outcome.
                if intent.intent == "run_valuation" and session.pending_action != "valuation":
                    session.pending_action = "valuation"
                    self.store.append_event(session.session_id, type="valuation.goal_recorded", stage="planning",
                        status="completed", summary="已记录持续估值目标")
                current_stage = "document"
                if turn.file_ids and not session.data_source_preference:
                    # Supplying a file is itself an explicit source choice.
                    session.data_source_preference = "upload"
                for file_id in dict.fromkeys([*retry_files, *turn.file_ids]):
                    existing_doc = next((d for d in session.documents if d.file_id == file_id), None)
                    if existing_doc:
                        if file_id not in retry_files or existing_doc.parse_status != "unreadable":
                            continue
                        session.documents.remove(existing_doc)
                    try:
                        meta = self.store.get_file(file_id)
                    except (KeyError, ValueError):
                        # A stale upload token is a per-file failure, just like
                        # an unreadable PDF. Never invent metadata for it or let
                        # it prevent the remaining valid uploads from parsing.
                        message = f"附件 {self._redact_text(file_id)} 的引用已失效或不存在；未读取该文件，已继续处理其他附件。需要时请重新上传。"
                        if message not in session.gaps:
                            session.gaps.append(message)
                        self.store.append_event(session_id, type="document.unavailable", stage="document", status="warning",
                            summary=message, payload={"file_id": self._redact_text(file_id)})
                        continue

                    def parse(meta=meta):
                        blocks, warnings = parse_document(meta, check_cancel=self._check_execution)
                        self.store.save_research_blocks(session_id, meta["file_id"], blocks)
                        doc = DocumentSummary(file_id=meta["file_id"], name=meta["original_name"], role=meta["role"],
                                              block_count=len(blocks), sha256=meta["sha256"],
                                              size_bytes=meta["size_bytes"], warnings=warnings,
                                              parse_status="unreadable" if not blocks else "partial" if warnings else "parsed")
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
                            warnings=[warning], parse_status="unreadable"))
                        self.store.append_event(session_id, type="document.unreadable", stage="document", status="warning",
                            summary=f"无法读取 {meta['original_name']}；继续处理其他来源", payload={"file_id": file_id, "warning": warning})
                llm = self._clients.get(session_id)
                current_stage = "agent"
                if confirmation_content and intent.intent == "run_valuation" and session.pending_action != "valuation":
                    session.pending_action = "valuation"
                    self.store.append_event(session.session_id, type="valuation.goal_recorded", stage="planning",
                        status="completed", summary="用户确认候选并请求继续估值")
                if (session.pending_action == "valuation" or intent.intent == "run_valuation") and not (
                        question_kind == "facts" and turn.option_id == "accept"):
                    self._refresh_pending_candidates(session)
                continue_with_model = turn.option_id != "report" and (
                    question_kind == "clarification"
                    or (question_kind == "data_source" and turn.option_id in {"online", "web"})
                    or (question_kind == "search_failed" and turn.option_id == "retry")
                )
                resume_pending = bool(
                    session.pending_action == "valuation"
                    and session.question is None
                    and turn.option_id
                    and turn.option_id != "report"
                    and (
                        question_kind in {"task", "data_source", "facts", "clarification", "search_failed"}
                        or (
                            question_kind == "recovery"
                            and turn.option_id == "defer"
                            and bool(recovery_context.get("candidate_ids"))
                        )
                    )
                    and not (question_kind == "data_source" and turn.option_id == "upload" and not session.documents)
                    and not (question_kind == "search_failed" and turn.option_id != "retry")
                    and not (question_kind == "facts" and pending_question.valuation_review and turn.option_id != "accept")
                )
                if resume_pending:
                    result = self._advance_pending_valuation(
                        session, llm, planning_hint, acknowledgement or ""
                    )
                elif acknowledgement and not turn.content and not turn.file_ids and not retry_requested and (
                    not continue_with_model or llm is None
                ):
                    result = {"answer": acknowledgement}
                elif content == "/prepare":
                    result = self._tool(session, "check_preparation", {}, lambda: self._prepare(session))
                elif intent.intent == "run_valuation":
                    if session.pending_action != "valuation":
                        session.pending_action = "valuation"
                        self.store.append_event(
                            session.session_id,
                            type="valuation.goal_recorded",
                            stage="planning",
                            status="completed",
                            summary="已记录持续估值目标",
                        )
                    result = self._advance_pending_valuation(
                        session, llm, planning_hint
                    )
                elif not turn.option_id and rejects_tushare(content):
                    session.data_source_preference = ""
                    result = self._tool(
                        session,
                        "propose_data_source_change",
                        {"source": "web", "reason": "用户明确要求不使用 Tushare"},
                        lambda: self._propose_data_source_change(
                            session, "web", "用户明确要求不使用 Tushare"
                        ),
                    )
                elif (
                    not turn.option_id
                    and intent.intent == "new_valuation"
                    and bool(intent.slots.get("ticker"))
                    and not (session.draft.company or session.draft.ticker)
                ):
                    # Persist the initial target through the same explicit
                    # confirmation contract used by CLI commands. The model
                    # may explain a target in prose, but prose alone must
                    # never make the application believe the scope was saved.
                    draft = session.draft.model_copy(update={
                        "ticker": intent.slots["ticker"],
                        "company": intent.slots.get("company", ""),
                        "industry": intent.slots.get("industry", ""),
                        "valuation_date": (
                            date.fromisoformat(intent.slots["valuation_date"])
                            if intent.slots.get("valuation_date")
                            else date.today()  # noqa: DTZ011 - local user date
                        ),
                        "methods": intent.slots.get("methods") or ["dcf", "pe", "ev_ebitda"],
                        "objective": content[:2000],
                    })
                    result = self._tool(
                        session,
                        "propose_task",
                        {"draft": draft.model_dump(mode="json")},
                        lambda: self._propose_task(session, draft),
                    )
                elif (session.pending_action == "valuation" and not turn.file_ids and not turn.option_id
                      and re.fullmatch(r"(?:请)?继续(?:估值|处理|补证|分析)?[。！!]*", content)):
                    result = self._advance_pending_valuation(session, llm, planning_hint)
                elif llm is not None:
                    current_stage = "model"
                    result = self._llm_turn(session, llm, planning_hint)
                elif retry_requested:
                    result = self._offline(session, retry_content) if retry_content else {"answer": acknowledgement}
                elif session.requires_model:
                    current_stage = "model"
                    raise LlmError("MODEL_CONNECTION_REQUIRED: 会话已保存，请重新连接模型后继续。")
                else:
                    result = self._offline(session, content)
                if retry_requested or (answered_question and question_kind == "recovery" and not turn.option_id):
                    self._resolve_issue(session)
                self._restore_pending_fact_review(session)
                action = result.get("_action")
                self._say(session, result["answer"])
                self.store.append_event(session_id, type="turn.completed", stage="research", status="completed", summary="本轮研究已保存")
            except LlmError as exc:
                message = self._recover(
                    session,
                    exc,
                    "model" if current_stage in {"model", "agent"} else current_stage,
                    {"file_ids": list(dict.fromkeys([*turn.file_ids, *recovery_context.get("file_ids", [])])),
                     "intent": intent.intent, "message_id": user_message.message_id},
                )
                self._say(session, message)
                self.store.append_event(session_id, type="turn.interrupted", stage=current_stage, status="waiting_confirmation", summary=session.last_issue.code)
            except (ValueError, OSError) as exc:
                stage = (
                    "document" if current_stage == "document" else
                    "tool" if current_stage in {"agent", "model"} else
                    "input"
                )
                message = self._recover(session, exc, stage, {
                    "file_ids": list(dict.fromkeys([*turn.file_ids, *recovery_context.get("file_ids", [])])),
                    "intent": intent.intent,
                    "message_id": user_message.message_id,
                })
                self._say(session, message)
                self.store.append_event(session_id, type="turn.interrupted", stage=stage, status="waiting_confirmation", summary=session.last_issue.code)
            except Exception as exc:  # noqa: BLE001 - top-level recovery boundary
                message = self._recover(session, exc, "unknown", {
                    "file_ids": list(turn.file_ids), "intent": intent.intent,
                    "message_id": user_message.message_id,
                })
                self._say(session, message)
                self.store.append_event(session_id, type="turn.interrupted", stage="unknown", status="waiting_confirmation", summary=session.last_issue.code)
            self.store.save_research(session)
            if session.pending_action == "valuation" or session.valuation_run_id:
                from valuationagent.application.result_document import ensure_result_document
                try:
                    ensure_result_document(self, session)
                except Exception:
                    self.store.append_event(session_id, type="report.failed", stage="reporting", status="failed",
                        summary="报告生成暂未完成；研究进度已保存，可重新下载生成。")
            stopped = session.last_issue and session.last_issue.status == "open" and session.last_issue.code in {"EXECUTION_CANCELLED", "EXECUTION_TIME_LIMIT"}
            self.store.update_research_job(session_id, request_id, status="cancelled" if stopped else "completed", stage="saved")
            snapshot = self.snapshot(session_id, compact=compact_result)
            if action:
                snapshot["action"] = action
            return snapshot
        finally:
            stop.set()
            worker.join(timeout=1)
            self.store.release(session_id, owner)
            job = self.store.research_job(session_id, request_id)
            if job and job["status"] == "running":
                self.store.update_research_job(session_id, request_id, status="failed")
            self._execution.current = None
