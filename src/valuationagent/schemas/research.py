"""Research can start with incomplete material; confirmed valuation inputs stay strict."""

from datetime import date, datetime, timezone
from typing import Any, Literal
from pydantic import Field, model_validator
from valuationagent.schemas.models import ApiModel, AssumptionInputs, JsonDecimal, Language


class ForecastInputs(ApiModel):
    """Explicit opinions, never historical facts or silently filled defaults."""

    revenue_growth_scenarios: dict[Literal["pessimistic", "base", "optimistic"], list[JsonDecimal]]
    ebit_margin_scenarios: dict[Literal["pessimistic", "base", "optimistic"], list[JsonDecimal]] | None = None
    wacc: JsonDecimal
    terminal_growth: JsonDecimal

    @model_validator(mode="after")
    def validate_projection(self):
        AssumptionInputs.model_validate(self.model_dump(exclude_none=True))
        if self.wacc <= self.terminal_growth:
            raise ValueError("WACC必须高于永续增长率")
        for key in ("revenue_growth_scenarios", "ebit_margin_scenarios"):
            paths = getattr(self, key)
            if paths is None:
                continue
            for path in paths.values():
                if len(path) != 10:
                    raise ValueError(f"{key}必须明确给出每个情景的10年路径，不能自动延长")
                if any(not value.is_finite() or value <= -1 or value > (2 if key == "revenue_growth_scenarios" else 1) for value in path):
                    raise ValueError(f"{key}必须使用有限小数比例，不是百分数；增长率范围(-100%,200%]，利润率范围(-100%,100%]")
            if any(not low <= mid <= high for low, mid, high in zip(paths["pessimistic"], paths["base"], paths["optimistic"])):
                raise ValueError(f"{key}每年的悲观、基准、乐观值必须依次不减")
        return self


class ForecastProposal(ApiModel):
    proposal_id: str
    scope_key: str
    status: Literal["proposed", "confirmed"] = "proposed"
    inputs: ForecastInputs
    rationale: str = Field(min_length=20, max_length=2400)
    risks: list[str] = Field(min_length=1, max_length=8)
    evidence_ids: list[str] = Field(min_length=1, max_length=30)


class ResearchDraft(ApiModel):
    company: str = Field(default="", max_length=200)
    ticker: str = Field(default="", max_length=20)
    industry: str = Field(default="", max_length=120)
    valuation_date: date | None = None
    objective: str = Field(default="", max_length=2000)
    methods: list[Literal["dcf", "pe", "ps", "ev_ebitda"]] = Field(default_factory=list)


class ResearchChoice(ApiModel):
    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=400)


class ResearchMemoryItem(ApiModel):
    """Durable conversational context; never a substitute for financial facts."""

    key: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_.:-]+$")
    kind: Literal["goal", "preference", "constraint", "decision", "definition"]
    content: str = Field(min_length=1, max_length=600)
    source_message_id: str = Field(default="", max_length=120)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchIssue(ApiModel):
    """Latest recoverable interruption shown to the user and audit trail."""

    issue_id: str
    code: str = Field(min_length=1, max_length=100)
    stage: Literal["model", "tool", "document", "input", "agent", "unknown"]
    message: str = Field(min_length=1, max_length=1200)
    retryable: bool = True
    status: Literal["open", "retrying", "resolved", "deferred"] = "open"
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchQuestion(ApiModel):
    question_id: str
    kind: Literal[
        "task", "data_source", "facts", "clarification",
        "search_unavailable", "search_failed", "recovery",
    ]
    title: str = Field(min_length=1, max_length=600)
    options: list[ResearchChoice] = Field(min_length=1, max_length=4)
    fact_ids: list[str] = Field(default_factory=list)
    superseded_fact_ids: list[str] = Field(default_factory=list)
    proposed_draft: ResearchDraft | None = None
    valuation_review: dict[str, Any] | None = None


class FactCandidate(ApiModel):
    fact_id: str = ""
    metric: str = Field(min_length=1, max_length=120)
    raw_value: str = Field(min_length=1, max_length=100)
    unit: Literal["元", "千元", "万元", "百万元", "亿元", "股", "千股", "万股", "百万股", "亿股", "%", "ratio", "unknown"] = "unknown"
    normalized_value: str | None = None
    period: str = Field(default="unknown", max_length=60)
    scope: Literal["consolidated", "parent", "issuer", "unknown"] = "unknown"
    role: Literal["historical", "assumption", "policy", "comparable"] = "historical"
    peer_ticker: str = Field(default="", max_length=24)
    peer_name: str = Field(default="", max_length=200)
    multiple_basis: Literal["FY", "TTM", "forward", "unknown"] = "unknown"
    block_id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=2400)
    source_type: Literal["document", "user_note"] = "document"
    status: Literal["proposed", "confirmed", "rejected"] = "proposed"
    warnings: list[str] = Field(default_factory=list)
    context_block_ids: list[str] = Field(default_factory=list, max_length=8)
    source_location: dict[str, Any] = Field(default_factory=dict)
    source_url: str = ""
    source_sha256: str = ""
    published_at: date | None = None
    verification: dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(ApiModel):
    file_id: str
    name: str
    role: str
    block_count: int
    sha256: str = Field(default="", max_length=64)
    size_bytes: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    parse_status: Literal["parsed", "partial", "unreadable"] = "parsed"


class ResearchSession(ApiModel):
    session_id: str
    revision: int = 1
    language: Language = Language.ZH_CN
    requires_model: bool = False
    data_source_preference: Literal["", "online", "web", "upload"] = ""
    # A user goal may span several evidence/confirmation turns.  Keep it in
    # authoritative session state so accepting one candidate batch resumes the
    # original valuation request instead of dropping back to an idle chat.
    pending_action: Literal["", "valuation"] = ""
    search_history: list[dict[str, Any]] = Field(default_factory=list, max_length=120)
    search_retry_epoch: int = Field(default=0, ge=0)
    # Empty until the user accepts the controller-owned combined plan.  A
    # reviewed subset lets one data-starved method be excluded without silently
    # changing the user's request or blocking every calculable method.
    valuation_methods_override: list[
        Literal["dcf", "pe", "ps", "ev_ebitda"]
    ] = Field(default_factory=list)
    valuation_method_exclusions: dict[str, str] = Field(default_factory=dict)
    forecast_proposal: ForecastProposal | None = None
    staged_supersessions: dict[str, list[str]] = Field(default_factory=dict)
    draft: ResearchDraft = Field(default_factory=ResearchDraft)
    status: Literal[
        "collecting", "waiting_confirmation", "awaiting_financial_model",
        "ready_for_valuation", "submitted"
    ] = "collecting"
    documents: list[DocumentSummary] = Field(default_factory=list)
    facts: list[FactCandidate] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    memory: list[ResearchMemoryItem] = Field(default_factory=list, max_length=80)
    question: ResearchQuestion | None = None
    last_issue: ResearchIssue | None = None
    summary: str = ""
    agent_protocol_version: str = "research-agent-v2"
    prompt_version: str = "research-2026-09-27.20"
    model_provider: str = ""
    model_name: str = ""
    valuation_run_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchCreate(ApiModel):
    language: Language = Language.ZH_CN
    model_session_id: str | None = None
    data_source_preference: Literal["", "web", "upload"] = ""


class ResearchTurn(ApiModel):
    request_id: str | None = Field(default=None, min_length=8, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    time_budget_seconds: int = Field(default=600, ge=30, le=1800)
    language: Language | None = None
    content: str = Field(default="", max_length=8000)
    file_ids: list[str] = Field(default_factory=list, max_length=8)
    question_id: str | None = None
    option_id: str | None = None

    @model_validator(mode="after")
    def meaningful_turn(self):
        if not self.content.strip() and not self.file_ids and not self.option_id:
            raise ValueError("请输入需求、上传文件或选择一个选项。")
        if self.option_id and not self.question_id:
            raise ValueError("选项必须关联当前确认问题。")
        return self
