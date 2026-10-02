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
    information_cutoff_date: date | None = None
    peer_pricing_date: date | None = Field(default=None, description="相对估值统一使用的可核验行情日期；默认估值日。非交易日等原因可明确选择估值日前七天内的日期，不修改估值日或信息截止日，不凭猜测声称该日是最近交易日。")
    peer_pricing_rationale: str = Field(default="", max_length=800, description="选择不同于估值日的行情日时说明依据及陈旧性风险，随可比样本冻结和披露。")
    objective: str = Field(default="", max_length=2000)
    methods: list[Literal["dcf", "pe", "ps", "ev_ebitda"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def pricing_date_bounds(self):
        if self.peer_pricing_date:
            if self.valuation_date and not 0 <= (self.valuation_date - self.peer_pricing_date).days <= 7:
                raise ValueError("可比行情日期须在估值日前七天内，不能晚于估值日")
            if self.information_cutoff_date and self.peer_pricing_date > self.information_cutoff_date:
                raise ValueError("可比行情日期不能晚于信息截止日")
            if self.valuation_date and self.peer_pricing_date != self.valuation_date and len(self.peer_pricing_rationale.strip()) < 12:
                raise ValueError("不同行情日期须说明选择依据和陈旧性风险，不得静默替换")
        return self


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


class SemanticMappingAlternative(ApiModel):
    """A plausible accounting interpretation retained for audit, not hidden."""

    standard_metric: str = Field(default="", max_length=120)
    semantic_role: Literal[
        "operating", "financing", "financial_subsidiary", "investing",
        "tax", "equity", "non_operating", "unknown",
    ] = "unknown"
    confidence: float = Field(default=0, ge=0, le=1)
    rationale: str = Field(default="", max_length=600)


class FactCandidate(ApiModel):
    fact_id: str = ""
    metric: str = Field(min_length=1, max_length=120)
    standard_metric: str = Field(
        default="",
        max_length=120,
        description="LLM基于完整上下文建议的标准字段；metric始终保留原始科目名。",
    )
    semantic_role: Literal[
        "operating", "financing", "financial_subsidiary", "investing",
        "tax", "equity", "non_operating", "unknown",
    ] = "unknown"
    ebit_treatment: Literal["include", "exclude", "review"] = "review"
    fcff_treatment: Literal["include", "exclude", "review"] = "review"
    equity_bridge_treatment: Literal["include", "exclude", "review"] = "review"
    mapping_confidence: float = Field(default=0, ge=0, le=1)
    mapping_rationale: str = Field(default="", max_length=1200)
    alternative_interpretations: list[SemanticMappingAlternative] = Field(
        default_factory=list,
        max_length=4,
    )
    raw_value: str = Field(min_length=1, max_length=100)
    unit: Literal["元", "千元", "万元", "百万元", "亿元", "股", "千股", "万股", "百万股", "亿股", "%", "ratio", "unknown"] = "unknown"
    normalized_value: str | None = None
    period: str = Field(default="unknown", max_length=60)
    scope: Literal["consolidated", "parent", "issuer", "unknown"] = "unknown"
    role: Literal["historical", "assumption", "policy", "comparable"] = "historical"
    peer_ticker: str = Field(default="", max_length=24)
    peer_name: str = Field(default="", max_length=200)
    multiple_basis: Literal["FY", "TTM", "forward", "unknown"] = "unknown"
    denominator_period_end: date | None = None
    block_id: str = Field(min_length=1, max_length=200)
    table_id: str = ""
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
    provenance_type: Literal[
        "user_upload", "official_index", "official_filing",
        "public_web", "search_snippet", "structured_provider", "unknown",
    ] = "unknown"
    authority_tier: Literal["A", "B", "C", "D", "E"] = "D"
    source_confidence: float = Field(default=0.5, ge=0, le=1)
    provider: str = Field(default="", max_length=120)
    source_url: str = Field(default="", max_length=2000)
    acquisition_ref: str = Field(default="", max_length=100)


class DecisionOption(ApiModel):
    label: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=600)


class DecisionPrompt(ApiModel):
    question: str = Field(min_length=1, max_length=1000)
    options: list[DecisionOption] = Field(min_length=2, max_length=4)


class ResearchSession(ApiModel):
    session_id: str
    revision: int = 1
    language: Language = Language.ZH_CN
    data_source_preference: Literal["", "online", "web", "upload"] = ""
    information_cutoff_date: date | None = None
    # A user goal may span several evidence/confirmation turns.  Keep it in
    # authoritative session state so accepting one candidate batch resumes the
    # original valuation request instead of dropping back to an idle chat.
    pending_action: Literal["", "valuation"] = ""
    # Terminal outcome of the latest bounded research attempt. This remains
    # visible until stronger evidence produces a new outcome.
    outcome_status: Literal["", "insufficient_data"] = ""
    outcome_reason: str = Field(default="", max_length=2400)
    search_history: list[dict[str, Any]] = Field(default_factory=list, max_length=120)
    plan: list[dict[str, str]] = Field(default_factory=list, max_length=12)
    pending_decision: DecisionPrompt | None = None
    resume_context: dict[str, Any] = Field(default_factory=dict)
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
        "collecting", "awaiting_input",
        "ready_for_valuation", "submitted"
    ] = "collecting"
    documents: list[DocumentSummary] = Field(default_factory=list)
    facts: list[FactCandidate] = Field(default_factory=list)
    table_interpretations: list[dict[str, Any]] = Field(default_factory=list)
    reading_attempts: list[dict[str, Any]] = Field(default_factory=list, max_length=128)
    gaps: list[str] = Field(default_factory=list)
    memory: list[ResearchMemoryItem] = Field(default_factory=list, max_length=80)
    last_issue: ResearchIssue | None = None
    summary: str = ""
    agent_protocol_version: str = "workspace-agent-v1"
    prompt_version: str = "workspace-agent-2026-10-01.8"
    model_provider: str = ""
    model_name: str = ""
    valuation_run_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchTurn(ApiModel):
    request_id: str | None = Field(default=None, min_length=8, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    time_budget_seconds: int = Field(default=600, ge=30, le=1800)
    language: Language | None = None
    content: str = Field(default="", max_length=8000)
    file_ids: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def meaningful_turn(self):
        if not self.content.strip() and not self.file_ids:
            raise ValueError("请输入需求或上传文件。")
        return self
