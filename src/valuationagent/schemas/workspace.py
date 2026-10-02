"""Contracts for the valuation-first workspace.

The workspace owns the task and calculation pointers. Evidence collection is
persisted separately from immutable calculation inputs; neither is a second
conversation agent or a selectable workflow version.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import Field, model_validator

from valuationagent.schemas.models import ApiModel, Language


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


WorkspaceStatus = Literal[
    "setup",
    "researching",
    "awaiting_user",
    "awaiting_approval",
    "ready",
    "calculating",
    "completed",
    "completed_with_warnings",
    "degraded",
    "failed",
    "paused",
    "cancelled",
]

RunPolicy = Literal["review", "automatic"]


class ValuationWorkspace(ApiModel):
    workspace_id: str = Field(min_length=8, max_length=120)
    research_session_id: str = Field(min_length=8, max_length=120)
    revision: int = Field(default=1, ge=1)
    language: Language = Language.ZH_CN
    title: str = Field(default="新估值任务", min_length=1, max_length=240)
    status: WorkspaceStatus = "setup"
    current_phase: Literal[
        "scope",
        "evidence",
        "model_design",
        "approval",
        "calculation",
        "challenge",
        "decision",
        "reporting",
    ] = "scope"
    objective: str = Field(default="", max_length=2400)
    run_policy: RunPolicy = "automatic"
    information_cutoff_date: date | None = None
    execution_date: date = Field(default_factory=date.today)
    currency: str = Field(default="CNY", min_length=3, max_length=3)
    company_type: str = Field(default="", max_length=120)
    industry_strategy: str = Field(default="", max_length=240)
    active_run_id: str | None = Field(default=None, max_length=120)
    active_version_id: str | None = Field(default=None, max_length=120)
    active_checkpoint_id: str | None = Field(default=None, max_length=120)
    degraded_reason: str = Field(default="", max_length=2400)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DataRequirement(ApiModel):
    requirement_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    category: Literal[
        "scope",
        "historical_financial",
        "capital_structure",
        "forecast_assumption",
        "market_parameter",
        "comparable",
        "risk",
        "deliverable",
    ]
    label: str = Field(min_length=1, max_length=300)
    metric: str = Field(default="", max_length=160)
    periods: list[str] = Field(default_factory=list, max_length=30)
    scope: str = Field(default="consolidated", max_length=120)
    methods: list[Literal["dcf", "pe", "ps", "ev_ebitda"]] = Field(
        default_factory=list
    )
    required: bool = True
    priority: Literal["critical", "high", "normal", "low"] = "normal"
    importance_weight: float = Field(default=0.5, ge=0, le=1)
    acceptable_sources: list[str] = Field(default_factory=list, max_length=30)
    fallback_chain: list[str] = Field(default_factory=list, max_length=30)
    depends_on: list[str] = Field(default_factory=list, max_length=100)
    unlocks: list[str] = Field(default_factory=list, max_length=100)
    status: Literal[
        "pending", "searching", "satisfied", "unavailable", "waived", "superseded"
    ] = "pending"
    attempt_count: int = Field(default=0, ge=0)
    attempt_limit: int = Field(default=3, ge=1, le=12)
    fact_ids: list[str] = Field(default_factory=list, max_length=200)
    evidence_ids: list[str] = Field(default_factory=list, max_length=200)
    resolution: str = Field(default="", max_length=2000)
    resolution_type: Literal[
        "", "direct", "derived", "proxy", "assumption", "scenario", "unavailable"
    ] = ""
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def bounded_search(self) -> "DataRequirement":
        if self.attempt_count >= self.attempt_limit and self.status == "searching":
            raise ValueError("reaching the attempt limit must resolve to unavailable or satisfied")
        return self


class EvidenceRecord(ApiModel):
    evidence_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    source_type: Literal[
        "regulatory_filing",
        "issuer_disclosure",
        "exchange_data",
        "licensed_market_data",
        "user_document",
        "user_statement",
        "reputable_media",
        "public_web",
        "calculation",
    ]
    authority_tier: Literal["A", "B", "C", "D", "E"]
    title: str = Field(min_length=1, max_length=500)
    provider: str = Field(default="", max_length=160)
    publisher: str = Field(default="", max_length=240)
    url: str = Field(default="", max_length=2000)
    locator: dict[str, Any] = Field(default_factory=dict)
    excerpt: str = Field(default="", max_length=4000)
    source_sha256: str = Field(default="", max_length=64)
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    information_cutoff_ok: bool | None = None
    content_type: str = Field(default="", max_length=160)
    snapshot_file_id: str = Field(default="", max_length=160)
    is_republication: bool = False
    original_evidence_id: str | None = Field(default=None, max_length=160)
    license: str = Field(default="unknown", max_length=240)
    usage_scope: str = Field(default="analysis_and_audit", max_length=500)
    extraction_tool: str = Field(default="valuationagent", max_length=160)
    extraction_version: str = Field(default="", max_length=120)
    binding_proof: dict[str, Any] = Field(default_factory=dict)
    authority_score: float = Field(default=0, ge=0, le=1)
    directness_score: float = Field(default=0, ge=0, le=1)
    entity_scope_score: float = Field(default=0, ge=0, le=1)
    timing_score: float = Field(default=0, ge=0, le=1)
    extraction_score: float = Field(default=0, ge=0, le=1)
    cross_source_score: float = Field(default=0, ge=0, le=1)
    confidence: float = Field(default=0, ge=0, le=1)
    status: Literal["candidate", "verified", "conflicted", "rejected"] = "candidate"
    conflict_evidence_ids: list[str] = Field(default_factory=list, max_length=100)
    adopted_fact_ids: list[str] = Field(default_factory=list, max_length=100)
    limitations: list[str] = Field(default_factory=list, max_length=30)


class EvidenceUsage(ApiModel):
    """Immutable adoption decision linking source evidence to one ModelSpec."""

    usage_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    run_id: str = Field(min_length=8, max_length=120)
    model_spec_id: str = Field(min_length=8, max_length=160)
    evidence_id: str = Field(min_length=8, max_length=160)
    fact_id: str = Field(min_length=8, max_length=160)
    adopted: bool = True
    rationale: str = Field(min_length=1, max_length=1200)
    created_at: datetime = Field(default_factory=utc_now)


class WorkspaceFact(ApiModel):
    fact_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    source_fact_id: str = Field(default="", max_length=160)
    metric: str = Field(min_length=1, max_length=160)
    issuer: str = Field(default="", max_length=240)
    raw_metric: str = Field(default="", max_length=240)
    raw_value: str = Field(min_length=1, max_length=500)
    normalized_value: str | None = Field(default=None, max_length=200)
    unit: str = Field(default="unknown", max_length=80)
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    period: str = Field(default="unknown", max_length=80)
    scope: str = Field(default="unknown", max_length=120)
    assertion_type: Literal[
        "source_fact", "user_input", "calculation", "model_inference", "opinion"
    ] = "source_fact"
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    directly_disclosed: bool = True
    derivation_formula: str = Field(default="", max_length=1200)
    upstream_fact_ids: list[str] = Field(default_factory=list, max_length=100)
    conflict_fact_ids: list[str] = Field(default_factory=list, max_length=100)
    valuation_impact: Literal["critical", "high", "normal", "low"] = "normal"
    confidence: float | None = Field(default=None, ge=0, le=1)
    status: Literal["proposed", "confirmed", "conflicted", "rejected"] = "proposed"
    supersedes: list[str] = Field(default_factory=list, max_length=20)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def source_facts_need_evidence(self) -> "WorkspaceFact":
        if self.assertion_type in {"source_fact", "calculation"} and not self.evidence_ids:
            raise ValueError("source facts and calculations require evidence lineage")
        return self


class WorkspaceAssumption(ApiModel):
    assumption_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    run_id: str = Field(default="", max_length=120)
    proposal_id: str = Field(default="", max_length=160)
    parameter: str = Field(min_length=1, max_length=160)
    scenario: Literal["pessimistic", "base", "optimistic", "all"] = "base"
    value: Any
    rationale: str = Field(min_length=1, max_length=2400)
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    source: Literal["user", "model", "industry_parameter", "market_data"]
    status: Literal["proposed", "approved", "rejected", "superseded"] = "proposed"
    user_locked: bool = False
    supersedes: str | None = Field(default=None, max_length=160)
    created_at: datetime = Field(default_factory=utc_now)


class AgentAction(ApiModel):
    action_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    actor: Literal[
        "orchestrator", "research", "extraction", "modeling", "challenge", "decision", "user"
    ]
    action_type: str = Field(min_length=1, max_length=120)
    status: Literal["planned", "running", "completed", "failed", "skipped"]
    summary: str = Field(min_length=1, max_length=1200)
    objective: str = Field(default="", max_length=1200)
    tool_name: str = Field(default="", max_length=160)
    tool_version: str = Field(default="", max_length=120)
    input_digest: str = Field(default="", max_length=64)
    output_digest: str = Field(default="", max_length=64)
    change_reason: str = Field(default="", max_length=1200)
    input_refs: list[str] = Field(default_factory=list, max_length=100)
    output_refs: list[str] = Field(default_factory=list, max_length=100)
    error_code: str | None = Field(default=None, max_length=120)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)


class PreValuationCheckpoint(ApiModel):
    checkpoint_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    research_revision: int = Field(ge=1)
    state_hash: str = Field(min_length=64, max_length=64)
    status: Literal["pending", "approved", "reopened", "superseded"] = "pending"
    company: str = Field(default="", max_length=240)
    ticker: str = Field(default="", max_length=32)
    valuation_date: str = Field(default="", max_length=40)
    information_cutoff_date: str = Field(default="", max_length=40)
    execution_date: str = Field(default="", max_length=40)
    industry: str = Field(default="", max_length=160)
    company_type: str = Field(default="", max_length=120)
    currency: str = Field(default="CNY", min_length=3, max_length=3)
    requested_methods: list[str] = Field(default_factory=list, max_length=10)
    executable_methods: list[str] = Field(default_factory=list, max_length=10)
    excluded_methods: dict[str, str] = Field(default_factory=dict)
    baseline_period: str | None = Field(default=None, max_length=40)
    inputs: dict[str, Any] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)
    wacc_components: dict[str, Any] = Field(default_factory=dict)
    peers: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    capital_actions: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    user_overrides: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    request_snapshot: dict[str, Any] = Field(default_factory=dict)
    evidence_coverage: float = Field(default=0, ge=0, le=1)
    evidence_grade: Literal["A", "B", "C", "D", "E"] = "E"
    preflight_findings: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    fact_ids: list[str] = Field(default_factory=list, max_length=500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=500)
    unresolved_items: list[str] = Field(default_factory=list, max_length=100)
    risks: list[str] = Field(default_factory=list, max_length=100)
    approval_note: str = Field(default="", max_length=2000)
    approved_by: Literal["user", "automatic_policy"] | None = None
    approved_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ChallengeFinding(ApiModel):
    finding_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    run_id: str = Field(min_length=8, max_length=120)
    category: Literal[
        "data_quality", "assumption", "model_risk", "method", "sensitivity", "scope"
    ]
    severity: Literal["info", "warning", "high", "blocking"]
    title: str = Field(min_length=1, max_length=300)
    analysis: str = Field(min_length=1, max_length=2400)
    evidence_refs: list[str] = Field(default_factory=list, max_length=100)
    recommendation: str = Field(default="", max_length=1200)
    status: Literal["open", "accepted", "mitigated", "dismissed"] = "open"
    created_at: datetime = Field(default_factory=utc_now)


class DecisionRecord(ApiModel):
    decision_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    run_id: str = Field(min_length=8, max_length=120)
    outcome: Literal["accepted", "accepted_with_warnings", "review_required", "rejected"]
    rationale: str = Field(min_length=1, max_length=3000)
    finding_ids: list[str] = Field(default_factory=list, max_length=100)
    disposition_ids: list[str] = Field(default_factory=list, max_length=100)
    alternatives: list[str] = Field(default_factory=list, max_length=20)
    selected_action: str = Field(default="", max_length=1000)
    actor: Literal["decision_layer", "user"] = "decision_layer"
    created_at: datetime = Field(default_factory=utc_now)


class FindingDisposition(ApiModel):
    """Immutable decision-ledger entry for one challenge finding."""

    disposition_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    run_id: str = Field(min_length=8, max_length=120)
    finding_id: str = Field(min_length=8, max_length=160)
    decision: Literal[
        "accepted", "partially_accepted", "mitigated", "dismissed", "blocking_recalculation"
    ]
    rationale: str = Field(min_length=1, max_length=2400)
    resulting_action: str = Field(default="", max_length=1200)
    scenario_change: dict[str, Any] = Field(default_factory=dict)
    actor: Literal["decision_layer", "user"] = "decision_layer"
    created_at: datetime = Field(default_factory=utc_now)


class ModelSpec(ApiModel):
    """Immutable snapshot passed through the existing financial-model adapter."""

    model_spec_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    checkpoint_id: str | None = Field(default=None, max_length=160)
    parent_model_spec_id: str | None = Field(default=None, max_length=160)
    run_id: str = Field(min_length=8, max_length=120)
    version_number: int = Field(ge=1)
    valuation_date: date
    information_cutoff_date: date
    execution_date: date
    methods: list[str] = Field(min_length=1, max_length=10)
    requested_methods: list[str] = Field(default_factory=list, max_length=10)
    fact_ids: list[str] = Field(default_factory=list, max_length=1000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=1000)
    assumption_ids: list[str] = Field(default_factory=list, max_length=500)
    peer_tickers: list[str] = Field(default_factory=list, max_length=200)
    request_snapshot: dict[str, Any]
    resolved_assumptions: dict[str, Any] = Field(default_factory=dict)
    formula_manifest: dict[str, str] = Field(default_factory=dict)
    calculation_order: list[str] = Field(default_factory=list, max_length=200)
    rounding_policy: str = Field(default="Decimal ROUND_HALF_UP; display 4 decimals unless specified", max_length=500)
    financial_model_name: str = Field(default="", max_length=160)
    financial_model_version: str = Field(default="", max_length=160)
    software_versions: dict[str, str] = Field(default_factory=dict)
    degradation_policy: dict[str, Any] = Field(default_factory=dict)
    snapshot_hash: str = Field(min_length=64, max_length=64)
    created_at: datetime = Field(default_factory=utc_now)


class CalculationRecord(ApiModel):
    """Calculation/version ledger; numbers can be replayed without the LLM."""

    calculation_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    run_id: str = Field(min_length=8, max_length=120)
    model_spec_id: str = Field(min_length=8, max_length=160)
    input_hash: str = Field(min_length=64, max_length=64)
    result_hash: str | None = Field(default=None, min_length=64, max_length=64)
    status: str = Field(min_length=1, max_length=80)
    formulas: dict[str, str] = Field(default_factory=dict)
    upstream_refs: list[str] = Field(default_factory=list, max_length=1500)
    calculation_order: list[str] = Field(default_factory=list, max_length=200)
    precision_policy: str = Field(default="Decimal", max_length=500)
    financial_model_version: str = Field(default="", max_length=160)
    replayable_offline: bool = True
    created_at: datetime = Field(default_factory=utc_now)


class ValuationVersion(ApiModel):
    version_id: str = Field(min_length=8, max_length=160)
    workspace_id: str = Field(min_length=8, max_length=120)
    number: int = Field(ge=1)
    run_id: str = Field(min_length=8, max_length=120)
    checkpoint_id: str | None = Field(default=None, max_length=160)
    model_spec_id: str | None = Field(default=None, max_length=160)
    calculation_id: str | None = Field(default=None, max_length=160)
    parent_version_id: str | None = Field(default=None, max_length=160)
    reason: str = Field(min_length=1, max_length=2000)
    request_hash: str = Field(min_length=64, max_length=64)
    result_hash: str | None = Field(default=None, min_length=64, max_length=64)
    changes: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="created", max_length=80)
    report_refs: dict[str, str] = Field(default_factory=dict)
    change_attribution: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class WorkspaceCreate(ApiModel):
    language: Language = Language.ZH_CN
    title: str = Field(default="新估值任务", min_length=1, max_length=240)
    objective: str = Field(default="", max_length=2400)
    model_session_id: str | None = None
    data_source_preference: Literal["", "web", "upload"] = ""
    run_policy: RunPolicy = "automatic"
    information_cutoff_date: date | None = None


class WorkspaceRevision(ApiModel):
    reason: str = Field(min_length=1, max_length=2000)
    changes: dict[str, Any] = Field(default_factory=dict)


class CheckpointDecision(ApiModel):
    checkpoint_id: str = Field(min_length=8, max_length=160)
    note: str = Field(default="", max_length=2000)
