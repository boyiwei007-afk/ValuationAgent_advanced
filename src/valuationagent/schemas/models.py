'''
这份文件用 Pydantic 定义了一整套**企业估值系统的数据契约**——
把"公司、财务、假设、估值方法、运行状态、API 请求响应"等所有数据都规定成严格的类型模型，
让 CLI、API、Agent 三方共享同一套"合法数据长什么样"的规矩，
**它不负责计算，只负责把关和定型**。
'''
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    SecretStr,
    field_validator,
    model_validator,
)

JsonDecimal = Annotated[
    Decimal,
    PlainSerializer(lambda value: str(value), return_type=str, when_used="json"),
]


class ApiModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
    )


class DataSourceType(StrEnum):
    STRUCTURED = "structured"
    TICKER = "ticker"
    UPLOAD = "upload"


class AssumptionSourceType(StrEnum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    UPLOAD = "upload"


class RunMode(StrEnum):
    DEMO = "demo"
    LIVE = "live"
    SNAPSHOT = "snapshot"


class Language(StrEnum):
    ZH_CN = "zh-CN"
    EN_US = "en-US"


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_REVIEW = "waiting_review"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValuationMethod(StrEnum):
    DCF = "dcf"
    PE = "pe"
    PS = "ps"
    EV_EBITDA = "ev_ebitda"


class CompanyInput(ApiModel):
    ticker: str | None = Field(default=None, max_length=24)
    name: str | None = Field(default=None, max_length=200)
    exchange: str | None = Field(default=None, max_length=32)
    industry: str | None = Field(default=None, max_length=120)
    currency: str = Field(default="CNY", min_length=3, max_length=3)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else value


class EvidenceRef(ApiModel):
    evidence_id: str
    source: str
    file_id: str | None = None
    page: int | None = Field(default=None, ge=1)
    sheet: str | None = None
    cell: str | None = None
    published_at: date | None = None
    source_url: str = ""
    source_sha256: str = ""
    note: str = ""


class FinancialSnapshot(ApiModel):
    period_end: date
    # Missing means unknown, never zero. Required fields depend on the selected
    # method and are checked before any financial calculations.
    revenue: JsonDecimal | None = Field(default=None, gt=0)
    ebit_margin: JsonDecimal | None = Field(default=None, ge=Decimal("-1"), le=Decimal("1"))
    tax_rate: JsonDecimal | None = Field(default=None, ge=0, le=Decimal("0.6"))
    depreciation_amortization: JsonDecimal | None = Field(default=None, ge=0)
    capital_expenditure: JsonDecimal | None = Field(default=None, ge=0)
    change_operating_nwc: JsonDecimal | None = None
    cash_and_non_operating_assets: JsonDecimal | None = Field(default=None, ge=0)
    interest_bearing_debt: JsonDecimal | None = Field(default=None, ge=0)
    common_shares: JsonDecimal = Field(gt=0)
    # Per-share denominator can be disclosed after the fiscal baseline.
    # This date is explicit so a later issuer total is not disguised as the
    # year-end balance-sheet share count.
    common_shares_as_of: date | None = None
    net_income_parent: JsonDecimal | None = None
    ebitda: JsonDecimal | None = None
    source_label: str = "user_structured_input"
    currency: str = "CNY"
    unit: Literal["base_currency"] = "base_currency"
    statement_scope: Literal["consolidated", "parent"] = "consolidated"
    published_at: date | None = None
    tax_policy: str | None = Field(default=None, max_length=200)
    tax_policy_expiry_year: int | None = Field(default=None, ge=2000, le=2200)
    comparability_status: Literal[
        "comparable", "adjusted", "review_required", "excluded"
    ] = "comparable"
    comparability_note: str = Field(default="", max_length=1000)
    evidence: dict[str, list[EvidenceRef]] = Field(default_factory=dict)
    statement_items: dict[str, JsonDecimal] = Field(default_factory=dict)
    # Machine-readable formulas used to turn confirmed statement lines into
    # valuation inputs.  Keeping this beside the snapshot makes derived values
    # independently reviewable and reproducible in exports and audit logs.
    calculation_methods: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def comparability_change_requires_note(self) -> "FinancialSnapshot":
        if self.comparability_status != "comparable" and not self.comparability_note.strip():
            raise ValueError("non-comparable financial years require comparability_note")
        return self


class PeerCompany(ApiModel):
    ticker: str
    name: str
    pe: JsonDecimal | None = Field(default=None, gt=0)
    ps: JsonDecimal | None = Field(default=None, gt=0)
    ev_ebitda: JsonDecimal | None = Field(default=None, gt=0)
    market_cap: JsonDecimal | None = Field(default=None, gt=0)
    revenue_growth: JsonDecimal | None = None
    ebit_margin: JsonDecimal | None = None
    selection_score: JsonDecimal | None = Field(default=None, ge=0)
    peer_tier: Literal["core", "broad", "user"] = "user"
    rationale: str = "user-provided comparable"
    as_of_date: date | None = None
    multiple_basis: Literal["FY", "TTM", "forward", "unknown"] = "unknown"
    evidence: dict[str, list[EvidenceRef]] = Field(default_factory=dict)


def required_financial_metrics(methods):
    required = {"common_shares"}
    by_method = {
        "dcf": {"revenue", "ebit_margin", "tax_rate", "depreciation_amortization",
                "capital_expenditure", "change_operating_nwc", "cash_and_non_operating_assets", "interest_bearing_debt"},
        "pe": {"net_income_parent"}, "ps": {"revenue"},
        "ev_ebitda": {"ebitda", "cash_and_non_operating_assets", "interest_bearing_debt"},
    }
    for method in methods:
        required.update(by_method.get(str(method), set()))
    return required


class AssumptionInputs(ApiModel):
    revenue_growth: list[JsonDecimal] | None = None
    ebit_margin: list[JsonDecimal] | None = None
    revenue_growth_scenarios: dict[
        Literal["pessimistic", "base", "optimistic"], list[JsonDecimal]
    ] | None = None
    ebit_margin_scenarios: dict[
        Literal["pessimistic", "base", "optimistic"], list[JsonDecimal]
    ] | None = None
    wacc: JsonDecimal | None = Field(default=None, gt=0, lt=Decimal("0.5"))
    terminal_growth: JsonDecimal | None = Field(
        default=None, ge=Decimal("-0.1"), lt=Decimal("0.2")
    )
    terminal_tax_rate: JsonDecimal | None = Field(default=None, ge=0, le=Decimal("0.4"))
    risk_free_rate: JsonDecimal | None = Field(default=None, ge=0, lt=Decimal("0.2"))
    equity_risk_premium: JsonDecimal | None = Field(default=None, gt=0, lt=Decimal("0.3"))
    beta: JsonDecimal | None = Field(default=None, gt=0, lt=Decimal("5"))
    debt_cost: JsonDecimal | None = Field(default=None, ge=0, lt=Decimal("0.5"))
    market_cap: JsonDecimal | None = Field(default=None, gt=0)
    quarterly_average_market_cap: JsonDecimal | None = Field(default=None, gt=0)
    annual_average_market_cap: JsonDecimal | None = Field(default=None, gt=0)
    market_cap_period_low: JsonDecimal | None = Field(default=None, gt=0)
    market_cap_period_high: JsonDecimal | None = Field(default=None, gt=0)
    capex_alpha: JsonDecimal | None = Field(default=None, ge=0, le=Decimal("5"))
    capex_kappa: JsonDecimal | None = Field(
        default=None, ge=Decimal("-5"), le=Decimal("5")
    )
    dso_days: JsonDecimal | None = Field(default=None, ge=0, le=Decimal("730"))
    dio_days: JsonDecimal | None = Field(default=None, ge=0, le=Decimal("730"))
    dpo_days: JsonDecimal | None = Field(default=None, ge=0, le=Decimal("730"))
    operating_cost_ratio: JsonDecimal | None = Field(
        default=None, ge=0, le=Decimal("2")
    )
    operating_cash_ratio: JsonDecimal | None = Field(
        default=None, ge=0, le=Decimal("0.2")
    )
    exit_multiple: JsonDecimal | None = Field(default=None, gt=0, le=Decimal("100"))
    tax_transition_years: int | None = Field(default=None, ge=0, le=10)

    @field_validator("revenue_growth", "ebit_margin")
    @classmethod
    def require_nonempty_series(
        cls, value: list[JsonDecimal] | None
    ) -> list[JsonDecimal] | None:
        if value is not None and not value:
            raise ValueError("series must contain at least one value")
        return value

    @model_validator(mode="after")
    def require_complete_scenario_sets(self) -> "AssumptionInputs":
        required = {"pessimistic", "base", "optimistic"}
        for field_name in (
            "revenue_growth_scenarios",
            "ebit_margin_scenarios",
        ):
            scenarios = getattr(self, field_name)
            if not scenarios:
                continue
            if set(scenarios) != required:
                missing = ", ".join(sorted(required - set(scenarios))) or "none"
                raise ValueError(
                    f"{field_name} must contain pessimistic, base and optimistic; "
                    f"missing: {missing}"
                )
            if any(not path for path in scenarios.values()):
                raise ValueError(f"{field_name} scenario paths must not be empty")
        return self


class ValuationRequest(ApiModel):
    company: CompanyInput
    valuation_date: date
    language: Language = Language.ZH_CN
    data_source: DataSourceType = DataSourceType.STRUCTURED
    assumption_source: AssumptionSourceType = AssumptionSourceType.AUTOMATIC
    mode: RunMode = RunMode.SNAPSHOT
    forecast_years: int = Field(default=5, ge=3, le=10)
    methods: list[ValuationMethod] = Field(
        default_factory=lambda: [
            ValuationMethod.DCF,
            ValuationMethod.PE,
            ValuationMethod.EV_EBITDA,
        ]
    )
    requested_methods: list[ValuationMethod] = Field(default_factory=list)
    excluded_methods: dict[str, str] = Field(default_factory=dict)
    financials: FinancialSnapshot | None = None
    historical_financials: list[FinancialSnapshot] = Field(default_factory=list)
    assumptions: AssumptionInputs = Field(default_factory=AssumptionInputs)
    peers: list[PeerCompany] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)
    assumption_file_ids: list[str] = Field(default_factory=list)
    assumption_evidence: dict[str, list[EvidenceRef]] = Field(default_factory=dict)
    discount_policy: Literal["annual_midyear_remaining", "year_end"] = (
        "annual_midyear_remaining"
    )
    user_goal: str = "完成可追溯的企业估值并解释关键假设"

    def agent_parameters(self) -> dict[str, Any]:
        """Stable, JSON-safe task choices shared by CLI, API and Agent tools.

        Financial plugins still receive this complete typed request. Credentials
        are configured separately and are never part of these task parameters.
        """
        return self.model_dump(
            mode="json",
            include={
                "company",
                "valuation_date",
                "language",
                "mode",
                "data_source",
                "assumption_source",
                "forecast_years",
                "methods",
                "requested_methods",
                "excluded_methods",
                "discount_policy",
                "assumptions",
                "file_ids",
                "assumption_file_ids",
            },
        )

    @field_validator("methods")
    @classmethod
    def methods_are_unique(cls, value: list[ValuationMethod]) -> list[ValuationMethod]:
        if not value:
            raise ValueError("at least one valuation method is required")
        if len(set(value)) != len(value):
            raise ValueError("valuation methods must be unique")
        return value

    @model_validator(mode="after")
    def source_inputs_are_present(self) -> "ValuationRequest":
        for snapshot in [*self.historical_financials, *([self.financials] if self.financials else [])]:
            if snapshot.common_shares_as_of and not (
                snapshot.period_end <= snapshot.common_shares_as_of <= self.valuation_date
            ):
                raise ValueError("issuer share-count date must fall between the financial period end and valuation date")
        if not self.requested_methods:
            self.requested_methods = list(self.methods)
        if not set(self.methods) <= set(self.requested_methods):
            raise ValueError("effective valuation methods must be a subset of requested_methods")
        unknown_exclusions = set(self.excluded_methods) - {
            str(method) for method in self.requested_methods
        }
        if unknown_exclusions:
            raise ValueError("excluded_methods contains methods that were not requested")
        if set(self.excluded_methods) & {str(method) for method in self.methods}:
            raise ValueError("an effective valuation method cannot also be excluded")
        if self.data_source == DataSourceType.TICKER and not self.company.ticker:
            raise ValueError("ticker data source requires company.ticker")
        if self.data_source == DataSourceType.UPLOAD and not self.file_ids:
            raise ValueError("upload data source requires file_ids")
        return self


class ModelConnectionInput(ApiModel):
    provider: Literal["openai_compatible", "openai"] = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    model: str = Field(min_length=1)
    api_key: SecretStr = Field(min_length=1)
    timeout_seconds: float = Field(default=90.0, gt=1, le=180)
    thinking: Literal["auto", "enabled", "disabled"] = "auto"

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value):
        from urllib.parse import urlsplit

        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url 必须是无账号、无查询参数的 HTTP(S) 接口地址")
        return value.rstrip("/")


class ModelSessionPublic(ApiModel):
    session_id: str
    provider: str
    base_url: str
    model: str
    created_at: datetime


class RunCreateBody(ApiModel):
    request: ValuationRequest
    model_session_id: str | None = None


class ForecastYear(ApiModel):
    year: int
    discount_period: JsonDecimal = Decimal("1")
    cash_flow_fraction: JsonDecimal = Decimal("1")
    revenue: JsonDecimal
    revenue_growth: JsonDecimal
    ebit_margin: JsonDecimal
    ebit: JsonDecimal
    tax_rate: JsonDecimal | None = None
    nopat: JsonDecimal
    depreciation_amortization: JsonDecimal
    capital_expenditure: JsonDecimal
    change_operating_nwc: JsonDecimal
    fcff: JsonDecimal
    operating_items: dict[str, JsonDecimal] = Field(default_factory=dict)
    calculation_methods: dict[str, str] = Field(default_factory=dict)


class AssumptionSet(ApiModel):
    revenue_growth: list[JsonDecimal]
    ebit_margin: list[JsonDecimal]
    wacc: JsonDecimal
    terminal_growth: JsonDecimal
    source: str
    rationale: dict[str, str]
    revenue_growth_scenarios: dict[str, list[JsonDecimal]] = Field(default_factory=dict)
    ebit_margin_scenarios: dict[str, list[JsonDecimal]] = Field(default_factory=dict)
    tax_rate_path: list[JsonDecimal] = Field(default_factory=list)
    wacc_components: dict[str, JsonDecimal] = Field(default_factory=dict)
    industry_parameters: dict[str, Any] = Field(default_factory=dict)
    model_decisions: list[str] = Field(default_factory=list)
    operating_drivers: dict[str, JsonDecimal] = Field(default_factory=dict)
    calculation_methods: dict[str, str] = Field(default_factory=dict)


class DcfResult(ApiModel):
    status: Literal["success", "not_applicable"] = "success"
    enterprise_value: JsonDecimal
    equity_value: JsonDecimal
    per_share_value: JsonDecimal
    range_low: JsonDecimal
    range_high: JsonDecimal
    terminal_value_share: JsonDecimal
    bridge: dict[str, JsonDecimal]
    scenario_warnings: list[str] = Field(default_factory=list)
    scenario_values: dict[str, JsonDecimal] = Field(default_factory=dict)
    present_value_explicit: JsonDecimal | None = None
    present_value_terminal: JsonDecimal | None = None
    terminal_value: JsonDecimal | None = None
    implied_exit_multiple: JsonDecimal | None = None
    exit_multiple_cross_check: JsonDecimal | None = None
    exit_multiple_per_share: JsonDecimal | None = None
    terminal_method_gap: JsonDecimal | None = None


class MultipleResult(ApiModel):
    method: Literal["pe", "ps", "ev_ebitda"]
    status: Literal["success", "not_applicable"]
    per_share_value: JsonDecimal | None = None
    range_low: JsonDecimal | None = None
    range_high: JsonDecimal | None = None
    sample_size: int = 0
    original_sample_size: int = 0
    outlier_count: int = 0
    sample_quality: Literal["adequate", "limited", "insufficient"] = "insufficient"
    statistic: str = "P25/P50/P75"
    peer_tickers: list[str] = Field(default_factory=list)
    reason: str | None = None


class SensitivityCell(ApiModel):
    wacc: JsonDecimal
    terminal_growth: JsonDecimal
    per_share_value: JsonDecimal | None
    valid: bool


class SensitivityStudy(ApiModel):
    study_id: str
    parameter: str
    baseline_input: str
    low_input: str = ""
    high_input: str = ""
    baseline_per_share: JsonDecimal | None = None
    low_per_share: JsonDecimal | None = None
    high_per_share: JsonDecimal | None = None
    max_relative_change: JsonDecimal | None = None
    classification: Literal["high", "medium", "low", "not_available"]
    status: Literal["completed", "not_available"]
    rationale: str = ""


class ReconciliationResult(ApiModel):
    dcf_range: tuple[JsonDecimal, JsonDecimal] | None
    relative_range: tuple[JsonDecimal, JsonDecimal] | None
    overlap_range: tuple[JsonDecimal, JsonDecimal] | None
    conclusion: str
    combined_range: None = None
    method_comparison: dict[str, str] = Field(default_factory=dict)


class DataQualityAssessment(ApiModel):
    historical_years: int = 0
    comparable_years: int = 0
    adjusted_years: list[int] = Field(default_factory=list)
    excluded_years: list[int] = Field(default_factory=list)
    evidence_coverage: JsonDecimal = Decimal(0)
    industry_parameter_quality: str = "unknown"
    industry_metadata_completeness: str = "unknown"
    peer_sample_quality: str = "not_requested"
    confidence: Literal["high", "medium", "low"] = "low"
    notes: list[str] = Field(default_factory=list)


class ValuationOutput(ApiModel):
    run_id: str = ""
    revision: int = 1
    company: CompanyInput | None = None
    valuation_date: date | None = None
    currency: str = "CNY"
    mode: RunMode = RunMode.SNAPSHOT
    language: Language = Language.ZH_CN
    input_hash: str = ""
    effective_input_hash: str = ""
    effective_financials: FinancialSnapshot | None = None
    effective_peers: list[PeerCompany] = Field(default_factory=list)
    assumption_evidence: dict[str, list[EvidenceRef]] = Field(default_factory=dict)
    discount_policy: str = "annual_midyear_remaining"
    model_version: str
    assumptions: AssumptionSet
    forecast: list[ForecastYear]
    dcf: DcfResult | None
    relative: list[MultipleResult]
    sensitivity: list[SensitivityCell]
    sensitivity_studies: list[SensitivityStudy] = Field(default_factory=list)
    reconciliation: ReconciliationResult
    data_quality: DataQualityAssessment = Field(default_factory=DataQualityAssessment)
    executive_summary: str
    warnings: list[str] = Field(default_factory=list)
    calculation_checks: list[str] = Field(default_factory=list)


class ValidationFinding(ApiModel):
    rule_id: str
    severity: Literal["info", "warning", "blocking"]
    message: str
    actual: JsonDecimal | str | None = None
    expected: JsonDecimal | str | None = None
    recommended_action: str | None = None


class RunEvent(ApiModel):
    run_id: str
    sequence: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    type: str
    stage: str | None = None
    status: str | None = None
    summary: str
    tool_call_id: str | None = None
    tool: str | None = None
    duration_ms: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(ApiModel):
    message_id: str
    run_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage: str | None = None
    related_run_id: str | None = None


class ChatInput(ApiModel):
    content: str = Field(min_length=1, max_length=8000)


class RunRecord(ApiModel):
    run_id: str
    status: RunStatus
    request: ValuationRequest
    input_hash: str
    workflow_version: str = "0.5.0"
    root_run_id: str | None = None
    parent_run_id: str | None = None
    revision: int = 1
    revision_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    result: ValuationOutput | None = None
    error: dict[str, Any] | None = None
    review: dict[str, Any] | None = None


class RunAccepted(ApiModel):
    run_id: str
    status: RunStatus


class Capability(ApiModel):
    capability_id: str
    available: bool
    detail: str


class RevisionInput(ApiModel):
    reason: str = Field(min_length=1, max_length=2000)
    changes: dict[str, Any] = Field(default_factory=dict)


class ResumeInput(ApiModel):
    model_session_id: str | None = None
