from __future__ import annotations

from typing import Any, Protocol

from valuationagent.core.tools import ToolSpec
from valuationagent.schemas.models import (
    AssumptionSet,
    DcfResult,
    FinancialSnapshot,
    ForecastYear,
    MultipleResult,
    PeerCompany,
    ReconciliationResult,
    SensitivityCell,
    ValidationFinding,
    ValuationRequest,
)


class AgentToolProvider(Protocol):
    """Adds allowlisted tools to the workspace Agent without changing its loop.

    Providers own their dependencies and return typed ``ToolSpec`` objects.
    Every invocation is still wrapped by the application event/audit layer.
    """

    provider_id: str
    version: str

    def tool_specs(self, session: Any) -> list[ToolSpec]: ...


class FinancialModelPlugin(Protocol):
    plugin_id: str
    version: str

    def validate(
        self, request: ValuationRequest, financials: FinancialSnapshot
    ) -> list[ValidationFinding]: ...

    def resolve_assumptions(
        self, request: ValuationRequest, financials: FinancialSnapshot
    ) -> AssumptionSet: ...

    def forecast(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        assumptions: AssumptionSet,
    ) -> list[ForecastYear]: ...

    def dcf(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        assumptions: AssumptionSet,
        forecast: list[ForecastYear],
    ) -> DcfResult: ...

    def relative(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        peers: list[PeerCompany],
    ) -> list[MultipleResult]: ...

    def sensitivity(
        self,
        request: ValuationRequest,
        financials: FinancialSnapshot,
        assumptions: AssumptionSet,
    ) -> list[SensitivityCell]: ...

    def reconcile(
        self, dcf: DcfResult | None, relative: list[MultipleResult]
    ) -> ReconciliationResult: ...
