"""Workspace state, frozen calculations and the unified agent tool boundary."""

from __future__ import annotations

import hashlib
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from valuationagent.application.challenge import (
    ValuationChallengeService,
    ValuationDecisionService,
)
from valuationagent.application.ledgers import build_calculation_record, build_model_spec, sync_action_ledger, sync_assumption_ledger, sync_evidence_and_fact_ledgers, sync_evidence_usage, sync_run_action_ledger
from valuationagent.application.requirements import (
    build_requirement_graph,
    requirement_edges,
)
from valuationagent.application.valuation_plan import preview_session, valuation_progress
from valuationagent.application.research_plan import research_plan
from valuationagent.llm.context import AGENT_PROMPT_VERSION
from valuationagent.core.tools import canonical
from valuationagent.schemas.models import AssumptionInputs, RevisionInput, ValuationRequest
from valuationagent.schemas.research import ResearchTurn
from valuationagent.schemas.workspace import (
    AgentAction,
    ChallengeFinding,
    CalculationRecord,
    DecisionRecord,
    DataRequirement,
    EvidenceRecord,
    EvidenceUsage,
    FindingDisposition,
    ModelSpec,
    PreValuationCheckpoint,
    ValuationVersion,
    ValuationWorkspace,
    WorkspaceAssumption,
    WorkspaceFact,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class ValuationWorkspaceService:
    """Own workspace mutations; expose deterministic finance as agent tools."""

    def __init__(self, store, research, runner):
        self.store = store
        self.research = research
        research.workspace_service = self
        self.runner = runner
        self.challenge = ValuationChallengeService()
        self.decisions = ValuationDecisionService()
        self._sync_locks_guard = threading.Lock()
        self._sync_locks: dict[str, threading.RLock] = {}

    def _sync_lock(self, workspace_id: str) -> threading.RLock:
        """Serialize read projections for one workspace inside this process."""

        with self._sync_locks_guard:
            return self._sync_locks.setdefault(workspace_id, threading.RLock())

    def delete(self, workspace_id: str, *, expected_revision: int):
        with self._sync_lock(workspace_id):
            removed = self.store.delete_workspace(workspace_id, expected_revision=expected_revision)
            session_id = removed["session_id"]
            for clients in (self.research._clients, self.research._search_clients,
                            self.research._search_client_epochs, self.research._market_clients):
                clients.pop(session_id, None)
            self.runner.forget_runs(removed["run_ids"])

    def create(
        self,
        *,
        language="zh-CN",
        title="新估值任务",
        objective="",
        llm=None,
        data_source_preference="",
        run_policy="automatic",
        information_cutoff_date=None,
    ) -> ValuationWorkspace:
        session = self.research.create(
            language,
            llm,
            data_source_preference=data_source_preference,
        )
        if information_cutoff_date is not None:
            session.information_cutoff_date = information_cutoff_date
            self.store.save_research(session)
        workspace = ValuationWorkspace(
            workspace_id=_id("workspace_"),
            research_session_id=session.session_id,
            language=language,
            title=title,
            objective=objective,
            run_policy=run_policy,
            information_cutoff_date=information_cutoff_date,
        )
        self.store.create_workspace(workspace)
        self._action(
            workspace,
            "orchestrator",
            "workspace.created",
            "completed",
            "已创建估值工作区；研究取证与金融计算使用独立、可追踪的状态。",
            output_refs=[session.session_id],
        )
        self._sync_requirements(workspace, session, progress=None)
        return workspace

    def get(self, workspace_id: str) -> ValuationWorkspace:
        return self.store.get_workspace(workspace_id)

    def list(self, limit=30) -> list[ValuationWorkspace]:
        return self.store.list_workspaces(limit)

    def _records(self, workspace_id, kind, model, limit=500):
        return self.store.list_workspace_records(
            workspace_id, kind, model, limit=limit
        )

    def _action(
        self,
        workspace,
        actor,
        action_type,
        status,
        summary,
        *,
        input_refs=(),
        output_refs=(),
        error_code=None,
    ):
        action = AgentAction(
            action_id=_id("action_"),
            workspace_id=workspace.workspace_id,
            actor=actor,
            action_type=action_type,
            status=status,
            summary=summary,
            input_refs=list(input_refs),
            output_refs=list(output_refs),
            error_code=error_code,
            completed_at=_now() if status in {"completed", "failed", "skipped"} else None,
        )
        self.store.save_workspace_record(workspace.workspace_id, "action", action, immutable=True)
        return action

    @staticmethod
    def _requirement_id(workspace_id: str, key: str) -> str:
        return "req_" + hashlib.sha256(f"{workspace_id}|{key}".encode()).hexdigest()[:24]

    def _sync_requirements(self, workspace, session, progress):
        requirements = build_requirement_graph(workspace, session, progress)
        active_ids = {item.requirement_id for item in requirements}
        for requirement in requirements:
            self.store.save_workspace_record(
                workspace.workspace_id, "requirement", requirement
            )
        # Method changes do not delete history.  Nodes that are no longer part
        # of the active graph are explicitly superseded for auditability.
        for previous in self._records(
            workspace.workspace_id, "requirement", DataRequirement, limit=2000
        ):
            if previous.requirement_id not in active_ids and previous.status != "superseded":
                previous.status = "superseded"
                previous.resolution = "估值方法或任务范围已变化；该节点不再属于当前需求图。"
                self.store.save_workspace_record(
                    workspace.workspace_id, "requirement", previous
                )
        return requirements

    def _ensure_version(self, workspace, record) -> ValuationVersion:
        version_id = f"version_{record.run_id}"
        try:
            return self.store.get_workspace_record(
                workspace.workspace_id, "version", version_id, ValuationVersion
            )
        except KeyError:
            pass
        versions = self._records(workspace.workspace_id, "version", ValuationVersion)
        checkpoint = None
        if workspace.active_checkpoint_id:
            try:
                checkpoint = self.store.get_workspace_record(
                    workspace.workspace_id,
                    "checkpoint",
                    workspace.active_checkpoint_id,
                    PreValuationCheckpoint,
                )
            except KeyError:
                checkpoint = None
        # Approval freezes the only authoritative input snapshot for this run.
        # Completion may add result-derived ledger entries, but those outputs
        # must never flow backwards and produce a second ModelSpec.
        existing_specs = sorted(
            (
                item for item in self._records(
                    workspace.workspace_id, "model_spec", ModelSpec, limit=1000
                )
                if item.run_id == record.run_id
            ),
            key=lambda item: item.created_at,
        )
        if existing_specs:
            model_spec = existing_specs[0]
        else:
            raise ValueError("计算缺少事前冻结的 ModelSpec；拒绝在完成后补造输入快照。")
        sync_evidence_usage(self.store, workspace, model_spec)
        calculation = build_calculation_record(workspace, record, model_spec)
        try:
            self.store.get_workspace_record(
                workspace.workspace_id,
                "calculation",
                calculation.calculation_id,
                CalculationRecord,
            )
        except KeyError:
            self.store.save_workspace_record(
                workspace.workspace_id, "calculation", calculation, immutable=True
            )
        result_hash = _digest(record.result.model_dump(mode="json")) if record.result else None
        request_changes = {}
        attribution = {}
        if record.parent_run_id:
            try:
                parent_request = self.store.get_run(record.parent_run_id).request.model_dump(mode="json")
            except KeyError:
                parent_request = {}
            current_request = record.request.model_dump(mode="json")
            request_changes = {
                key: {"before": parent_request.get(key), "after": current_request.get(key)}
                for key in sorted(set(parent_request) | set(current_request))
                if parent_request.get(key) != current_request.get(key)
            }
            groups = {
                "financial_data": {"financials", "historical_financials"},
                "assumptions": {"assumptions", "assumption_source"},
                "methods": {"methods", "requested_methods", "excluded_methods"},
                "comparables": {"peers"},
                "capital_structure": {"financials"},
                "sources": {"data_source", "file_ids", "assumption_file_ids"},
            }
            attribution = {
                group: sorted(fields & set(request_changes))
                for group, fields in groups.items()
                if fields & set(request_changes)
            }
        version = ValuationVersion(
            version_id=version_id,
            workspace_id=workspace.workspace_id,
            number=max([item.number for item in versions], default=0) + 1,
            run_id=record.run_id,
            checkpoint_id=checkpoint.checkpoint_id if checkpoint else None,
            model_spec_id=model_spec.model_spec_id,
            calculation_id=calculation.calculation_id,
            parent_version_id=workspace.active_version_id,
            reason=record.revision_reason or ("首次正式估值" if not versions else "估值任务更新"),
            request_hash=record.input_hash,
            result_hash=result_hash,
            changes=request_changes,
            status=str(record.status),
            report_refs={
                format_name: f"/api/runs/{record.run_id}/export?format={format_name}"
                for format_name in ("pdf", "xlsx", "json")
            },
            change_attribution=attribution,
        )
        self.store.save_workspace_record(
            workspace.workspace_id, "version", version, immutable=True
        )
        return version

    def _ensure_challenge_and_decision(self, workspace, record):
        findings = [
            item for item in self._records(
                workspace.workspace_id, "finding", ChallengeFinding
            )
            if item.run_id == record.run_id
        ]
        if not findings:
            findings = self.challenge.review(workspace.workspace_id, record)
            for finding in findings:
                self.store.save_workspace_record(
                    workspace.workspace_id, "finding", finding, immutable=True
                )
            self._action(
                workspace,
                "challenge",
                "challenge.completed",
                "completed",
                f"独立挑战层完成，形成 {len(findings)} 项可复核发现。",
                input_refs=[record.run_id],
                output_refs=[item.finding_id for item in findings],
            )
        decisions = [
            item for item in self._records(
                workspace.workspace_id, "decision", DecisionRecord
            )
            if item.run_id == record.run_id
        ]
        dispositions = [
            item for item in self._records(
                workspace.workspace_id, "disposition", FindingDisposition
            )
            if item.run_id == record.run_id
        ]
        if not dispositions:
            dispositions = self.decisions.dispositions(
                workspace.workspace_id, record, findings
            )
            for disposition in dispositions:
                self.store.save_workspace_record(
                    workspace.workspace_id,
                    "disposition",
                    disposition,
                    immutable=True,
                )
        if not decisions:
            decision = self.decisions.decide(
                workspace.workspace_id, record, findings, dispositions
            )
            self.store.save_workspace_record(
                workspace.workspace_id, "decision", decision, immutable=True
            )
            self._action(
                workspace,
                "decision",
                "decision.completed",
                "completed",
                decision.selected_action,
                input_refs=[record.run_id, *decision.finding_ids],
                output_refs=[decision.decision_id],
            )
            decisions = [decision]
        return findings, decisions[-1]

    def sync(self, workspace_id: str) -> ValuationWorkspace:
        """Refresh derived workspace state without leaking poll races as 500s.

        Browser polling and a background research completion can legitimately
        observe the same revision.  Optimistic locking must still protect user
        mutations, but a read-side state projection can simply reload and
        recompute from the authoritative research/run records.
        """
        with self._sync_lock(workspace_id):
            for attempt in range(5):
                try:
                    return self._sync_once(workspace_id)
                except ValueError as exc:
                    message = str(exc)
                    optimistic_race = (
                        "工作区已被更新" in message
                        or (
                            message.startswith("immutable ")
                            and " already exists:" in message
                        )
                    )
                    if not optimistic_race or attempt == 4:
                        raise
        return self.get(workspace_id)  # pragma: no cover - loop always returns

    def _sync_once(self, workspace_id: str) -> ValuationWorkspace:
        workspace = self.get(workspace_id)
        session = self.store.get_research(workspace.research_session_id)
        sync_evidence_and_fact_ledgers(self.store, workspace, session)
        sync_action_ledger(self.store, workspace, session)
        sync_assumption_ledger(self.store, workspace, session)
        execution = self.research.execution_state(session.session_id)
        progress = valuation_progress(session, self.research.valuation_assembler)
        if workspace.status in {"paused", "cancelled"} and not execution.get("active"):
            self._sync_requirements(workspace, session, progress)
            return workspace
        new_status = workspace.status
        phase = workspace.current_phase
        active_run_id = workspace.active_run_id or session.valuation_run_id
        active_version_id = workspace.active_version_id
        degraded_reason = workspace.degraded_reason

        if active_run_id:
            try:
                record = self.store.get_run(active_run_id)
            except KeyError:
                record = None
            if record is not None:
                sync_run_action_ledger(self.store, workspace, record)
                sync_assumption_ledger(self.store, workspace, session, record)
                if str(record.status) in {"created", "running"}:
                    new_status, phase = "calculating", "calculation"
                elif str(record.status) in {"completed", "completed_with_warnings"}:
                    version = self._ensure_version(workspace, record)
                    active_version_id = version.version_id
                    findings, decision = self._ensure_challenge_and_decision(workspace, record)
                    if decision.outcome == "review_required":
                        new_status = "degraded"
                        degraded_reason = decision.selected_action
                    else:
                        new_status = (
                            "completed_with_warnings"
                            if decision.outcome == "accepted_with_warnings"
                            else "completed"
                        )
                    phase = "reporting"
                elif str(record.status) in {"failed", "waiting_review", "cancelled"}:
                    new_status, phase = "failed", "calculation"
                    degraded_reason = (
                        (record.error or record.review or {}).get("message", "正式估值未完成")
                    )
        elif execution.get("active"):
            new_status, phase = "researching", "evidence"
        elif session.status == "awaiting_input":
            new_status, phase = "awaiting_user", "scope"
        elif progress.get("ready_for_review"):
            new_status, phase = "awaiting_approval", "approval"
        elif session.outcome_status == "insufficient_data":
            new_status, phase = "degraded", "reporting"
            degraded_reason = session.outcome_reason or "有界取证已结束，当前数据不足以形成可靠数值估值。"
        elif session.draft.company or session.draft.ticker:
            new_status, phase = "researching", "evidence"
        else:
            new_status, phase = "setup", "scope"

        self._sync_requirements(workspace, session, progress)
        generated_title = workspace.title
        automatic_titles = {
            "新估值任务",
            (workspace.objective or "")[:40],
        }
        subject = session.draft.company or session.draft.ticker
        if subject and workspace.title in automatic_titles:
            method_label = "/".join(
                method.upper() for method in session.draft.methods[:3]
            )
            generated_title = (
                f"{subject} · {method_label} 估值"
                if method_label
                else f"{subject}估值"
            )[:240]
        changes = {
            "title": generated_title,
            "status": new_status,
            "current_phase": phase,
            "active_run_id": active_run_id,
            "active_version_id": active_version_id,
            "degraded_reason": degraded_reason,
            "information_cutoff_date": (
                session.information_cutoff_date
                or workspace.information_cutoff_date
                or session.draft.valuation_date
            ),
            "industry_strategy": workspace.industry_strategy or session.draft.industry,
            "company_type": workspace.company_type or (
                "financial" if any(
                    token in (session.draft.industry or "")
                    for token in ("银行", "保险", "证券", "券商", "金融")
                ) else "non_financial" if session.draft.industry else ""
            ),
        }
        if any(getattr(workspace, key) != value for key, value in changes.items()):
            for key, value in changes.items():
                setattr(workspace, key, value)
            self.store.save_workspace(workspace)
        return workspace

    def _freeze_prevaluation_request(self, session, progress):
        """Resolve data and assumptions before the user's single approval gate.

        The returned request is self-contained.  Formal calculation therefore
        never needs to re-fetch market data or reinterpret an LLM response after
        the checkpoint is approved.
        """

        from valuationagent.application.input_baseline import validate_baseline_source

        validate_baseline_source(self.store, session)
        preview = preview_session(session)
        preview.valuation_methods_override = list(progress.get("methods") or [])
        preview.valuation_method_exclusions = dict(
            progress.get("excluded_methods") or {}
        )
        request = self.research.valuation_assembler.build(preview)
        if request.input_records:
            from valuationagent.application.file_workspace import source_bytes
            from valuationagent.application.provider_inputs import validate_provider_input
            from valuationagent.application.issuer_identity import audit_request_identities
            from valuationagent.schemas.inputs import InputRecord

            audit_request_identities(self.store, session, request)
            for file_id in {row["source"].get("file_id") for row in request.input_records if row["source"]["kind"] != "user"}:
                _, raw = source_bytes(self.store, session, file_id)
                for row in request.input_records:
                    if row["source"].get("file_id") == file_id and row["source"].get("provider_binding"):
                        validate_provider_input(session, InputRecord.model_validate(row), raw)
        provider_warnings = []
        if str(request.data_source) == "ticker":
            cutoff = session.information_cutoff_date or request.valuation_date
            if cutoff != request.valuation_date:
                raise ValueError(
                    "当前结构化行情适配器尚不能把估值日与更早的信息截止日分开取数；"
                    "请改用可保存历史快照的公开披露/上传资料，或把两日期设为一致。"
                )
            provider = self.research._market_clients.get(  # noqa: SLF001 - application boundary
                session.session_id, self.runner.data
            )
            bundle = provider.resolve(request, self.store)
            provider_values = bundle.assumptions.model_dump(
                mode="json", exclude_none=True
            )
            explicit_values = request.assumptions.model_dump(
                mode="json", exclude_none=True, exclude_defaults=True
            )
            merged_assumptions = AssumptionInputs.model_validate(
                {**provider_values, **explicit_values}
            )
            request = request.model_copy(update={
                "company": bundle.company or request.company,
                "data_source": "structured",
                "financials": bundle.financials,
                "historical_financials": bundle.historical_financials,
                "peers": bundle.peers,
                "assumptions": merged_assumptions,
                "assumption_evidence": {
                    **bundle.assumption_evidence,
                    **request.assumption_evidence,
                },
                "file_ids": [],
                "assumption_file_ids": [],
            })
            provider_warnings = list(bundle.warnings)
        if request.financials is None:
            raise ValueError("冻结估值方案前未取得可计算的基期财务快照。")
        findings = self.runner.finance.validate(request, request.financials)
        blocking = [item for item in findings if item.severity == "blocking"]
        if blocking:
            raise ValueError("；".join(item.message for item in blocking))
        resolved = self.runner.finance.resolve_assumptions(
            request, request.financials
        )

        frozen_values = request.assumptions.model_dump(
            mode="json", exclude_none=True
        )
        resolved_keys = (
            "revenue_growth", "ebit_margin", "revenue_growth_scenarios",
            "ebit_margin_scenarios", "wacc", "terminal_growth",
        ) if "dcf" in request.methods else ()
        for key in resolved_keys:
            value = getattr(resolved, key, None)
            if value not in (None, [], {}):
                frozen_values[key] = value
        resolved_sources = (resolved.wacc_components, resolved.operating_drivers) if "dcf" in request.methods else ()
        for source in resolved_sources:
            for key, value in source.items():
                if key in AssumptionInputs.model_fields:
                    frozen_values[key] = value
        frozen_assumptions = AssumptionInputs.model_validate(frozen_values)
        frozen_request = request.model_copy(update={
            "assumption_source": "manual" if "dcf" in request.methods else request.assumption_source,
            "assumptions": frozen_assumptions,
        })
        # Prove the approved numeric assumptions will be consumed unchanged.
        replay = self.runner.finance.resolve_assumptions(
            frozen_request, frozen_request.financials
        )
        approved_values = {
            "revenue_growth": resolved.revenue_growth,
            "ebit_margin": resolved.ebit_margin,
            "wacc": resolved.wacc,
            "terminal_growth": resolved.terminal_growth,
            "revenue_growth_scenarios": resolved.revenue_growth_scenarios,
            "ebit_margin_scenarios": resolved.ebit_margin_scenarios,
        }
        replay_values = {
            key: getattr(replay, key) for key in approved_values
        }
        if canonical(approved_values) != canonical(replay_values):
            raise ValueError("估值前假设冻结校验未通过，不能提交可能静默变化的方案。")
        return frozen_request, resolved, findings, provider_warnings

    def prevaluation_review(self, workspace_id: str) -> PreValuationCheckpoint:
        current_workspace = self.get(workspace_id)
        if self.store.active(current_workspace.research_session_id) and not self.research._owns_turn(current_workspace.research_session_id):
            raise ValueError("Agent 正在执行；请等待完成或先暂停，再修改计算方案。")
        workspace = self.sync(workspace_id)
        session = self.store.get_research(workspace.research_session_id)
        progress = valuation_progress(session, self.research.valuation_assembler)
        cached = self._records(
            workspace_id, "checkpoint", PreValuationCheckpoint, limit=1000
        )
        for checkpoint in reversed(cached):
            if (
                checkpoint.research_revision == session.revision
                and checkpoint.status in {"pending", "approved"}
                and checkpoint.request_snapshot
            ):
                return checkpoint
        selected_ids = {row.source.source_id for row in session.input_dataset.active_records() if row.source.kind != "user"} if session.input_dataset else None
        clean_facts = [
            fact for fact in session.facts
            if fact.status in {"confirmed", "proposed"} and not fact.warnings
            and (selected_ids is None or fact.fact_id in selected_ids)
        ]
        current_ids = {fact.fact_id for fact in clean_facts}
        ledger_facts = [
            fact for fact in self._records(
                workspace_id, "fact", WorkspaceFact, limit=2000
            )
            if fact.source_fact_id in current_ids and fact.status != "rejected"
        ]
        evidence_ids = sorted({
            evidence_id for fact in ledger_facts for evidence_id in fact.evidence_ids
        })
        evidence = [
            item for item in self._records(
                workspace_id, "evidence", EvidenceRecord, limit=2000
            )
            if item.evidence_id in evidence_ids
        ]
        requirements = [
            item for item in self._records(
                workspace_id, "requirement", DataRequirement, limit=2000
            )
            if item.status != "superseded"
        ]
        model_requirements = [
            item for item in requirements
            if item.category in {
                "historical_financial", "capital_structure",
                "forecast_assumption", "comparable",
            }
        ]
        coverage = (
            sum(item.status == "satisfied" for item in model_requirements)
            / len(model_requirements)
            if model_requirements else 0
        )
        used_tiers = [item.authority_tier for item in evidence]
        grade = max(used_tiers, key="ABCDE".index) if used_tiers else "E"
        future_evidence = [
            item for item in evidence if item.information_cutoff_ok is False
        ]
        preflight_findings = []
        corroborated_facts = [fact for fact in clean_facts
                              if fact.verification.get("source_assessment", {}).get("admission") == "corroborated_draft"]
        if corroborated_facts:
            preflight_findings.append({
                "severity": "warning", "category": "third_party_evidence",
                "title": "输入含跨源一致的第三方财务数据，仍属带来源限制的估值草案依据",
                "fact_ids": [fact.fact_id for fact in corroborated_facts],
                "action": "保留C级及上游独立性未证明的限制；不能描述为官方核验或独立审计。",
            })
        if future_evidence:
            preflight_findings.append({
                "severity": "blocking",
                "category": "information_cutoff",
                "title": "发现信息截止日之后发布的证据",
                "evidence_ids": [item.evidence_id for item in future_evidence],
                "action": "从当时可知口径中剔除，或由用户明确创建事后修订版本。",
            })
        unresolved = [] if progress.get("ready_for_review") else [
            progress.get("blocking_detail") or progress.get("blocking_reason") or "估值输入尚未齐备"
        ]
        if future_evidence:
            unresolved.append("存在信息截止日之后发布并被当前输入引用的资料，不能进入当时可知口径。")
        frozen_request = None
        resolved_assumptions = None
        provider_warnings = []
        if progress.get("ready_for_review") and not future_evidence:
            try:
                (
                    frozen_request,
                    resolved_assumptions,
                    finance_findings,
                    provider_warnings,
                ) = self._freeze_prevaluation_request(session, progress)
                preflight_findings.extend({
                    "severity": item.severity,
                    "category": "financial_model_preflight",
                    "title": item.message,
                    "rule_id": item.rule_id,
                    "action": item.recommended_action or "在报告中披露并纳入敏感性/风险复核。",
                } for item in finance_findings)
                preflight_findings.extend({
                    "severity": "warning",
                    "category": "data_provider",
                    "title": warning,
                    "action": "保留在数据质量与局限披露中。",
                } for warning in provider_warnings)
            except (ValueError, NotImplementedError) as exc:
                unresolved.append(str(exc))
                preflight_findings.append({
                    "severity": "blocking",
                    "category": "model_preflight",
                    "title": "估值输入尚不能冻结",
                    "action": str(exc),
                })
        assumptions = (
            resolved_assumptions.model_dump(mode="json")
            if resolved_assumptions
            else dict(progress.get("assumptions") or {})
        )
        wacc_keys = {
            "wacc", "risk_free_rate", "equity_risk_premium", "beta",
            "debt_cost", "market_cap",
        }
        peer_rows = {}
        for fact in clean_facts:
            if fact.role == "comparable" and fact.peer_ticker:
                peer_rows.setdefault(fact.peer_ticker, {
                    "ticker": fact.peer_ticker,
                    "name": fact.peer_name,
                    "multiples": {},
                })["multiples"][fact.standard_metric or fact.metric] = fact.normalized_value or fact.raw_value
        cutoff = workspace.information_cutoff_date or session.draft.valuation_date
        payload = {
            "research_revision": session.revision,
            "draft": session.draft.model_dump(mode="json"),
            "information_cutoff_date": str(cutoff or ""),
            "execution_date": str(workspace.execution_date),
            "data_source_preference": session.data_source_preference,
            "progress": progress,
            "fact_ids": [fact.fact_id for fact in ledger_facts],
            "evidence_ids": evidence_ids,
            "evidence_coverage": coverage,
            "evidence_grade": grade,
            "preflight_findings": preflight_findings,
            "request_snapshot": (
                frozen_request.model_dump(mode="json") if frozen_request else {}
            ),
        }
        state_hash = _digest(payload)
        existing = self._records(workspace_id, "checkpoint", PreValuationCheckpoint)
        for checkpoint in reversed(existing):
            if checkpoint.state_hash == state_hash and checkpoint.status in {"pending", "approved"}:
                return checkpoint
        for checkpoint in existing:
            if checkpoint.status == "pending":
                checkpoint.status = "superseded"
                self.store.save_workspace_record(workspace_id, "checkpoint", checkpoint)
        checkpoint = PreValuationCheckpoint(
            checkpoint_id=_id("checkpoint_"),
            workspace_id=workspace_id,
            research_revision=session.revision,
            state_hash=state_hash,
            company=session.draft.company,
            ticker=session.draft.ticker,
            valuation_date=str(session.draft.valuation_date or ""),
            information_cutoff_date=str(cutoff or ""),
            execution_date=str(workspace.execution_date),
            industry=session.draft.industry,
            company_type=workspace.company_type,
            currency=workspace.currency,
            requested_methods=list(session.draft.methods),
            executable_methods=list(progress.get("methods") or []),
            excluded_methods=dict(progress.get("excluded_methods") or {}),
            baseline_period=progress.get("baseline_period"),
            inputs=(
                frozen_request.financials.model_dump(mode="json", exclude={"evidence"})
                if frozen_request and frozen_request.financials
                else dict(progress.get("financials") or {})
            ),
            assumptions=assumptions,
            wacc_components=(
                resolved_assumptions.wacc_components
                if resolved_assumptions
                else {key: value for key, value in assumptions.items() if key in wacc_keys}
            ),
            peers=(
                [peer.model_dump(mode="json", exclude={"evidence"}) for peer in frozen_request.peers]
                if frozen_request
                else list(peer_rows.values())
            ),
            capital_actions=[
                {
                    "metric": fact.standard_metric or fact.metric,
                    "period": fact.period,
                    "value": fact.normalized_value or fact.raw_value,
                    "source": fact.block_id,
                }
                for fact in clean_facts
                if any(token in (fact.standard_metric or fact.metric).lower()
                       for token in ("share", "股数", "股本", "回购", "增发"))
            ],
            user_overrides=[
                {
                    "metric": fact.standard_metric or fact.metric,
                    "value": fact.normalized_value or fact.raw_value,
                    "period": fact.period,
                }
                for fact in clean_facts if fact.source_type == "user_note"
            ],
            request_snapshot=(
                frozen_request.model_dump(mode="json") if frozen_request else {}
            ),
            evidence_coverage=coverage,
            evidence_grade=grade,
            preflight_findings=preflight_findings,
            fact_ids=[fact.fact_id for fact in ledger_facts],
            evidence_ids=evidence_ids,
            unresolved_items=unresolved,
            risks=list(progress.get("risks") or []),
        )
        self.store.save_workspace_record(workspace_id, "checkpoint", checkpoint, immutable=True)
        workspace.active_checkpoint_id = checkpoint.checkpoint_id
        if progress.get("ready_for_review"):
            workspace.status, workspace.current_phase = "awaiting_approval", "approval"
        self.store.save_workspace(workspace)
        self._action(
            workspace,
            "orchestrator",
            "checkpoint.created",
            "completed",
            "已生成估值前集中复核快照。" if not unresolved else "已生成准备度快照；仍有不可忽略的输入缺口。",
            input_refs=[session.session_id],
            output_refs=[checkpoint.checkpoint_id],
        )
        return checkpoint

    def approve(self, workspace_id: str, checkpoint_id: str, note="", *, automatic=False):
        with self._sync_lock(workspace_id):
            return self._approve(workspace_id, checkpoint_id, note, automatic=automatic)

    def _approve(self, workspace_id: str, checkpoint_id: str, note="", *, automatic=False):
        workspace = self.sync(workspace_id)
        if automatic and workspace.run_policy != "automatic":
            raise ValueError("审阅模式不能使用自动批准")
        checkpoint = self.store.get_workspace_record(
            workspace_id, "checkpoint", checkpoint_id, PreValuationCheckpoint
        )
        current = self.prevaluation_review(workspace_id)
        if current.state_hash != checkpoint.state_hash:
            if checkpoint.status == "pending":
                checkpoint.status = "superseded"
                self.store.save_workspace_record(workspace_id, "checkpoint", checkpoint)
            raise ValueError("工作区数据已变化，请复核最新方案后再批准。")
        if checkpoint.status == "approved":
            if workspace.active_run_id:
                return self.store.get_run(workspace.active_run_id)
            raise ValueError("方案已批准但未找到运行记录，请刷新工作区。")
        if (
            checkpoint.unresolved_items
            or not checkpoint.executable_methods
            or not checkpoint.request_snapshot
        ):
            raise ValueError("当前方案仍有阻塞缺口，不能批准为正式数值估值。")

        session = self.research.approve_valuation_plan(
            workspace.research_session_id,
            {
                "methods": checkpoint.executable_methods,
                "requested_methods": checkpoint.requested_methods,
                "excluded_methods": checkpoint.excluded_methods,
                "baseline_period": checkpoint.baseline_period,
            },
            automatic=automatic,
        )
        frozen_request = ValuationRequest.model_validate(checkpoint.request_snapshot)
        record = self.research.submit_valuation(
            session.session_id, self.runner, request_override=frozen_request
        )
        # The accepted forecast package becomes an approved assumption ledger
        # entry before ModelSpec is frozen; later calculations cannot silently
        # replace it.
        session = self.store.get_research(workspace.research_session_id)
        sync_evidence_and_fact_ledgers(self.store, workspace, session)
        sync_assumption_ledger(self.store, workspace, session)
        model_spec = build_model_spec(
            self.store, workspace, record, checkpoint, self.runner.finance.version
        )
        try:
            self.store.get_workspace_record(
                workspace_id, "model_spec", model_spec.model_spec_id, ModelSpec
            )
        except KeyError:
            self.store.save_workspace_record(
                workspace_id, "model_spec", model_spec, immutable=True
            )
        sync_evidence_usage(self.store, workspace, model_spec)
        self.store.freeze_run_sources(record.run_id, session)
        checkpoint.status = "approved"
        checkpoint.approval_note = note
        checkpoint.approved_by = "automatic_policy" if automatic else "user"
        checkpoint.approved_at = _now()
        self.store.save_workspace_record(workspace_id, "checkpoint", checkpoint)
        workspace = self.get(workspace_id)
        workspace.active_checkpoint_id = checkpoint_id
        workspace.active_run_id = record.run_id
        workspace.status, workspace.current_phase = "calculating", "calculation"
        self.store.save_workspace(workspace)
        self._action(
            workspace,
            "orchestrator" if automatic else "user",
            "checkpoint.approved",
            "completed",
            "自动策略已接受可复现输入并创建估值草案计算任务；不代表人工批准。" if automatic else "用户已批准可复现的估值输入快照，正式计算任务已创建。",
            input_refs=[checkpoint_id],
            output_refs=[model_spec.model_spec_id, record.run_id],
        )
        return record

    def reopen(self, workspace_id: str, checkpoint_id: str, note=""):
        workspace = self.get(workspace_id)
        checkpoint = self.store.get_workspace_record(
            workspace_id, "checkpoint", checkpoint_id, PreValuationCheckpoint
        )
        if checkpoint.status == "approved" and workspace.active_run_id:
            record = self.store.get_run(workspace.active_run_id)
            if str(record.status) in {"created", "running"}:
                raise ValueError("正式计算已开始，不能改写已批准快照；请在完成后创建新版本。")
        checkpoint.status = "reopened"
        checkpoint.approval_note = note
        self.store.save_workspace_record(workspace_id, "checkpoint", checkpoint)
        workspace.status, workspace.current_phase = "researching", "model_design"
        workspace.active_checkpoint_id = None
        self.store.save_workspace(workspace)
        self._action(
            workspace, "user", "checkpoint.reopened", "completed",
            note or "用户要求修改估值方案。", input_refs=[checkpoint_id]
        )
        return checkpoint

    def revise(self, workspace_id: str, reason: str, changes: dict[str, Any]):
        with self._sync_lock(workspace_id):
            return self._revise(workspace_id, reason, changes)

    def _revise(self, workspace_id: str, reason: str, changes: dict[str, Any]):
        current_workspace = self.get(workspace_id)
        if self.store.active(current_workspace.research_session_id) and not self.research._owns_turn(current_workspace.research_session_id):
            raise ValueError("Agent 正在执行；请等待完成或先暂停，再修改计算方案。")
        workspace = self.sync(workspace_id)
        if not workspace.active_run_id:
            raise ValueError("尚无正式估值版本；请先完成估值前复核。")
        changes = dict(changes)
        if "methods" in changes:
            # A method switch is a complete user choice for this version.  Keep
            # requested/effective/excluded sets internally consistent even when
            # the UI expands beyond the methods used by the prior version.
            methods = list(dict.fromkeys(changes["methods"] or []))
            if not methods:
                raise ValueError("至少保留一种估值方法。")
            changes["methods"] = methods
            changes["requested_methods"] = methods
            changes["excluded_methods"] = {}
        child = self.runner.revise(
            workspace.active_run_id,
            RevisionInput(reason=reason, changes=changes),
            execute=False,
        )
        session = self.store.get_research(workspace.research_session_id)
        sync_evidence_and_fact_ledgers(self.store, workspace, session)
        model_spec = build_model_spec(
            self.store, workspace, child, None, self.runner.finance.version
        )
        self.store.save_workspace_record(
            workspace_id, "model_spec", model_spec, immutable=True
        )
        sync_evidence_usage(self.store, workspace, model_spec)
        self.store.freeze_run_sources(child.run_id, session)
        session.valuation_run_id = child.run_id
        self.store.save_research(session)
        workspace.active_run_id = child.run_id
        workspace.status, workspace.current_phase = "calculating", "calculation"
        workspace.active_checkpoint_id = None
        self.store.save_workspace(workspace)
        self._action(
            workspace,
            "user",
            "version.requested",
            "completed",
            reason,
            input_refs=[child.parent_run_id] if child.parent_run_id else [],
            output_refs=[child.run_id],
        )
        return child

    def message(self, workspace_id: str, turn: ResearchTurn, *, reserved=False):
        workspace = self.get(workspace_id)
        if workspace.status == "cancelled":
            raise ValueError("工作区已取消，请创建新工作区。")
        if workspace.active_run_id:
            record = self.store.get_run(workspace.active_run_id)
            if str(record.status) in {"created", "running"}:
                raise ValueError("正在计算，请等待完成或先暂停。")
        if workspace.status == "paused":
            workspace.status = "researching"
            self.store.save_workspace(workspace)
        try:
            return self.research.turn(
                workspace.research_session_id, turn,
                reserved=reserved, compact_result=reserved,
            )
        finally:
            self.sync(workspace_id)

    def calculate_from_agent(self, session_id, args):
        workspace = self.store.workspace_for_research(session_id)
        if workspace is None:
            raise ValueError("当前会话不属于工作区。")
        if args.changes:
            if workspace.run_policy == "review":
                return {"status": "approval_required", "changes": args.changes,
                        "instruction": "向用户展示修改建议，等待用户通过调整参数面板确认；本次没有计算。"}
            record = self.revise(workspace.workspace_id, args.reason, args.changes)
        else:
            checkpoint = self.prevaluation_review(workspace.workspace_id)
            if checkpoint.unresolved_items:
                return {"status": "inputs_missing", "gaps": checkpoint.unresolved_items,
                        "preflight_findings": checkpoint.preflight_findings}
            if workspace.run_policy == "review":
                return {"status": "approval_required",
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "instruction": "方案已冻结，等待用户审批；尚未计算。"}
            record = self.approve(
                workspace.workspace_id, checkpoint.checkpoint_id,
                "自动模式生成估值草案；机器核验不代表人工审核",
                automatic=True,
            )
        self.research._check_execution()
        if str(record.status) not in {"completed", "completed_with_warnings"}:
            self.runner.execute(record.run_id)
        self.sync(workspace.workspace_id)
        return self.read_valuation(session_id, "summary")

    def read_valuation(self, session_id, section="summary"):
        workspace = self.store.workspace_for_research(session_id)
        if workspace is None or not workspace.active_run_id:
            return {"status": "not_calculated",
                    "run_policy": workspace.run_policy if workspace else None}
        record = self.store.get_run(workspace.active_run_id)
        output = {"run_id": record.run_id, "status": str(record.status),
                  "run_policy": workspace.run_policy,
                  "input_hash": record.input_hash, "error": record.error}
        from valuationagent.application.result_delivery import method_completion

        output["method_completion"] = method_completion(record)
        if section == "request":
            output["request"] = record.request.model_dump(mode="json")
        elif section == "findings":
            output["findings"] = [item.model_dump(mode="json") for item in
                self._records(workspace.workspace_id, "finding", ChallengeFinding)
                if item.run_id == record.run_id]
        elif record.result:
            from valuationagent.application.result_views import financial_display

            output["financial_display"] = financial_display(record.result)
            fields = None if section == "result" else {
                "executive_summary", "dcf", "relative", "assumptions", "warnings",
                "data_quality", "reconciliation", "currency", "analysis_basis",
            }
            output["result"] = record.result.model_dump(mode="json", include=fields)
            output["delivery_instruction"] = "完成估值/报告时，finish_response的answer只写定性说明，不抄数字、日期、代码或下载链接。系统将从当前冻结结果渲染各方法数值表、真实基期、可比样本和报告链接；未绑定的定量重述不发布，也不重新计算或下载。"
        return output

    def execute(self, run_id):
        try:
            return self.runner.execute(run_id)
        finally:
            workspace = self.store.workspace_for_run(run_id)
            if workspace:
                self.sync(workspace.workspace_id)


    def pause(self, workspace_id: str):
        workspace = self.get(workspace_id)
        self.research.cancel_turn(workspace.research_session_id)
        if workspace.active_run_id:
            try:
                record = self.store.get_run(workspace.active_run_id)
                if str(record.status) in {"created", "running"}:
                    self.runner.request_pause(record.run_id)
            except KeyError:
                pass
        workspace.status = "paused"
        self.store.save_workspace(workspace)
        self._action(
            workspace, "user", "workspace.paused", "completed",
            "用户暂停任务；已取得资料、不可变快照和版本均保留。",
        )
        return workspace

    def cancel(self, workspace_id: str):
        workspace = self.get(workspace_id)
        self.research.cancel_turn(workspace.research_session_id)
        if workspace.active_run_id:
            self.runner.request_pause(workspace.active_run_id)
        workspace.status = "cancelled"
        self.store.save_workspace(workspace)
        self._action(
            workspace, "user", "workspace.cancelled", "completed",
            "用户取消后续执行；历史证据、账本和已完成版本仍可导出。",
        )
        return workspace

    def resume(self, workspace_id: str):
        workspace = self.get(workspace_id)
        if workspace.status != "paused":
            raise ValueError("工作区当前不是暂停状态。")
        run_id = None
        if workspace.active_run_id:
            try:
                record = self.store.get_run(workspace.active_run_id)
            except KeyError:
                record = None
            if record and str(record.status) == "running":
                raise ValueError("计算正在到达安全暂停点，请稍后再继续。")
            if record and str(record.status) == "waiting_review" and (
                record.review or {}
            ).get("code") == "USER_PAUSED":
                run_id = record.run_id
                workspace.status, workspace.current_phase = "calculating", "calculation"
            else:
                workspace.status = "researching"
        else:
            workspace.status = "researching"
        self.store.save_workspace(workspace)
        self._action(
            workspace, "user", "workspace.resumed", "completed",
            "用户继续任务；从已保存的需求图和证据账本恢复。",
        )
        return {"workspace": workspace, "run_id": run_id}

    def add_evidence(self, workspace_id: str, evidence: EvidenceRecord):
        self.get(workspace_id)
        if evidence.workspace_id != workspace_id:
            raise ValueError("evidence workspace mismatch")
        return self.store.save_workspace_record(workspace_id, "evidence", evidence, immutable=True)

    def add_fact(self, workspace_id: str, fact: WorkspaceFact):
        self.get(workspace_id)
        if fact.workspace_id != workspace_id:
            raise ValueError("fact workspace mismatch")
        known = {item.evidence_id for item in self._records(workspace_id, "evidence", EvidenceRecord)}
        if not set(fact.evidence_ids) <= known:
            raise ValueError("事实引用了工作区中不存在的证据。")
        return self.store.save_workspace_record(workspace_id, "fact", fact, immutable=True)

    def add_assumption(self, workspace_id: str, assumption: WorkspaceAssumption):
        self.get(workspace_id)
        if assumption.workspace_id != workspace_id:
            raise ValueError("assumption workspace mismatch")
        return self.store.save_workspace_record(workspace_id, "assumption", assumption, immutable=True)

    def compare_versions(self, workspace_id: str, left_id: str, right_id: str):
        left = self.store.get_workspace_record(workspace_id, "version", left_id, ValuationVersion)
        right = self.store.get_workspace_record(workspace_id, "version", right_id, ValuationVersion)
        left_run, right_run = self.store.get_run(left.run_id), self.store.get_run(right.run_id)
        left_request = left_run.request.model_dump(mode="json")
        right_request = right_run.request.model_dump(mode="json")
        request_changes = {
            key: {"left": left_request.get(key), "right": right_request.get(key)}
            for key in sorted(set(left_request) | set(right_request))
            if left_request.get(key) != right_request.get(key)
        }

        def outputs(record):
            if not record.result:
                return {"status": str(record.status)}
            return {
                "status": str(record.status),
                "dcf_per_share": str(record.result.dcf.per_share_value) if record.result.dcf else None,
                "relative": {
                    item.method: str(item.per_share_value)
                    for item in record.result.relative if item.status == "success"
                },
                "confidence": record.result.data_quality.confidence,
                "grade": record.result.data_quality.result_grade,
            }
        return {
            "left": left.model_dump(mode="json"),
            "right": right.model_dump(mode="json"),
            "request_changes": request_changes,
            "output_changes": {"left": outputs(left_run), "right": outputs(right_run)},
        }

    def timeline(self, workspace_id: str):
        workspace = self.get(workspace_id)
        actions = self._records(workspace_id, "action", AgentAction, limit=1000)
        rows = [
            {
                "id": action.action_id,
                "timestamp": action.started_at.isoformat(),
                "actor": action.actor,
                "type": action.action_type,
                "status": action.status,
                "summary": action.summary,
                "objective": action.objective,
                "tool": action.tool_name,
                "duration_ms": action.duration_ms,
                "input_refs": action.input_refs,
                "output_refs": action.output_refs,
            }
            for action in actions
        ]
        if workspace.active_run_id:
            try:
                run_events = self.store.list_events(workspace.active_run_id)
            except KeyError:
                run_events = []
            rows.extend({
                "id": f"run_event_{event.sequence}",
                "timestamp": event.timestamp.isoformat(),
                "actor": "modeling",
                "type": event.type,
                "status": event.status or "completed",
                "summary": event.summary,
                "input_refs": [],
                "output_refs": [workspace.active_run_id],
            } for event in run_events)
        return sorted(rows, key=lambda item: (item["timestamp"], item["id"]))

    def snapshot(self, workspace_id: str):
        workspace = self.get(workspace_id)
        session = self.store.get_research(workspace.research_session_id)
        research = self.research.snapshot(workspace.research_session_id, compact=True)
        active_run = None
        active_record = None
        if workspace.active_run_id:
            try:
                active_record = self.store.get_run(workspace.active_run_id)
                active_run = active_record.model_dump(mode="json")
            except KeyError:
                pass
        requirements = self._records(
            workspace_id, "requirement", DataRequirement, limit=2000
        )
        evidence = self._records(
            workspace_id, "evidence", EvidenceRecord, limit=2000
        )
        evidence_usage = self._records(
            workspace_id, "evidence_usage", EvidenceUsage, limit=2000
        )
        facts = self._records(
            workspace_id, "fact", WorkspaceFact, limit=2000
        )
        assumptions = self._records(
            workspace_id, "assumption", WorkspaceAssumption, limit=1000
        )
        actions = self._records(
            workspace_id, "action", AgentAction, limit=2000
        )
        model_specs = self._records(
            workspace_id, "model_spec", ModelSpec, limit=1000
        )
        calculations = self._records(
            workspace_id, "calculation", CalculationRecord, limit=1000
        )
        dispositions = self._records(
            workspace_id, "disposition", FindingDisposition, limit=1000
        )
        return {
            "workspace": workspace.model_dump(mode="json"),
            "research": research,
            "messages": research["messages"],
            "execution": research["execution"],
            "plan": session.plan,
            "artifacts": self.store.list_artifacts(session.session_id),
            "research_plan": research_plan(session, self.research.valuation_assembler, active_record),
            "runtime": {"agent_version": AGENT_PROMPT_VERSION, "session_agent_version": session.prompt_version},
            "active_run": active_run,
            "connections": {
                "model": {
                    "available": self.research.model_connected(
                        workspace.research_session_id
                    ),
                    "provider": session.model_provider,
                    "name": session.model_name,
                }
            },
            "requirements": [item.model_dump(mode="json") for item in requirements],
            "requirement_edges": requirement_edges(requirements),
            "evidence_ledger": [item.model_dump(mode="json") for item in evidence],
            "evidence_usage": [item.model_dump(mode="json") for item in evidence_usage],
            "fact_ledger": [item.model_dump(mode="json") for item in facts],
            "assumption_ledger": [item.model_dump(mode="json") for item in assumptions],
            "action_ledger": [item.model_dump(mode="json") for item in actions],
            "checkpoints": [item.model_dump(mode="json") for item in self._records(workspace_id, "checkpoint", PreValuationCheckpoint)],
            "model_specs": [item.model_dump(mode="json") for item in model_specs],
            "calculation_ledger": [item.model_dump(mode="json") for item in calculations],
            "versions": [item.model_dump(mode="json") for item in self._records(workspace_id, "version", ValuationVersion)],
            "findings": [item.model_dump(mode="json") for item in self._records(workspace_id, "finding", ChallengeFinding)],
            "finding_dispositions": [item.model_dump(mode="json") for item in dispositions],
            "decisions": [item.model_dump(mode="json") for item in self._records(workspace_id, "decision", DecisionRecord)],
        }

    def reproducibility_manifest(self, workspace_id: str):
        snapshot = self.snapshot(workspace_id)
        workspace = self.get(workspace_id)
        session = self.store.get_research(workspace.research_session_id)
        active = self.store.get_run(workspace.active_run_id) if workspace.active_run_id else None
        manifest = {
            "schema": "valuation-workspace-replay-v2",
            "workspace_id": workspace_id,
            "workspace_revision": workspace.revision,
            "research_session_id": session.session_id,
            "research_revision": session.revision,
            "agent_protocol_version": session.agent_protocol_version,
            "prompt_version": session.prompt_version,
            "model": {"provider": session.model_provider, "name": session.model_name},
            "active_run_id": workspace.active_run_id,
            "active_version_id": workspace.active_version_id,
            "information_cutoff_date": str(workspace.information_cutoff_date or ""),
            "execution_date": str(workspace.execution_date),
            "input_hash": active.input_hash if active else None,
            "result_hash": _digest(active.result.model_dump(mode="json")) if active and active.result else None,
            "source_hashes": sorted(
                item.get("sha256", "")
                for item in snapshot["research"]["session"].get("documents", [])
                if item.get("sha256")
            ),
            "checkpoint_hashes": [item["state_hash"] for item in snapshot["checkpoints"]],
            "model_specs": [
                {
                    "model_spec_id": item["model_spec_id"],
                    "snapshot_hash": item["snapshot_hash"],
                    "financial_model_version": item["financial_model_version"],
                }
                for item in snapshot["model_specs"]
            ],
            "calculations": [
                {
                    "calculation_id": item["calculation_id"],
                    "model_spec_id": item["model_spec_id"],
                    "input_hash": item["input_hash"],
                    "result_hash": item["result_hash"],
                    "replayable_offline": item["replayable_offline"],
                }
                for item in snapshot["calculation_ledger"]
            ],
            "reproducibility": {
                "calculation": "offline_exact" if snapshot["model_specs"] else "not_yet_frozen",
                "acquisition": "snapshot_or_excerpt_with_hash",
                "note": "外部网页可变化；计算使用冻结 ModelSpec，数据获取使用快照、引文、URL、抓取时间和哈希复核。",
            },
            "timeline": self.timeline(workspace_id),
        }
        manifest["manifest_hash"] = _digest(manifest)
        return manifest
