"""Four auditable ledgers for evidence, actions, decisions and calculations.

This adapter deliberately stores references, hashes and structured outcomes —
never model chain-of-thought and never credentials. Evidence-session storage
stays separate from the workspace contract exposed to users and reports.
"""

from __future__ import annotations

import hashlib
import platform
import re
from datetime import date, datetime, timezone
from typing import Any

from valuationagent import __version__
from valuationagent.core.tools import canonical
from valuationagent.application.evidence_status import observation_verified
from valuationagent.schemas.workspace import (
    AgentAction,
    CalculationRecord,
    EvidenceRecord,
    EvidenceUsage,
    ModelSpec,
    WorkspaceAssumption,
    WorkspaceFact,
)


FORMULA_MANIFEST = {
    "fcff": "EBIT × (1 - tax_rate) + depreciation_amortization - capital_expenditure - change_operating_nwc",
    "terminal_value": "FCFF_(n+1) / (WACC - terminal_growth)",
    "dcf_equity_bridge": "enterprise_value + cash_and_non_operating_assets - interest_bearing_debt - other_claims",
    "dcf_per_share": "common_equity_value / common_shares",
    "pe_per_share": "net_income_parent × comparable_PE / common_shares",
    "ps_per_share": "revenue × comparable_PS / common_shares",
    "ev_ebitda_per_share": "(EBITDA × comparable_EV_EBITDA + cash - debt) / common_shares",
}

CALCULATION_ORDER = [
    "validate_scope_and_information_cutoff",
    "validate_financial_statements_and_units",
    "resolve_assumptions",
    "forecast_operating_results",
    "calculate_fcff",
    "discount_explicit_cash_flows",
    "calculate_terminal_value",
    "bridge_enterprise_to_common_equity",
    "calculate_relative_valuation",
    "run_sensitivity",
    "reconcile_methods",
]

_SECRET = re.compile(
    r"(?i)(?:api[_ -]?key|token|secret|authorization)\s*[:=]\s*[^\s,;]+|"
    r"\b(?:sk|tvly)-[A-Za-z0-9_-]{12,}\b"
)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *values: Any) -> str:
    return prefix + hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()[:28]


def _safe_text(value: Any, limit=1200) -> str:
    return _SECRET.sub("[REDACTED]", str(value or ""))[:limit]


def _source_type(document, fact) -> str:
    if fact.source_type == "user_note":
        return "user_statement"
    return {
        "official_filing": "regulatory_filing",
        "official_index": "exchange_data",
        "user_upload": "user_document",
        "public_web": "public_web",
        "structured_provider": "licensed_market_data",
        "search_snippet": "public_web",
    }.get(getattr(document, "provenance_type", "unknown"), "public_web")


def _document_for_fact(session, fact):
    file_id = fact.block_id.split(":", 1)[0]
    return next((item for item in session.documents if item.file_id == file_id), None)


def _published_at(fact, location):
    value = fact.published_at or location.get("published_at")
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _cutoff_ok(published_at, cutoff):
    if not published_at or not cutoff:
        return None
    return published_at.date() <= cutoff


def sync_evidence_and_fact_ledgers(store, workspace, session) -> None:
    """Project source-bound research facts into immutable workspace records."""

    known_evidence = {
        item.evidence_id
        for item in store.list_workspace_records(
            workspace.workspace_id, "evidence", EvidenceRecord, limit=2000
        )
    }
    known_facts = {
        item.fact_id
        for item in store.list_workspace_records(
            workspace.workspace_id, "fact", WorkspaceFact, limit=2000
        )
    }
    all_blocks = {
        block["block_id"]: block
        for document in session.documents
        for block in store.research_blocks(session.session_id, document.file_id)
    }
    user_messages = {
        "message:" + message.message_id: message
        for message in store.list_messages(session.session_id)
        if message.role == "user"
    }
    cutoff = (
        session.information_cutoff_date
        or workspace.information_cutoff_date
        or session.draft.valuation_date
    )

    def fact_identity(fact):
        entity = (fact.peer_ticker or fact.peer_name) if fact.role == "comparable" else (session.draft.ticker or session.draft.company)
        return (entity.casefold(), fact.role, fact.standard_metric or fact.metric,
                fact.period, fact.scope, fact.multiple_basis, fact.unit)

    identities: dict[tuple, list] = {}
    for fact in session.facts:
        if fact.status == "rejected":
            continue
        key = fact_identity(fact)
        identities.setdefault(key, []).append(fact)

    def evidence_identity(candidate):
        candidate_document = _document_for_fact(session, candidate)
        candidate_block = all_blocks.get(candidate.block_id)
        candidate_message = user_messages.get(candidate.block_id)
        candidate_excerpt = candidate.quote or (candidate_block or {}).get("text") or (
            candidate_message.content if candidate_message else ""
        )
        return stable_id(
            "evidence_",
            workspace.workspace_id,
            candidate.block_id,
            getattr(candidate_document, "sha256", ""),
            digest(candidate_excerpt),
            str(cutoff or ""),
            digest(candidate.verification.get("source_assessment", {})),
            digest(candidate.verification.get("table_interpretation", {})),
            digest(candidate.verification.get("observation", {})),
            digest(candidate.verification.get("reading_proof", {})),
            digest(candidate.verification.get("semantic_review", {})),
            tuple(candidate.warnings),
        )

    for fact in session.facts:
        if fact.status == "rejected":
            continue
        document = _document_for_fact(session, fact)
        block = all_blocks.get(fact.block_id)
        message = user_messages.get(fact.block_id)
        location = dict(fact.source_location or {})
        if block:
            location = {**(block.get("location") or {}), **location}
        elif message:
            location = {"message_id": message.message_id, "source_type": "user_note"}
        excerpt = fact.quote or (block or {}).get("text") or (message.content if message else "")
        authority_tier = getattr(document, "authority_tier", "D") if document else "D"
        authority_score = {"A": .98, "B": .85, "C": .65, "D": .4, "E": .2}[authority_tier]
        published_at = _published_at(fact, location)
        cutoff_ok = _cutoff_ok(published_at, cutoff)
        directness = .95 if document and document.provenance_type in {
            "official_filing", "official_index", "user_upload"
        } else .35 if document and document.provenance_type == "search_snippet" else .65
        security_flags = list(location.get("security_flags") or [])
        extraction_score = .5 if security_flags else .95 if observation_verified(fact) else .35
        entity_score = .9 if fact.scope != "unknown" else .45
        timing_score = .95 if cutoff_ok is True else .1 if cutoff_ok is False else .5
        conflicts = [
            other for other in identities.get(
                fact_identity(fact), []
            )
            if other.fact_id != fact.fact_id
            and other.normalized_value != fact.normalized_value
        ]
        evidence_id = evidence_identity(fact)
        if evidence_id not in known_evidence:
            evidence = EvidenceRecord(
                evidence_id=evidence_id,
                workspace_id=workspace.workspace_id,
                source_type=_source_type(document, fact),
                authority_tier=authority_tier,
                title=(document.name if document else "用户对话输入"),
                provider=(getattr(document, "provider", "") if document else "user"),
                publisher=(getattr(document, "provider", "") if document else "user"),
                url=fact.source_url or (getattr(document, "source_url", "") if document else ""),
                locator=location,
                excerpt=_safe_text(excerpt, 4000),
                source_sha256=fact.source_sha256 or getattr(document, "sha256", ""),
                published_at=published_at,
                information_cutoff_ok=cutoff_ok,
                snapshot_file_id=getattr(document, "file_id", "") if document else "",
                extraction_tool="valuationagent.document_parser+semantic_mapper",
                extraction_version=__version__,
                binding_proof={key: fact.verification[key] for key in ("observation", "reading_proof", "semantic_review", "table_interpretation") if key in fact.verification},
                authority_score=authority_score,
                directness_score=directness,
                entity_scope_score=entity_score,
                timing_score=timing_score,
                extraction_score=extraction_score,
                cross_source_score=.8 if conflicts == [] and len(identities.get(
                    fact_identity(fact), []
                )) > 1 else .5,
                confidence=min(
                    getattr(document, "source_confidence", authority_score) if document else .5,
                    extraction_score,
                    entity_score,
                    timing_score,
                ),
                status="conflicted" if conflicts else "verified" if observation_verified(fact) and not fact.verification.get("source_assessment", {}).get("source_issues") and fact.verification.get("semantic_review", {}).get("status", "supported") == "supported" else "candidate",
                conflict_evidence_ids=[evidence_identity(other) for other in conflicts],
                adopted_fact_ids=[fact.fact_id] if fact.status == "confirmed" else [],
                limitations=[
                    *fact.warnings,
                    *fact.verification.get("source_assessment", {}).get("limitations", []),
                    *(["来源含疑似提示注入文本；已作为不可信数据隔离，未执行其中指令。"] if security_flags else []),
                ],
            )
            store.save_workspace_record(
                workspace.workspace_id, "evidence", evidence, immutable=True
            )
            known_evidence.add(evidence_id)

        # The adapter makes the state transition explicit instead of mutating
        # an earlier immutable observation.
        ledger_fact_id = stable_id(
            "fact_", workspace.workspace_id, fact.fact_id, fact.status,
            fact.normalized_value, tuple(fact.warnings), evidence_id
        )
        if ledger_fact_id in known_facts:
            continue
        impact = "critical" if (fact.standard_metric or fact.metric) in {
            "common_shares", "revenue", "cash_and_non_operating_assets",
            "interest_bearing_debt", "net_income_parent",
        } else "high" if fact.role in {"historical", "comparable"} else "normal"
        conflict_ids = [
            stable_id("fact_", workspace.workspace_id, other.fact_id, other.status,
                      other.normalized_value, tuple(other.warnings), evidence_identity(other))
            for other in conflicts
        ]
        workspace_fact = WorkspaceFact(
            fact_id=ledger_fact_id,
            workspace_id=workspace.workspace_id,
            source_fact_id=fact.fact_id,
            issuer=(fact.peer_name or fact.peer_ticker) if fact.role == "comparable" else (session.draft.company or session.draft.ticker),
            metric=fact.standard_metric or fact.metric,
            raw_metric=fact.metric,
            raw_value=fact.raw_value,
            normalized_value=fact.normalized_value,
            unit=fact.unit,
            currency=fact.verification["reading_proof"]["basis"].get("currency") if fact.verification.get("reading_proof") else "CNY",
            period=fact.period,
            scope=fact.scope,
            assertion_type="user_input" if fact.source_type == "user_note" else "source_fact",
            evidence_ids=[evidence_id],
            directly_disclosed=not bool(fact.verification.get("derived")),
            derivation_formula=str(fact.verification.get("formula") or ""),
            conflict_fact_ids=conflict_ids,
            valuation_impact=impact,
            confidence=None if fact.verification.get("reading_proof") else min(fact.mapping_confidence or .5, extraction_score),
            status="conflicted" if conflicts else "confirmed" if fact.status == "confirmed" else "proposed",
            supersedes=[stable_id("fact_", workspace.workspace_id, item, "confirmed", "", ()) for item in session.staged_supersessions.get(fact.fact_id, [])],
        )
        store.save_workspace_record(
            workspace.workspace_id, "fact", workspace_fact, immutable=True
        )
        known_facts.add(ledger_fact_id)


def sync_action_ledger(store, workspace, session) -> None:
    """Copy visible research actions without persisting raw arguments or secrets."""

    known = {
        item.action_id
        for item in store.list_workspace_records(
            workspace.workspace_id, "action", AgentAction, limit=2000
        )
    }
    for event in store.list_events(session.session_id):
        action_id = stable_id(
            "action_", workspace.workspace_id, session.session_id, event.sequence
        )
        if action_id in known:
            continue
        payload_digest = digest(_safe_text(canonical(event.payload), 4000))
        action = AgentAction(
            action_id=action_id,
            workspace_id=workspace.workspace_id,
            actor="research" if not event.tool else "extraction",
            action_type=event.type,
            status=(
                "failed" if event.status == "failed"
                else "running" if event.status == "running"
                else "completed"
            ),
            summary=_safe_text(event.summary),
            objective=_safe_text((event.payload or {}).get("reason", "")),
            tool_name=event.tool or "",
            tool_version=__version__,
            input_digest=payload_digest,
            output_digest=digest({"status": event.status, "summary": _safe_text(event.summary)}),
            change_reason=_safe_text((event.payload or {}).get("instruction", "")),
            output_refs=[event.tool_call_id] if event.tool_call_id else [],
            error_code=(event.type if event.status == "failed" else None),
            started_at=event.timestamp,
            completed_at=(None if event.status == "running" else event.timestamp),
            duration_ms=event.duration_ms,
        )
        store.save_workspace_record(
            workspace.workspace_id, "action", action, immutable=True
        )
        known.add(action_id)


def sync_run_action_ledger(store, workspace, record) -> None:
    """Persist deterministic financial-tool events in the same action book."""

    known = {
        item.action_id
        for item in store.list_workspace_records(
            workspace.workspace_id, "action", AgentAction, limit=2000
        )
    }
    for event in store.list_events(record.run_id):
        action_id = stable_id(
            "action_", workspace.workspace_id, record.run_id, event.sequence
        )
        if action_id in known:
            continue
        safe_payload = _safe_text(canonical(event.payload), 4000)
        action = AgentAction(
            action_id=action_id,
            workspace_id=workspace.workspace_id,
            actor="modeling",
            action_type=event.type,
            status=(
                "failed" if event.status == "failed"
                else "running" if event.status == "running"
                else "completed"
            ),
            summary=_safe_text(event.summary),
            objective="执行已冻结 ModelSpec 的确定性计算或校验",
            tool_name=event.tool or "financial_pipeline",
            tool_version=record.workflow_version,
            input_digest=digest({"run_id": record.run_id, "payload": safe_payload}),
            output_digest=digest({"status": event.status, "summary": _safe_text(event.summary)}),
            output_refs=[record.run_id, *([event.tool_call_id] if event.tool_call_id else [])],
            error_code=event.type if event.status == "failed" else None,
            started_at=event.timestamp,
            completed_at=None if event.status == "running" else event.timestamp,
            duration_ms=event.duration_ms,
        )
        store.save_workspace_record(
            workspace.workspace_id, "action", action, immutable=True
        )
        known.add(action_id)


def sync_assumption_ledger(store, workspace, session, record=None) -> None:
    """Materialize proposed/approved assumptions separately from source facts."""

    existing = store.list_workspace_records(
        workspace.workspace_id, "assumption", WorkspaceAssumption, limit=2000
    )
    known = {item.assumption_id for item in existing}
    facts = store.list_workspace_records(
        workspace.workspace_id, "fact", WorkspaceFact, limit=2000
    )
    evidence_by_source_fact = {
        item.source_fact_id: evidence_id
        for item in facts
        for evidence_id in item.evidence_ids
        if item.source_fact_id
    }

    proposal = session.forecast_proposal
    if proposal:
        evidence_ids = sorted({
            evidence_by_source_fact[ref]
            for ref in proposal.evidence_ids
            if ref in evidence_by_source_fact
        })
        for parameter, value in proposal.inputs.model_dump(
            mode="json", exclude_none=True
        ).items():
            assumption_id = stable_id(
                "assumption_", workspace.workspace_id, proposal.proposal_id,
                parameter, proposal.status
            )
            if assumption_id in known:
                continue
            assumption = WorkspaceAssumption(
                assumption_id=assumption_id,
                workspace_id=workspace.workspace_id,
                proposal_id=proposal.proposal_id,
                parameter=parameter,
                value=value,
                rationale=proposal.rationale,
                evidence_ids=evidence_ids,
                source="model",
                status="approved" if proposal.status == "confirmed" else "proposed",
            )
            store.save_workspace_record(
                workspace.workspace_id, "assumption", assumption, immutable=True
            )
            known.add(assumption_id)

    if not record or not record.result:
        return
    result_assumptions = record.result.assumptions
    values = {
        "revenue_growth": result_assumptions.revenue_growth,
        "ebit_margin": result_assumptions.ebit_margin,
        "wacc": result_assumptions.wacc,
        "terminal_growth": result_assumptions.terminal_growth,
        "revenue_growth_scenarios": result_assumptions.revenue_growth_scenarios,
        "ebit_margin_scenarios": result_assumptions.ebit_margin_scenarios,
        "tax_rate_path": result_assumptions.tax_rate_path,
        "wacc_components": result_assumptions.wacc_components,
        "operating_drivers": result_assumptions.operating_drivers,
    }
    for parameter, value in values.items():
        if value in (None, [], {}):
            continue
        assumption_id = stable_id(
            "assumption_", workspace.workspace_id, record.run_id, parameter
        )
        if assumption_id in known:
            continue
        evidence_refs = record.result.assumption_evidence.get(parameter, [])
        referenced = sorted({
            ref.evidence_id for ref in evidence_refs
            if any(ref.evidence_id in item.evidence_ids for item in facts)
        })
        rationale = result_assumptions.rationale.get(
            parameter,
            f"冻结于 {record.run_id} 的确定性估值输入；来源为 {result_assumptions.source}。",
        )
        source = (
            "market_data" if parameter in {"wacc", "wacc_components"}
            else "industry_parameter" if result_assumptions.source == "industry_model"
            else "model"
        )
        assumption = WorkspaceAssumption(
            assumption_id=assumption_id,
            workspace_id=workspace.workspace_id,
            run_id=record.run_id,
            parameter=parameter,
            value=value,
            rationale=rationale,
            evidence_ids=referenced,
            source=source,
            status="approved",
        )
        store.save_workspace_record(
            workspace.workspace_id, "assumption", assumption, immutable=True
        )
        known.add(assumption_id)


def build_model_spec(store, workspace, record, checkpoint, finance_version) -> ModelSpec:
    """Freeze the exact request accepted by the existing financial engine."""

    facts = store.list_workspace_records(
        workspace.workspace_id, "fact", WorkspaceFact, limit=2000
    )
    parent_model_spec = None
    if record.parent_run_id:
        specs = store.list_workspace_records(
            workspace.workspace_id, "model_spec", ModelSpec, limit=1000
        )
        parent_model_spec = next(
            (item for item in specs if item.run_id == record.parent_run_id), None
        )
    confirmed = [item for item in facts if item.status == "confirmed"]
    if checkpoint:
        checkpoint_sources = {
            item.source_fact_id
            for item in facts
            if item.fact_id in set(checkpoint.fact_ids)
        }
        eligible = [
            item for item in confirmed if item.source_fact_id in checkpoint_sources
        ]
        # A proposed ledger entry becomes a distinct confirmed observation at
        # approval. Keep only the latest state for each source fact.
        latest = {}
        for item in eligible:
            previous = latest.get(item.source_fact_id)
            if previous is None or item.created_at > previous.created_at:
                latest[item.source_fact_id] = item
        accepted_facts = list(latest.values())
    elif parent_model_spec:
        by_id = {item.fact_id: item for item in facts}
        accepted_facts = [
            by_id[fact_id] for fact_id in parent_model_spec.fact_ids if fact_id in by_id
        ]
    else:
        latest = {}
        for item in confirmed:
            key = item.source_fact_id or item.fact_id
            previous = latest.get(key)
            if previous is None or item.created_at > previous.created_at:
                latest[key] = item
        accepted_facts = list(latest.values())
    fact_ids = [item.fact_id for item in accepted_facts]
    evidence_ids = sorted({ref for item in accepted_facts for ref in item.evidence_ids})
    assumptions = store.list_workspace_records(
        workspace.workspace_id, "assumption", WorkspaceAssumption, limit=2000
    )
    assumption_ids = [
        item.assumption_id
        for item in assumptions
        if item.status == "approved" and item.run_id in {"", record.run_id}
    ]
    request_snapshot = record.request.model_dump(mode="json")
    cutoff = workspace.information_cutoff_date or record.request.valuation_date
    body = {
        "workspace_id": workspace.workspace_id,
        "checkpoint_id": checkpoint.checkpoint_id if checkpoint else None,
        "parent_model_spec_id": (
            parent_model_spec.model_spec_id if parent_model_spec else None
        ),
        "run_id": record.run_id,
        "version_number": record.revision,
        "valuation_date": record.request.valuation_date,
        "information_cutoff_date": cutoff,
        "execution_date": workspace.execution_date,
        "methods": [str(item) for item in record.request.methods],
        "requested_methods": [str(item) for item in record.request.requested_methods],
        "fact_ids": fact_ids,
        "evidence_ids": evidence_ids,
        "assumption_ids": assumption_ids,
        "peer_tickers": [item.ticker for item in record.request.peers],
        "request_snapshot": request_snapshot,
        "resolved_assumptions": dict(checkpoint.assumptions) if checkpoint else (
            record.result.assumptions.model_dump(mode="json") if record.result else {}
        ),
        "formula_manifest": {
            key: value for key, value in FORMULA_MANIFEST.items()
            if (
                (key in {"fcff", "terminal_value", "dcf_equity_bridge", "dcf_per_share"}
                 and "dcf" in record.request.methods)
                or (key.startswith("pe_") and "pe" in record.request.methods)
                or (key.startswith("ps_") and "ps" in record.request.methods)
                or (key.startswith("ev_ebitda_") and "ev_ebitda" in record.request.methods)
            )
        },
        "calculation_order": CALCULATION_ORDER,
        # This value is deliberately independent of execution state.  The
        # ModelSpec is frozen before calculation and must hash to the exact
        # same object when the completed run is later projected into ledgers.
        "financial_model_name": "valuationagent_finance_engine",
        "financial_model_version": finance_version,
        "software_versions": {
            "valuationagent": __version__,
            "python": platform.python_version(),
            "workflow": record.workflow_version,
        },
        "degradation_policy": {
            "excluded_methods": record.request.excluded_methods,
            "data_source": str(record.request.data_source),
        },
    }
    snapshot_hash = digest(body)
    return ModelSpec(
        model_spec_id="modelspec_" + snapshot_hash[:28],
        snapshot_hash=snapshot_hash,
        **body,
    )


def build_calculation_record(workspace, record, model_spec) -> CalculationRecord:
    result_hash = digest(record.result.model_dump(mode="json")) if record.result else None
    return CalculationRecord(
        calculation_id=stable_id("calculation_", workspace.workspace_id, record.run_id),
        workspace_id=workspace.workspace_id,
        run_id=record.run_id,
        model_spec_id=model_spec.model_spec_id,
        input_hash=record.input_hash,
        result_hash=result_hash,
        status=str(record.status),
        formulas=model_spec.formula_manifest,
        upstream_refs=[
            model_spec.model_spec_id,
            *model_spec.fact_ids,
            *model_spec.evidence_ids,
            *model_spec.assumption_ids,
        ],
        calculation_order=model_spec.calculation_order,
        precision_policy=model_spec.rounding_policy,
        financial_model_version=(record.result.model_version if record.result else model_spec.financial_model_version),
        replayable_offline=True,
    )


def sync_evidence_usage(store, workspace, model_spec) -> None:
    """Record exactly which evidence-backed facts entered a frozen version."""

    facts = {
        item.fact_id: item
        for item in store.list_workspace_records(
            workspace.workspace_id, "fact", WorkspaceFact, limit=2000
        )
    }
    known = {
        item.usage_id
        for item in store.list_workspace_records(
            workspace.workspace_id, "evidence_usage", EvidenceUsage, limit=2000
        )
    }
    for fact_id in model_spec.fact_ids:
        fact = facts.get(fact_id)
        if not fact:
            continue
        for evidence_id in fact.evidence_ids:
            usage_id = stable_id(
                "usage_", model_spec.model_spec_id, evidence_id, fact_id
            )
            if usage_id in known:
                continue
            usage = EvidenceUsage(
                usage_id=usage_id,
                workspace_id=workspace.workspace_id,
                run_id=model_spec.run_id,
                model_spec_id=model_spec.model_spec_id,
                evidence_id=evidence_id,
                fact_id=fact_id,
                adopted=True,
                rationale="该已确认事实被写入本版本 ModelSpec，并参与模型输入冻结。",
            )
            store.save_workspace_record(
                workspace.workspace_id, "evidence_usage", usage, immutable=True
            )
            known.add(usage_id)
