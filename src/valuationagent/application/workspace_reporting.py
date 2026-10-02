"""Resolve one immutable valuation run back to its auditable workspace state.

The financial engine intentionally knows nothing about conversations or
evidence acquisition.  Exporters use this read-only adapter to enrich an
otherwise deterministic run report with the four ledgers and version context.
"""

from __future__ import annotations

from typing import Any

from valuationagent.schemas.workspace import (
    AgentAction,
    CalculationRecord,
    ChallengeFinding,
    DecisionRecord,
    EvidenceRecord,
    EvidenceUsage,
    FindingDisposition,
    ModelSpec,
    PreValuationCheckpoint,
    ValuationVersion,
    WorkspaceAssumption,
    WorkspaceFact,
)


def _records(store, workspace_id: str, kind: str, model, limit: int = 2000):
    return store.list_workspace_records(workspace_id, kind, model, limit=limit)


def build_workspace_report_context(store, run_id: str) -> dict[str, Any] | None:
    """Return only persisted, user-visible audit data for ``run_id``.

    No LLM or network call is made here.  The returned object can therefore be
    embedded in JSON exports and generated again offline.
    """
    if store is None or not hasattr(store, "workspace_for_run"):
        return None
    workspace = store.workspace_for_run(run_id)
    if workspace is None:
        return None
    workspace_id = workspace.workspace_id
    versions = _records(store, workspace_id, "version", ValuationVersion, 1000)
    version = next((item for item in versions if item.run_id == run_id), None)
    specs = _records(store, workspace_id, "model_spec", ModelSpec, 1000)
    model_spec = next((item for item in specs if item.run_id == run_id), None)
    calculations = _records(store, workspace_id, "calculation", CalculationRecord, 1000)
    calculation = next((item for item in calculations if item.run_id == run_id), None)
    checkpoints = _records(store, workspace_id, "checkpoint", PreValuationCheckpoint, 1000)
    checkpoint_id = (
        (version.checkpoint_id if version else None)
        or (model_spec.checkpoint_id if model_spec else None)
    )
    checkpoint = next(
        (item for item in checkpoints if item.checkpoint_id == checkpoint_id), None
    )
    findings = [
        item for item in _records(store, workspace_id, "finding", ChallengeFinding)
        if item.run_id == run_id
    ]
    dispositions = [
        item for item in _records(store, workspace_id, "disposition", FindingDisposition)
        if item.run_id == run_id
    ]
    decisions = [
        item for item in _records(store, workspace_id, "decision", DecisionRecord)
        if item.run_id == run_id
    ]

    def dump(value):
        return value.model_dump(mode="json") if value is not None else None

    return {
        "workspace": dump(workspace),
        "version": dump(version),
        "model_spec": dump(model_spec),
        "calculation": dump(calculation),
        "checkpoint": dump(checkpoint),
        "findings": [dump(item) for item in findings],
        "finding_dispositions": [dump(item) for item in dispositions],
        "decisions": [dump(item) for item in decisions],
        "evidence_ledger": [
            dump(item) for item in _records(store, workspace_id, "evidence", EvidenceRecord)
        ],
        "evidence_usage": [
            dump(item) for item in _records(
                store, workspace_id, "evidence_usage", EvidenceUsage
            )
            if item.run_id == run_id
        ],
        "fact_ledger": [
            dump(item) for item in _records(store, workspace_id, "fact", WorkspaceFact)
        ],
        "assumption_ledger": [
            dump(item) for item in _records(store, workspace_id, "assumption", WorkspaceAssumption)
        ],
        "action_ledger": [
            dump(item) for item in _records(store, workspace_id, "action", AgentAction)
        ],
        "versions": [dump(item) for item in versions],
        "reproducibility": {
            "calculation": "offline_exact" if model_spec and calculation else "not_packaged",
            "data_acquisition": "snapshot_or_excerpt_replay",
            "network_required_for_calculation": False,
        },
    }

