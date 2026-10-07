"""Layered and bounded conversational context for long valuation engagements.

The manager intentionally keeps raw document text out of the prompt.  Source
fragments are retrieved through tools only when the current question needs
them.  This makes long-running workspaces predictable and prevents a late chat
turn from displacing approved scope, constraints, facts or assumptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from valuationagent.core.tools import canonical
from valuationagent.schemas.agent import ContextSnapshot


_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_ -]?key|token|secret|password)\s*[:=]\s*\S+"),
    re.compile(r"(?<![A-Za-z0-9_-])(?:sk|tvly)-(?:[A-Za-z0-9_-]{12,})(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{16,}-infoway(?![A-Za-z0-9_-])", re.I),
)


def redact_context_text(value: str) -> str:
    text = value
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def sensitive_spans(value: str) -> list[tuple[int, int]]:
    return sorted({match.span() for pattern in _SECRET_PATTERNS for match in pattern.finditer(value)})


def _terms(value: str) -> set[str]:
    return {
        item.lower()
        for item in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,}|[\u4e00-\u9fff]{2,}", value)
    }


@dataclass(frozen=True)
class ContextBudget:
    total_chars: int = 46000
    facts_chars: int = 13000
    memory_chars: int = 9000
    documents_chars: int = 8000
    turns_chars: int = 13000
    max_turns: int = 20


class LayeredContextManager:
    """Compose prompt state by authority and relevance, not transcript length."""

    def __init__(self, budget: ContextBudget | None = None):
        self.budget = budget or ContextBudget()

    @staticmethod
    def _fit(rows: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        used = 0
        for row in rows:
            safe = LayeredContextManager._redact_value(row)
            size = len(canonical(safe))
            if selected and used + size > limit:
                continue
            if not selected and size > limit:
                # Metadata should be small, but a malformed oversized row must
                # not monopolise the whole prompt.
                continue
            selected.append(safe)
            used += size
        return selected

    @staticmethod
    def _redact_value(value: Any) -> Any:
        if isinstance(value, str):
            return redact_context_text(value)
        if isinstance(value, list):
            return [LayeredContextManager._redact_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: "[REDACTED]"
                if any(mark in str(key).lower() for mark in ("api_key", "token", "password", "secret"))
                else LayeredContextManager._redact_value(item)
                for key, item in value.items()
            }
        return value

    def _facts(self, session, query: str) -> list[dict[str, Any]]:
        query_terms = _terms(query)
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for index, fact in enumerate(session.facts):
            row = fact.model_dump(
                mode="json",
                include={
                    "fact_id",
                    "metric",
                    "standard_metric",
                    "raw_value",
                    "unit",
                    "normalized_value",
                    "period",
                    "scope",
                    "role",
                    "block_id",
                    "table_id",
                    "source_type",
                    "status",
                    "warnings",
                    "peer_ticker",
                    "peer_name",
                    "multiple_basis",
                    "denominator_period_end",
                    "context_block_ids",
                    "verification",
                    "mapping_confidence",
                },
            )
            verification = row.pop("verification", {})
            row["verification"] = {key: verification[key] for key in ("binding", "period_end", "scope", "source_assessment") if key in verification}
            if proof := verification.get("reading_proof"):
                row["verification"]["reading_proof"] = {"proof_id": proof["proof_id"], "file_id": proof["file_id"], "schema": proof["schema"]}
            if review := verification.get("semantic_review"):
                row["verification"]["semantic_review"] = {key: review[key] for key in ("status", "independent_audit") if key in review}
            haystack = canonical(row)
            overlap = len(query_terms & _terms(haystack))
            authority = 80 if fact.status == "confirmed" else 40 if not fact.warnings else 10
            ranked.append((authority + overlap * 12, index, row))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen = self._fit((item[2] for item in ranked), self.budget.facts_chars)
        # Stable chronological order makes diffs and LLM behaviour reproducible.
        order = {fact.fact_id: index for index, fact in enumerate(session.facts)}
        return sorted(chosen, key=lambda row: order.get(row["fact_id"], 0))

    def _memory(self, session, query: str) -> list[dict[str, Any]]:
        query_terms = _terms(query)
        kind_priority = {"constraint": 100, "decision": 95, "goal": 90, "definition": 70, "preference": 60}
        ranked = []
        for index, item in enumerate(session.memory):
            row = item.model_dump(mode="json")
            overlap = len(query_terms & _terms(item.content))
            ranked.append((kind_priority.get(item.kind, 0) + overlap * 10, index, row))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen = self._fit((item[2] for item in ranked), self.budget.memory_chars)
        order = {item.key: index for index, item in enumerate(session.memory)}
        return sorted(chosen, key=lambda row: order.get(row["key"], 0))

    def _documents(self, session) -> list[dict[str, Any]]:
        # Only source manifests enter the context; document content is tool-retrieved.
        rows = [item.model_dump(mode="json") for item in reversed(session.documents)]
        return list(reversed(self._fit(rows, self.budget.documents_chars)))

    def _turns(self, messages) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        used = 0
        for message in reversed(messages):
            if message.role not in ("user", "assistant"):
                continue
            content = redact_context_text(message.content)
            if rows and used + len(content) > self.budget.turns_chars:
                break
            rows.append({"role": message.role, "content": content})
            used += len(content)
            if len(rows) >= self.budget.max_turns:
                break
        return list(reversed(rows))

    def snapshot(self, session, messages, *, query: str = "") -> ContextSnapshot:
        messages = list(messages)
        latest = next((message for message in reversed(messages) if message.role == "user"), None)
        if not query:
            query = next(
                (message.content for message in reversed(messages) if message.role == "user"),
                "",
            )
        facts = self._facts(session, query)
        memory = self._memory(session, query)
        documents = self._documents(session)
        recent = self._turns(messages)
        state = {
            "draft": session.draft.model_dump(mode="json"),
            "status": session.status,
            "data_source_preference": session.data_source_preference,
            "information_cutoff_date": str(session.information_cutoff_date or ""),
            "pending_action": session.pending_action,
            "turn_control": session.turn_control.model_dump(mode="json") if session.turn_control else None,
            "execution_permissions": {name: item.model_dump() for name, item in session.execution_permissions.items()},
            "pending_decision": session.pending_decision.model_dump() if session.pending_decision else None,
            "recent_searches": [
                {key: item.get(key) for key in ("query", "purpose", "status", "source_ids")}
                for item in session.search_history[-8:]
            ],
            "forecast_proposal": session.forecast_proposal.model_dump(mode="json")
            if session.forecast_proposal
            else None,
            "documents": documents,
            "document_count": len(session.documents),
            "documents_omitted": max(0, len(session.documents) - len(documents)),
            "gaps": self._redact_value(session.gaps),
            "memory": memory,
            "memory_count": len(session.memory),
            "memory_omitted": max(0, len(session.memory) - len(memory)),
            "last_issue": self._redact_value(
                session.last_issue.model_dump(mode="json") if session.last_issue else None
            ),
            "agent_protocol_version": session.agent_protocol_version,
            "prompt_version": session.prompt_version,
            "model_provider": session.model_provider,
            "model_name": session.model_name,
            "facts": facts,
            "facts_omitted": max(0, len(session.facts) - len(facts)),
            "fact_counts": {
                status: sum(f.status == status for f in session.facts)
                for status in ("confirmed", "proposed", "rejected")
            },
            "user_note_ids": [
                {
                    "block_id": "message:" + message.message_id,
                    "excerpt": redact_context_text(message.content[:160]),
                }
                for message in messages
                if message.role == "user"
            ][-8:],
            "context_policy": {
                "strategy": "authority_then_relevance_then_recency",
                "raw_documents_in_prompt": False,
                "total_budget_chars": self.budget.total_chars,
                "facts_budget_chars": self.budget.facts_chars,
                "memory_budget_chars": self.budget.memory_chars,
                "turns_budget_chars": self.budget.turns_chars,
                "retrieval_required_when_omitted": True,
            },
        }
        snapshot = ContextSnapshot(
            session_id=session.session_id,
            revision=session.revision,
            language=session.language,
            current_request={"message_id": latest.message_id, "content": redact_context_text(latest.content)} if latest else {},
            summary="" if any(turn["content"] == redact_context_text(session.summary) for turn in recent) else redact_context_text(session.summary),
            task_state=state,
            confirmed_fact_ids=[
                row["fact_id"] for row in facts if row["status"] == "confirmed"
            ][-500:],
            evidence_ids=list(dict.fromkeys(row["block_id"] for row in facts))[-500:],
            recent_turns=recent,
        )
        for layer in ("recent_turns", "facts", "documents", "memory", "recent_searches", "gaps", "user_note_ids"):
            rows = snapshot.recent_turns if layer == "recent_turns" else snapshot.task_state[layer]
            while rows and len(canonical(snapshot)) > self.budget.total_chars:
                rows.pop(0)
                if layer in {"facts", "documents", "memory"}:
                    snapshot.task_state[layer + "_omitted"] += 1
        for layer in ("last_issue", "forecast_proposal"):
            if len(canonical(snapshot)) > self.budget.total_chars:
                snapshot.task_state[layer] = {"context_omitted": True, "retrieve_with": "inspect_context"}
        snapshot.confirmed_fact_ids = [
            row["fact_id"] for row in snapshot.task_state["facts"] if row["status"] == "confirmed"
        ][-500:]
        snapshot.evidence_ids = list(dict.fromkeys(
            row["block_id"] for row in snapshot.task_state["facts"]
        ))[-500:]
        if len(canonical(snapshot)) > self.budget.total_chars:
            snapshot.summary = ""
        if len(canonical(snapshot)) > self.budget.total_chars:
            raise ValueError("权威任务范围超过上下文预算，不能静默截断用户约束。")
        return snapshot


DEFAULT_CONTEXT_MANAGER = LayeredContextManager()
