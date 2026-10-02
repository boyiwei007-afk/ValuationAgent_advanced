"""Cross-observation source-cell checks without document-layout heuristics."""
from collections import defaultdict


PERIOD_CELL_CONFLICT = "SOURCE_PERIOD_COLLISION: 同一原文数值位置被赋给不同期间；须更正/撤回错误解释或取得各期间独立证据，LLM复核不能直接消除此冲突"
PERIOD_READBACK_REQUIRED = "PERIOD_READBACK_REQUIRED: 时点观察须脱离候选日期从原文复读截止日，不能仅凭年报标题推定年末；请重新prepare_observation_review复核"


def period_readback_issue(fact):
    row = fact.verification.get("reading_proof", {}).get("row", {})
    if row.get("period_kind") != "instant":
        return ""
    review = fact.verification.get("semantic_review", {})
    if review.get("source_period_end") != row.get("period_end") or review.get("period_readback_version") != 1:
        return PERIOD_READBACK_REQUIRED
    return ""


def invalidate_unread_periods(session):
    changed = []
    for fact in session.facts:
        if fact.status != "confirmed" or not period_readback_issue(fact):
            continue
        fact.status = "proposed"
        fact.warnings = list(dict.fromkeys([*fact.warnings, PERIOD_READBACK_REQUIRED]))
        fact.verification.setdefault("semantic_review", {}).update(status="needs_evidence")
        fact.verification.setdefault("source_assessment", {}).update(admission="blocked")
        changed.append(fact.fact_id)
    return changed


def period_cell_conflicts(facts):
    groups = defaultdict(list)
    for fact in facts:
        if fact.status == "rejected":
            continue
        proof = fact.verification.get("reading_proof", {})
        anchor = proof.get("resolved_value", {})
        if not anchor or not proof.get("source_sha256"):
            continue
        basis = proof.get("basis", {})
        location = anchor.get("location", {})
        key = (proof["source_sha256"], anchor.get("block_sha256"), location.get("page"), location.get("sheet"),
               anchor.get("quote"), fact.standard_metric, fact.scope, fact.role,
               basis.get("entity_ticker") or basis.get("entity_name"))
        groups[key].append((fact, proof, anchor))
    conflicts = defaultdict(list)
    for group in groups.values():
        for index, (current, proof, anchor) in enumerate(group):
            for other, other_proof, other_anchor in group[index + 1:]:
                if current.period == other.period:
                    continue
                if (other.fact_id in proof.get("row", {}).get("replaces", [])
                        or current.fact_id in other_proof.get("row", {}).get("replaces", [])):
                    continue
                if "text_offset" in anchor and "text_offset" in other_anchor:
                    same_cell = anchor["text_offset"] == other_anchor["text_offset"]
                else:
                    same_cell = all(anchor.get(key) == other_anchor.get(key) for key in ("start_line", "end_line", "char_offset"))
                if same_cell:
                    conflicts[current.fact_id].append(other.fact_id)
                    conflicts[other.fact_id].append(current.fact_id)
    return dict(conflicts)


def invalidate_conflicting_periods(session):
    conflicts = period_cell_conflicts(session.facts)
    changed = []
    for fact in session.facts:
        if fact.fact_id not in conflicts:
            continue
        if fact.status != "proposed" or PERIOD_CELL_CONFLICT not in fact.warnings:
            changed.append(fact.fact_id)
        fact.status = "proposed"
        fact.warnings = list(dict.fromkeys([*fact.warnings, PERIOD_CELL_CONFLICT]))
        fact.verification.setdefault("source_assessment", {}).update(admission="blocked", consistency="period_collision")
    return changed


def retire_contradicted_entities(facts):
    changed = []
    for fact in facts:
        semantic = fact.verification.get("semantic_review", {})
        if (fact.status == "rejected" or not fact.verification.get("reading_proof")
                or semantic.get("reviewer") != "workspace_llm" or not semantic.get("reviewed_packet_id")
                or semantic.get("checks", {}).get("entity") != "contradicted"):
            continue
        fact.status = "rejected"
        fact.verification["disposition"] = {
            "actor": "semantic_review_policy", "reason": "entity_contradicted",
            "packet_id": semantic["reviewed_packet_id"],
            "instruction": "原文与所声明主体矛盾，此解释已撤回；原始来源及复核记录保留。不能改写目标公司或直接改为可比事实，须按真实主体重新提取并复核。",
        }
        changed.append(fact.fact_id)
    return changed
