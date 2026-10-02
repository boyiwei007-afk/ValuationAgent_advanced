"""Observation quality is independent of valuation admission."""
from collections import Counter


def observation_verified(fact):
    if fact.status == "rejected":
        return False
    observation = fact.verification.get("observation")
    if observation:
        return observation.get("status") == "verified"
    return fact.status == "confirmed" and not fact.warnings


def binding_issues(fact):
    if fact.status == "rejected":
        return []
    observation = fact.verification.get("observation")
    if observation:
        return observation.get("issues", [])
    return [warning for warning in fact.warnings if any(marker in warning for marker in
            ("单位缺少", "年度列", "年度表头", "口径缺少", "表格列无法", "原文标题"))]


def repair_groups(facts):
    groups = {}
    for fact in facts:
        for issue in binding_issues(fact):
            key = (fact.block_id.rsplit(":", 1)[0], issue)
            group = groups.setdefault(key, {"file_id": key[0], "issue": issue, "fact_ids": [], "block_ids": set()})
            group["fact_ids"].append(fact.fact_id)
            group["block_ids"].add(fact.block_id)
    return [{**group, "block_ids": sorted(group["block_ids"]), "affected_count": len(group["fact_ids"]),
             "kind": "binding_repair", "action": "原文已存在；核对证据片段并切换读取视图，不重复搜索。"}
            for group in sorted(groups.values(), key=lambda group: -len(group["fact_ids"]))]


def evidence_counts(facts):
    active = [fact for fact in facts if fact.status != "rejected"]
    return {"candidates": len(active), "observations_verified": sum(observation_verified(fact) for fact in active),
            "semantic_review_pending": sum(fact.verification.get("semantic_review", {}).get("status") in {"pending", "needs_evidence"} for fact in active),
            "semantic_review_supported": sum(fact.verification.get("semantic_review", {}).get("status") == "supported" for fact in active),
            "model_eligible": sum(fact.status == "confirmed" and not fact.warnings for fact in active),
            "binding_issues": dict(Counter(issue for fact in active for issue in binding_issues(fact)))}


def evidence_references(session, blocks, ids):
    facts = {fact.fact_id: fact for fact in session.facts}
    documents = {document.file_id: document for document in session.documents}
    references = []
    seen = set()
    for key in ids:
        block_id = facts[key].block_id if key in facts else key
        if block_id in seen:
            continue
        seen.add(block_id)
        block = blocks.get(block_id, {})
        location = block.get("location", {})
        document = documents.get(block_id.rsplit(":", 1)[0])
        references.append({"block_id": block_id, "name": document.name if document else "用户提供资料",
                           "page": location.get("page"), "sheet": location.get("sheet"),
                           "source_url": location.get("source_url") or location.get("url") or ""})
    return references
