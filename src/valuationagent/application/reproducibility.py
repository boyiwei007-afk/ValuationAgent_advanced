"""Offline, content-addressed review packages. No model or network calls in replay."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from valuationagent.core.tools import canonical
from valuationagent.finance.factory import create_financial_model
from valuationagent.schemas.models import RunRecord, ValuationRequest


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def model_files():
    root = Path(__file__).parents[1]
    paths = [path for path in sorted((root / "finance").rglob("*")) if path.suffix in {".py", ".json"}]
    paths += [root / "application" / name for name in ("input_calculations.py", "input_derivations.py", "input_balances.py", "input_peers.py", "input_peer_bridge.py", "user_peers.py", "forecast_inputs.py", "provider_inputs.py")]
    paths.append(root / "market" / "tushare_contracts.py")
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def build_valuation_bundle(store, record):
    if record.result is None:
        raise ValueError("估值尚未完成，不能导出复算包")
    artifacts = store.artifacts(record.run_id) if store else []
    events = [e.model_dump(mode="json") for e in store.list_events(record.run_id)] if store else []
    effective = record.request.model_dump(mode="json")
    by_id = {a["artifact_id"]: a for a in artifacts}
    for event in reversed(events):
        if event.get("tool") == "resolve_financial_input" and event.get("payload", {}).get("artifact_id") in by_id:
            bundle = by_id[event["payload"]["artifact_id"]]["output"]
            for key in ("financials", "historical_financials", "peers", "assumptions", "assumption_evidence"):
                if key in bundle:
                    effective[key] = bundle[key]
            break
    effective.update(company=record.result.company.model_dump(mode="json") if record.result.company else effective["company"],
                     financials=record.result.effective_financials.model_dump(mode="json"),
                     peers=[p.model_dump(mode="json") for p in record.result.effective_peers],
                     data_source="structured", file_ids=[], assumption_file_ids=[])
    references = list(record.request.file_ids + record.request.assumption_file_ids)
    research = None
    frozen_sources = store.run_sources(record.run_id) if store else None
    if frozen_sources:
        research = frozen_sources["research"]
        references += [document["file_id"] for document in research["documents"]]
    manifest, blocks = [], {}
    for file_id in dict.fromkeys(references):
        if file_id.startswith("web_"):
            doc = next(d for d in research["documents"] if d["file_id"] == file_id)
            manifest.append({"file_id": file_id, "original_name": doc["name"], "role": "search_lead",
                             "size_bytes": doc["size_bytes"], "sha256": doc["sha256"]})
        else:
            meta = store.get_file(file_id)
            manifest.append({k: meta[k] for k in ("file_id", "original_name", "role", "size_bytes", "sha256")})
        if research:
            if any(d["file_id"] == file_id for d in research["documents"]):
                blocks[file_id] = frozen_sources["source_blocks"][file_id]
    package = {"schema_version": "valuation-review-v1", "run": record.model_dump(mode="json"),
               "effective_request": effective, "artifacts": artifacts, "events": events,
               "research": research, "source_manifest": manifest, "source_blocks": blocks,
               "research_events": frozen_sources["research_events"] if frozen_sources else [],
               "model_files": model_files(),
               "dependencies": {name: version(name) for name in ("pydantic", "langgraph", "httpx")},
               "replay_scope": "复算已锁定输入的财务预测、DCF、相对估值与敏感性；不重放随机模型回复或实时搜索。"}
    if store:
        from valuationagent.application.workspace_reporting import (
            build_workspace_report_context,
        )

        workspace_audit = build_workspace_report_context(store, record.run_id)
        if workspace_audit:
            package["workspace_audit"] = workspace_audit
    package["package_sha256"] = digest(package)
    return package


def replay_bundle(package):
    package = dict(package)
    expected_hash = package.pop("package_sha256", None)
    if not expected_hash or expected_hash != digest(package):
        raise ValueError("复算包哈希不匹配，内容可能被更改")
    if package.get("schema_version") != "valuation-review-v1":
        raise ValueError("不支持的复算包版本")
    if package["model_files"] != model_files():
        raise ValueError("本地金融源码或参数库与导出时不一致，请使用现场展示版本")
    record = RunRecord.model_validate(package["run"])
    request = ValuationRequest.model_validate(package["effective_request"])
    from valuationagent.application.input_calculations import verify_frozen_calculations
    from valuationagent.application.input_derivations import verify_frozen_derivations
    from valuationagent.application.input_balances import verify_balance_changes
    from valuationagent.application.user_peers import verify_user_peers
    from valuationagent.application.input_peer_bridge import verify_source_peers
    from valuationagent.application.forecast_inputs import verify_frozen_forecast_inputs

    verify_frozen_calculations(request)
    verify_frozen_derivations(request)
    verify_balance_changes(request)
    verify_user_peers(request)
    verify_source_peers(request)
    verify_frozen_forecast_inputs(request)
    result = record.result
    finance = create_financial_model()
    actual_version = finance.model_version_for(request) if hasattr(finance, "model_version_for") else finance.version
    if actual_version != result.model_version:
        raise ValueError("金融模型版本与复算包不一致")
    financials, assumptions, peers = request.financials, result.assumptions, result.effective_peers
    forecast = finance.forecast(request, financials, assumptions) if "dcf" in request.methods else []
    outputs = {"forecast": forecast,
        "dcf": finance.dcf(request, financials, assumptions, forecast) if "dcf" in request.methods else None,
        "relative": finance.relative(request, financials, peers) if any(m != "dcf" for m in request.methods) else [],
        "sensitivity": finance.sensitivity(request, financials, assumptions) if "dcf" in request.methods else [],
        "sensitivity_studies": finance.sensitivity_studies(request, financials, assumptions) if "dcf" in request.methods and hasattr(finance, "sensitivity_studies") else []}
    from valuationagent.finance.relative_sensitivity import relative_sensitivity
    outputs["sensitivity_studies"] += relative_sensitivity(finance, request, financials, peers)
    checks = {key: canonical(value) == canonical(getattr(result, key)) for key, value in outputs.items()}
    return {"passed": all(checks.values()), "checks": checks, "model_version": actual_version,
            "package_sha256": expected_hash, "network_used": False}
