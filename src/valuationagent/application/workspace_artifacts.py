"""Immutable report outputs; generated text never becomes source evidence."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from valuationagent.application.evidence_status import evidence_references
from valuationagent.application.research_export import build_research_export
from valuationagent.application.result_document import ensure_result_document, document_sections
from valuationagent.schemas.models import ApiModel


class ReportWrite(ApiModel):
    format: Literal["md", "json", "html", "pdf"] = "md"


class NoteWrite(ApiModel):
    title: str = Field(min_length=1, max_length=160)
    body: str = Field(min_length=1, max_length=12000)
    basis: Literal["source_analysis", "concept_note"] = Field(default="source_analysis", description="source_analysis为基于资料的数据研究，必须引用真实原文块；concept_note仅为不声称公司数据的概念/方法说明。")
    evidence_ids: list[str] = Field(default_factory=list, max_length=30, description="资料型笔记必填引用的真实block_id；不能只写文件名或空列表。")


class ArtifactRead(ApiModel):
    artifact_id: str = Field(min_length=1, max_length=100)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=8000, ge=1, le=16000)


def report_delivery(service, session, metadata):
    workspace = service.store.workspace_for_research(session.session_id)
    return {**metadata, "download_url": f"/api/workspaces/{workspace.workspace_id}/artifacts/{metadata['artifact_id']}" if workspace else None,
            "instruction": "已保存报告，可使用真实download_url或文件卡片下载；不要编造sandbox:/mnt/data路径。状态来自系统，不代表所有事实或假设已由用户批准。"}


def write_report(service, session, args):
    service._check_execution()
    document = ensure_result_document(service, session)
    input_hash = service.store.get_run(document["valuation_run_id"]).input_hash if document["numeric_result_available"] and document["valuation_run_id"] else None
    payload, media_type = None, None
    identity_document = document
    if input_hash and args.format != "json":
        identity_document = {key: value for key, value in document.items()
            if key not in {"report_id", "source_revision", "generated_at"}}
    identity = {"policy": "substantive-report-v1", "format": args.format,
        "document": identity_document, "input_hash": input_hash}
    if args.format == "json":
        content, media_type = build_research_export(service, session.session_id, args.format, session=session)
        payload = content.encode("utf-8")
        identity["export_sha256"] = hashlib.sha256(payload).hexdigest()
    report_identity = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    service._check_execution()
    for artifact in service.store.list_artifacts(session.session_id):
        if artifact.get("kind") == "result_report" and artifact.get("report_identity") == report_identity and artifact["filename"] == "valuation-report." + args.format:
            metadata, _ = service.store.get_artifact(session.session_id, artifact["artifact_id"])
            service._check_execution()
            return report_delivery(service, session, metadata)
    if args.format == "md":
        content = "# 估值研究报告\n\n" + document["status_label"] + "\n\n"
        for heading, paragraphs in document_sections(document):
            content += "## " + heading + "\n\n" + "\n\n".join(paragraphs) + "\n\n"
        content += f"报告标识：{document['report_id']}\n输入修订：{document['source_revision']}\n"
        media_type = "text/markdown"
    elif payload is None:
        content, media_type = build_research_export(service, session.session_id, args.format, session=session)
    if payload is None:
        payload = content.encode("utf-8") if isinstance(content, str) else content
    service._check_execution()
    metadata = service.store.save_artifact(session.session_id, {
        "kind": "result_report", "filename": "valuation-report." + args.format, "media_type": media_type,
        "source_revision": document["source_revision"], "report_id": document["report_id"],
        "report_identity": report_identity, "input_hash": input_hash,
        "status": document["status"], "numeric_result_available": document["numeric_result_available"],
        "valuation_run_id": document["valuation_run_id"], "prompt_version": session.prompt_version,
    }, payload)
    return report_delivery(service, session, metadata)


def save_interruption_report(service, session, document, request_id):
    """Persist terminal bookkeeping without more model, network or parsing work."""
    from valuationagent.application.turn_control import artifact_output_allowed

    if not artifact_output_allowed(session) or session.pending_action != "valuation" or document["numeric_result_available"]:
        return None
    for artifact in service.store.list_artifacts(session.session_id):
        if artifact.get("kind") == "interruption_report" and artifact.get("request_id") == request_id:
            service.store.get_artifact(session.session_id, artifact["artifact_id"])
            return artifact
    paragraphs = ["# 估值执行中断与输入缺口报告",
                  "> 本轮执行未完成。本文件不是数值估值报告，不表示已穷尽公开资料；没有后台任务继续运行。"]
    for heading, content in document_sections(document):
        paragraphs.extend(["## " + heading, *content])
    paragraphs.extend(["## 续做检查点", json.dumps(session.resume_context, ensure_ascii=False, indent=2)])
    payload = service._redact_text("\n\n".join(paragraphs)).encode("utf-8")
    artifact = service.store.save_artifact(session.session_id, {
        "kind": "interruption_report", "filename": "valuation-interruption.md", "media_type": "text/markdown",
        "source_revision": document["source_revision"], "report_id": document["report_id"],
        "request_id": request_id, "status": "execution_incomplete", "numeric_result_available": False,
        "valuation_run_id": None, "prompt_version": session.prompt_version,
    }, payload)
    service.store.append_event(session.session_id, type="report.interruption_saved", stage="reporting", status="completed",
        summary="已保存非数值的执行中断与缺口报告；估值未完成", payload={"artifact_id": artifact["artifact_id"], "request_id": request_id})
    return artifact


def write_note(service, session, args):
    if args.basis == "source_analysis" and not args.evidence_ids:
        raise ValueError("NOTE_REFERENCES_REQUIRED: 资料型研究笔记须提供读到的真实block_id作为evidence_ids；正文中的文件名不是可追溯引用。无公司数据的概念说明才可用concept_note。")
    blocks = service._blocks(session)
    if any(block_id not in blocks for block_id in args.evidence_ids):
        raise ValueError("NOTE_REFERENCE_INVALID: 笔记引用必须来自当前工作区的原文。")
    references = evidence_references(session, blocks, args.evidence_ids)
    documents = {document.file_id: document for document in session.documents}
    for reference in references:
        block_id = reference["block_id"]
        document = documents.get(block_id.rsplit(":", 1)[0])
        reference["source_sha256"] = document.sha256 if document else ""
        reference["block_text_sha256"] = hashlib.sha256(blocks[block_id]["text"].encode("utf-8")).hexdigest()
        reference["location"] = blocks[block_id].get("location", {})
    body = service._redact_text(args.body)
    title = service._redact_text(args.title).replace("\n", " ").replace("\r", " ")
    content = f"# {title}\n\n> LLM研究笔记 · 未审阅 · 不是确定性估值报告。正文观点与数值未因保存文件而通过核验。\n\n{body}\n\n## 原文定位\n\n"
    for item in references:
        location = item["location"]
        position = location.get("json_pointer") or (f"第{item['page']}页" if item["page"] else location.get("cell_range") or "原文块")
        content += f"- {item['name']} · {position} · {item['block_id']}\n"
    if not references:
        content += "概念说明，无外部公司数据引用。\n"
    service._check_execution()
    return service.store.save_artifact(session.session_id, {
        "kind": "research_note", "filename": "research-note.md", "media_type": "text/markdown",
        "status": "unreviewed", "numeric_result_available": False, "source_revision": session.revision,
        "prompt_version": session.prompt_version, "evidence_refs": references, "basis": args.basis,
    }, content.encode("utf-8"))


def read_artifact(store, session_id, args):
    metadata, payload = store.get_artifact(session_id, args.artifact_id)
    if metadata["media_type"] == "application/pdf":
        return {"metadata": metadata, "instruction": "PDF为二进制报告；下载查看排版，文本复核可生成同输入的Markdown报告。"}
    text = payload.decode("utf-8")
    end = min(len(text), args.offset + args.limit)
    return {"metadata": metadata, "content": text[args.offset:end], "next_offset": end if end < len(text) else None,
            "trust": "这是生成文件，不是原始财务证据；不可将模型笔记重新用作事实来源。"}


def save_local_artifact(store, session_id, artifact_id, directory):
    metadata, content = store.get_artifact(session_id, artifact_id)
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    suffix = Path(metadata["filename"]).suffix
    if suffix not in {".md", ".json", ".html", ".pdf"}:
        raise ValueError("ARTIFACT_FORMAT_INVALID: 输出格式不可保存。")
    target = root / (metadata["artifact_id"] + suffix)
    if target.resolve().parent != root or target.is_symlink():
        raise ValueError("ARTIFACT_PATH_INVALID: 输出路径越界。")
    try:
        with target.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        if target.read_bytes() != content:
            raise ValueError("ARTIFACT_EXISTS: 同名本地文件不同，未覆盖。") from None
    manifest = {**metadata, "local_filename": target.name}
    manifest_path = root / (metadata["artifact_id"] + ".manifest.json")
    if manifest_path.resolve().parent != root or manifest_path.is_symlink():
        raise ValueError("ARTIFACT_PATH_INVALID: 清单路径越界。")
    if not manifest_path.exists():
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return target
