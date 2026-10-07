"""The single public agent workspace API."""

from __future__ import annotations

from fastapi import BackgroundTasks, HTTPException, Query
from fastapi.responses import Response

from pydantic import Field, SecretStr, model_validator
from valuationagent.schemas.models import ApiModel
from valuationagent.application.workspace_artifacts import ReportWrite, write_report
from valuationagent.application.file_workspace import FileList, FileReference, PageView, list_files, inspect_file, render_page
from valuationagent.llm.client import LlmError
from valuationagent.market import TushareApiClient, TushareDataProvider
from valuationagent.market.infoway import InfowayApiClient, InfowayDataProvider
from valuationagent.search.providers import TavilySearchProvider
from valuationagent.schemas.agent import SearchQuery
from valuationagent.schemas.models import ResumeInput
from valuationagent.schemas.research import ResearchTurn
from valuationagent.schemas.workspace import CheckpointDecision, WorkspaceCreate, WorkspaceRevision


class DataServicesInput(ApiModel):
    verify_search: bool = False
    tavily_api_key: SecretStr | None = Field(default=None)
    tushare_token: SecretStr | None = Field(default=None)
    infoway_api_key: SecretStr | None = Field(default=None)

    @model_validator(mode="after")
    def one_market_provider(self):
        if (self.tushare_token and self.tushare_token.get_secret_value().strip()
                and self.infoway_api_key and self.infoway_api_key.get_secret_value().strip()):
            raise ValueError("一次只连接一个结构化供应商，避免静默覆盖；网页搜索可同时配置。")
        return self

    def has_values(self):
        return bool(
            (self.tavily_api_key and self.tavily_api_key.get_secret_value().strip())
            or (self.tushare_token and self.tushare_token.get_secret_value().strip())
            or (self.infoway_api_key and self.infoway_api_key.get_secret_value().strip())
        )


def register_workspace_routes(
    app,
    workspaces,
    sessions,
    execute_background,
):
    def get_workspace(workspace_id):
        try:
            return workspaces.get(workspace_id)
        except KeyError:
            raise HTTPException(404, "valuation workspace not found") from None

    @app.post("/api/workspaces", status_code=201)
    def create_workspace(body: WorkspaceCreate):
        llm = None
        if body.model_session_id:
            try:
                llm = sessions.client(body.model_session_id)
            except KeyError:
                raise HTTPException(404, "model session not found") from None
        workspace = workspaces.create(
            language=body.language,
            title=body.title,
            objective=body.objective,
            llm=llm,
            data_source_preference=body.data_source_preference,
            run_policy=body.run_policy,
            information_cutoff_date=body.information_cutoff_date,
        )
        return workspaces.snapshot(workspace.workspace_id)

    @app.get("/api/workspaces")
    def list_workspaces(limit: int = Query(default=30, ge=1, le=100)):
        return [item.model_dump(mode="json") for item in workspaces.list(limit)]

    @app.get("/api/workspaces/{workspace_id}")
    def workspace_snapshot(workspace_id: str):
        get_workspace(workspace_id)
        return workspaces.snapshot(workspace_id)

    @app.delete("/api/workspaces/{workspace_id}", status_code=204)
    def delete_workspace(workspace_id: str, revision: int = Query(ge=0)):
        try:
            workspaces.delete(workspace_id, expected_revision=revision)
        except KeyError:
            raise HTTPException(404, "valuation workspace not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return Response(status_code=204)

    def execute_workspace_message(workspace_id: str, body: ResearchTurn):
        try:
            workspaces.message(workspace_id, body, reserved=True)
        except Exception as exc:
            workspace = workspaces.get(workspace_id)
            detail = str(exc)
            visible = (
                detail
                if isinstance(exc, LlmError)
                or detail.startswith(("LLM_", "EXECUTION_"))
                else "本轮任务未完成，已保留此前进度。请检查连接或补充输入后重试。"
            )
            workspaces.store.update_research_job(
                workspace.research_session_id,
                body.request_id,
                status="failed",
                stage="interrupted",
            )
            workspaces.store.add_message(
                workspace.research_session_id,
                "assistant",
                visible,
                "interrupted",
            )
            workspaces.store.append_event(
                workspace.research_session_id,
                type="turn.interrupted",
                stage="interrupted",
                status="failed",
                summary=visible,
                payload={"request_id": body.request_id, "error_type": type(exc).__name__},
            )

    @app.post("/api/workspaces/{workspace_id}/messages", status_code=202)
    def workspace_message(
        workspace_id: str,
        body: ResearchTurn,
        background_tasks: BackgroundTasks,
    ):
        workspace = get_workspace(workspace_id)
        try:
            request_id, created = workspaces.research.reserve_turn(
                workspace.research_session_id, body
            )
        except KeyError:
            raise HTTPException(404, "valuation workspace not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        if created:
            queued = body.model_copy(update={"request_id": request_id})
            background_tasks.add_task(execute_workspace_message, workspace_id, queued)
        return {
            "mode": "agent_queued",
            "request_id": request_id,
            "execution": workspaces.research.execution_state(workspace.research_session_id),
        }

    @app.post("/api/workspaces/{workspace_id}/model-session", status_code=204)
    def attach_model(workspace_id: str, body: ResumeInput):
        workspace = get_workspace(workspace_id)
        if not body.model_session_id:
            raise HTTPException(422, "model_session_id is required")
        try:
            workspaces.research.attach(
                workspace.research_session_id,
                sessions.client(body.model_session_id),
            )
        except KeyError:
            raise HTTPException(404, "model session not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/workspaces/{workspace_id}/data-services")
    def data_services(workspace_id: str):
        workspace = get_workspace(workspace_id)
        return workspaces.research.data_service_status(
            workspace.research_session_id,
            default_market=workspaces.runner.data,
        )

    @app.post("/api/workspaces/{workspace_id}/data-services")
    def attach_data_services(workspace_id: str, body: DataServicesInput):
        workspace = get_workspace(workspace_id)
        if not body.has_values():
            raise HTTPException(422, "至少填写一个 Tavily Key、Infoway Key 或 Tushare Token。")
        try:
            if body.tavily_api_key and body.tavily_api_key.get_secret_value().strip():
                provider = TavilySearchProvider(body.tavily_api_key.get_secret_value())
                if body.verify_search:
                    result = provider.search(SearchQuery(
                        query="上市公司 年度报告 官方公告",
                        purpose="other",
                        candidate_limit=1,
                    ))
                    if result.status not in {"completed", "no_results"}:
                        raise HTTPException(502, result.error_message or "搜索连接验证失败")
                    provider.connection_verified = True
                workspaces.research.attach_search(workspace.research_session_id, provider)
            if body.tushare_token and body.tushare_token.get_secret_value().strip():
                workspaces.research.attach_market(
                    workspace.research_session_id,
                    TushareDataProvider(TushareApiClient(body.tushare_token.get_secret_value())),
                )
            if body.infoway_api_key and body.infoway_api_key.get_secret_value().strip():
                workspaces.research.attach_market(
                    workspace.research_session_id,
                    InfowayDataProvider(InfowayApiClient(body.infoway_api_key.get_secret_value())),
                )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return data_services(workspace_id)

    @app.post("/api/workspaces/{workspace_id}/prevaluation-review")
    def prevaluation_review(workspace_id: str):
        get_workspace(workspace_id)
        return workspaces.prevaluation_review(workspace_id)

    @app.post("/api/workspaces/{workspace_id}/approvals", status_code=202)
    def approve(
        workspace_id: str,
        body: CheckpointDecision,
        background_tasks: BackgroundTasks,
    ):
        get_workspace(workspace_id)
        try:
            record = workspaces.approve(workspace_id, body.checkpoint_id, body.note)
        except KeyError:
            raise HTTPException(404, "checkpoint not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        if str(record.status) == "created":
            background_tasks.add_task(execute_background, record.run_id)
        return {"run_id": record.run_id, "status": record.status}

    @app.post("/api/workspaces/{workspace_id}/reopen")
    def reopen(workspace_id: str, body: CheckpointDecision):
        get_workspace(workspace_id)
        try:
            return workspaces.reopen(workspace_id, body.checkpoint_id, body.note)
        except KeyError:
            raise HTTPException(404, "checkpoint not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/workspaces/{workspace_id}/versions", status_code=202)
    def create_version(
        workspace_id: str,
        body: WorkspaceRevision,
        background_tasks: BackgroundTasks,
    ):
        get_workspace(workspace_id)
        try:
            record = workspaces.revise(workspace_id, body.reason, body.changes)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        background_tasks.add_task(execute_background, record.run_id)
        return {"run_id": record.run_id, "status": record.status}

    @app.post("/api/workspaces/{workspace_id}/pause")
    def pause_workspace(workspace_id: str):
        get_workspace(workspace_id)
        return workspaces.pause(workspace_id)

    @app.post("/api/workspaces/{workspace_id}/resume", status_code=202)
    def resume_workspace(
        workspace_id: str,
        background_tasks: BackgroundTasks,
    ):
        workspace = get_workspace(workspace_id)
        try:
            resumed = workspaces.resume(workspace_id)
            if resumed.get("run_id"):
                background_tasks.add_task(execute_background, resumed["run_id"])
                return {"run_id": resumed["run_id"], "status": "queued"}
            turn = ResearchTurn(
                content="继续完成上一项估值任务，复用已保存的资料、需求图和确认结果。"
            )
            request_id, created = workspaces.research.reserve_turn(
                workspace.research_session_id, turn
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        if created:
            queued = turn.model_copy(update={"request_id": request_id})
            background_tasks.add_task(
                execute_workspace_message, workspace_id, queued
            )
        return {"request_id": request_id, "status": "queued"}

    @app.post("/api/workspaces/{workspace_id}/cancel")
    def cancel_workspace(workspace_id: str):
        get_workspace(workspace_id)
        return workspaces.cancel(workspace_id)

    @app.get("/api/workspaces/{workspace_id}/versions/compare")
    def compare_versions(workspace_id: str, left: str, right: str):
        get_workspace(workspace_id)
        try:
            return workspaces.compare_versions(workspace_id, left, right)
        except KeyError:
            raise HTTPException(404, "valuation version not found") from None

    @app.get("/api/workspaces/{workspace_id}/timeline")
    def timeline(workspace_id: str):
        get_workspace(workspace_id)
        return workspaces.timeline(workspace_id)

    @app.get("/api/workspaces/{workspace_id}/reproducibility")
    def reproducibility(workspace_id: str):
        get_workspace(workspace_id)
        return workspaces.reproducibility_manifest(workspace_id)

    @app.get("/api/workspaces/{workspace_id}/sources/{file_id}")
    def source_blocks(workspace_id: str, file_id: str, offset: int = 0, block_id: str = ""):
        workspace = get_workspace(workspace_id)
        session = workspaces.store.get_research(workspace.research_session_id)
        if not any(document.file_id == file_id for document in session.documents):
            raise HTTPException(404, "source not found in this workspace")
        try:
            blocks = workspaces.store.research_blocks(workspace.research_session_id, file_id)
        except ValueError:
            raise HTTPException(404, "source not found in this workspace") from None
        start = max(0, offset)
        if block_id:
            matches = [index for index, block in enumerate(blocks) if block["block_id"] == block_id]
            if not matches:
                raise HTTPException(404, "source block not found")
            start = max(0, matches[0] - 1)
        end = min(len(blocks), start + 12)
        return {"total": len(blocks), "blocks": blocks[start:end],
                "next_offset": end if end < len(blocks) else None}

    @app.get("/api/workspaces/{workspace_id}/export")
    def export_workspace(workspace_id: str, format: str = "json"):
        workspace = get_workspace(workspace_id)
        try:
            session = workspaces.store.get_research(workspace.research_session_id)
            artifact = write_report(workspaces.research, session, ReportWrite(format=format))
            metadata, content = workspaces.store.get_artifact(session.session_id, artifact["artifact_id"])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return Response(
            content,
            media_type=metadata["media_type"],
            headers={
                "Content-Disposition": f'attachment; filename="{workspace_id}.{format}"',
                "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox",
                "X-Artifact-Id": metadata["artifact_id"], "ETag": '"' + metadata["sha256"] + '"',
            },
        )

    @app.get("/api/workspaces/{workspace_id}/files")
    def workspace_files(workspace_id: str, query: str = "", offset: int = 0):
        workspace = get_workspace(workspace_id)
        session = workspaces.store.get_research(workspace.research_session_id)
        try:
            return list_files(workspaces.store, session, FileList(query=query, offset=offset))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/workspaces/{workspace_id}/files/{file_id}")
    def workspace_file(workspace_id: str, file_id: str):
        workspace = get_workspace(workspace_id)
        session = workspaces.store.get_research(workspace.research_session_id)
        try:
            return inspect_file(workspaces.store, session, FileReference(file_id=file_id))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/workspaces/{workspace_id}/files/{file_id}/pages/{page}")
    def workspace_page(workspace_id: str, file_id: str, page: int):
        workspace = get_workspace(workspace_id)
        session = workspaces.store.get_research(workspace.research_session_id)
        try:
            content, reference = render_page(workspaces.store, session, PageView(file_id=file_id, page=page))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return Response(content, media_type="image/png", headers={"X-Content-Type-Options": "nosniff",
            "ETag": '"' + reference["image_sha256"] + '"', "Cache-Control": "private, no-store"})

    @app.get("/api/workspaces/{workspace_id}/artifacts")
    def workspace_artifacts(workspace_id: str):
        workspace = get_workspace(workspace_id)
        return {"artifacts": workspaces.store.list_artifacts(workspace.research_session_id)}

    @app.get("/api/workspaces/{workspace_id}/artifacts/{artifact_id}")
    def download_saved_artifact(workspace_id: str, artifact_id: str):
        workspace = get_workspace(workspace_id)
        try:
            metadata, content = workspaces.store.get_artifact(workspace.research_session_id, artifact_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        return Response(content, media_type=metadata["media_type"], headers={
            "Content-Disposition": f'attachment; filename="{metadata["filename"]}"',
            "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox",
            "ETag": '"' + metadata["sha256"] + '"', "Cache-Control": "private, no-store",
        })
