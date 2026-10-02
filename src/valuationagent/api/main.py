from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.sse import EventSourceResponse

from valuationagent.application.runner import ValuationRunner
from valuationagent.application.research import ResearchService
from valuationagent.application.workspaces import ValuationWorkspaceService
from valuationagent.application.reporting import ValuationReportExporter
from valuationagent.api.workspaces import register_workspace_routes
from valuationagent.api.access import register_access
from valuationagent.finance.factory import create_financial_model
from valuationagent.finance.tools import FinanceResearchToolProvider
from valuationagent.market import create_data_provider
from valuationagent.search.providers import UnavailableSearchProvider, create_search_provider
from valuationagent.llm.client import (
    LlmError,
    ModelSessionRegistry,
    OpenAICompatibleClient,
)
from valuationagent.schemas.models import Capability, ChatMessage, ModelConnectionInput, ModelSessionPublic, RunEvent, RunRecord, RunStatus, ValuationOutput
from valuationagent.storage.sqlite import SQLiteRunStore, WorkspaceStateError
from valuationagent.llm.context import AGENT_PROMPT_VERSION


TERMINAL_STATUSES = {
    RunStatus.WAITING_REVIEW,
    RunStatus.COMPLETED,
    RunStatus.COMPLETED_WITH_WARNINGS,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
}


def create_app(data_dir: Path | str | None = None) -> FastAPI:
    runtime_dir = Path(data_dir or os.getenv("VALUATION_DATA_DIR", "var"))
    store = SQLiteRunStore(runtime_dir)
    finance = create_financial_model()
    data_provider = create_data_provider()
    search_provider = create_search_provider()
    runner = ValuationRunner(store, finance, data_provider)
    sessions = ModelSessionRegistry()
    reports = ValuationReportExporter()

    app = FastAPI(
        title="ValuationAgent API",
        version="0.6.0-dev",
        description="Valuation-first agent workspace with evidence, approval, deterministic modeling, challenge and versioning.",
    )
    origins = [
        item.strip()
        for item in os.getenv(
            "VALUATION_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if item.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )
    register_access(app, origins)

    app.state.store = store
    @app.exception_handler(WorkspaceStateError)
    async def invalid_workspace_state(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.middleware("http")
    async def cache_policy(request: Request, call_next):
        response = await call_next(request)
        # Reload after deployment must never refer to removed hashed assets;
        # task data and credentials must not enter shared HTTP caches.
        if request.url.path.startswith("/api/") or "text/html" in response.headers.get("content-type", ""):
            response.headers["Cache-Control"] = "no-store"
        return response
    app.state.runner = runner
    app.state.sessions = sessions
    app.state.research = ResearchService(
        store,
        search_provider=search_provider,
        tool_providers=(FinanceResearchToolProvider(),),
    )
    app.state.workspaces = ValuationWorkspaceService(
        store, app.state.research, runner
    )

    def execute_background(run_id):
        try:
            app.state.workspaces.execute(run_id)
        except ValueError:
            # A concurrent request may already own the execution lease.
            return

    register_workspace_routes(
        app, app.state.workspaces, sessions, execute_background
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": e["loc"], "type": e["type"], "msg": e["msg"]}
                    for e in exc.errors()
                ]
            },
        )

    def get_run_or_404(run_id: str) -> RunRecord:
        try:
            return store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "valuationagent", "version": "0.6.0-dev", "agent_version": AGENT_PROMPT_VERSION}

    @app.get("/api/capabilities", response_model=list[Capability])
    def capabilities() -> list[Capability]:
        return [
            Capability(
                capability_id="valuation_workspaces",
                available=True,
                detail="估值优先工作区、数据需求、集中复核、不可变版本、挑战层、决策层和复现清单",
            ),
            Capability(capability_id="research_document_parsing", available=True,
                       detail="研究会话读取文本 PDF、XLSX、CSV、JSON、TXT；语义抽取需要模型，OCR 待接入"),
            Capability(
                capability_id="structured_financial_input",
                available=True,
                detail="结构化财务快照可完整运行",
            ),
            Capability(
                capability_id="synthetic_demo",
                available=True,
                detail="合成数据演示，结果明确标记为 demo",
            ),
            Capability(
                capability_id="agent_conversation",
                available=True,
                detail="运行消息与结果问答共用消息接口",
            ),
            Capability(
                capability_id="execution_events",
                available=True,
                detail="SQLite 审计事件 + SSE 增量流",
            ),
            Capability(
                capability_id="finance_team_dcf",
                available=True,
                detail="非金融A股十年收入预测、FCFF、WACC、三情景DCF与敏感性分析",
            ),
            Capability(
                capability_id="finance_team_relative",
                available=True,
                detail="P/E、P/S 与 EV/EBITDA 独立相对估值及方法差异复核",
            ),
            Capability(
                capability_id="review_resume_revisions",
                available=True,
                detail="复核更正、版本、检查点恢复与重算",
            ),
            Capability(
                capability_id="json_extraction",
                available=True,
                detail="按角色导入结构化 JSON，保留来源引用",
            ),
            Capability(
                capability_id="typed_agent_tools",
                available=True,
                detail="受限工具调用循环，要求模型支持 function calling",
            ),
            Capability(
                capability_id="durable_conversation_context",
                available=True,
                detail="权威任务状态、可见长期记忆、近期对话与按需证据组成分层上下文",
            ),
            Capability(
                capability_id="interactive_agent_recovery",
                available=True,
                detail="模型、工具和文件异常保留进度，可重新连接或补充输入后继续",
            ),
            Capability(
                capability_id="agent_tool_extensions",
                available=True,
                detail="AgentToolProvider 可注册金融与数据工具，并复用统一参数校验和审计事件",
            ),
            Capability(
                capability_id="agent_application_contracts",
                available=True,
                detail="单一工具循环共享工作区状态、分层上下文、来源证据和搜索查询契约",
            ),
            Capability(
                capability_id="search_provider_contract",
                available=True,
                detail="Tavily真实联网搜索、零网络mock与未配置安全降级；结果保留URL、时点与供应商版本",
            ),
            Capability(
                capability_id="web_search_runtime",
                available=not isinstance(search_provider, UnavailableSearchProvider),
                detail=(
                    f"当前进程已连接 {search_provider.provider_id}"
                    if not isinstance(search_provider, UnavailableSearchProvider)
                    else "当前进程未配置Tavily API Key；CLI可用/search会话级连接"
                ),
            ),
            Capability(
                capability_id="ticker_data_provider",
                available=True,
                detail="Tushare Pro点时A股年报、每日估值指标及非金融行业可比公司筛选",
            ),
            Capability(
                capability_id="pdf_excel_reporting",
                available=True,
                detail="正式估值结果支持JSON、Excel审计工作簿与PDF报告",
            ),
            Capability(
                capability_id="research_document_ingestion",
                available=all(importlib.util.find_spec(name) is not None for name in ("pypdf", "openpyxl", "docx")),
                detail="研究工作台支持文本PDF、DOCX、XLSX、CSV/TSV、HTML、JSON、TXT、MD；字段语义提取需模型，扫描件OCR未接入",
            ),
            Capability(
                capability_id="research_outcome_reports", available=True,
                detail="即使没有有效数值估值也生成HTML/JSON说明报告；PDF需reports依赖。读取失败、搜索上限和待核验字段均如实披露",
            ),
        ]

    @app.get("/api/workflow-definition")
    def workflow_definition() -> dict:
        from valuationagent.workflow.graph import STAGES

        return {
            "version": "workspace-agent-v1",
            "stages": [
                {"id": stage, "label": label, "order": index}
                for index, (stage, label) in enumerate(STAGES, 1)
            ],
        }

    @app.post("/api/model-sessions", response_model=ModelSessionPublic, status_code=201)
    def create_model_session(config: ModelConnectionInput) -> ModelSessionPublic:
        return sessions.create(config)

    @app.post("/api/model-connections/test")
    def test_model_connection(config: ModelConnectionInput) -> dict[str, str]:
        try:
            reply = OpenAICompatibleClient(config).test_connection()
        except LlmError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"status": "ok", "model": config.model, "reply": reply[:60]}

    @app.delete("/api/model-sessions/{session_id}", status_code=204)
    def delete_model_session(session_id: str) -> None:
        try:
            sessions.delete(session_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="model session not found"
            ) from exc

    @app.post("/api/files", status_code=201)
    async def upload_file(
        file: Annotated[UploadFile, File(description="PDF, Excel, CSV or JSON")],
        role: Annotated[str, Form()] = "historical_financials",
    ) -> dict:
        content = await file.read(50 * 1024 * 1024 + 1)
        try:
            return store.save_upload(
                file.filename or "upload", role, file.content_type, content
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/runs", response_model=list[RunRecord])
    def list_runs(limit: int = 50) -> list[RunRecord]:
        return store.list_runs(limit)

    @app.get("/api/runs/{run_id}", response_model=RunRecord)
    def get_run(run_id: str) -> RunRecord:
        return get_run_or_404(run_id)

    @app.get("/api/runs/{run_id}/results", response_model=ValuationOutput)
    def get_results(run_id: str) -> ValuationOutput:
        record = get_run_or_404(run_id)
        if record.result is None or record.status not in {
            RunStatus.COMPLETED,
            RunStatus.COMPLETED_WITH_WARNINGS,
        }:
            raise HTTPException(
                status_code=409,
                detail={
                    "status": record.status,
                    "review": record.review,
                    "error": record.error,
                },
            )
        return record.result

    @app.get("/api/runs/{run_id}/export")
    def export_run(run_id: str, format: str = "json"):
        record = get_run_or_404(run_id)
        try:
            content, media_type, extension = reports.export(record, format, store=store)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return Response(
            content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="valuation-{run_id}.{extension}"'
            },
        )

    @app.get("/api/runs/{run_id}/events/history", response_model=list[RunEvent])
    def event_history(run_id: str, after: int = 0) -> list[RunEvent]:
        get_run_or_404(run_id)
        return store.list_events(run_id, after=max(after, 0))

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(
        run_id: str,
        request: Request,
        last_event_id: Annotated[str | None, Header()] = None,
        after: int = 0,
    ):
        get_run_or_404(run_id)
        try:
            initial_cursor = max(0, after, int(last_event_id or 0))
        except ValueError:
            raise HTTPException(
                status_code=422, detail="Last-Event-ID must be an integer"
            ) from None

        async def events():
            cursor = initial_cursor
            while True:
                if await request.is_disconnected():
                    return
                for event in store.list_events(run_id, after=cursor):
                    cursor = event.sequence
                    yield f"id: {cursor}\nevent: {event.type}\ndata: {event.model_dump_json()}\nretry: 1500\n\n"
                if store.get_run(
                    run_id
                ).status in TERMINAL_STATUSES and not store.list_events(
                    run_id, after=cursor
                ):
                    return
                await asyncio.sleep(0.15)

        return EventSourceResponse(events())

    @app.get("/api/runs/{run_id}/messages", response_model=list[ChatMessage])
    def list_messages(run_id: str) -> list[ChatMessage]:
        get_run_or_404(run_id)
        return store.list_messages(run_id)

    @app.get("/api/runs/{run_id}/revisions", response_model=list[RunRecord])
    def revisions(run_id: str):
        get_run_or_404(run_id)
        return store.revisions(run_id)

    @app.get("/api/runs/{run_id}/artifacts")
    def artifacts(run_id: str):
        get_run_or_404(run_id)
        return store.artifacts(run_id)

    # The built web app shares this origin and the same runner/database as CLI.
    # Mount only public build artifacts, never project files or uploaded data.
    web_dir = Path(
        os.getenv(
            "VALUATION_WEB_DIR",
            str(Path(__file__).resolve().parents[3] / "web" / "dist"),
        )
    )
    if (web_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app


app = create_app()
