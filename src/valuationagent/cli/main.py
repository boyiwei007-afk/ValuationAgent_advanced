"""Terminal client for the same workspace agent used by the web client."""
from __future__ import annotations

import json
import mimetypes
import os
import re
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.text import Text

from valuationagent.application.reproducibility import replay_bundle
from valuationagent.application.file_workspace import FileList, FileRead, list_files, read_file
from valuationagent.application.workspace_artifacts import ReportWrite, write_report, save_local_artifact
from valuationagent.cli.ui import console, welcome, conversation, decision_view, workbench, execute_with_display, panel
from valuationagent.llm.client import OpenAICompatibleClient
from valuationagent.market.tushare import TushareApiClient, TushareDataProvider
from valuationagent.schemas.models import ModelConnectionInput
from valuationagent.schemas.research import ResearchTurn
from valuationagent.search.providers import TavilySearchProvider

app = typer.Typer(no_args_is_help=False, pretty_exceptions_enable=False)

HELP = """直接输入需求即可对话、检索或估值。
/model                 配置或更换模型（密钥隐藏，仅本进程保存）
/search                配置 Tavily 搜索（可稍后配置）
/market                配置 Tushare 结构化财务数据（可选）
/upload 文件路径        上传附件，随下一条消息交给 Agent
/files                 查看本工作区原文文件
/read 文件ID            读取已保存原文（精细页码/表格可直接让 Agent 读取）
/vision                明确启用/停用当前模型图片输入（接口须支持图片）
/export md|pdf|html|json 生成报告并保存至当前目录 artifacts
/artifacts             查看已生成的报告与笔记
/save 文件ID            将生成文件保存到当前目录 artifacts
/status                查看当前工作区
/status --json         查看完整工作区数据
/approve               审阅并批准冻结的估值方案
/help                  查看命令
/exit                  保存并退出"""


def show_error(exc):
    if isinstance(exc, ValidationError):
        message = "；".join(
            ".".join(str(part) for part in error["loc"]) + ": " + error["msg"]
            for error in exc.errors(include_input=False, include_url=False)
        )
    else:
        message = str(exc)
    console.print(panel(Text(message, style="#FB7185"), "需要处理"))


def configure_model(current=None):
    previous = current.config if current is not None else None
    console.print(panel(Text("配置 OpenAI-compatible 接口\n连接检查会发送一次工具调用请求。密钥隐藏输入，仅本进程使用。"), "MODEL / 模型连接"))
    base_url = typer.prompt(
        "Base URL",
        default=previous.base_url if previous else os.getenv("VALUATION_LLM_BASE_URL", "https://api.deepseek.com"),
    )
    model_name = typer.prompt(
        "模型名称（填写供应商提供的模型 ID）",
        default=previous.model if previous else os.getenv("VALUATION_LLM_MODEL", ""),
    )
    tool_call_format = typer.prompt(
        "工具格式（native 标准 / json_content 文本JSON网关）",
        default=previous.tool_call_format if previous else os.getenv("VALUATION_LLM_TOOL_CALL_FORMAT", "native"),
    )
    reasoning_protocol = typer.prompt(
        "推理参数协议（auto 默认 / chat_template 用于 SGLang、vLLM）",
        default=previous.reasoning_protocol if previous else os.getenv("VALUATION_LLM_REASONING_PROTOCOL", "auto"),
    )
    thinking = typer.prompt(
        "思考模式（auto / enabled / disabled）",
        default=previous.thinking if previous else os.getenv("VALUATION_LLM_THINKING", "auto"),
    )
    temperature = typer.prompt(
        "采样温度（0至2，按模型部署建议设置）",
        default=previous.temperature if previous and previous.temperature is not None else float(os.getenv("VALUATION_LLM_TEMPERATURE", "0")),
        type=float,
    )
    api_key = typer.prompt("API Key（隐藏输入）", hide_input=True, show_default=False)
    config = ModelConnectionInput(base_url=base_url.strip(), model=model_name.strip(), api_key=api_key.strip(),
                                  tool_call_format=tool_call_format.strip(), reasoning_protocol=reasoning_protocol.strip(),
                                  thinking=thinking.strip(), temperature=temperature)
    candidate = OpenAICompatibleClient(config)
    candidate.test_connection()
    console.print("模型已连接；密钥不会写入数据库或配置文件。", style="green")
    return candidate


def run_chat(workspace_id=None, review=False):
    from valuationagent.api.main import create_app

    service = create_app().state.workspaces
    workspace = service.get(workspace_id) if workspace_id else service.create(
        data_source_preference="web", run_policy="review" if review else "automatic",
    )
    session_id = workspace.research_session_id
    console.print(welcome())
    console.print("Workspace: " + workspace.workspace_id, markup=False)
    console.print(panel(Text(HELP), "快捷命令 · 自由对话"))
    if workspace_id:
        snapshot = service.snapshot(workspace.workspace_id)
        console.print(workbench(snapshot))
        previous_answer = next((message for message in reversed(snapshot["messages"]) if message["role"] == "assistant"), None)
        if previous_answer:
            console.print(conversation(previous_answer["content"]))
        if decision := snapshot["research"]["session"].get("pending_decision"):
            console.print(decision_view(decision))
    model = None
    pending_files = []
    try:
        if os.getenv("VALUATION_LLM_MODEL") and os.getenv("VALUATION_LLM_API_KEY"):
            model = OpenAICompatibleClient.from_environment()
        else:
            model = configure_model()
        service.research.attach(session_id, model)
    except typer.Abort:
        raise
    except (ValueError, RuntimeError) as exc:
        show_error(exc)
        console.print("工作区已保留。使用 /model 重新配置，或 /exit 退出。")
    while True:
        content = console.input("\n你 > ").strip()
        if content == "/exit":
            console.print("工作区已保存。下次继续：")
            console.print("valuationagent --workspace " + workspace.workspace_id, markup=False)
            return
        if not content:
            continue
        try:
            if content == "/help":
                console.print(panel(Text(HELP), "快捷命令 · 自由对话"))
            elif content == "/model":
                candidate = configure_model(model)
                service.research.attach(session_id, candidate)
                model = candidate
            elif content == "/search":
                api_key = typer.prompt("Tavily API Key（隐藏输入）", hide_input=True, show_default=False)
                service.research.attach_search(session_id, TavilySearchProvider(api_key.strip()))
                console.print("搜索已配置；未发送测试查询，密钥仅在本进程有效。", style="green")
            elif content == "/market":
                token = typer.prompt("Tushare Token（隐藏输入）", hide_input=True, show_default=False)
                service.research.attach_market(session_id, TushareDataProvider(TushareApiClient(token.strip())))
                console.print("结构化数据已配置；尚未请求数据，Token 仅在本进程有效。没有此服务也可继续网页取证。", style="green")
            elif content == "/status --json":
                console.print_json(data=service.snapshot(workspace.workspace_id))
            elif content == "/status":
                console.print(workbench(service.snapshot(workspace.workspace_id)))
            elif content == "/vision":
                if not isinstance(model, OpenAICompatibleClient):
                    raise ValueError("请先用 /model 连接支持图片的模型接口。")
                enabled = typer.confirm("允许向当前模型发送选定的原始页图？接口必须支持图片；可能产生图片计费。", default=False)
                model.config.supports_images = enabled
                console.print("图片输入已启用。" if enabled else "图片输入已关闭。", markup=False)
            elif content == "/files":
                session = service.store.get_research(session_id)
                console.print_json(data=list_files(service.store, session, FileList(limit=40)))
            elif content.startswith("/read "):
                session = service.store.get_research(session_id)
                console.print_json(data=read_file(service.store, session, FileRead(file_id=content[6:].strip())))
            elif content == "/artifacts":
                console.print_json(data={"artifacts": service.store.list_artifacts(session_id)})
            elif content == "/export" or content.startswith("/export "):
                session = service.store.get_research(session_id)
                artifact = write_report(service.research, session, ReportWrite(format=content[7:].strip() or "md"))
                target = save_local_artifact(service.store, session_id, artifact["artifact_id"], Path.cwd() / "artifacts")
                console.print(panel(Text(f"报告已保存：{target}\n状态：{artifact['status']}\nSHA-256：{artifact['sha256']}"), "ARTIFACT / 文件交付"))
            elif content.startswith("/save "):
                target = save_local_artifact(service.store, session_id, content[6:].strip(), Path.cwd() / "artifacts")
                console.print("文件已保存：" + str(target), markup=False)
            elif content == "/approve":
                checkpoint = service.prevaluation_review(workspace.workspace_id)
                console.print_json(data=checkpoint.model_dump(mode="json"))
                if typer.confirm("批准这份冻结输入并计算？", default=False):
                    record = service.approve(workspace.workspace_id, checkpoint.checkpoint_id, "CLI 用户审批")
                    service.execute(record.run_id)
                    console.print_json(data=service.read_valuation(session_id))
            elif content.startswith("/upload "):
                path = Path(content[len("/upload "):].strip().strip('"')).expanduser()
                if path.stat().st_size > 50 * 1024 * 1024:
                    raise ValueError("文件超过50 MB，请拆分后上传。")
                if len(pending_files) >= 8:
                    raise ValueError("每条消息最多附带 8 个文件，请先发送当前附件。")
                metadata = service.store.save_upload(
                    path.name, "evidence", mimetypes.guess_type(path.name)[0] or "application/octet-stream", path.read_bytes(),
                )
                pending_files.append(metadata["file_id"])
                console.print("已添加附件：" + path.name + "；发送下一条消息开始解析。", markup=False)
            elif content.startswith("/"):
                console.print("未知命令；输入 /help 查看支持的命令。")
            elif model is None:
                console.print("请先使用 /model 连接模型。附件仍保留在待发送列表中。")
            else:
                decision = service.store.get_research(session_id).pending_decision
                selection = re.fullmatch(r"([1-4A-Da-d])(?:[、.：:\s]+(.*))?", content, re.S)
                if decision and selection:
                    index = int(selection[1]) - 1 if selection[1].isdigit() else ord(selection[1].upper()) - ord('A')
                    if index >= len(decision.options):
                        raise ValueError("请选择列出的方案，或直接输入自己的要求。")
                    option = decision.options[index]
                    content = f"关于“{decision.question}”，我选择{option.label}。{option.description}" + (f"\n补充：{selection[2]}" if selection[2] else "")
                turn = ResearchTurn(content=content, file_ids=pending_files)
                console.print(conversation(content, "user"))
                execute_with_display(service, workspace.workspace_id, turn, target=console)
                pending_files = []
                snapshot = service.snapshot(workspace.workspace_id)
                console.print(workbench(snapshot))
                console.print(conversation(snapshot["messages"][-1]["content"]))
                if decision := snapshot["research"]["session"].get("pending_decision"):
                    console.print(decision_view(decision))
        except typer.Abort:
            raise
        except (ValueError, RuntimeError, KeyError, OSError) as exc:
            show_error(exc)
            console.print("工作区已保留，可以修正输入后继续。")


def launch(workspace_id, review):
    try:
        run_chat(workspace_id, review)
    except (EOFError, KeyboardInterrupt, typer.Abort):
        console.print("\n已退出；已提交的工作区记录保留。")
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        show_error(exc)
        raise typer.Exit(1) from None


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace_id: str | None = typer.Option(None, "--workspace"),
    review: bool = typer.Option(False, "--review"),
):
    """不带子命令直接进入 Agent；模型和搜索可以在终端内配置。"""
    if ctx.invoked_subcommand is None:
        launch(workspace_id, review)


@app.command()
def chat(
    workspace_id: str | None = typer.Option(None, "--workspace"),
    review: bool = typer.Option(False, "--review"),
):
    """Create or continue a workspace conversation."""
    launch(workspace_id, review)


@app.command()
def replay(package: Path):
    """Replay frozen calculations without model or network access."""
    try:
        result = replay_bundle(json.loads(package.read_text(encoding="utf-8-sig")))
        console.print_json(data=result)
        if not result["passed"]:
            raise typer.Exit(1)
    except (ValueError, OSError) as exc:
        show_error(exc)
        raise typer.Exit(1) from None


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the workspace API and built web client on one origin."""
    import uvicorn
    from valuationagent.api.access import validate_bind

    try:
        validate_bind(host)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    uvicorn.run("valuationagent.api.main:app", host=host, port=port)


if __name__ == "__main__":
    app()
