from io import StringIO

import pytest
from rich.cells import cell_len
from rich.console import Console
from typer.testing import CliRunner

from valuationagent.api.main import create_app
from valuationagent.cli import main as cli
from valuationagent.cli.ui import welcome
from valuationagent.llm.client import LlmError, OpenAICompatibleClient
from valuationagent.schemas.models import ModelConnectionInput
from valuationagent.schemas.research import ResearchTurn
from test_unified_workspace_agent import ScriptedModel


def test_workbench_separates_observation_binding_from_model_admission():
    from valuationagent.cli.ui import workbench

    stream = StringIO()
    target = Console(file=stream, width=140, force_terminal=False)
    snapshot = {"research": {"session": {"facts": [], "documents": []}}, "research_plan": {
        "evidence_counts": {"observations_verified": 2, "semantic_review_pending": 1, "semantic_review_supported": 1},
        "extraction_recovery": {"files": [{"file_id": "file_example", "failure_count": 1}]},
        "annual_coverage": [{"year": 2025, "candidate_metrics": ["营业收入", "利息费用"], "bound_metrics": ["营业收入", "利息费用"], "confirmed_metrics": ["revenue"]}],
        "table_repairs": [{"affected_count": 3, "issue": "年度列未绑定"}],
        "method_readiness": [{"method": "pe", "status": "inputs_ready", "reason": "尚未计算"}, {"method": "dcf", "status": "blocked", "reason": "桥接未确认"}],
    }}
    target.print(workbench(snapshot))
    output = stream.getvalue()
    assert "2 原文已绑定" in output and "2025: 2/2/1" in output
    assert "补读原文或换视图后修正解释" in output
    assert "1 待复核 / 1 已支持" in output
    assert "独立审计" in output and "读取恢复" in output
    assert "PE · 输入准备通过" in output and "DCF · 输入未就绪" in output


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    monkeypatch.delenv("VALUATION_LLM_API_KEY", raising=False)
    monkeypatch.delenv("VALUATION_LLM_MODEL", raising=False)
    application = create_app(tmp_path)
    monkeypatch.setattr("valuationagent.api.main.create_app", lambda: application)
    return CliRunner(), application.state.workspaces


def test_bare_command_configures_model_and_uses_workspace_agent(terminal, monkeypatch):
    runner, service = terminal
    model = ScriptedModel(("finish_response", {"answer": "先讨论，不进行计算。"}))
    configured = []
    monkeypatch.setattr(cli, "configure_model", lambda current=None: configured.append(current) or model)
    result = runner.invoke(cli.app, [], input="先讨论估值方法\n/exit\n")
    assert result.exit_code == 0, result.output
    assert configured == [None]
    workspace = service.list()[0]
    snapshot = service.snapshot(workspace.workspace_id)
    assert snapshot["messages"][-1]["content"] == "先讨论，不进行计算。"
    assert workspace.active_run_id is None
    assert "valuationagent --workspace " + workspace.workspace_id in result.output


def test_in_terminal_configuration_hides_key_and_checks_connection(monkeypatch):
    responses = iter(["https://models.example.test/v1", "test-model", "json_content", "chat_template", "disabled", 0.6, 8192, 16384, 180, "synthetic-secret"])
    prompts = []
    checked = []

    def prompt(label, **kwargs):
        prompts.append((label, kwargs))
        return next(responses)

    monkeypatch.setattr(cli.typer, "prompt", prompt)
    monkeypatch.setattr(OpenAICompatibleClient, "test_connection", lambda client: checked.append(client.config))
    model = cli.configure_model()
    assert prompts[-1][1]["hide_input"] is True
    assert prompts[-1][1]["show_default"] is False
    assert checked == [model.config]
    assert model.config.model == "test-model"
    assert model.config.tool_call_format == "json_content"
    assert model.config.reasoning_protocol == "chat_template"
    assert model.config.thinking == "disabled"


@pytest.mark.parametrize("provider_name", ["tushare", "infoway"])
def test_optional_market_connection_stays_in_process_without_fetching(terminal, monkeypatch, provider_name):
    runner, service = terminal
    monkeypatch.setattr(cli, "configure_model", lambda current=None: ScriptedModel())
    prompts = []
    def prompt(label, **kwargs):
        prompts.append(kwargs)
        return "synthetic-market-token"
    monkeypatch.setattr(cli.typer, "prompt", prompt)
    monkeypatch.setattr(cli.TushareApiClient, "query", lambda *args, **kwargs: pytest.fail("configuration should not fetch"))
    monkeypatch.setattr(cli.InfowayApiClient, "query", lambda *args, **kwargs: pytest.fail("configuration should not fetch"))
    result = runner.invoke(cli.app, [], input=f"/market {provider_name}\n/exit\n")
    assert result.exit_code == 0, result.output
    assert prompts[-1]["hide_input"]
    assert "synthetic-market-token" not in result.output
    workspace = service.list()[0]
    assert workspace.research_session_id in service.research._market_clients
    assert "synthetic-market-token" not in str(service.snapshot(workspace.workspace_id))


def test_cli_exports_reports_without_model_calls_and_keeps_file_commands(terminal, monkeypatch, tmp_path):
    runner, service = terminal
    monkeypatch.setattr(cli, "configure_model", lambda current=None: ScriptedModel())
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, [], input="/files\n/export md\n/artifacts\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "ARTIFACT / 文件交付" in result.output
    workspace = service.list()[0]
    artifacts = service.store.list_artifacts(workspace.research_session_id)
    assert len(artifacts) == 1
    assert artifacts[0]["numeric_result_available"] is False
    assert len(list((tmp_path / "artifacts").glob("*.md"))) == 1
    assert len(list((tmp_path / "artifacts").glob("*.manifest.json"))) == 1


def test_configuration_failure_can_retry_without_losing_workspace(terminal, monkeypatch):
    runner, service = terminal
    model = ScriptedModel(("finish_response", {"answer": "连接恢复，继续讨论。"}))
    attempts = []

    def configure(current=None):
        attempts.append(current)
        if len(attempts) == 1:
            raise LlmError("LLM_HTTP_401: 认证失败")
        return model

    monkeypatch.setattr(cli, "configure_model", configure)
    result = runner.invoke(cli.app, [], input="/model\n继续讨论\n/exit\n")
    assert result.exit_code == 0, result.output
    assert len(service.list()) == 1
    assert len(attempts) == 2
    assert "连接恢复，继续讨论。" in result.output


def test_invalid_input_does_not_close_conversation(terminal, monkeypatch):
    runner, service = terminal
    model = ScriptedModel(("finish_response", {"answer": "继续。"}))
    monkeypatch.setattr(cli, "configure_model", lambda current=None: model)
    result = runner.invoke(cli.app, [], input="x" * 8001 + "\n有效输入\n/exit\n")
    assert result.exit_code == 0, result.output
    snapshot = service.snapshot(service.list()[0].workspace_id)
    assert [message["content"] for message in snapshot["messages"] if message["role"] == "user"] == ["有效输入"]
    assert "8000" in result.output


def test_resume_and_review_option_keep_existing_workspace(terminal, monkeypatch):
    runner, service = terminal
    model = ScriptedModel()
    monkeypatch.setattr(cli, "configure_model", lambda current=None: model)
    assert runner.invoke(cli.app, ["--review"], input="/exit\n").exit_code == 0
    workspace = service.list()[0]
    assert workspace.run_policy == "review"
    result = runner.invoke(cli.app, ["--workspace", workspace.workspace_id], input="/exit\n")
    assert result.exit_code == 0, result.output
    assert len(service.list()) == 1


def test_resume_renders_saved_progress_and_decision_before_new_input(terminal, monkeypatch):
    from valuationagent.schemas.research import DecisionPrompt, DecisionOption
    runner, service = terminal
    monkeypatch.setattr(cli, "configure_model", lambda current=None: ScriptedModel())
    workspace = service.create()
    session = service.store.get_research(workspace.research_session_id)
    session.pending_decision = DecisionPrompt(question="选择研究范围", options=[
        DecisionOption(label="主营业务", description="需要业务数据"),
        DecisionOption(label="整体公司", description="保留适用边界"),
    ])
    service.store.save_research(session)
    service.store.add_message(session.session_id, "assistant", "现有事实已保存，尚未计算。", "agent")
    result = runner.invoke(cli.app, ["--workspace", workspace.workspace_id], input="/exit\n")
    assert result.exit_code == 0, result.output
    assert "流程工作台" in result.output and "选择研究范围" in result.output
    assert "现有事实已保存，尚未计算。" in result.output
    assert "主营业务" in result.output and "整体公司" in result.output


def test_search_credentials_never_enter_persisted_context(terminal, monkeypatch):
    runner, service = terminal
    monkeypatch.setattr(cli, "configure_model", lambda current=None: ScriptedModel())
    prompts = []

    def prompt(label, **kwargs):
        prompts.append(kwargs)
        return "synthetic-search-secret"

    monkeypatch.setattr(cli.typer, "prompt", prompt)
    result = runner.invoke(cli.app, [], input="/search\n/status\n/exit\n")
    assert result.exit_code == 0, result.output
    assert prompts[0]["hide_input"]
    snapshot = service.snapshot(service.list()[0].workspace_id)
    assert "synthetic-search-secret" not in str(snapshot)
    assert "synthetic-search-secret" not in result.output


def test_environment_configuration_remains_optional(terminal, monkeypatch):
    runner, service = terminal
    monkeypatch.setenv("VALUATION_LLM_MODEL", "test-model")
    monkeypatch.setenv("VALUATION_LLM_API_KEY", "synthetic-env-secret")
    monkeypatch.setenv("VALUATION_LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(cli, "configure_model", lambda *args: pytest.fail("unnecessary prompt"))
    result = runner.invoke(cli.app, [], input="/exit\n")
    assert result.exit_code == 0, result.output
    workspace = service.list()[0]
    assert service.research.model_connected(workspace.research_session_id)
    assert "synthetic-env-secret" not in str(service.snapshot(workspace.workspace_id))


def test_cancelling_configuration_exits_without_starting_a_turn(terminal, monkeypatch):
    runner, service = terminal

    def cancel(current=None):
        raise cli.typer.Abort()

    monkeypatch.setattr(cli, "configure_model", cancel)
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert "已退出" in result.output
    assert not any(message["role"] == "user" for message in service.snapshot(service.list()[0].workspace_id)["messages"])


def test_cli_choice_with_free_text_is_a_normal_workspace_message(terminal, monkeypatch):
    runner, service = terminal
    model = ScriptedModel(
        ("finish_response", {"answer": "请选口径或补充。", "outcome": "needs_input", "decision": {
            "question": "选择口径", "options": [{"label": "合并口径", "description": "披露范围限制"},
                                              {"label": "分部口径", "description": "需分部数据"}]}}),
        ("finish_response", {"answer": "已记录额外要求。"}),
    )
    monkeypatch.setattr(cli, "configure_model", lambda current=None: model)
    result = runner.invoke(cli.app, [], input="讨论方案\nA 补充敏感性分析\n/status\n/exit\n")
    assert result.exit_code == 0, result.output
    snapshot = service.snapshot(service.list()[0].workspace_id)
    submitted = [message["content"] for message in snapshot["messages"] if message["role"] == "user"][-1]
    assert "合并口径" in submitted and "补充敏感性分析" in submitted
    assert "流程工作台" in result.output
    assert "方案选择" in result.output


def test_upload_attaches_to_next_agent_message(terminal, monkeypatch, tmp_path):
    runner, service = terminal
    model = ScriptedModel(("finish_response", {"answer": "已读取附件，不进行估值。"}))
    monkeypatch.setattr(cli, "configure_model", lambda current=None: model)
    document = tmp_path / "user note.txt"
    document.write_text("测试资料，不包含估值数值。", encoding="utf-8")
    result = runner.invoke(cli.app, [], input=f'/upload "{document}"\n总结附件\n/exit\n')
    assert result.exit_code == 0, result.output
    session = service.store.get_research(service.list()[0].research_session_id)
    assert len(session.documents) == 1
    assert session.documents[0].name == document.name


@pytest.mark.parametrize("width", [40, 70, 112])
def test_terminal_welcome_fits(width):
    stream = StringIO()
    Console(file=stream, width=width).print(welcome())
    assert "ValuationAgent" in stream.getvalue()
    assert all(cell_len(line) <= width for line in stream.getvalue().splitlines())


def test_old_protocol_and_commands_are_not_reintroduced():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResearchTurn(content="确认", question_id="old", option_id="confirm")
    for command in ("wizard", "interactive", "research", "demo", "run"):
        assert CliRunner().invoke(cli.app, [command]).exit_code == 2
    assert CliRunner().invoke(cli.app, ["--help"]).exit_code == 0


def test_configuration_validation_never_echoes_credentials(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream))
    try:
        ModelConnectionInput(model="", api_key="synthetic-hidden-secret", base_url="https://example.test?key=hidden")
    except ValueError as exc:
        cli.show_error(exc)
    assert "synthetic-hidden-secret" not in stream.getvalue()
    assert "?key=hidden" not in stream.getvalue()
