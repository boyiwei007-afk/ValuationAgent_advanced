from __future__ import annotations
import json
import re
import threading
import uuid
from decimal import Decimal
from typing import Literal
from pydantic import Field
from valuationagent.core.data import LocalDataProvider
from valuationagent.core.i18n import agent_context, translator
from valuationagent.core.plugins import FinancialModelPlugin
from valuationagent.core.tools import ToolRegistry, ToolSpec, canonical
from valuationagent.llm.agent import run_tool_loop
from valuationagent.llm.client import LlmError
from valuationagent.schemas.models import (
    ApiModel,
    AssumptionInputs,
    RevisionInput,
    RunStatus,
    ValuationRequest,
)
from valuationagent.storage.sqlite import SQLiteRunStore
from valuationagent.workflow.graph import WorkflowServices, build_workflow


class ExplainArguments(ApiModel):
    topic: Literal[
        "overview", "assumptions", "dcf", "relative", "sensitivity", "risks", "history"
    ] = "overview"


class ChangeArguments(ApiModel):
    assumptions: AssumptionInputs
    reason: str = Field(min_length=1, max_length=1000)


def merge(base, patch):
    result = dict(base)
    for key, value in patch.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


class ValuationRunner:
    def __init__(self, store: SQLiteRunStore, finance: FinancialModelPlugin, data=None):
        self.store = store
        self.finance = finance
        self.data = data or LocalDataProvider()
        self._run_clients = {}
        self._run_data = {}
        self._client_lock = threading.RLock()
        self._pause_events = {}

    def request_pause(self, run_id):
        with self._client_lock:
            self._pause_events.setdefault(run_id, threading.Event()).set()

    def create_run(self, request, llm=None, *, parent_id=None, reason=None):
        if request.mode == "live" and llm is None:
            raise ValueError("live 模式需要模型会话或 VALUATION_LLM_* 环境变量。")
        record = self.store.create_run(
            "run_" + uuid.uuid4().hex, request, parent_id=parent_id, reason=reason
        )
        if llm is not None:
            self.attach_model(record.run_id, llm)
        self.store.add_message(record.run_id, "user", request.user_goal, "intake")
        self._say(
            record.run_id,
            translator(request.language)(
                "任务已创建 · 版本 {revision} · {mode}"
            ).format(revision=record.revision, mode=request.mode),
            "intake",
        )
        return record

    def attach_model(self, run_id, llm):
        self.store.get_run(run_id)
        with self._client_lock:
            self._run_clients[run_id] = llm

    def attach_data(self, run_id, provider):
        """Attach an in-memory data provider to one run only.

        Runtime credentials stay outside the persisted request, events and
        reports. This also prevents one Web research session from changing the
        provider used by another concurrent run.
        """
        self.store.get_run(run_id)
        if not callable(getattr(provider, "resolve", None)):
            raise ValueError("A股取数服务配置无效。")
        with self._client_lock:
            self._run_data[run_id] = provider

    def execute(self, run_id):
        record = self.store.get_run(run_id)
        if record.status in ("completed", "completed_with_warnings", "cancelled"):
            return record
        owner = uuid.uuid4().hex
        if not self.store.acquire(run_id, owner):
            raise ValueError("该任务正在执行。进程意外退出后，最多等待30秒再恢复。")
        stopped = threading.Event()
        with self._client_lock:
            pause = self._pause_events.setdefault(run_id, threading.Event())
            pause.clear()

        def heartbeat():
            while not stopped.wait(5):
                self.store.heartbeat(run_id, owner)

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            # Recheck under lease to avoid racing a just-completed execution.
            record = self.store.get_run(run_id)
            if record.status in ("completed", "completed_with_warnings", "cancelled"):
                return record
            self.store.update_run(run_id, status=RunStatus.RUNNING)
            self.store.append_event(
                run_id,
                type="run.started",
                status="running",
                summary="开始执行",
                payload={
                    "mode": record.request.mode,
                    "parameters": record.request.agent_parameters(),
                    "revision": record.revision,
                    "model_version": self.finance.version,
                },
            )
            services = WorkflowServices(
                self.store,
                self.finance,
                self._run_clients.get(run_id),
                self._run_data.get(run_id, self.data),
                pause.is_set,
            )
            state = build_workflow(services).invoke(
                {
                    "run_id": run_id,
                    "request": record.request,
                    "blocked": False,
                    "warnings": [],
                }
            )
            if state.get("blocked"):
                self.store.update_run(
                    run_id, status=RunStatus.WAITING_REVIEW, review=state["review"]
                )
            else:
                result = state["result"]
                status = (
                    RunStatus.COMPLETED_WITH_WARNINGS
                    if result.warnings
                    else RunStatus.COMPLETED
                )
                self.store.update_run(run_id, status=status, result=result)
                self.store.append_event(
                    run_id,
                    type="run.completed",
                    status=status.value,
                    summary="估值完成",
                )
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, (ValueError, LlmError))
                else "工作流异常，请检查插件后恢复；已保留成功步骤。"
            )
            self.store.update_run(
                run_id,
                status=RunStatus.FAILED,
                error={
                    "code": "WORKFLOW_FAILED",
                    "type": type(exc).__name__,
                    "message": message,
                },
            )
            self.store.append_event(
                run_id, type="run.failed", status="failed", summary=message
            )
        finally:
            stopped.set()
            worker.join(timeout=1)
            self.store.release(run_id, owner)
        return self.store.get_run(run_id)

    def run(self, request, llm=None):
        return self.execute(self.create_run(request, llm).run_id)

    def revise(self, run_id, revision: RevisionInput, *, execute=True):
        parent = self.store.get_run(run_id)
        if parent.status in ("running", "created"):
            raise ValueError("请等待当前任务结束，再创建更正版本。")
        allowed = set(ValuationRequest.model_fields) - {"user_goal"}
        if not revision.changes or not set(revision.changes) <= allowed:
            raise ValueError("更正需要有效的请求字段；不接受任务状态、结果或空更正。")
        original = parent.request.model_dump(mode="json")
        if "assumptions" in revision.changes:
            # Use the effective uploaded assumptions as the base before switching to manual.
            if parent.result:
                # Preserve advanced request inputs (for example beta and market
                # rates) and carry forward the assumptions the model may
                # already have resolved or the user may have revised.
                effective_assumptions = dict(original.get("assumptions") or {})
                for field_name in (
                    "revenue_growth",
                    "ebit_margin",
                    "revenue_growth_scenarios",
                    "ebit_margin_scenarios",
                    "wacc",
                    "terminal_growth",
                ):
                    effective_assumptions[field_name] = getattr(
                        parent.result.assumptions, field_name
                    )
                original["assumptions"] = effective_assumptions
            original["assumption_source"] = "manual"
            original["assumption_file_ids"] = []
        changed = ValuationRequest.model_validate(merge(original, revision.changes))
        llm = self._run_clients.get(run_id)
        child = self.create_run(changed, llm, parent_id=run_id, reason=revision.reason)
        if run_id in self._run_data:
            self.attach_data(child.run_id, self._run_data[run_id])
        self.store.append_event(
            child.run_id,
            type="revision.created",
            status="created",
            summary=revision.reason,
            payload={
                "parent_run_id": run_id,
                "revision": child.revision,
                "changed_fields": list(revision.changes),
            },
        )
        return self.execute(child.run_id) if execute else child

    def resume(self, run_id, llm=None):
        if llm is not None:
            self.attach_model(run_id, llm)
        return self.execute(run_id)

    def _say(self, run_id, text, stage="conversation", related_run_id=None):
        msg = self.store.add_message(run_id, "assistant", text, stage, related_run_id)
        self.store.append_event(
            run_id,
            type="conversation.message",
            stage=stage,
            status="completed",
            summary=text,
            payload={
                "message_id": msg.message_id,
                "role": "assistant",
                "related_run_id": related_run_id,
            },
        )
        return msg

    def _explain(self, record, topic):
        _ = translator(record.request.language)
        if record.status == "waiting_review":
            return _("任务等待复核：") + (record.review or {}).get(
                "message", _("请补充数据。")
            )
        if record.result is None:
            return _("任务状态为 {status}，还没有可用的估值结果。").format(
                status=record.status
            )
        result = record.result
        if topic == "assumptions":
            a = result.assumptions
            return (
                _("WACC {wacc:.2%}，永续增长率 {growth:.2%}；收入增长：").format(
                    wacc=a.wacc, growth=a.terminal_growth
                )
                + ", ".join(f"{v:.2%}" for v in a.revenue_growth)
                + _("。来源：")
                + _(a.source)
            )
        if topic == "risks":
            model_note = (
                _("模型为参考版本，尚待金融团队核准。")
                if record.result.model_version.endswith("-reference")
                else _("正式模型已接入；请重点复核数据等级、自动降级和市场参数时点。")
            )
            review_focus = _("请复核资本成本、终值、同业选择和数据口径。")
            details = [review_focus, *result.warnings]
            return model_note + " " + "; ".join(details)
        if topic == "relative":
            return "; ".join(
                f"{v.method.upper()}: {v.per_share_value:.2f}" + _("/股")
                if v.status == "success"
                else f"{v.method}: {v.reason}"
                for v in result.relative
            )
        if topic == "sensitivity":
            return _(
                "已生成 {count} 个 WACC × 永续增长率组合；无效组合明确标记。"
            ).format(count=len(result.sensitivity))
        if topic == "history":
            return "\n".join(
                f"v{r.revision} · {r.status} · {r.run_id}"
                for r in self.store.revisions(record.run_id)
            )
        return result.executive_summary

    def converse(self, run_id, content):
        record = self.store.get_run(run_id)
        _ = translator(record.request.language)
        self.store.add_message(run_id, "user", content, "conversation")
        llm = self._run_clients.get(run_id)

        def change(args):
            if not args.assumptions.model_dump(
                exclude_none=True, exclude_defaults=True
            ):
                raise ValueError("请指定要修改的假设。")
            child = self.revise(
                run_id,
                RevisionInput(
                    reason=args.reason,
                    changes={
                        "assumptions": args.assumptions.model_dump(
                            exclude_none=True, exclude_defaults=True, mode="json"
                        )
                    },
                ),
            )
            return {
                "_terminal": True,
                "related_run_id": child.run_id,
                "answer": _("已创建版本 {revision}，旧版本保留。").format(
                    revision=child.revision
                )
                + self._explain(child, "overview"),
            }

        def explain(args):
            return {"_terminal": True, "answer": self._explain(record, args.topic)}

        if llm is not None:
            registry = ToolRegistry(
                [
                    ToolSpec(
                        "explain_valuation",
                        "读取已有结果与假设、风险或历史版本，数字由系统渲染。",
                        ExplainArguments,
                        explain,
                    ),
                    ToolSpec(
                        "revise_assumptions",
                        "仅在用户明确要求修改假设时创建版本并重算；比例用小数。",
                        ChangeArguments,
                        change,
                    ),
                ]
            )
            chain = [record]
            while chain[0].parent_run_id and len(chain) < 12:
                chain.insert(0, self.store.get_run(chain[0].parent_run_id))
            history = [
                m
                for ancestor in chain
                for m in self.store.list_messages(ancestor.run_id)
            ][-24:]
            messages = [
                {
                    "role": "system",
                    "content": "你是估值任务助手。根据用户意图选择工具。询问只读结果调用 explain_valuation；"
                    "明确修改假设才调用 revise_assumptions。不要生成估值数字。历史对话用于理解上下文。"
                    + "当前假设："
                    + canonical(
                        record.result.assumptions
                        if record.result
                        else record.request.assumptions
                    )
                    + agent_context(record.request),
                }
            ]
            messages += [
                {"role": m.role, "content": m.content}
                for m in history
                if m.role in ("user", "assistant")
            ]
            services = WorkflowServices(self.store, self.finance, llm, self.data)
            decision = run_tool_loop(
                llm,
                messages,
                registry,
                lambda name, args, fn: services.tool(
                    run_id,
                    "conversation",
                    name,
                    {"arguments": json.loads(args)},
                    fn,
                    cache=False,
                ),
            )
        else:
            # Offline mode supports a deliberately small, explicit command grammar.
            pattern = r"(?i)(wacc|terminal[_ ]growth|revenue[_ ]growth|ebit[_ ]margin|永续增长率|收入增长率|营业利润率)\s*(?:改为|调整为|设为|=|为|to\b)\s*(-?\d+(?:\.\d+)?)\s*(%|％)?"
            changes = {}
            for name, value, percent in re.findall(pattern, content):
                key = {
                    "wacc": "wacc",
                    "永续增长率": "terminal_growth",
                    "收入增长率": "revenue_growth",
                    "营业利润率": "ebit_margin",
                    "terminal_growth": "terminal_growth",
                    "revenue_growth": "revenue_growth",
                    "ebit_margin": "ebit_margin",
                }[name.lower().replace(" ", "_")]
                number = Decimal(value) / (100 if percent else 1)
                changes[key] = (
                    [number] * record.request.forecast_years
                    if key in ("revenue_growth", "ebit_margin")
                    else number
                )
            if changes:
                decision = change(
                    ChangeArguments(
                        assumptions=AssumptionInputs(**changes), reason=content
                    )
                )
            else:
                topic = next(
                    (
                        topic
                        for topic, words in (
                            ("sensitivity", ("敏感", "sensitivity")),
                            ("risks", ("风险", "risk")),
                            ("history", ("版本", "history", "revision")),
                            ("relative", ("相对", "relative", "multiples")),
                            (
                                "assumptions",
                                ("wacc", "假设", "增长率", "assumption", "growth"),
                            ),
                        )
                        if any(word in content.lower() for word in words)
                    ),
                    "overview",
                )
                decision = explain(ExplainArguments(topic=topic))
        return self._say(
            run_id, decision["answer"], related_run_id=decision.get("related_run_id")
        )

    def answer(self, run_id, content):
        return self.converse(run_id, content).content
