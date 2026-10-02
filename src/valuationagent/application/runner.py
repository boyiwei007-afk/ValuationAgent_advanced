from __future__ import annotations
import threading
import uuid
from valuationagent.core.data import LocalDataProvider
from valuationagent.core.i18n import translator
from valuationagent.core.plugins import FinancialModelPlugin
from valuationagent.schemas.models import RevisionInput, RunStatus, ValuationRequest
from valuationagent.storage.sqlite import SQLiteRunStore
from valuationagent.workflow.graph import WorkflowServices, build_workflow


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
        self._run_data = {}
        self._client_lock = threading.RLock()
        self._pause_events = {}

    def request_pause(self, run_id):
        with self._client_lock:
            self._pause_events.setdefault(run_id, threading.Event()).set()

    def forget_runs(self, run_ids):
        with self._client_lock:
            for run_id in run_ids:
                self._run_data.pop(run_id, None)
                self._pause_events.pop(run_id, None)

    def create_run(self, request, *, parent_id=None, reason=None):
        record = self.store.create_run(
            "run_" + uuid.uuid4().hex, request, parent_id=parent_id, reason=reason
        )
        self.store.add_message(record.run_id, "user", request.user_goal, "intake")
        self._say(
            record.run_id,
            translator(request.language)(
                "任务已创建 · 版本 {revision} · {mode}"
            ).format(revision=record.revision, mode=request.mode),
            "intake",
        )
        return record


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
                if isinstance(exc, ValueError)
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

    def run(self, request):
        return self.execute(self.create_run(request).run_id)

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
        child = self.create_run(changed, parent_id=run_id, reason=revision.reason)
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
