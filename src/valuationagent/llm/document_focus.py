"""Bounded single-file attention mode within the existing model/tool loop."""
import json
import copy
from typing import Literal

from pydantic import Field

from valuationagent.application.file_workspace import source_document
from valuationagent.application.observation_extraction import observation_next_action
from valuationagent.core.tools import canonical
from valuationagent.llm.context import AGENT_PROMPT
from valuationagent.schemas.models import ApiModel


READING_TOOLS = {"inspect_file", "search_file", "list_source_links", "read_file", "view_pdf_page", "extract_observations",
                 "prepare_observation_review", "review_observations", "reject_candidates", "end_file_task"}
EXPLORATION_TOOLS = {"inspect_file", "search_file", "list_source_links", "read_file", "view_pdf_page", "end_file_task"}


class FileTask(ApiModel):
    file_id: str = Field(min_length=1, max_length=100)
    entity_ticker: str = Field(max_length=24, description="本次读取的真实主体代码。可比文件填可比公司代码，不填估值目标的代码；无代码的文件讨论可填空字符串。")
    role: Literal["historical", "comparable", "reference"] = Field(description="historical为目标公司，comparable为已明确的其他企业；reference用于主体未知、多公司行业文章或纯资料探索，此时可留空entity_ticker，先读资料识别主体，不提交金融事实。仅约束本次读取，不修改研究目标。")
    objective: str = Field(min_length=12, max_length=800, description="有界阅读目标，包括所需科目/年度、主体与待解决歧义，不包含预设答案。")
    metrics: list[str] = Field(min_length=1, max_length=6, description="本批优先关注的标准字段，不是允许提交字段的白名单；原文出现其他有用字段时保留其真实含义，不强行映射为重点字段。不修改估值方法。")


class FileTaskEnd(ApiModel):
    summary: str = Field(min_length=10, max_length=1200, description="只总结真实完成情况与具体剩余问题，不能自行声称核验通过。")


class DocumentFocus:
    def __init__(self, runtime):
        self.runtime = runtime
        self.task = None
        self.steps = 0

    def begin(self, args):
        if self.task is not None:
            raise ValueError("FILE_TASK_ACTIVE: 先结束当前阅读任务，再选择其他文件")
        document = source_document(self.runtime.session, args.file_id)
        if document.provenance_type in {"search_snippet", "official_index"}:
            raise ValueError("SOURCE_NOT_DOWNLOADED: 先取得原文，再开始阅读任务")
        target = self.runtime.session.draft.ticker
        if args.role == "historical" and target and args.entity_ticker != target:
            raise ValueError("FILE_TASK_ENTITY: 其他公司的文件须选择comparable，不将其作为目标公司的historical数据")
        if args.role == "comparable" and (not args.entity_ticker or args.entity_ticker == target):
            raise ValueError("FILE_TASK_ENTITY: 可比提取须明确另一家公司自身代码，不能用目标公司自比；仅探索行业文章或尚未识别主体时用role=reference阅读，确定主体后另开comparable任务")
        self.task, self.steps = args, 0
        return {"file_task": args.model_dump(), "document": document.model_dump(mode="json"),
                "instruction": "进入单文件阅读上下文，仍使用同一模型和同一总预算。只读取、解释和复核，不联网、不改任务、不计算。完成或达到局部限制用end_file_task返回主循环。"}

    def end(self, args):
        if self.task is None:
            raise ValueError("FILE_TASK_MISSING: 当前没有单文件阅读任务")
        facts = [fact for fact in self.runtime.session.facts
                 if fact.block_id.startswith(self.task.file_id + ":") and fact.status != "rejected"
                 and fact.role == self.task.role]
        result = {"summary": args.summary, "file_id": self.task.file_id,
                  "facts": [{"fact_id": fact.fact_id, "metric": fact.standard_metric, "period": fact.period,
                             "role": fact.role, "peer_ticker": fact.peer_ticker, "scope": fact.scope,
                             "status": fact.status, "warnings": fact.warnings} for fact in facts],
                  "instruction": "已返回主循环；状态由程序给出。此步骤不代表完成估值，按原用户目标继续或结束。"}
        self.task = None
        return result

    def guard(self, name, arguments):
        if self.task is None:
            return
        permitted = EXPLORATION_TOOLS if self.task.role == "reference" else READING_TOOLS
        if name not in permitted:
            raise ValueError("FILE_TASK_SCOPE: 当前仅处理已选文件；先end_file_task返回主循环才能检索、改任务或计算")
        params = json.loads(arguments) if isinstance(arguments, str) else arguments
        if params.get("file_id", self.task.file_id) != self.task.file_id:
            raise ValueError("FILE_TASK_SCOPE: 不能在单文件阅读任务中切换文件")
        if name == "extract_observations":
            from valuationagent.application.observation_extraction import ExtractObservations

            extraction = ExtractObservations.model_validate(params)
            if any(row.role != self.task.role for row in extraction.rows):
                raise ValueError(f"FILE_TASK_ROLE: 本次阅读明确为{self.task.role}，每行role须一致。不要改写主体来迁就角色；若阅读目标改变，先结束当前任务。")
            if any((row.basis or extraction.basis).entity_ticker != self.task.entity_ticker for row in extraction.rows):
                raise ValueError("FILE_TASK_ENTITY: 提取主体与本次读取声明不一致；保留原文真实公司，核对任务，不自动改写公司代码")
        identities = params.get("fact_ids", []) or [review.get("fact_id") for review in params.get("reviews", [])]
        allowed = {fact.fact_id for fact in self.runtime.session.facts if fact.block_id.startswith(self.task.file_id + ":")}
        if any(identity not in allowed for identity in identities):
            raise ValueError("FILE_TASK_SCOPE: 复核对象必须来自当前文件")
        self.steps += 1
        if self.steps > 16 and name != "end_file_task" and not (name == "review_observations" and self.steps == 17):
            raise ValueError("FILE_TASK_BUDGET: 单文件阅读窗口已用尽，请end_file_task说明剩余问题，不继续重复读取")

    def adapt(self, messages, tools):
        if self.task is None:
            return messages, tools
        session = self.runtime.session
        state = json.loads(messages[1]["content"]) if len(messages) > 1 else {}
        context = state.get("context", {})
        task_state = context.get("task_state", {})
        facts = [{**fact.model_dump(mode="json", include={"fact_id", "metric", "standard_metric", "period", "scope", "role", "peer_ticker",
                                                       "unit", "normalized_value", "status", "warnings", "block_id"}),
                  "next_action": observation_next_action(fact)}
                 for fact in session.facts if fact.block_id.startswith(self.task.file_id + ":")
                 and fact.status != "rejected" and fact.role == self.task.role][-12:]
        focus = {"file_task": self.task.model_dump(), "remaining_local_steps": max(0, 16 - self.steps),
                 "draft": session.draft.model_dump(mode="json"), "information_cutoff": str(session.information_cutoff_date),
                 "source_permission": session.data_source_preference, "facts": facts,
                 "acquisition_targets": state.get("research_plan", {}).get("acquisition_targets", []),
                 "user_constraints": task_state.get("memory", []),
                 "user_turns": [turn for turn in context.get("recent_turns", []) if turn.get("role") == "user"][-2:]}
        groups = []
        for message in messages[2:]:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                calls = message["tool_calls"]
                names = {call["function"]["name"] for call in calls}
                if "begin_file_task" in names:
                    groups = []
                groups.append([message])
            elif groups:
                groups[-1].append(message)
        selected = [group for group in groups if all(call["function"]["name"] in READING_TOOLS | {"begin_file_task"}
                                                    for call in group[0].get("tool_calls", []))][-8:]
        while len(selected) > 1 and len(canonical(selected)) > 42000:
            selected.pop(0)
        focused = [{"role": "system", "content": AGENT_PROMPT + "\n当前为单文件阅读阶段。只处理当前文件，不计算。先理解原文及真实依据，再提交与复核；完成后end_file_task。若本文件属于可比企业，basis保留其真实名称/代码，rows[].role必须填comparable；historical仅属于draft中的估值目标，不能因主体校验失败而将可比公司改成目标公司。更正旧解释用replaces；确需撤回当前文件的错误候选可用reject_candidates并说明理由，不重复留下已知错误来阻断后续。不要修饰数据来满足模型。"},
                   {"role": "user", "content": canonical(focus)}, *[message for group in selected for message in group]]
        permitted = {"end_file_task"} if self.steps >= 16 else EXPLORATION_TOOLS if self.task.role == "reference" else READING_TOOLS
        selected_tools = copy.deepcopy([tool for tool in tools if tool["function"]["name"] in permitted])
        for tool in selected_tools:
            if tool["function"]["name"] == "extract_observations":
                schema = tool["function"]["parameters"]
                schema["properties"]["file_id"]["const"] = self.task.file_id
                definitions = schema["$defs"]
                definitions["ReadingBasis"]["properties"]["entity_ticker"]["const"] = self.task.entity_ticker
                row = definitions["Observation"]
                row["properties"]["role"]["const"] = self.task.role
                row["properties"]["role"].pop("default", None)
                row["required"] = list(dict.fromkeys([*row["required"], "role"]))
        return focused, selected_tools
