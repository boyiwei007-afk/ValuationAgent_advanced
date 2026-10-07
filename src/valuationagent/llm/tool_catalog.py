import copy

from pydantic import Field

from valuationagent.core.tools import canonical
from valuationagent.schemas.models import ApiModel


CORE_TOOLS = {"load_tools", "finish_response", "update_task", "update_plan", "read_user_input", "record_user_inputs",
    "inspect_inputs", "inspect_requirements", "check_preparation", "calculate_valuation", "read_valuation",
    "write_workspace_report", "write_research_note", "list_files", "inspect_file", "read_file", "begin_file_task",
    "acquire_financial_inputs"}


class LoadTools(ApiModel):
    names: list[str] = Field(min_length=1, max_length=4,
        description="从可用工具目录选择本步骤需要的1至4个名称。替换上次加载的工具集，不执行它们、不改变权限；下一次调用会提供完整参数定义。")


def compact_schema(value):
    if isinstance(value, list):
        return [compact_schema(item) for item in value]
    if isinstance(value, dict):
        return {key: ({name: compact_schema(schema) for name, schema in item.items()}
                      if key in {"properties", "$defs"} else compact_schema(item))
                for key, item in value.items() if key != "title"}
    return value


class ToolCatalog:
    def __init__(self):
        self.loaded = []
        self.available = {}
        self.deferred = {}

    def load(self, args):
        missing = [name for name in args.names if name not in self.available]
        if any(name not in self.deferred for name in missing):
            raise ValueError("TOOL_UNAVAILABLE: 工具不在本轮许可目录中；加载工具不能解除用户或平台限制。")
        if missing:
            raise ValueError("TOOL_PREREQUISITE: " + "；".join(f"{name}: {self.deferred[name]}" for name in missing))
        self.loaded = list(dict.fromkeys(args.names))
        return {"loaded": self.loaded, "executed": False,
                "instruction": "下一次模型请求提供这些工具的完整参数；按schema调用，不猜参数。加载本身不代表已取数、读取、计算或生成文件。"}

    def adapt(self, messages, tools, *, deferred=None, suggested=()):
        if not any(tool["function"]["name"] == "load_tools" for tool in tools):
            return messages, tools
        self.available = {tool["function"]["name"]: tool for tool in tools}
        self.deferred = {name: reason for name, reason in (deferred or {}).items() if name not in self.available}
        selected = [compact_schema(tool) for tool in tools
                    if tool["function"]["name"] in CORE_TOOLS | set(self.loaded) | set(suggested)]
        directory = [{"name": name, "description": tool["function"].get("description", "")}
                     for name, tool in self.available.items() if name not in CORE_TOOLS]
        projected = copy.deepcopy(messages)
        guidance = ("工具按需加载，目录不是参数schema。除当前暴露的工具外，先load_tools(names=[所需工具名])，"
                    "等待返回后按完整定义调用。每次加载替换上次选择，不执行业务、不扩大权限。读取、录入、计算可独立切换，"
                    "不因暂未加载而声称不支持。避免一次加载不相关的大型工具。\n可用目录：" + canonical(directory))
        if self.deferred:
            guidance += "\n尚未满足前置条件的工具（不是权限拒绝，不能靠重复加载解决）：" + canonical(self.deferred)
        ready = sorted(set(suggested) & self.available.keys())
        if ready:
            guidance += "\n根据刚完成的读取/提取，已直接提供后续工具schema：" + canonical(ready) + "。无需load_tools；是否调用由你依据用户目标决定，不是强制执行或核验通过。"
        if projected and projected[0].get("role") == "system":
            projected[0]["content"] += "\n" + guidance
        else:
            projected.insert(0, {"role": "system", "content": guidance})
        return projected, selected
