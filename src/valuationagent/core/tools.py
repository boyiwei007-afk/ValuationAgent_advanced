"""Typed allowlist for model tools; arbitrary code and tool names are never executed."""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Callable
from pydantic import BaseModel
from valuationagent.schemas.models import ApiModel


class NoArguments(ApiModel):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    arguments: type[BaseModel]
    handler: Callable[[Any], Any]
    effects: tuple[str, ...] | None = None

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]):
        self.specs = {s.name: s for s in specs}
        if len(self.specs) != len(specs):
            raise ValueError("duplicate tool registration")

    def schemas(self) -> list[dict]:
        return [spec.schema() for spec in self.specs.values()]

    def invoke(self, name: str, arguments: str) -> Any:
        if name not in self.specs:
            raise ValueError("工具不在当前任务的白名单中")
        spec = self.specs[name]
        parsed = spec.arguments.model_validate_json(arguments)
        return spec.handler(parsed)


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def canonical(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
