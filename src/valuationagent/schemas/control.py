from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from valuationagent.schemas.models import ApiModel


TurnAction = Literal["discuss", "read", "ingest", "research", "value", "sensitivity", "report", "update", "pause"]
Permission = Literal["network", "web", "structured_data", "calculation", "files", "artifacts"]


class PermissionChange(ApiModel):
    permission: Permission
    allowed: bool = Field(description="最新用户授权为true，最新用户禁止为false；用户可改变自己的旧限制，但不可越过系统权限上限。")
    scope: Literal["turn", "workspace"] = "workspace"
    user_quote: str = Field(min_length=1, max_length=1000)


class TurnDecision(ApiModel):
    summary: str = Field(min_length=1, max_length=500)
    actions: list[TurnAction] = Field(min_length=1, max_length=8, description="research=获取任何外部数据，包括已连接的结构化API，不只网页搜索。read/ingest只能读本地已有资料，不能请求API。value=创建或修改基准；sensitivity=已有基准试算，不改变基准；只解释用discuss。")
    permission_changes: list[PermissionChange] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def unique_permissions(self):
        if len({change.permission for change in self.permission_changes}) != len(self.permission_changes):
            raise ValueError("同一权限每轮只能修改一次。")
        if "pause" in self.actions and len(set(self.actions)) != 1:
            raise ValueError("暂停不能同时授权其他操作。")
        self.actions = list(dict.fromkeys(self.actions))
        return self


class SavedPermission(ApiModel):
    allowed: bool
    message_id: str
    user_quote: str


class TurnControl(ApiModel):
    message_id: str
    decision: TurnDecision
    permissions: dict[Permission, bool]
    effects: list[str]
    decision_context: dict[str, Any] = Field(default_factory=dict)
