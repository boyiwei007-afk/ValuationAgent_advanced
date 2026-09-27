from __future__ import annotations

import os
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from valuationagent.schemas.models import ModelConnectionInput, ModelSessionPublic


class LlmError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Minimal chat-completions adapter with no process-global secret mutation."""

    def __init__(self, config: ModelConnectionInput):
        self.config = config
        self.revoked = threading.Event()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        max_tokens: int = 900,
    ) -> dict:
        if self.revoked.is_set():
            raise LlmError("MODEL_SESSION_REVOKED: 模型会话已删除，请重新配置。")
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if tools:
            payload.update(
                tools=tools, tool_choice=tool_choice, parallel_tool_calls=False
            )
        # DeepSeek's default thinking mode and forced tool selection differ from
        # generic Chat Completions. Scope vendor-specific fields to its host.
        from urllib.parse import urlsplit
        if urlsplit(self.config.base_url).hostname == "api.deepseek.com":
            thinking = "disabled" if self.config.thinking == "auto" else self.config.thinking
            payload["thinking"] = {"type": thinking}
            payload.pop("temperature", None)
            if thinking == "enabled":
                payload["max_tokens"] = max(max_tokens, 4096)
                if tools:
                    payload["tool_choice"] = "auto"
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                for attempt in range(3):
                    try:
                        response = client.post(endpoint, headers=headers, json=payload)
                    except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError):
                        # One bounded transport retry; a timed-out generation
                        # may already have consumed provider credits.
                        if attempt >= 1:
                            raise
                        if self.revoked.wait(.4):
                            raise LlmError("MODEL_SESSION_REVOKED: 模型会话已删除。")
                        continue
                    if response.status_code not in {429, 502, 503, 504} or attempt == 2:
                        break
                    if self.revoked.wait(.4 * (2 ** attempt)):
                        raise LlmError("MODEL_SESSION_REVOKED: 模型会话已删除。")
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 402:
                raise LlmError(
                    "LLM_HTTP_402: 模型账户额度不足或计费不可用；当前进度已保留。"
                    "请在供应商后台检查余额/计费，处理后重新连接，或改用获准的模型。不要直接反复重试。"
                ) from None
            category = (
                "认证失败"
                if status in (401, 403)
                else "请求限流"
                if status == 429
                else "供应商请求失败"
            )
            hint = ""
            if status == 400:
                # Never copy vendor bodies into persisted conversations: some
                # gateways echo credentials or uploaded content in error strings.
                try:
                    detail = str(exc.response.json()).lower()
                    names = [name for name in ("tool_choice", "thinking", "temperature", "reasoning_content", "max_tokens", "model", "messages", "tools") if name in detail]
                    if names:
                        hint = " 涉及参数：" + ", ".join(names) + "。"
                except ValueError:
                    pass
            raise LlmError(
                f"LLM_HTTP_{status}: {category}，请检查模型配置或稍后恢复。{hint}"
            ) from None
        except httpx.TimeoutException:
            raise LlmError(
                f"LLM_TIMEOUT: 模型服务在 {self.config.timeout_seconds:g} 秒内未返回；当前进度已保留，可以重试。"
            ) from None
        except httpx.ConnectError:
            raise LlmError(
                "LLM_CONNECTION_FAILED: 无法连接模型服务，请检查接口地址、网络、代理或防火墙。"
            ) from None
        except httpx.RequestError:
            raise LlmError(
                "LLM_NETWORK_FAILED: 模型请求在传输过程中失败，请检查网络后重试。"
            ) from None
        except ValueError:
            raise LlmError(
                "LLM_RESPONSE_INVALID_JSON: 模型服务返回的内容不是有效 JSON，请稍后重试或更换接口。"
            ) from None
        try:
            message = body["choices"][0]["message"]
            if not isinstance(message, dict):
                raise TypeError()
        except (KeyError, IndexError, TypeError):
            raise LlmError("LLM_RESPONSE_INVALID: 响应缺少有效 message") from None
        # Some OpenAI-compatible gateways serialize function arguments as an
        # object while others return the JSON text required by the protocol.
        # Normalize both forms before the finite tool loop validates them.
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_calls = []
            for item in tool_calls:
                if not isinstance(item, dict):
                    normalized_calls.append(item)
                    continue
                function = item.get("function")
                if isinstance(function, dict) and isinstance(function.get("arguments"), (dict, list)):
                    item = dict(item)
                    item["function"] = dict(function)
                    item["function"]["arguments"] = json.dumps(
                        function["arguments"], ensure_ascii=False, separators=(",", ":")
                    )
                normalized_calls.append(item)
            message = dict(message)
            message["tool_calls"] = normalized_calls
        if self.revoked.is_set():
            raise LlmError("MODEL_SESSION_REVOKED: 模型会话已删除。")
        return message

    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int = 900) -> str:
        content = self.chat(messages, max_tokens=max_tokens).get("content")
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str) or not content.strip():
            raise LlmError("LLM returned empty content")
        return content.strip()

    def test_connection(self) -> str:
        response = self.chat(
            [{"role": "user", "content": "Call connection_check with no arguments."}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "connection_check",
                        "description": "Connection test",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            tool_choice="required",
            max_tokens=150,
        )
        calls = response.get("tool_calls") or []
        if (
            len(calls) != 1
            or calls[0].get("function", {}).get("name") != "connection_check"
        ):
            raise LlmError("TOOL_CALLS_UNSUPPORTED: 模型未返回有效工具调用。")
        import json

        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except (ValueError, KeyError, TypeError):
            raise LlmError("TOOL_ARGUMENTS_INVALID: 工具参数无效。") from None
        if args != {}:
            raise LlmError("TOOL_ARGUMENTS_INVALID: 工具参数无效。")
        return "OK · 工具调用可用"

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleClient":
        values = {
            "provider": os.getenv("VALUATION_LLM_PROVIDER", "openai_compatible"),
            "base_url": os.getenv(
                "VALUATION_LLM_BASE_URL", "https://api.openai.com/v1"
            ),
            "model": os.getenv("VALUATION_LLM_MODEL", ""),
            "api_key": os.getenv("VALUATION_LLM_API_KEY", ""),
            "thinking": os.getenv("VALUATION_LLM_THINKING", "auto"),
        }
        if not values["model"] or not values["api_key"]:
            raise LlmError(
                "VALUATION_LLM_MODEL and VALUATION_LLM_API_KEY are required for live CLI mode"
            )
        return cls(ModelConnectionInput(**values))


class ModelSessionRegistry:
    """In-memory session-scoped model configs. API keys never enter SQLite or API responses."""

    def __init__(self):
        self._sessions: dict[str, tuple[ModelConnectionInput, datetime]] = {}
        self._clients: dict[str, list[OpenAICompatibleClient]] = {}
        self._lock = threading.RLock()

    def create(self, config: ModelConnectionInput) -> ModelSessionPublic:
        session_id = f"llm_{uuid.uuid4().hex}"
        created_at = datetime.now(timezone.utc)
        with self._lock:
            self._sessions[session_id] = (config, created_at)
        return ModelSessionPublic(
            session_id=session_id,
            provider=config.provider,
            base_url=config.base_url,
            model=config.model,
            created_at=created_at,
        )

    def client(self, session_id: str) -> OpenAICompatibleClient:
        with self._lock:
            item = self._sessions.get(session_id)
            if item is None:
                raise KeyError(session_id)
            client = OpenAICompatibleClient(item[0])
            self._clients.setdefault(session_id, []).append(client)
        return client

    def delete(self, session_id: str) -> None:
        with self._lock:
            if self._sessions.pop(session_id, None) is None:
                raise KeyError(session_id)
            for client in self._clients.pop(session_id, []):
                client.revoked.set()
