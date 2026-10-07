import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import httpx

from valuationagent.application.turn_control import CONTROL_PROMPT
from valuationagent.core.tools import canonical
from valuationagent.llm.qwen_tools import normalize_qwen_call
from valuationagent.schemas.control import TurnDecision


def repetition(text):
    if len(text) < 64:
        return 0
    windows = Counter(text[index:index + 64] for index in range(len(text) - 63))
    return round(sum(count - 1 for count in windows.values()) / (len(text) - 63), 4)


def main():
    parser = argparse.ArgumentParser(description="Explicit sampling comparison; only stores counts and parsed tool decisions, never private reasoning.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    prompt = Path("tests/fixtures/user_manufacturing_scenario.txt").read_text(encoding="utf-8")
    tool = {"type": "function", "function": {"name": "set_turn_plan", "description": "解释本轮请求并确定执行边界，不执行任务。",
        "parameters": TurnDecision.model_json_schema()}}
    messages = [{"role": "system", "content": CONTROL_PROMPT}, {"role": "user", "content": canonical({
        "current_request": {"content": prompt}, "platform_policy": {"network_allowed": False},
        "prior_user_permissions": {}, "existing_result_available": False})}]
    receipts = []
    for name, temperature, penalty in [("coding", .6, 0), ("general", 1.0, 1.5)]:
        started = time.monotonic()
        payload = {"model": "qwen36-teacher", "messages": messages, "tools": [tool], "tool_choice": "auto",
            "max_tokens": args.max_tokens, "temperature": temperature, "top_p": .95, "top_k": 20,
            "presence_penalty": penalty, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": True}}
        with httpx.Client(timeout=180, trust_env=False) as client:
            response = client.post("http://127.0.0.1:18000/v1/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        decision = None
        if choice["finish_reason"] != "length":
            try:
                parsed = message if message.get("tool_calls") else normalize_qwen_call(message, [tool])
                decision = TurnDecision.model_validate_json(parsed["tool_calls"][0]["function"]["arguments"]).model_dump()
            except (ValueError, KeyError, IndexError):
                pass
        receipt = {"profile": name, "temperature": temperature, "presence_penalty": penalty,
            "input_sha256": hashlib.sha256(canonical(messages).encode()).hexdigest(),
            "seconds": round(time.monotonic() - started, 2), "finish_reason": choice["finish_reason"],
            "usage": body.get("usage"), "content_chars": len(message.get("content") or ""),
            "reasoning_chars": len(message.get("reasoning_content") or ""),
            "reasoning_repeated_64char_fraction": repetition(message.get("reasoning_content") or ""),
            "decision": decision}
        receipts.append(receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipts, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
