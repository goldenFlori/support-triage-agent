"""Ollama backend: a free LLM that runs locally on your own machine.

Talks to Ollama's `/api/chat` endpoint with tool calling, using only the Python
standard library (no extra dependency). The agent keeps its conversation in one
internal format; this module translates it to Ollama's format on every call and
translates the answer back into `LLMTurn` / `ToolCall` objects.

Small local models are less reliable than hosted ones, so the parser also
recovers tool calls that a model writes as JSON text instead of a real tool call.
Everything the model returns still goes through the agent's validation and the
deterministic policy checks.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from triage_agent.llm import LLMError, LLMTurn, ToolCall


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434",
                 timeout: float = 180.0, max_retries: int = 2) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._calls = 0

    # ------------------------------------------------------------------ API

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMTurn:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + to_ollama_messages(messages),
            "tools": [to_ollama_tool(t) for t in tools],
            "stream": False,
            "options": {"temperature": 0},     # as deterministic as a local model gets
        }
        data = self._post("/api/chat", payload)
        message = data.get("message") or {}
        text = message.get("content") or ""
        tool_names = {t["name"] for t in tools}

        self._calls += 1
        calls = [_to_call(tc.get("function", {}), f"ollama_{self._calls}_{i}")
                 for i, tc in enumerate(message.get("tool_calls") or [])]
        if not calls:
            calls = recover_tool_calls(text, tool_names, prefix=f"ollama_{self._calls}")
            if calls:
                text = ""

        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
        usage = {"input_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")}
        stop = "tool_use" if calls else ("max_tokens" if data.get("done_reason") == "length" else "end_turn")
        return LLMTurn(stop, calls, text, content, usage)

    def is_available(self) -> bool:
        try:
            tags = self._get("/api/tags", timeout=1.5)
        except LLMError:
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def installed_models(self) -> list[str]:
        try:
            return sorted(m.get("name", "") for m in self._get("/api/tags", timeout=1.5).get("models", []))
        except LLMError:
            return []

    # ------------------------------------------------------------------ HTTP

    def _get(self, path: str, timeout: float) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(self.base_url + path, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise LLMError(f"Ollama is not reachable at {self.base_url}. Is the Ollama app running?") from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for _ in range(self.max_retries + 1):
            req = urllib.request.Request(self.base_url + path, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                if exc.code == 404 or "not found" in detail.lower():
                    raise LLMError(f"Model '{self.model}' is not installed. Run: ollama pull {self.model}") from exc
                if exc.code < 500:
                    raise LLMError(f"Ollama rejected the request ({exc.code}): {detail[:200]}") from exc
                last = exc                       # 5xx: retry
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc                       # network / timeout: retry
            except ValueError as exc:
                raise LLMError("Ollama returned invalid JSON.") from exc
        raise LLMError(f"Ollama is not reachable at {self.base_url} after retries. Is the Ollama app running? ({last})")


# ---------------------------------------------------------------------- format conversion


def to_ollama_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": tool["name"], "description": tool.get("description", ""), "parameters": tool["input_schema"]}}


def to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the agent's (Anthropic-style) conversation into Ollama chat messages."""
    out: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        if msg["role"] == "assistant":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = []
            for b in content:
                if b.get("type") == "tool_use":
                    names[b["id"]] = b["name"]
                    calls.append({"function": {"name": b["name"], "arguments": b["input"]}})
            entry: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
        else:
            for b in content:
                if b.get("type") == "tool_result":
                    out.append({"role": "tool", "tool_name": names.get(b["tool_use_id"], ""), "content": str(b["content"])})
                elif b.get("type") == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


def _to_call(fn: dict[str, Any], call_id: str) -> ToolCall:
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {"_raw": args}            # the agent will report invalid arguments back to the model
    return ToolCall(call_id, str(fn.get("name", "")), args if isinstance(args, dict) else {"_raw": args})


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def recover_tool_calls(text: str, tool_names: set[str], prefix: str) -> list[ToolCall]:
    """Some local models write the tool call as JSON text. Turn that back into a real call."""
    match = _JSON_BLOCK.search(text or "")
    if not match:
        return []
    try:
        obj = json.loads(match.group(0))
    except ValueError:
        return []
    if not isinstance(obj, dict):
        return []
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    args = obj.get("arguments") or obj.get("parameters") or obj.get("input")
    if isinstance(name, str) and name in tool_names and isinstance(args, (dict, str)):
        return [_to_call({"name": name, "arguments": args}, f"{prefix}_r")]
    if "submit_triage" in tool_names and {"decision", "customer_reply"} <= obj.keys():
        return [ToolCall(f"{prefix}_r", "submit_triage", obj)]
    return []
