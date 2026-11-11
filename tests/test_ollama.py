"""Ollama backend tests against a fake local Ollama server (real HTTP, no model needed)."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from triage_agent.agent import TriageAgent
from triage_agent.config import Settings
from triage_agent.llm import LLMError
from triage_agent.ollama import OllamaBackend, recover_tool_calls, to_ollama_messages
from triage_agent.schemas import Decision

from conftest import ticket


class FakeOllama:
    """Tiny HTTP server that mimics /api/tags and /api/chat with scripted replies."""

    def __init__(self, replies: list[dict[str, Any]], models: tuple[str, ...] = ("qwen2.5:7b",), chat_status: int = 200):
        self.replies, self.requests = list(replies), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence test output
                pass

            def _send(self, code: int, body: dict[str, Any]) -> None:
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._send(200, {"models": [{"name": m} for m in models]})

            def do_POST(self):
                outer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if chat_status != 200:
                    return self._send(chat_status, {"error": "model 'qwen2.5:7b' not found"})
                msg = outer.replies.pop(0) if outer.replies else {"role": "assistant", "content": "done"}
                self._send(200, {"message": msg, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 5})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


def call(name: str, args: Any) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


SUBMIT = {"intent": "damaged_item", "order_id": "1001", "decision": "approve_refund", "priority": "medium",
          "refund_amount_eur": 45.0, "policy_references": ["refunds"], "reasoning": "Within 30 days.",
          "customer_reply": "Your refund of 45.00 EUR has been approved. Customer Support"}


def agent_for(fake: FakeOllama) -> TriageAgent:
    settings = Settings(backend="ollama", ollama_url=fake.url)
    return TriageAgent(settings, backend=OllamaBackend(base_url=fake.url, timeout=5, max_retries=0))


def test_full_agent_run_through_ollama():
    fake = FakeOllama([
        call("lookup_order", {"order_id": "1001"}),
        call("check_refund_eligibility", json.dumps({"order_id": "1001", "reason": "damaged"})),  # args as a string
        call("submit_triage", SUBMIT),
    ])
    try:
        result = agent_for(fake).run(ticket("Order #1001 arrived broken, refund please"))
    finally:
        fake.close()
    assert result.backend == "ollama"
    assert result.decision == Decision.APPROVE_REFUND and result.refund_amount_eur == 45.0
    assert [c.tool for c in result.tool_calls] == ["lookup_order", "check_refund_eligibility"]
    # what was sent to Ollama
    first, last = fake.requests[0], fake.requests[-1]
    assert first["messages"][0]["role"] == "system" and first["stream"] is False
    assert {t["function"]["name"] for t in first["tools"]} >= {"lookup_order", "search_policy", "submit_triage"}
    tool_msgs = [m for m in last["messages"] if m["role"] == "tool"]
    assert [m["tool_name"] for m in tool_msgs] == ["lookup_order", "check_refund_eligibility"]


def test_tool_call_written_as_text_is_recovered():
    fake = FakeOllama([
        {"role": "assistant", "content": 'Sure! {"name": "lookup_order", "arguments": {"order_id": "1001"}}'},
        {"role": "assistant", "content": "Here is my decision:\n```json\n" + json.dumps({**SUBMIT, "decision": "auto_resolve", "refund_amount_eur": None, "customer_reply": "Your order was delivered."}) + "\n```"},
    ])
    try:
        result = agent_for(fake).run(ticket("Where is order #1001?"))
    finally:
        fake.close()
    assert result.decision == Decision.AUTO_RESOLVE
    assert result.tool_calls[0].tool == "lookup_order" and result.tool_calls[0].ok


def test_local_model_cannot_bypass_policy_rules():
    fake = FakeOllama([call("submit_triage", {**SUBMIT, "order_id": "1003", "refund_amount_eur": 649.0})])
    try:
        result = agent_for(fake).run(ticket("TV #1003 broken", email="carla@example.com"))
    finally:
        fake.close()
    assert result.decision == Decision.ESCALATE_TO_HUMAN      # never verified + no eligibility check
    assert result.overrides


def test_ollama_not_running_falls_back_to_human():
    backend = OllamaBackend(base_url="http://127.0.0.1:9", timeout=1, max_retries=0)
    assert backend.is_available() is False
    result = TriageAgent(Settings(backend="ollama"), backend=backend).run(ticket("Order #1001 broken"))
    assert result.decision == Decision.ESCALATE_TO_HUMAN
    assert "llm_unavailable" in result.guardrail_flags


def test_missing_model_gives_clear_error():
    fake = FakeOllama([], models=(), chat_status=404)
    try:
        backend = OllamaBackend(base_url=fake.url, max_retries=0)
        assert backend.is_available() is False
        with pytest.raises(LLMError, match="ollama pull qwen2.5:7b"):
            backend.complete("sys", [{"role": "user", "content": "hi"}], [])
    finally:
        fake.close()


def test_message_conversion():
    msgs = [
        {"role": "user", "content": "ticket"},
        {"role": "assistant", "content": [{"type": "text", "text": "checking"},
                                          {"type": "tool_use", "id": "a1", "name": "lookup_order", "input": {"order_id": "1"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a1", "content": "Error: nope", "is_error": True}]},
        {"role": "user", "content": "Please submit."},
    ]
    out = to_ollama_messages(msgs)
    assert out[1] == {"role": "assistant", "content": "checking",
                      "tool_calls": [{"function": {"name": "lookup_order", "arguments": {"order_id": "1"}}}]}
    assert out[2] == {"role": "tool", "tool_name": "lookup_order", "content": "Error: nope"}
    assert out[3] == {"role": "user", "content": "Please submit."}


@pytest.mark.parametrize("text, expected", [
    ('{"name": "search_policy", "arguments": {"query": "refund"}}', "search_policy"),
    ('{"tool": "search_policy", "parameters": {"query": "x"}}', "search_policy"),
    ('{"decision": "refuse", "customer_reply": "No."}', "submit_triage"),
    ('{"name": "delete_everything", "arguments": {}}', None),          # unknown tool is ignored
    ("Just a normal sentence.", None),
    ("{not json}", None),
])
def test_recover_tool_calls(text, expected):
    calls = recover_tool_calls(text, {"search_policy", "submit_triage"}, prefix="t")
    assert (calls[0].name if calls else None) == expected


@pytest.mark.live
@pytest.mark.skipif(not OllamaBackend().is_available(), reason="needs a local Ollama with qwen2.5:7b")
def test_live_ollama():
    agent = TriageAgent(Settings(backend="ollama"))
    result = agent.run(ticket("My order #1001 arrived broken, I want a refund", email="anna@example.com"))
    assert result.decision in {Decision.APPROVE_REFUND, Decision.ESCALATE_TO_HUMAN}, result.summary
