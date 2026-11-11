from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from triage_agent.config import Settings
from triage_agent.llm import LLMError, LLMTurn, ToolCall
from triage_agent.tools import ToolContext, Toolbox


@pytest.fixture
def settings() -> Settings:
    return Settings(backend="mock")


@pytest.fixture
def toolbox(settings: Settings) -> Toolbox:
    return Toolbox(settings)


@pytest.fixture
def ctx_anna(settings: Settings) -> ToolContext:
    return ToolContext(customer_email="anna@example.com", today=date(2026, 11, 11), settings=settings)


def ticket(message: str, email: str = "anna@example.com", ticket_id: str = "T-1") -> dict[str, Any]:
    return {"ticket_id": ticket_id, "customer_email": email, "message": message, "received_at": "2026-11-11"}


def tool_turn(name: str, args: dict[str, Any], idx: int = 0) -> LLMTurn:
    call = ToolCall(f"s{idx}_{name}", name, args)
    return LLMTurn("tool_use", [call], "", [{"type": "tool_use", "id": call.id, "name": name, "input": args}])


def submit(**overrides: Any) -> dict[str, Any]:
    base = {"intent": "damaged_item", "order_id": "1001", "decision": "approve_refund", "priority": "medium",
            "refund_amount_eur": 45.0, "policy_references": ["refunds"], "reasoning": "ok",
            "customer_reply": "Your refund of 45.00 EUR has been approved. Customer Support"}
    base.update(overrides)
    return base


class ScriptedBackend:
    """Plays back a fixed list of turns - lets tests simulate any model behaviour."""

    name = "scripted"

    def __init__(self, turns: list[LLMTurn | Exception]) -> None:
        self.turns = list(turns)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMTurn:
        self.calls.append(list(messages))
        if not self.turns:
            return LLMTurn("end_turn", [], "I am done.", [{"type": "text", "text": "I am done."}])
        nxt = self.turns.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


__all__ = ["ScriptedBackend", "tool_turn", "submit", "ticket", "LLMError"]
