"""Deterministic guardrails that run before and after the LLM.

Design rule: the LLM is never the only line of defence. Anything that can be
checked with plain code (input shape, injection patterns, money limits, data
ownership, what the reply promises) is checked with plain code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from triage_agent.schemas import Decision, Ticket

# Phrases that try to override the agent's instructions or extract its internals.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_instructions", re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|all|your|the)\b.{0,20}\b(instructions?|rules?|prompts?|polic(y|ies))\b", re.I | re.S)),
    ("role_hijack", re.compile(r"\b(you are now|act as|pretend to be|new instructions?|developer mode|jailbreak)\b", re.I)),
    ("system_prompt_probe", re.compile(r"\b(system prompt|your (instructions|prompt|rules) (are|say)|reveal (your|the) (prompt|instructions))\b", re.I)),
    ("fake_role_tags", re.compile(r"(</?\s*(system|assistant|tool|instructions?)\s*>|\[\s*(system|admin)\s*\]|^\s*(system|assistant)\s*:)", re.I | re.M)),
    ("forced_action", re.compile(r"\b(approve|issue|process)\b.{0,30}\brefund\b.{0,40}\b(immediately|without (checking|verification|review)|no matter what)\b", re.I | re.S)),
]

# Requests the agent must refuse regardless of how they are phrased.
_OUT_OF_SCOPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("other_customer_data", re.compile(r"\b(another|other|someone else'?s?|different)\b.{0,30}\b(customer|user|person|account)'?s?\b.{0,40}\b(orders?|address|email|data|details|card)\b", re.I | re.S)),
    ("other_customer_data", re.compile(r"\b(orders?|address|email|phone|data|details|card)\b.{0,20}\b(of|for|from)\b.{0,10}\b(another|other|a different|someone else'?s?)\b.{0,15}\b(customers?|users?|persons?|people|accounts?)\b", re.I | re.S)),
    ("other_customer_data", re.compile(r"\b(someone|somebody|anyone) else'?s?\b.{0,20}\b(orders?|address|email|phone|data|details|card)\b", re.I | re.S)),
    ("credential_change", re.compile(r"\b(change|reset|update)\b.{0,20}\b(password|payment method|card number|account email)\b", re.I | re.S)),
]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩]")
_ORDER_ID = re.compile(r"(?:order\s*(?:no\.?|number|id)?\s*#?\s*|#)(\d{3,10})\b", re.I)


@dataclass
class InputCheck:
    ok: bool
    ticket: Ticket | None = None
    errors: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    refuse_reason: str | None = None


def sanitize(text: str) -> str:
    """Strip invisible / control characters often used to hide injected text."""
    return _CONTROL_CHARS.sub("", text).strip()


def detect_injection(text: str) -> list[str]:
    return [name for name, pattern in _INJECTION_PATTERNS if pattern.search(text)]


def detect_out_of_scope(text: str) -> list[str]:
    return [name for name, pattern in _OUT_OF_SCOPE_PATTERNS if pattern.search(text)]


def extract_order_ids(text: str) -> list[str]:
    seen: list[str] = []
    for match in _ORDER_ID.finditer(text):
        if match.group(1) not in seen:
            seen.append(match.group(1))
    return seen


def check_input(raw: dict) -> InputCheck:
    """Validate and screen a raw ticket before any LLM call is made."""
    try:
        ticket = Ticket.model_validate(raw)
    except ValidationError as exc:
        errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return InputCheck(ok=False, errors=errors)

    clean = sanitize(ticket.message)
    if not clean:
        return InputCheck(ok=False, errors=["message: empty after removing control characters"])
    flags: list[str] = []
    if clean != ticket.message.strip():
        flags.append("control_characters_removed")
    ticket = ticket.model_copy(update={"message": clean})

    injection = detect_injection(clean)
    flags += [f"prompt_injection:{name}" for name in injection]
    scope = detect_out_of_scope(clean)
    flags += [f"out_of_scope:{name}" for name in scope]

    refuse_reason = None
    if injection:
        refuse_reason = "The message tries to override the assistant's instructions."
    elif scope:
        refuse_reason = "The request is outside what support chat is allowed to do."
    return InputCheck(ok=True, ticket=ticket, flags=flags, refuse_reason=refuse_reason)


# ---------------------------------------------------------------- output checks

_REFUND_PROMISE = re.compile(
    r"\b(refund\b.{0,40}\b(has been|is|was|will be) (approved|issued|processed)|we('ve| have| will)( now)? (refund|issue (a|your) refund))",
    re.I | re.S,
)


def reply_promises_refund(reply: str) -> bool:
    return bool(_REFUND_PROMISE.search(reply))


def reply_leaks_internals(reply: str) -> bool:
    lowered = reply.lower()
    return any(t in lowered for t in ("system prompt", "submit_triage", "tool_use", "my instructions"))


def decision_allows_refund_promise(decision: Decision) -> bool:
    return decision == Decision.APPROVE_REFUND
