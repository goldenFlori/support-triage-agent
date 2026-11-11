"""Typed data contracts for the agent.

Everything that crosses a boundary (user input, tool output, final result) is a
Pydantic model, so malformed data fails loudly instead of silently flowing on.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


class Intent(str, Enum):
    REFUND_REQUEST = "refund_request"
    DAMAGED_ITEM = "damaged_item"
    LATE_DELIVERY = "late_delivery"
    ORDER_STATUS = "order_status"
    CANCELLATION = "cancellation"
    GENERAL_QUESTION = "general_question"
    OUT_OF_SCOPE = "out_of_scope"


class Decision(str, Enum):
    AUTO_RESOLVE = "auto_resolve"          # agent can answer / act on its own
    APPROVE_REFUND = "approve_refund"      # refund approved within policy limits
    ESCALATE_TO_HUMAN = "escalate_to_human"
    NEED_MORE_INFO = "need_more_info"      # e.g. no order number given
    REFUSE = "refuse"                      # unsafe, out of scope, or injection attempt


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------- input


class Ticket(BaseModel):
    """An incoming customer message."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(min_length=1, max_length=64)
    customer_email: str = Field(max_length=254)
    message: str = Field(min_length=1, max_length=4000)
    received_at: date = Field(
        default_factory=date.today,
        description="Reference date for policy windows (makes runs reproducible).",
    )

    @field_validator("customer_email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        if not re.fullmatch(EMAIL_PATTERN, v):
            raise ValueError("must be a valid email address, e.g. anna@example.com")
        return v.lower()

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


# ---------------------------------------------------------------- tool data


class Order(BaseModel):
    order_id: str
    customer_email: str
    item: str
    amount_eur: float = Field(ge=0)
    status: str
    order_date: date
    delivered_date: Optional[date] = None
    expected_delivery: Optional[date] = None


class PolicySnippet(BaseModel):
    source: str
    text: str
    score: float


class EligibilityResult(BaseModel):
    order_id: str
    eligible: bool
    requires_human: bool
    reasons: list[str]
    max_refund_eur: float


# ---------------------------------------------------------------- output


class TriageSubmission(BaseModel):
    """What the LLM must submit through the `submit_triage` tool."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    order_id: Optional[str] = None
    decision: Decision
    priority: Priority
    refund_amount_eur: Optional[float] = Field(default=None, ge=0)
    policy_references: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1, max_length=1500)
    customer_reply: str = Field(min_length=1, max_length=2000)


class ToolCallRecord(BaseModel):
    step: int
    tool: str
    input: dict[str, Any]
    ok: bool
    output_preview: str
    duration_ms: float


class TriageResult(BaseModel):
    """Final structured output of one agent run."""

    ticket_id: str
    intent: Intent
    order_id: Optional[str]
    decision: Decision
    priority: Priority
    refund_amount_eur: Optional[float]
    policy_references: list[str]
    customer_reply: str
    reasoning: str
    guardrail_flags: list[str] = Field(default_factory=list)
    overrides: list[str] = Field(
        default_factory=list,
        description="Deterministic rule changes applied on top of the LLM decision.",
    )
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    backend: str
    steps_used: int
    summary: str = ""
