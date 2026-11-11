"""Tools the agent can call.

Each tool has a JSON schema (sent to the LLM) and a plain-Python handler.
Handlers never trust identity coming from the LLM: the customer's email is
taken from the validated ticket (the `ToolContext`), so a manipulated model
cannot look up somebody else's order.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from triage_agent.config import Settings
from triage_agent.schemas import EligibilityResult, Order, PolicySnippet


class ToolError(Exception):
    """A tool failed in a way the model should be told about (sent back as is_error)."""


@dataclass(frozen=True)
class ToolContext:
    customer_email: str
    today: date
    settings: Settings


# ---------------------------------------------------------------- data stores


class OrderStore:
    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._orders = {o["order_id"]: Order.model_validate(o) for o in raw}

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)


_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "for", "in", "on", "my", "i", "it", "be", "can", "we", "our", "with", "by", "if", "not", "that", "this", "was", "has"}


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


class PolicyIndex:
    """Tiny TF-IDF search over policy sentences. No external services needed."""

    def __init__(self, folder: Path) -> None:
        self._docs: list[tuple[str, str, list[str]]] = []
        for path in sorted(folder.glob("*.md")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    self._docs.append((path.stem, line, _tokens(line)))
        n = len(self._docs)
        df: dict[str, int] = {}
        for _, _, toks in self._docs:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, top_k: int = 3) -> list[PolicySnippet]:
        q = set(_tokens(query))
        scored = []
        for source, text, toks in self._docs:
            score = sum(self._idf.get(t, 0.0) for t in q if t in toks)
            if score > 0:
                scored.append(PolicySnippet(source=source, text=text, score=round(score, 3)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------- tool registry


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], dict[str, Any]]

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


REFUND_REASONS = ["damaged", "late_delivery", "not_received", "change_of_mind"]


class Toolbox:
    """Holds data stores and exposes the tools to the agent."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.orders = OrderStore(settings.data_dir / "orders.json")
        self.policies = PolicyIndex(settings.data_dir / "policies")
        self.tools: dict[str, Tool] = {t.name: t for t in self._build()}

    def api_specs(self) -> list[dict[str, Any]]:
        return [t.to_api() for t in self.tools.values()]

    def run(self, name: str, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool '{name}'. Available: {', '.join(self.tools)}")
        return tool.handler(args, ctx)

    # -- handlers ---------------------------------------------------------

    def _owned_order(self, order_id: Any, ctx: ToolContext) -> Order:
        if not isinstance(order_id, str) or not re.fullmatch(r"\d{3,10}", order_id.strip()):
            raise ToolError("order_id must be a string of 3-10 digits, e.g. '1001'.")
        order = self.orders.get(order_id.strip())
        # Same message for "missing" and "belongs to someone else": no data leak.
        if order is None or order.customer_email.lower() != ctx.customer_email.lower():
            raise ToolError(f"No order {order_id} found for this customer's email address.")
        return order

    def lookup_order(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        order = self._owned_order(args.get("order_id"), ctx)
        data = order.model_dump(mode="json")
        data.pop("customer_email")
        if order.expected_delivery and not order.delivered_date:
            data["days_late"] = max(0, (ctx.today - order.expected_delivery).days)
        if order.delivered_date:
            data["days_since_delivery"] = (ctx.today - order.delivered_date).days
        return data

    def search_policy(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string.")
        hits = self.policies.search(query[:300], top_k=3)
        return {"results": [h.model_dump() for h in hits]}

    def check_refund_eligibility(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        order = self._owned_order(args.get("order_id"), ctx)
        reason = args.get("reason")
        if reason not in REFUND_REASONS:
            raise ToolError(f"reason must be one of {REFUND_REASONS}.")
        return evaluate_eligibility(order, reason, ctx.today, self.settings).model_dump()

    def _build(self) -> list[Tool]:
        return [
            Tool(
                name="lookup_order",
                description=(
                    "Look up one order of the current customer by order number. Returns item, amount, "
                    "status and delivery dates. Only works for orders placed with the ticket's email."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"order_id": {"type": "string", "description": "Digits only, e.g. '1001'."}},
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
                handler=self.lookup_order,
            ),
            Tool(
                name="search_policy",
                description="Search the company support policies (refunds, shipping, damaged items, privacy). Returns the most relevant policy lines.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Short keyword query, e.g. 'damaged item refund window'."}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=self.search_policy,
            ),
            Tool(
                name="check_refund_eligibility",
                description=(
                    "Deterministic policy check: is this order eligible for a refund for the given reason, "
                    "and does it need human approval? Always call this before approving a refund."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "reason": {"type": "string", "enum": REFUND_REASONS},
                    },
                    "required": ["order_id", "reason"],
                    "additionalProperties": False,
                },
                handler=self.check_refund_eligibility,
            ),
        ]


def evaluate_eligibility(order: Order, reason: str, today: date, settings: Settings) -> EligibilityResult:
    """Pure business rules: same input, same answer, easy to unit test."""
    reasons: list[str] = []
    eligible = True

    if order.status == "refunded":
        return EligibilityResult(order_id=order.order_id, eligible=False, requires_human=False,
                                 reasons=["Order was already refunded."], max_refund_eur=0.0)

    if reason in ("damaged", "change_of_mind"):
        if not order.delivered_date:
            eligible = False
            reasons.append("Item has not been delivered yet.")
        else:
            days = (today - order.delivered_date).days
            window = settings.refund_window_days if reason == "damaged" else 14
            if days > window:
                eligible = False
                reasons.append(f"Delivered {days} days ago; the {reason.replace('_', '-')} refund window is {window} days.")
            else:
                reasons.append(f"Delivered {days} days ago, within the {window}-day window.")
            if reason == "change_of_mind" and eligible:
                reasons.append("Refund is issued after the item is returned unused.")
    elif reason in ("late_delivery", "not_received"):
        if order.delivered_date:
            eligible = False
            reasons.append("Order is marked as delivered.")
        elif order.status == "processing":
            eligible = True
            reasons.append("Order has not shipped yet and can be cancelled free of charge.")
        else:
            late = (today - order.expected_delivery).days if order.expected_delivery else 0
            if late > settings.late_delivery_threshold_days:
                reasons.append(f"Order is {late} days late (> {settings.late_delivery_threshold_days}).")
            else:
                eligible = False
                reasons.append(f"Order is {max(late, 0)} days late; refund only after {settings.late_delivery_threshold_days} days.")

    requires_human = eligible and order.amount_eur > settings.auto_refund_limit_eur
    if requires_human:
        reasons.append(f"Amount {order.amount_eur:.2f} EUR exceeds the {settings.auto_refund_limit_eur:.0f} EUR auto-approval limit.")
    return EligibilityResult(order_id=order.order_id, eligible=eligible, requires_human=requires_human,
                             reasons=reasons, max_refund_eur=order.amount_eur if eligible else 0.0)
