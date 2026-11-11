"""LLM backends behind one small interface.

- `ClaudeBackend` calls the Anthropic Messages API with function calling.
- `MockBackend` is a deterministic, rule-based stand-in that speaks the same
  protocol (it emits tool calls and a final `submit_triage`). It lets anyone
  run the full agent loop and the test-suite offline, for free, and reproducibly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from triage_agent.guardrails import extract_order_ids


class LLMError(Exception):
    """The model could not be reached after the SDK's own retries."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMTurn:
    stop_reason: str                      # end_turn | tool_use | max_tokens | refusal
    tool_calls: list[ToolCall]
    text: str = ""
    assistant_content: Any = None         # exact content to append to the conversation
    usage: dict[str, Any] = field(default_factory=dict)


class LLMBackend(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMTurn: ...


# ---------------------------------------------------------------- Claude


class ClaudeBackend:
    name = "claude"

    def __init__(self, model: str, effort: str = "medium", max_retries: int = 3, use_fallbacks: bool = True) -> None:
        import anthropic  # imported lazily so mock mode needs no SDK setup

        self._anthropic = anthropic
        # The SDK retries 408/409/429/5xx and connection errors with backoff.
        self._client = anthropic.Anthropic(max_retries=max_retries, timeout=120.0)
        self.model = model
        self.effort = effort
        self.use_fallbacks = use_fallbacks

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMTurn:
        a = self._anthropic
        extra: dict[str, Any] = {}
        if self.use_fallbacks:
            # On a safety decline, the API re-runs the request on a suitable fallback model.
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=messages,
                tools=tools,
                output_config={"effort": self.effort},
                **extra,
            )
        except a.AuthenticationError as exc:
            raise LLMError(f"Authentication failed - check ANTHROPIC_API_KEY ({exc.status_code}).") from exc
        except a.BadRequestError as exc:
            raise LLMError(f"Request rejected by the API: {exc.message}") from exc
        except a.RateLimitError as exc:
            raise LLMError("Rate limited after retries.") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code} after retries.") from exc
        except a.APIConnectionError as exc:
            raise LLMError("Could not reach the API after retries.") from exc

        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        text = "".join(b.text for b in response.content if b.type == "text")
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        # Append response.content unchanged (it can include thinking blocks the API expects back).
        return LLMTurn(response.stop_reason or "end_turn", calls, text, response.content, usage)


# ---------------------------------------------------------------- Mock

_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("damaged_item", re.compile(r"\b(damaged|broken|defective|cracked|smashed|doesn'?t work|not working|faulty)\b", re.I)),
    ("cancellation", re.compile(r"\bcancel", re.I)),
    ("late_delivery", re.compile(r"\b(late|not (yet )?arrived|hasn'?t arrived|never arrived|still waiting|not received|where is my)\b", re.I)),
    ("refund_request", re.compile(r"\b(refund|money back|return (it|this|the))\b", re.I)),
    ("order_status", re.compile(r"\b(status|track|tracking|when will)\b", re.I)),
]

_POLICY_QUERY = {
    "damaged_item": "damaged defective item refund",
    "late_delivery": "late delivery expected refund re-shipment",
    "refund_request": "refund change of mind returned unused",
    "cancellation": "processing cancelled free",
    "order_status": "track orders delivery",
    "general_question": "",
}

_ELIGIBILITY_REASON = {
    "damaged_item": "damaged",
    "late_delivery": "late_delivery",
    "refund_request": "change_of_mind",
    "cancellation": "not_received",
}

SIGN = "\n\nBest regards,\nCustomer Support"


class MockBackend:
    """Deterministic planner that follows the same tool protocol as the real model."""

    name = "mock"

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMTurn:
        message = _ticket_text(messages)
        intent = next((name for name, rx in _INTENT_RULES if rx.search(message)), "general_question")
        order_ids = extract_order_ids(message)
        order_id = order_ids[0] if order_ids else None
        done = _tool_history(messages)
        step = len(done)

        def call(name: str, args: dict[str, Any]) -> LLMTurn:
            tc = ToolCall(f"mock_{step}_{name}", name, args)
            content = [{"type": "tool_use", "id": tc.id, "name": name, "input": args}]
            return LLMTurn("tool_use", [tc], "", content)

        # 1) general questions: policy search, then answer
        if intent == "general_question" or (intent == "order_status" and not order_id):
            if "search_policy" not in done:
                return call("search_policy", {"query": _POLICY_QUERY["general_question"] or message[:200]})
            return self._submit(step, intent, None, "auto_resolve", "low", None, ["general"],
                                "General question answered from policy.", _general_reply(done["search_policy"]))

        # 2) everything else needs an order number
        if not order_id:
            return self._submit(step, intent, None, "need_more_info", "low", None, [],
                                "No order number in the message.",
                                "Thanks for reaching out! Could you send us your order number (for example #1001) so we can look into this?" + SIGN)

        if "lookup_order" not in done:
            return call("lookup_order", {"order_id": order_id})
        order, order_err = done["lookup_order"]
        if order_err:
            return self._submit(step, intent, None, "need_more_info", "low", None, ["privacy"],
                                f"Order {order_id} not found for this email.",
                                f"We could not find order #{order_id} for your email address. Could you double-check the order number?" + SIGN)

        if "search_policy" not in done:
            return call("search_policy", {"query": _POLICY_QUERY.get(intent) or intent})

        if intent == "order_status":
            status = order.get("status")
            return self._submit(step, intent, order_id, "auto_resolve", "low", None, ["general"],
                                f"Order status is {status}.",
                                f"Your order #{order_id} ({order.get('item')}) is currently '{status}'. You can follow it on the 'My orders' page." + SIGN)

        if "check_refund_eligibility" not in done:
            return call("check_refund_eligibility", {"order_id": order_id, "reason": _ELIGIBILITY_REASON[intent]})
        elig, elig_err = done["check_refund_eligibility"]
        refs = sorted({r["source"] for r in done["search_policy"][0].get("results", [])})
        if elig_err:
            return self._submit(step, intent, order_id, "escalate_to_human", "medium", None, refs,
                                "Eligibility check failed.", _escalate_reply(order_id))
        why = " ".join(elig["reasons"])
        if elig["eligible"] and not elig["requires_human"]:
            amount = elig["max_refund_eur"]
            if intent == "cancellation":
                return self._submit(step, intent, order_id, "auto_resolve", "low", None, refs, why,
                                    f"Your order #{order_id} has not shipped yet, so we have cancelled it free of charge. Any payment will be returned to your original payment method." + SIGN)
            return self._submit(step, intent, order_id, "approve_refund", "medium", amount, refs, why,
                                f"We're sorry about your order #{order_id} ({order.get('item')}). Your refund of {amount:.2f} EUR has been approved and will reach your original payment method within 5 business days." + SIGN)
        if elig["eligible"] and elig["requires_human"]:
            return self._submit(step, intent, order_id, "escalate_to_human", "high", None, refs, why, _escalate_reply(order_id))
        return self._submit(step, intent, order_id, "auto_resolve", "low", None, refs, why,
                            f"Thanks for contacting us about order #{order_id}. Unfortunately it does not qualify for a refund right now: {why} If anything changes, just reply to this message." + SIGN)

    @staticmethod
    def _submit(step: int, intent: str, order_id: str | None, decision: str, priority: str,
                amount: float | None, refs: list[str], reasoning: str, reply: str) -> LLMTurn:
        args = {"intent": intent, "order_id": order_id, "decision": decision, "priority": priority,
                "refund_amount_eur": amount, "policy_references": refs, "reasoning": reasoning,
                "customer_reply": reply}
        tc = ToolCall(f"mock_{step}_submit", "submit_triage", args)
        return LLMTurn("tool_use", [tc], "", [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": args}])


def _ticket_text(messages: list[dict[str, Any]]) -> str:
    first = messages[0]["content"]
    text = first if isinstance(first, str) else json.dumps(first)
    m = re.search(r"<customer_message>\n(.*)\n</customer_message>", text, re.S)
    return m.group(1) if m else text


def _tool_history(messages: list[dict[str, Any]]) -> dict[str, tuple[Any, bool]]:
    """Map tool name -> (parsed result, is_error) for tools already called."""
    names: dict[str, str] = {}
    out: dict[str, tuple[Any, bool]] = {}
    for msg in messages:
        if not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                names[block["id"]] = block["name"]
            elif block.get("type") == "tool_result":
                name = names.get(block["tool_use_id"], "?")
                try:
                    parsed = json.loads(block["content"])
                except (TypeError, ValueError):
                    parsed = block["content"]
                out[name] = (parsed, bool(block.get("is_error")))
    return out


def _escalate_reply(order_id: str) -> str:
    return (f"Thanks for your message about order #{order_id}. A member of our support team will review your request "
            "and get back to you within 24 hours." + SIGN)


def _general_reply(result: tuple[Any, bool]) -> str:
    hits = result[0].get("results", []) if isinstance(result[0], dict) else []
    if not hits:
        return "Thanks for your question! A member of our team will get back to you within 24 hours." + SIGN
    return f"Thanks for your question! {hits[0]['text']}" + SIGN
