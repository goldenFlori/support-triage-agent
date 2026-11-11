"""The orchestrator: guardrails -> tool-using agent loop -> policy enforcement -> result.

    raw ticket
       |
    [1] check_input        validate schema, strip hidden chars, detect injection / out-of-scope
       |  (refuse here deterministically, without calling the LLM)
    [2] agent loop         LLM plans and calls tools (lookup_order, search_policy,
       |                   check_refund_eligibility) until it calls submit_triage
       |                   - bad tool args / tool errors go back as is_error results
       |                   - invalid or missing submission -> re-prompt (bounded)
       |                   - step cap, API failure, model refusal -> safe escalation
    [3] enforce_policy     deterministic checks override the LLM if it broke a rule
       |
    [4] TriageResult       structured JSON + human summary + tool-call trace
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from triage_agent import guardrails
from triage_agent.config import Settings
from triage_agent.llm import ClaudeBackend, LLMBackend, LLMError, MockBackend
from triage_agent.prompts import PROMPT_VERSION, SUBMIT_TOOL, SYSTEM_PROMPT, render_ticket
from triage_agent.schemas import (
    Decision,
    EligibilityResult,
    Intent,
    Priority,
    Ticket,
    ToolCallRecord,
    TriageResult,
    TriageSubmission,
)
from triage_agent.tools import ToolContext, ToolError, Toolbox
from triage_agent.tracing import Trace, preview

SIGN = "\n\nBest regards,\nCustomer Support"


class InvalidTicketError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class _RunState:
    """Everything the agent learned during one run (used for deterministic checks)."""

    def __init__(self) -> None:
        self.verified_orders: dict[str, dict[str, Any]] = {}
        self.eligibility: dict[str, EligibilityResult] = {}
        self.tool_calls: list[ToolCallRecord] = []
        self.flags: list[str] = []
        self.steps = 0


def build_backend(settings: Settings) -> LLMBackend:
    if settings.backend == "claude":
        return ClaudeBackend(model=settings.model, effort=settings.effort)
    if settings.backend == "mock":
        return MockBackend()
    raise ValueError(f"Unknown backend '{settings.backend}' (use 'mock' or 'claude').")


class TriageAgent:
    def __init__(self, settings: Settings | None = None, backend: LLMBackend | None = None,
                 toolbox: Toolbox | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.backend = backend or build_backend(self.settings)
        self.toolbox = toolbox or Toolbox(self.settings)

    # ------------------------------------------------------------------ public

    def run(self, raw_ticket: dict[str, Any]) -> TriageResult:
        trace = Trace(str(raw_ticket.get("ticket_id", "?")))
        trace.event("run_started", backend=self.backend.name, prompt_version=PROMPT_VERSION)

        # [1] input guardrails
        check = guardrails.check_input(raw_ticket)
        if not check.ok or check.ticket is None:
            trace.event("input_rejected", errors=check.errors)
            raise InvalidTicketError(check.errors)
        ticket = check.ticket
        state = _RunState()
        state.flags += check.flags
        trace.event("input_checked", flags=check.flags)

        if check.refuse_reason:
            submission = TriageSubmission(
                intent=Intent.OUT_OF_SCOPE, order_id=None, decision=Decision.REFUSE, priority=Priority.MEDIUM,
                reasoning=f"Refused by input guardrail: {check.refuse_reason}",
                customer_reply=("Sorry, we can't help with that request through support chat. "
                                "For account or security changes please use the secure account page. "
                                "If you have a question about one of your own orders, just send us the order number." + SIGN),
            )
            trace.event("refused_by_guardrail", reason=check.refuse_reason)
        else:
            # [2] agent loop
            submission = self._agent_loop(ticket, state, trace)

        # [3] deterministic policy enforcement
        submission, overrides = self.enforce_policy(submission, state)
        for o in overrides:
            trace.event("policy_override", detail=o)

        # [4] result
        result = TriageResult(
            ticket_id=ticket.ticket_id,
            intent=submission.intent,
            order_id=submission.order_id,
            decision=submission.decision,
            priority=submission.priority,
            refund_amount_eur=submission.refund_amount_eur,
            policy_references=submission.policy_references,
            customer_reply=submission.customer_reply,
            reasoning=submission.reasoning,
            guardrail_flags=state.flags,
            overrides=overrides,
            tool_calls=state.tool_calls,
            backend=self.backend.name,
            steps_used=state.steps,
        )
        result.summary = summarize(result)
        trace.event("run_finished", decision=result.decision.value, intent=result.intent.value,
                    order_id=result.order_id, overrides=overrides, summary=result.summary)
        return result

    # ------------------------------------------------------------------ loop

    def _agent_loop(self, ticket: Ticket, state: _RunState, trace: Trace) -> TriageSubmission:
        ctx = ToolContext(customer_email=ticket.customer_email, today=ticket.received_at, settings=self.settings)
        tools = self.toolbox.api_specs() + [SUBMIT_TOOL]
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": render_ticket(ticket.ticket_id, ticket.customer_email, ticket.received_at.isoformat(), ticket.message),
        }]
        submit_failures = 0

        while state.steps < self.settings.max_steps:
            state.steps += 1
            try:
                turn = self.backend.complete(SYSTEM_PROMPT, messages, tools)
            except LLMError as exc:
                trace.event("llm_error", step=state.steps, error=str(exc))
                state.flags.append("llm_unavailable")
                return self._fallback(ticket, f"LLM unavailable: {exc}")

            trace.event("llm_turn", step=state.steps, stop_reason=turn.stop_reason,
                        tool_calls=[c.name for c in turn.tool_calls], text=preview(turn.text), usage=turn.usage)

            if turn.stop_reason == "refusal":
                state.flags.append("model_refusal")
                return self._fallback(ticket, "The model declined to process this ticket.")

            messages.append({"role": "assistant", "content": turn.assistant_content})

            if not turn.tool_calls:
                submit_failures += 1
                if submit_failures > self.settings.max_submit_retries:
                    state.flags.append("no_submission")
                    return self._fallback(ticket, "Model did not submit a decision.")
                messages.append({"role": "user", "content": "Please finish by calling the submit_triage tool with your decision."})
                trace.event("reprompt", step=state.steps, reason="no tool call")
                continue

            results: list[dict[str, Any]] = []
            submission: TriageSubmission | None = None
            for call in turn.tool_calls:
                if call.name == "submit_triage":
                    try:
                        submission = TriageSubmission.model_validate(call.input)
                        results.append(_tool_result(call.id, '{"status": "received"}'))
                        trace.event("submission_valid", step=state.steps, decision=submission.decision.value)
                    except ValidationError as exc:
                        submit_failures += 1
                        msg = "Invalid submission: " + "; ".join(
                            f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                        trace.event("submission_invalid", step=state.steps, error=msg)
                        results.append(_tool_result(call.id, msg, is_error=True))
                    continue
                results.append(self._execute_tool(call.id, call.name, call.input, ctx, state, trace))

            if submission is not None:
                return submission
            if submit_failures > self.settings.max_submit_retries:
                state.flags.append("invalid_submission")
                return self._fallback(ticket, "Model kept submitting invalid output.")
            messages.append({"role": "user", "content": results})

        state.flags.append("max_steps_reached")
        trace.event("max_steps_reached", steps=state.steps)
        return self._fallback(ticket, f"Stopped after {self.settings.max_steps} steps without a decision.")

    def _execute_tool(self, call_id: str, name: str, args: dict[str, Any], ctx: ToolContext,
                      state: _RunState, trace: Trace) -> dict[str, Any]:
        t0 = time.perf_counter()
        ok = True
        try:
            output: Any = self.toolbox.run(name, args, ctx)
            content = json.dumps(output, default=str)
            if name == "lookup_order":
                state.verified_orders[output["order_id"]] = output
            elif name == "check_refund_eligibility":
                elig = EligibilityResult.model_validate(output)
                state.eligibility[elig.order_id] = elig
                state.verified_orders.setdefault(elig.order_id, {"order_id": elig.order_id})
        except ToolError as exc:
            ok, content = False, f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001 - a tool bug must not crash the run
            ok, content = False, f"Error: internal tool failure ({type(exc).__name__})."
        ms = round((time.perf_counter() - t0) * 1000, 2)
        record = ToolCallRecord(step=state.steps, tool=name, input=args, ok=ok, output_preview=preview(content, 200), duration_ms=ms)
        state.tool_calls.append(record)
        trace.event("tool_call", **record.model_dump())
        return _tool_result(call_id, content, is_error=not ok)

    def _fallback(self, ticket: Ticket, reason: str) -> TriageSubmission:
        """Safe default when the agent cannot finish: hand the ticket to a human."""
        order_ids = guardrails.extract_order_ids(ticket.message)
        return TriageSubmission(
            intent=Intent.GENERAL_QUESTION, order_id=None, decision=Decision.ESCALATE_TO_HUMAN,
            priority=Priority.MEDIUM, reasoning=f"Fallback escalation: {reason}",
            customer_reply=_escalate_reply(order_ids[0] if order_ids else None),
        )

    # ------------------------------------------------------------------ rules

    def enforce_policy(self, sub: TriageSubmission, state: _RunState) -> tuple[TriageSubmission, list[str]]:
        """Deterministic checks the LLM cannot talk its way around."""
        s = sub.model_copy()
        overrides: list[str] = []
        limit = self.settings.auto_refund_limit_eur

        if s.order_id and s.order_id not in state.verified_orders:
            overrides.append(f"order_id {s.order_id} was never verified by a tool; cleared.")
            s.order_id = None

        if s.decision == Decision.APPROVE_REFUND:
            elig = state.eligibility.get(s.order_id or "")
            if elig is None:
                overrides.append("approve_refund without a successful eligibility check -> escalate_to_human.")
                s.decision = Decision.ESCALATE_TO_HUMAN
            elif not elig.eligible or elig.requires_human:
                overrides.append(f"approve_refund contradicts eligibility (eligible={elig.eligible}, requires_human={elig.requires_human}) -> escalate_to_human.")
                s.decision = Decision.ESCALATE_TO_HUMAN
            else:
                amount = s.refund_amount_eur
                if amount is None or amount > elig.max_refund_eur or amount > limit:
                    overrides.append(f"refund amount {amount} corrected to {elig.max_refund_eur}.")
                    s.refund_amount_eur = elig.max_refund_eur
            if s.decision != Decision.APPROVE_REFUND:
                s.priority = Priority.HIGH if (elig and elig.max_refund_eur > limit) else Priority.MEDIUM
                s.customer_reply = _escalate_reply(s.order_id)

        if s.decision != Decision.APPROVE_REFUND and s.refund_amount_eur is not None:
            overrides.append("refund_amount cleared because no refund was approved.")
            s.refund_amount_eur = None

        if not guardrails.decision_allows_refund_promise(s.decision) and guardrails.reply_promises_refund(s.customer_reply):
            overrides.append("reply promised a refund that was not approved; replaced with safe template.")
            s.customer_reply = _escalate_reply(s.order_id)
            if s.decision == Decision.AUTO_RESOLVE:
                s.decision = Decision.ESCALATE_TO_HUMAN

        if guardrails.reply_leaks_internals(s.customer_reply):
            overrides.append("reply mentioned internal details; replaced with safe template.")
            s.customer_reply = _escalate_reply(s.order_id)
            s.decision = Decision.ESCALATE_TO_HUMAN

        return s, overrides


# ---------------------------------------------------------------------- helpers


def _tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def _escalate_reply(order_id: str | None) -> str:
    about = f" about order #{order_id}" if order_id else ""
    return (f"Thanks for your message{about}. A member of our support team will review it and get back to you "
            "within 24 hours." + SIGN)


def summarize(r: TriageResult) -> str:
    parts = [f"Ticket {r.ticket_id}: {r.intent.value.replace('_', ' ')}"]
    if r.order_id:
        parts.append(f"order {r.order_id}")
    decision = r.decision.value.replace("_", " ")
    if r.refund_amount_eur is not None:
        decision += f" ({r.refund_amount_eur:.2f} EUR)"
    parts.append(f"-> {decision} [priority {r.priority.value}]")
    text = ", ".join(parts[:-1]) + " " + parts[-1] + "."
    text += f" Reason: {r.reasoning}"
    if r.overrides:
        text += f" Rule overrides: {len(r.overrides)}."
    if r.guardrail_flags:
        text += f" Flags: {', '.join(r.guardrail_flags)}."
    used = ", ".join(c.tool for c in r.tool_calls) or "none"
    text += f" Tools used: {used}."
    return text
