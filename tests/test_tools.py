from datetime import date

import pytest

from triage_agent.config import Settings
from triage_agent.schemas import Order
from triage_agent.tools import ToolError, evaluate_eligibility

TODAY = date(2026, 11, 11)


def order(**kw) -> Order:
    base = dict(order_id="1", customer_email="a@b.co", item="x", amount_eur=50.0, status="delivered",
                order_date=date(2026, 11, 1), delivered_date=date(2026, 11, 5), expected_delivery=date(2026, 11, 5))
    base.update(kw)
    return Order(**base)


def test_lookup_own_order(toolbox, ctx_anna):
    out = toolbox.run("lookup_order", {"order_id": "1001"}, ctx_anna)
    assert out["item"] == "Wireless headphones"
    assert "customer_email" not in out          # no PII echoed back to the model
    assert out["days_since_delivery"] == 6


def test_lookup_other_customers_order_is_hidden(toolbox, ctx_anna):
    with pytest.raises(ToolError, match="No order 1003 found"):
        toolbox.run("lookup_order", {"order_id": "1003"}, ctx_anna)   # belongs to carla


@pytest.mark.parametrize("bad", ["abc", "", "1001; DROP TABLE", 1001, None])
def test_lookup_rejects_malformed_ids(toolbox, ctx_anna, bad):
    with pytest.raises(ToolError):
        toolbox.run("lookup_order", {"order_id": bad}, ctx_anna)


def test_unknown_tool(toolbox, ctx_anna):
    with pytest.raises(ToolError, match="Unknown tool"):
        toolbox.run("delete_all_orders", {}, ctx_anna)


def test_policy_search_finds_relevant_rule(toolbox, ctx_anna):
    out = toolbox.run("search_policy", {"query": "refund above 200 EUR approval"}, ctx_anna)
    assert out["results"][0]["source"] == "refunds"
    assert "200 EUR" in out["results"][0]["text"]


def test_policy_search_rejects_empty(toolbox, ctx_anna):
    with pytest.raises(ToolError):
        toolbox.run("search_policy", {"query": "  "}, ctx_anna)


def test_eligibility_rejects_unknown_reason(toolbox, ctx_anna):
    with pytest.raises(ToolError, match="reason must be one of"):
        toolbox.run("check_refund_eligibility", {"order_id": "1001", "reason": "because"}, ctx_anna)


def test_tool_specs_are_strict(toolbox):
    for spec in toolbox.api_specs():
        assert spec["strict"] is True
        assert spec["input_schema"]["additionalProperties"] is False


S = Settings()


@pytest.mark.parametrize("o, reason, eligible, human", [
    (order(), "damaged", True, False),
    (order(amount_eur=650), "damaged", True, True),                          # above auto limit
    (order(delivered_date=date(2026, 9, 1)), "damaged", False, False),        # outside 30 days
    (order(delivered_date=date(2026, 10, 20)), "change_of_mind", False, False),  # outside 14 days
    (order(status="refunded"), "damaged", False, False),                     # double refund
    (order(status="shipped", delivered_date=None, expected_delivery=date(2026, 11, 1)), "late_delivery", True, False),
    (order(status="shipped", delivered_date=None, expected_delivery=date(2026, 11, 8)), "late_delivery", False, False),
    (order(status="processing", delivered_date=None), "not_received", True, False),
    (order(), "late_delivery", False, False),                                # already delivered
])
def test_eligibility_rules(o, reason, eligible, human):
    r = evaluate_eligibility(o, reason, TODAY, S)
    assert (r.eligible, r.requires_human) == (eligible, human), r.reasons
    assert r.max_refund_eur == (o.amount_eur if eligible else 0.0)


def test_eligibility_boundary_exactly_at_limit():
    r = evaluate_eligibility(order(amount_eur=200.0), "damaged", TODAY, S)
    assert r.eligible and not r.requires_human
