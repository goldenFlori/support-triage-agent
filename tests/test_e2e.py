"""End-to-end: every sample ticket through the real pipeline (mock LLM) and the CLI."""

import json
import os
from pathlib import Path

import pytest

from triage_agent.agent import InvalidTicketError, TriageAgent
from triage_agent.cli import main
from triage_agent.config import Settings

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted(p for p in (ROOT / "samples").glob("*.json") if not p.name.startswith("99_"))
KEYS = ["intent", "order_id", "decision", "priority", "refund_amount_eur", "policy_references", "guardrail_flags"]


@pytest.mark.parametrize("path", SAMPLES, ids=[p.stem for p in SAMPLES])
def test_sample_matches_expected_output(path):
    agent = TriageAgent(Settings(backend="mock"))
    result = agent.run(json.loads(path.read_text())).model_dump(mode="json")
    expected = json.loads((ROOT / "samples" / "expected" / path.name).read_text())
    assert {k: result[k] for k in KEYS} == {k: expected[k] for k in KEYS}
    assert result["summary"] and result["customer_reply"].endswith("Customer Support")


def test_invalid_sample_is_rejected():
    with pytest.raises(InvalidTicketError):
        TriageAgent(Settings(backend="mock")).run(json.loads((ROOT / "samples" / "99_invalid_input.json").read_text()))


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TRIAGE_BACKEND", "mock")
    code = main([str(ROOT / "samples" / "01_damaged_item_auto_refund.json"), "--log-file", str(tmp_path / "t.jsonl")])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["decision"] == "approve_refund"
    events = [json.loads(line)["event"] for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert events[0] == "run_started" and events[-1] == "run_finished"
    assert events.count("tool_call") == 3


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
@pytest.mark.parametrize("name, allowed", [
    ("01_damaged_item_auto_refund", {"approve_refund"}),
    ("02_damaged_expensive_escalate", {"escalate_to_human"}),
    ("06_someone_elses_order", {"need_more_info", "refuse", "escalate_to_human"}),
])
def test_live_claude(name, allowed):
    agent = TriageAgent(Settings(backend="claude"))
    result = agent.run(json.loads((ROOT / "samples" / f"{name}.json").read_text()))
    assert result.decision.value in allowed, result.summary
