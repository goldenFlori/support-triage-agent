"""Web API tests (FastAPI TestClient, mock backend, no network)."""

from fastapi.testclient import TestClient

from triage_agent.web.server import app

client = TestClient(app)


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200 and "Support Triage Agent" in r.text


def test_config_customers_and_samples():
    assert "claude_available" in client.get("/api/config").json()
    emails = [c["email"] for c in client.get("/api/customers").json()]
    assert "anna@example.com" in emails
    samples = client.get("/api/samples").json()
    assert len(samples) == 12 and samples[0]["ticket"]["ticket_id"] == "T-001"


def test_triage_returns_result_and_trace():
    r = client.post("/api/triage", json={"customer_email": "anna@example.com", "received_at": "2026-11-11",
                                         "message": "Order #1001 arrived broken, refund please"})
    body = r.json()
    assert r.status_code == 200
    assert body["result"]["decision"] == "approve_refund"
    events = [e["event"] for e in body["trace"]]
    assert events[0] == "run_started" and events[-1] == "run_finished" and "tool_call" in events


def test_invalid_ticket_returns_422_with_details():
    r = client.post("/api/triage", json={"customer_email": "bad", "message": "  "})
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "invalid_ticket"
    assert len(r.json()["detail"]["details"]) == 2


def test_claude_backend_without_key_is_rejected(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/api/triage", json={"customer_email": "anna@example.com", "message": "hi", "backend": "claude"})
    assert r.status_code == 400
