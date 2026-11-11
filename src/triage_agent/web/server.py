"""FastAPI app that exposes the agent to the browser.

    triage-web                # starts http://127.0.0.1:8000 and opens the browser

Endpoints
    GET  /                    the single-page UI
    GET  /api/config          which backends are available
    GET  /api/customers       mock customers and their orders (for the form)
    GET  /api/samples         the sample tickets from samples/
    POST /api/triage          run the agent on one ticket -> result + step-by-step trace
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from triage_agent.agent import InvalidTicketError, TriageAgent
from triage_agent.config import PROJECT_ROOT, Settings
from triage_agent.tracing import Trace

STATIC = Path(__file__).parent / "static"
SAMPLES = PROJECT_ROOT / "samples"

app = FastAPI(title="Support Triage Agent", version="0.1.0")


class TriageRequest(BaseModel):
    # Loose on purpose: the agent's own guardrails do the validation and
    # return readable errors, which the UI shows to the user.
    ticket_id: str = "WEB-1"
    customer_email: str = ""
    message: str = ""
    received_at: date | None = None
    backend: Literal["mock", "claude"] = "mock"


def claude_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=2)
def get_agent(backend: str) -> TriageAgent:
    base = Settings.from_env()
    return TriageAgent(Settings(**{**base.__dict__, "backend": backend}))


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config() -> dict[str, Any]:
    s = Settings.from_env()
    return {"claude_available": claude_available(), "model": s.model, "default_date": "2026-11-11"}


@app.get("/api/customers")
def customers() -> list[dict[str, Any]]:
    orders = json.loads((Settings.from_env().data_dir / "orders.json").read_text(encoding="utf-8"))
    by_email: dict[str, list[dict[str, Any]]] = {}
    for o in orders:
        by_email.setdefault(o["customer_email"], []).append(
            {"order_id": o["order_id"], "item": o["item"], "amount_eur": o["amount_eur"], "status": o["status"]})
    return [{"email": e, "orders": os_} for e, os_ in sorted(by_email.items())]


@app.get("/api/samples")
def samples() -> list[dict[str, Any]]:
    out = []
    for path in sorted(SAMPLES.glob("*.json")):
        title = path.stem.split("_", 1)[1].replace("_", " ").capitalize()
        out.append({"name": path.stem, "title": title, "ticket": json.loads(path.read_text(encoding="utf-8"))})
    return out


@app.post("/api/triage")
def triage(req: TriageRequest) -> dict[str, Any]:
    if req.backend == "claude" and not claude_available():
        raise HTTPException(400, "Claude backend needs ANTHROPIC_API_KEY on the server. Use mock mode instead.")
    raw = {"ticket_id": req.ticket_id or "WEB-1", "customer_email": req.customer_email, "message": req.message,
           "received_at": (req.received_at or date.today()).isoformat()}
    trace = Trace(raw["ticket_id"])
    try:
        result = get_agent(req.backend).run(raw, trace=trace)
    except InvalidTicketError as exc:
        raise HTTPException(422, {"error": "invalid_ticket", "details": exc.errors}) from exc
    return {"result": result.model_dump(mode="json"), "trace": trace.events}


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(prog="triage-web", description="Start the Support Triage Agent web UI.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)
    url = f"http://{args.host}:{args.port}"
    print(f"Support Triage Agent UI running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
