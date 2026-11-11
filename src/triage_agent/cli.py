"""Command-line entry point.

    triage-agent samples/01_damaged_item.json          # one ticket file
    triage-agent --message "Order #1001 arrived broken" --email anna@example.com
    triage-agent samples/*.json --summary-only          # batch
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from triage_agent.agent import InvalidTicketError, TriageAgent
from triage_agent.config import PROJECT_ROOT, Settings
from triage_agent.tracing import configure_logging


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="triage-agent", description="Triage customer-support tickets with an AI agent.")
    p.add_argument("files", nargs="*", type=Path, help="Ticket JSON files.")
    p.add_argument("--message", help="Ticket text (instead of a file).")
    p.add_argument("--email", default="anna@example.com", help="Customer email for --message.")
    p.add_argument("--date", default=None, help="Reference date YYYY-MM-DD for --message (default: today).")
    p.add_argument("--backend", choices=["mock", "claude", "ollama"], help="Override TRIAGE_BACKEND.")
    p.add_argument("--summary-only", action="store_true", help="Print only the human-readable summary.")
    p.add_argument("--log-level", default="WARNING", help="INFO shows the full trace on stderr.")
    p.add_argument("--log-file", type=Path, default=PROJECT_ROOT / "logs" / "trace.jsonl")
    args = p.parse_args(argv)

    configure_logging(args.log_level, args.log_file)
    settings = Settings.from_env()
    if args.backend:
        settings = Settings(**{**settings.__dict__, "backend": args.backend})
    agent = TriageAgent(settings)

    tickets: list[dict] = []
    if args.message:
        tickets.append({"ticket_id": "cli-1", "customer_email": args.email, "message": args.message,
                        "received_at": args.date or date.today().isoformat()})
    for path in args.files:
        try:
            tickets.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"file": str(path), "error": f"cannot read ticket: {exc}"}), file=sys.stderr)
            return 2
    if not tickets:
        p.error("give at least one ticket file or --message")

    exit_code = 0
    for raw in tickets:
        try:
            result = agent.run(raw)
        except InvalidTicketError as exc:
            print(json.dumps({"ticket_id": raw.get("ticket_id"), "error": "invalid_ticket", "details": exc.errors}, indent=2))
            exit_code = 2
            continue
        if args.summary_only:
            print(result.summary)
        else:
            print(result.model_dump_json(indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
