"""Observability: structured JSON-lines logs of every step of a run.

Each event carries the run id, so one ticket's whole story can be grepped out
of a shared log file. Messages and tool inputs are logged; secrets never are.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("triage_agent")


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Console shows `level` and above; the JSONL file always receives the full trace."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(level)
    stream.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
    logger.propagate = False


class Trace:
    """Collects events for one agent run and mirrors them to the logger."""

    def __init__(self, ticket_id: str) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.ticket_id = ticket_id
        self.events: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def event(self, kind: str, **data: Any) -> None:
        record = {
            "run_id": self.run_id,
            "ticket_id": self.ticket_id,
            "t_ms": round((time.perf_counter() - self._t0) * 1000, 1),
            "event": kind,
            **data,
        }
        self.events.append(record)
        logger.info(json.dumps(record, default=str, ensure_ascii=False))


def preview(value: Any, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."
