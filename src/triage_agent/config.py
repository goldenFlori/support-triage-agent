"""Runtime settings. Read once from environment variables, with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    backend: str = "mock"                       # "mock" (offline), "claude" (API) or "ollama" (local, free)
    model: str = "claude-opus-5-5"
    ollama_model: str = "qwen2.5:7b"            # any Ollama model that supports tool calling
    ollama_url: str = "http://localhost:11434"
    effort: str = "medium"                      # triage is routine work; raise for harder cases
    max_steps: int = 8                          # hard cap on agent loop iterations
    max_submit_retries: int = 2                 # re-prompts when the final JSON is invalid / missing
    auto_refund_limit_eur: float = 200.0        # above this a human must approve
    late_delivery_threshold_days: int = 5
    refund_window_days: int = 30
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")

    @classmethod
    def from_env(cls) -> "Settings":
        backend = os.getenv("TRIAGE_BACKEND", "").strip().lower()
        if not backend:
            backend = "claude" if os.getenv("ANTHROPIC_API_KEY") else "mock"
        return cls(
            backend=backend,
            model=os.getenv("TRIAGE_MODEL", cls.model),
            ollama_model=os.getenv("TRIAGE_OLLAMA_MODEL", cls.ollama_model),
            ollama_url=_ollama_url(),
            effort=os.getenv("TRIAGE_EFFORT", cls.effort),
            max_steps=int(os.getenv("TRIAGE_MAX_STEPS", cls.max_steps)),
            data_dir=Path(os.getenv("TRIAGE_DATA_DIR", str(PROJECT_ROOT / "data"))),
        )


def _ollama_url() -> str:
    host = os.getenv("OLLAMA_HOST", "").strip() or Settings.ollama_url
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")
