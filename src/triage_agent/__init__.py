"""Support Triage Agent: an AI agent that triages e-commerce support tickets."""

from triage_agent.agent import TriageAgent
from triage_agent.schemas import Decision, Intent, Ticket, TriageResult

__all__ = ["TriageAgent", "Ticket", "TriageResult", "Decision", "Intent"]
__version__ = "0.1.0"
