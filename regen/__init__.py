"""
REGEN — Unified Python API layer.

All consumers (CLI, Streamlit demo, the agent runtime stages) call through here.
Pure deterministic Python — no LLM client, no agent runtime, no network.

This is the consolidation layer the project needed: one source of truth for
ingest, run, screen, and result-loading logic.
"""

from .api import ingest, run_campaign, screen, get_results, load_synthetic

__all__ = [
    "ingest",
    "run_campaign",
    "screen",
    "get_results",
    "load_synthetic",
]
