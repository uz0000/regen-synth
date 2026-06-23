"""
Ingest — load a raw tabular file into an IngestResult and isolate rare events.

This is the engine's front door. Pure Python: it loads data, cleans it,
splits normal from rare, builds a typed field dictionary, and (optionally)
persists everything to the on-disk layout the API and CLI read.

No LLM, no network — the field dictionary and rare split are derived
deterministically from the data and the RareEventDef.
"""

from .loader import ingest, persist_ingest

__all__ = ["ingest", "persist_ingest"]
