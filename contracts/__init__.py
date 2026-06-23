"""
Contracts — shared types and schemas that cross the engine/API boundary.

Anything in this package must be importable by both engine/ (pure Python, no LLM)
and regen/ (the API/orchestration layer). Keep it to dataclasses, enums, and type
aliases. No LLM clients, no networking, no agent framework imports.
"""

from .types import (
    BatchManifest,
    ColumnFidelity,
    FidelityReport,
    FieldDict,
    FieldMeta,
    FieldType,
    IngestResult,
    LiftReport,
    RareEventDef,
    RareMode,
    SchemaGraph,
    TableEdge,
)

__all__ = [
    "BatchManifest",
    "ColumnFidelity",
    "FidelityReport",
    "FieldDict",
    "FieldMeta",
    "FieldType",
    "IngestResult",
    "LiftReport",
    "RareEventDef",
    "RareMode",
    "SchemaGraph",
    "TableEdge",
]
