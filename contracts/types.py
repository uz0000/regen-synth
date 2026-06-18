"""
Shared dataclasses for all REGEN components.

All data contracts between engine stages and the control plane live here.
No imports from engine/, agent-runtime/, or any LLM/network library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Rare event definition ─────────────────────────────────────────────────────

class RareMode(str, Enum):
    LABEL      = "label"           # explicit binary label column
    PERCENTILE = "percentile"      # bottom N% of a continuous target
    IMBALANCE  = "imbalance_ratio" # rows in the minority class


@dataclass
class RareEventDef:
    """
    How to identify rare events in the dataset.

    Exactly one of label_value / percentile / imbalance_ratio should be set,
    matching the mode.
    """
    mode: RareMode = RareMode.PERCENTILE
    label_value: Optional[Any] = None
    percentile: Optional[float] = None
    imbalance_ratio: Optional[float] = None


# ── Schema graph ──────────────────────────────────────────────────────────────

@dataclass
class TableEdge:
    """A foreign-key relationship: child_table references parent_table."""
    parent_table: str
    child_table: str
    join_key: str


@dataclass
class SchemaGraph:
    """
    Relational structure between tables.
    Empty when input is a single flat table; components check .is_empty().
    """
    tables: List[str] = field(default_factory=list)
    edges: List[TableEdge] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.edges) == 0

    def schema_hash(self) -> str:
        """Deterministic hash of the schema structure for manifest tracking."""
        payload = {
            "tables": sorted(self.tables),
            "edges": sorted(
                [{"p": e.parent_table, "c": e.child_table, "k": e.join_key}
                 for e in self.edges],
                key=lambda x: (x["p"], x["c"]),
            ),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ── Field dictionary ──────────────────────────────────────────────────────────

class FieldType(str, Enum):
    CONTINUOUS  = "continuous"
    CATEGORICAL = "categorical"
    BINARY      = "binary"
    IDENTIFIER  = "identifier"


@dataclass
class FieldMeta:
    name: str
    field_type: FieldType
    nullable: bool = False
    cardinality: Optional[int] = None  # for categorical fields
    min_val: Optional[float] = None    # for continuous fields
    max_val: Optional[float] = None


FieldDict = Dict[str, FieldMeta]


# ── Ingest result (engine input) ──────────────────────────────────────────────

@dataclass
class IngestResult:
    normal_df: pd.DataFrame
    rare_df: pd.DataFrame
    schema_graph: SchemaGraph
    field_dict: FieldDict
    label_col: str = ""


# ── Batch manifest ────────────────────────────────────────────────────────────

@dataclass
class BatchManifest:
    """
    Everything needed to reproduce a batch exactly.

    Invariant (test_reproducibility.py): same manifest → same data.
    The manifest is sealed at generation time and travels with every batch.
    """
    seed: int
    schema_hash: str
    prior_config: Dict[str, Any]     # serialisable subset of RegenConfig
    target_region: Dict[str, Any]    # Scout's chosen region description
    amplifier_params: Dict[str, Any]
    code_version: str                # git commit hash or package version
    n_rows: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "schema_hash": self.schema_hash,
                "prior_config": self.prior_config,
                "target_region": self.target_region,
                "amplifier_params": self.amplifier_params,
                "code_version": self.code_version,
                "n_rows": self.n_rows,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, s: str) -> "BatchManifest":
        d = json.loads(s)
        return cls(**d)


# ── Fidelity report (auditor output) ─────────────────────────────────────────

@dataclass
class ColumnFidelity:
    col: str
    tvd: Optional[float] = None         # Total Variation Distance
    wasserstein: Optional[float] = None # Wasserstein-1 (continuous only)
    passed: bool = True


@dataclass
class FidelityReport:
    coverage_rate: float         # PRIMARY METRIC: fraction of rare events covered
    coverage_passed: bool
    column_results: List[ColumnFidelity] = field(default_factory=list)
    overall_passed: bool = True
    n_real: int = 0
    n_synthetic: int = 0
    manifest: Optional[BatchManifest] = None


# ── Examiner output ───────────────────────────────────────────────────────────

@dataclass
class LiftReport:
    """
    Detection lift from the Examiner (M3).

    baseline_* = detector trained on real data only.
    amplified_* = detector trained on real + synthetic amplified data.
    tail_lift = amplified_recall - baseline_recall (Scout's reward signal).
    """
    baseline_recall: float
    baseline_precision: float
    amplified_recall: float
    amplified_precision: float
    tail_lift: float          # primary number Scout uses
    n_synthetic_used: int
    manifest: Optional[BatchManifest] = None
