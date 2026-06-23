"""
Shared dataclasses for all REGEN components.

All data contracts between engine stages and the control plane live here.
No imports from engine/, regen/, or any LLM/network library.
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


@dataclass
class TargetDetection:
    """
    Result of structural auto-detection of the rare-event target.

    Model-agnostic by design: the label column and rare value are chosen purely
    from the data's structure (cardinality + class imbalance), never from a
    model-specific lift measurement — detection lift depends on the downstream
    model, so it must not drive *what* the rare event is (see INVARIANTS.md §1).
    """
    label_col: str
    rare_value: Any
    n_rare: int
    minority_ratio: float
    cardinality: int
    score: float = 0.0
    auto_label: bool = False   # was the column auto-detected (vs user-supplied)?
    auto_rare: bool = False    # was the rare value auto-detected?
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label_col": self.label_col,
            "rare_value": self.rare_value,
            "n_rare": self.n_rare,
            "minority_ratio": round(self.minority_ratio, 4),
            "cardinality": self.cardinality,
            "score": round(self.score, 4),
            "auto_label": self.auto_label,
            "auto_rare": self.auto_rare,
            "alternatives": self.alternatives,
        }


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
    is_integer: bool = False           # continuous field whose real values are all integral
                                       # (counts, hour, Time) → round synthetic output back to int
    categories: Optional[List[Any]] = None  # canonical category order (categorical fields),
                                             # computed from the FULL dataset so encode/decode agree


FieldDict = Dict[str, FieldMeta]


# ── Ingest result (engine input) ──────────────────────────────────────────────

@dataclass
class IngestResult:
    normal_df: pd.DataFrame
    rare_df: pd.DataFrame
    schema_graph: SchemaGraph
    field_dict: FieldDict
    label_col: str = ""
    detection: Optional[TargetDetection] = None  # set when label/rare auto-detected


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


# ── Campaign output ───────────────────────────────────────────────────────────

@dataclass
class PassDetail:
    """One pass of an amplification campaign: accept/reject + metrics."""
    pass_num: int
    status: str                     # "accepted" | "rejected"
    tail_lift: float = 0.0
    baseline_recall: float = 0.0
    amplified_recall: float = 0.0
    baseline_precision: float = 0.0
    amplified_precision: float = 0.0
    coverage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pass_num": self.pass_num,
            "status": self.status,
            "tail_lift": self.tail_lift,
            "baseline_recall": self.baseline_recall,
            "amplified_recall": self.amplified_recall,
            "baseline_precision": self.baseline_precision,
            "amplified_precision": self.amplified_precision,
            "coverage": self.coverage,
        }


@dataclass
class CampaignResult:
    """Full outcome of a multi-pass REGEN campaign."""
    best_lift: float
    passes: List[PassDetail] = field(default_factory=list)
    n_accepted: int = 0
    n_rejected: int = 0
    n_normal: int = 0
    n_rare: int = 0
    n_features: int = 0
    n_rows_per_pass: int = 0
    output_dir: str = ""
    best_batch_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "best_lift": self.best_lift,
            "passes": [p.to_dict() for p in self.passes],
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "n_normal": self.n_normal,
            "n_rare": self.n_rare,
            "n_features": self.n_features,
            "n_rows_per_pass": self.n_rows_per_pass,
            "output_dir": self.output_dir,
            "best_batch_path": self.best_batch_path,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Screen result ──────────────────────────────────────────────────────────────

@dataclass
class ScreenResult:
    """Win-boundary prediction: which method is likely to win on this data.

    The prediction rule is calibrated against the breadth benchmark
    (benchmark/RESULTS_BREADTH.md, ~75% accuracy). The metric is the
    coefficient of variation (CV = σ/μ) of the fitted ARD kernel
    inverse-lengthscales — high spread means features vary in
    informativeness (REGEN-favorable), low spread means features are
    homogeneous/redundant (SMOTE-favorable). Two known misclassifications
    are both conservative (predicted SMOTE, REGEN actually won).
    """
    recommended_method: str         # "REGEN" | "SMOTE"
    heterogeneity_score: float      # CV of ARD inverse-lengthscales
    confidence: float               # distance from decision boundary [0, 1]
    predicted_lift_band: str        # rough estimate range
    rationale: str                  # one-line explanation
    n_rare: int = 0
    n_features: int = 0

    def to_dict(self) -> dict:
        return {
            "recommended_method": self.recommended_method,
            "heterogeneity_score": self.heterogeneity_score,
            "confidence": self.confidence,
            "predicted_lift_band": self.predicted_lift_band,
            "rationale": self.rationale,
            "n_rare": self.n_rare,
            "n_features": self.n_features,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
