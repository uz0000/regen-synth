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
    tail: str = "lower"  # PERCENTILE mode: "lower" → bottom pct, "upper" → top pct


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
    is_identifier: bool = False        # near-unique key column (order_id, uuid, email) →
                                       # regenerate as fresh unique values, not Gaussian noise


FieldDict = Dict[str, FieldMeta]


@dataclass
class ColumnConstraint:
    """A per-column validity rule the engine enforces on a synthetic batch (L1).

    Built deterministically from the data today (engine.constraints.build_constraints);
    in a future model-advised milestone the same spec is filled by a vetted semantic
    proposal. Enforcement (clamp/round/snap) never invents values — it only folds
    out-of-support output back onto what the column can actually be.
    """
    name: str
    kind: str  # "continuous" | "binary" | "categorical" | "label"
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    is_integer: bool = False
    binary_values: Optional[List[Any]] = None  # the (two) observed values, for snapping


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
    # Full-synthesis split: fraction of the batch that is the amplified rare class
    # (the rest is the synthetic normal part). 0.0 = rare-only (legacy/pre-split
    # batches); recorded so a full-dataset run can be reproduced row-for-row.
    rare_ratio: float = 0.0
    # Privacy mode the batch was generated under. "none" = legacy grounded
    # sampling (no copy prevention); "floored" = parametric generation + the
    # enforced δ-distance floor. Recorded so the privacy regime is reproducible
    # and auditable from disk. delta is the floor in σ-normalized units (0.0
    # when privacy="none").
    privacy: str = "none"
    delta: float = 0.0
    # The vetted ScenarioSpec (as a dict) this batch was generated under, so the
    # use-case context is reproducible from disk — Invariant 2 extended to the
    # contract. None for legacy batches generated before the contract.
    scenario: Optional[Dict[str, Any]] = None
    # Audit bundle: a version stamp, the SHA-256 of every artifact in the
    # bundle (so tampering is detectable), and the version of every metric used
    # (so results are never silently compared across metric-definition changes).
    manifest_schema_version: int = 1
    artifact_sha256: Optional[Dict[str, str]] = None
    metric_versions: Optional[Dict[str, int]] = None

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
                "rare_ratio": self.rare_ratio,
                "privacy": self.privacy,
                "delta": self.delta,
                "scenario": self.scenario,
                "manifest_schema_version": self.manifest_schema_version,
                "artifact_sha256": self.artifact_sha256,
                "metric_versions": self.metric_versions,
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
    # Cross-column correlation structure: mean absolute difference between the
    # real-rare and synthetic correlation matrices (0 = identical). None when
    # too few numeric columns/rows to estimate. A batch with right marginals but
    # scrambled dependence fails here even though every column passes.
    correlation_delta: Optional[float] = None
    correlation_passed: bool = True


# ── Privacy report (distance-floor output) ───────────────────────────────────

@dataclass
class PrivacyReport:
    """Outcome of the privacy δ-distance floor (engine.privacy).

    A checked, enforced per-record guarantee: when ``passed`` is True, *every*
    released row is at least ``delta`` (in σ-normalized numeric space) from
    *every* real row. This closes the near-copy leak that grounded sampling
    creates. It is NOT differential privacy — it does not bound aggregate or
    membership-inference attacks that don't rely on near-copies.

    Attributes:
        mode:         "floored" (floor enforced) or "none".
        delta:        The enforced floor, in σ-normalized units.
        min_distance: Smallest released-row → nearest-real-row distance found
                      (≥ delta when passed).
        n_moved:      Rows projected out to the δ-shell.
        n_respawned:  Rows that couldn't be resolved by projection and were
                      re-drawn from the generator.
        passed:       True iff min_distance ≥ delta (the guarantee holds).
        distance_p10/p50/p90: Spread of nearest-neighbor distances, for judging
                      how much headroom the batch has above the floor.
        floor_applied: Whether the δ-distance floor was actually enforced on the
                      rare part. False when the data can't support it (no
                      continuous features, or the label/rare class couldn't be
                      resolved) — the batch still gets parametric sampling + the
                      verbatim guard, but the reader must not assume a δ-shell
                      was carved. Never silently implied (fail loud).
        floor_skip_reason: Why the floor was skipped ("no_continuous_features" /
                      "no_label"), or None when it was applied.
    """
    mode: str = "floored"
    delta: float = 0.0
    min_distance: float = 0.0
    n_moved: int = 0
    n_respawned: int = 0
    passed: bool = True
    distance_p10: Optional[float] = None
    distance_p50: Optional[float] = None
    distance_p90: Optional[float] = None
    floor_applied: bool = True
    floor_skip_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "delta": self.delta,
            "min_distance": self.min_distance,
            "n_moved": self.n_moved,
            "n_respawned": self.n_respawned,
            "passed": self.passed,
            "distance_p10": self.distance_p10,
            "distance_p50": self.distance_p50,
            "distance_p90": self.distance_p90,
            "floor_applied": self.floor_applied,
            "floor_skip_reason": self.floor_skip_reason,
        }


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
    # Held-out real rare rows the lift was measured on. Below a floor the
    # estimate is degenerate (a handful of test rows → recall snaps between a few
    # discrete values, and a 0.0 is indistinguishable from "no benefit"). status
    # flags that so a bare 0.0 is never reported as if it were a measurement
    #. "ok" | "insufficient_rare_rows".
    n_test_rare: int = 0
    status: str = "ok"


# ── Surrogate quality: TSTR (Train on Synthetic, Test on Real) ────────────────

@dataclass
class TSTRReport:
    """Does the synthetic surrogate *stand in* for real data?

    Train a model on synthetic only (TSTR) and on real only (TRTR), score both on
    a held-out REAL test set, and report the ratio `recovered = TSTR / TRTR` — how
    much of the real-data performance the surrogate recovers. Reported across a
    small model panel and averaged over seeds (no single lucky number). The gap
    below 1.0 is expected — a perfect match would signal memorization, so read
    `recovered` alongside the privacy min-distance.

    status: "ok" | "insufficient_real_test" | "insufficient_train_classes".
    per_model entries carry tstr/trtr ROC-AUC + PR-AUC and the per-metric
    recovered ratios; the medians are the headline.
    """
    status: str = "ok"
    metric: str = "roc_auc+pr_auc"
    n_real_test: int = 0
    n_real_test_rare: int = 0
    n_synth_train: int = 0
    seeds: List[int] = field(default_factory=list)
    per_model: List[Dict[str, Any]] = field(default_factory=list)
    recovered_roc_auc_median: Optional[float] = None
    recovered_pr_auc_median: Optional[float] = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "metric": self.metric,
            "n_real_test": self.n_real_test,
            "n_real_test_rare": self.n_real_test_rare,
            "n_synth_train": self.n_synth_train,
            "seeds": list(self.seeds),
            "per_model": self.per_model,
            "recovered_roc_auc_median": self.recovered_roc_auc_median,
            "recovered_pr_auc_median": self.recovered_pr_auc_median,
            "note": self.note,
        }


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
    (benchmark/superseded/RESULTS_2026-06-22_BREADTH.md, ~75% accuracy). The metric is the
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
