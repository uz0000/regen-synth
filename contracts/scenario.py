"""
ScenarioSpec — one typed object that carries a whole use case (G-A).

Today a use case is smeared across loose ``generate()`` parameters (label_col,
rare_def, rare_ratio, mode, privacy, delta, ...) plus implicit structural
inference. A ScenarioSpec collects all of it into a single, serializable artifact
that states *what situation is being simulated*: the per-column semantics, the
intent (task + rare-event definition + focus), the gates (fidelity thresholds +
privacy regime + minimum-utility policy), and the provenance of every field (who
filled it — user / structural inference / model — and with what confidence).

A spec is the unit a researcher saves, shares, and re-runs. It is persisted in
the batch manifest, so a batch is reproducible from disk *including* its
use-case context (Invariant 2 extends to the contract).

Pure dataclasses. Like ``contracts.types``, this module imports nothing from
engine/ or regen/ — only its sibling ``contracts.types`` for the rare-event
enum. The deterministic *filling* of a spec from data (Source 1) and the *vetting*
of proposals (G-B) live above the contracts boundary; this file is just the shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from contracts.types import (
    FieldDict,
    FieldType,
    RareEventDef,
    RareMode,
)

# Closed vocabularies (the vetting gate in G-B enforces membership).
ROLES = ("feature", "identifier", "timestamp", "target", "free_text")
DTYPES = ("integer", "float", "categorical", "boolean", "datetime")
TASKS = ("detector_training", "data_sharing", "benchmarking", "exploration")
SOURCES = ("user", "structural", "model")
FAMILIES = ("ols", "logit")  # regression families an estimand may declare


# ── Per-column semantics (the L1 contract from SEMANTIC_FIDELITY_PLAN §3) ──────

@dataclass
class ColumnSemantics:
    """What a column *means*, plus how we came to believe it.

    Only metadata — never a data value (Invariant 4). ``source``/``confidence``/
    ``proposal_id`` are the per-column provenance the audit requires.
    """
    name: str
    role: str = "feature"          # one of ROLES
    dtype: str = "float"           # one of DTYPES
    unit: Optional[str] = None     # "currency" | "ratio[0,1]" | "count" | ...
    min: Optional[float] = None    # semantic bounds (must contain observed range)
    max: Optional[float] = None
    integer: bool = False          # values are integral
    categories: Optional[List[Any]] = None  # value-set for categoricals
    notes: str = ""
    confidence: float = 1.0        # per-column; low → structural fallback (rule 6)
    source: str = "structural"     # who filled it — one of SOURCES (rule 5/7)
    proposal_id: Optional[str] = None  # raw model proposal id (G-B Source 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "role": self.role, "dtype": self.dtype,
            "unit": self.unit, "min": self.min, "max": self.max,
            "integer": self.integer, "categories": self.categories,
            "notes": self.notes, "confidence": self.confidence,
            "source": self.source, "proposal_id": self.proposal_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColumnSemantics":
        return cls(**{k: d.get(k) for k in (
            "name", "role", "dtype", "unit", "min", "max", "integer",
            "categories", "notes", "confidence", "source", "proposal_id",
        ) if k in d})


# ── Vetting verdict (G-B: nothing silent, rule 7) ─────────────────────────────

@dataclass
class VettingVerdict:
    """One record of how a proposed constraint was resolved by the vetting gate.

    Every accept / reject / override / fallback is logged with the rule that
    decided it and a human rationale, so the contract is explainable (rule 7) and
    the record can be surfaced in explanation.json (G-C).
    """
    field: str                 # "<column>.<attr>" e.g. "amount.min"
    decision: str              # "accepted" | "rejected" | "fallback" | "unchanged"
    rule: str                  # which vetting rule fired (e.g. "data_is_ground_truth")
    source: str                # who proposed it ("user" | "model" | "structural")
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "decision": self.decision, "rule": self.rule,
                "source": self.source, "rationale": self.rationale}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VettingVerdict":
        return cls(**{k: d.get(k, "") for k in
                      ("field", "decision", "rule", "source", "rationale")})


# ── Intent: what is being simulated and why ───────────────────────────────────

@dataclass
class ScenarioIntent:
    task: str = "detector_training"      # one of TASKS
    label_col: str = ""                  # "" → auto-detect the target column
    rare_mode: str = "label"             # "label" | "percentile" | "imbalance_ratio"
    rare_value: Optional[Any] = None     # label mode; None → auto minority class
    percentile: Optional[float] = None   # percentile mode
    tail: str = "lower"                  # percentile mode: "lower" | "upper"
    imbalance_ratio: Optional[float] = None
    rare_ratio: Optional[float] = None   # None → auto (amplify minority to ≥25%)
    focus_features: List[str] = field(default_factory=list)  # columns worth observing
    n_rows: int = 300
    seed: int = 42
    mode: str = "balanced"               # "faithful" | "balanced" | "boost"

    def rare_def(self) -> RareEventDef:
        """Build the engine's RareEventDef from the declared intent."""
        mode = RareMode(self.rare_mode)
        return RareEventDef(
            mode=mode,
            label_value=self.rare_value,
            percentile=self.percentile,
            imbalance_ratio=self.imbalance_ratio,
            tail=self.tail,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task, "label_col": self.label_col,
            "rare_mode": self.rare_mode, "rare_value": self.rare_value,
            "percentile": self.percentile, "tail": self.tail,
            "imbalance_ratio": self.imbalance_ratio, "rare_ratio": self.rare_ratio,
            "focus_features": list(self.focus_features), "n_rows": self.n_rows,
            "seed": self.seed, "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScenarioIntent":
        d = dict(d or {})
        d["focus_features"] = list(d.get("focus_features") or [])
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


# ── Gates: fidelity thresholds, privacy regime, minimum-utility policy ────────

@dataclass
class ScenarioGates:
    # None → let generate() pick its per-mode default (kept so a spec can say
    # "use defaults" without pinning a number that might drift).
    coverage_threshold: Optional[float] = None
    correlation_threshold: Optional[float] = None
    tvd_threshold: Optional[float] = None
    wasserstein_threshold: Optional[float] = None
    privacy: str = "floored"             # "floored" | "none"
    delta: float = 0.5
    min_tail_lift: Optional[float] = None  # minimum-utility policy (advisory report)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage_threshold": self.coverage_threshold,
            "correlation_threshold": self.correlation_threshold,
            "tvd_threshold": self.tvd_threshold,
            "wasserstein_threshold": self.wasserstein_threshold,
            "privacy": self.privacy, "delta": self.delta,
            "min_tail_lift": self.min_tail_lift,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScenarioGates":
        d = dict(d or {})
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


# ── Estimand: the analysis whose ESTIMATE the synthetic data must preserve ─────

@dataclass
class EstimandSpec:
    """A declared analysis whose *estimate* the synthetic data must preserve.

    An estimand is the target quantity of an analysis. v1 supports **regression
    coefficients**: fit ``outcome ~ predictors`` (``family`` = ols | logit) on the
    real reference to get θ_real ± CI, fit the *same* spec on the delivered
    synthetic to get θ_synth, and CERTIFY preservation iff every
    coefficient-of-interest's θ_synth lands within θ_real's confidence interval.

    This is a guarantee distinct from fidelity (marginals/correlations) and TSTR
    (prediction): a batch can pass both while a coefficient silently shifts (a
    copula flattens an interaction; tail amplification re-weights the rare
    stratum). Certification is recomputed by ``regen verify`` from the delivered
    data + the disclosed θ_real ± CI in ``reference_aggregates.json`` — never from
    a cached verdict, and never from raw real rows.

    Empty (``outcome == ""``) → no estimand declared; the certificate omits it.
    Only decisions/metrics are derived from a spec — never a synthetic value
    (Invariant 4). The spec persists in the manifest, so the estimand and its
    verdict reproduce bit-for-bit (Invariant 7).
    """
    outcome: str = ""                                     # dependent-variable column
    predictors: List[str] = field(default_factory=list)   # regressor columns
    family: str = "ols"                                   # one of FAMILIES
    # Subset of predictors whose recovery is certified; [] → every predictor.
    coefficients_of_interest: List[str] = field(default_factory=list)
    ci_level: float = 0.95                                # confidence level for θ_real / the test
    # Certification rule. "consistent" (default): θ preserved iff θ_real and
    # θ_synth are indistinguishable beyond their combined standard error — a
    # two-sample Wald test, |Δ| ≤ z·√(se_real²+se_synth²). "within_ci" (stricter,
    # ignores synth uncertainty): θ_synth must lie inside θ_real's CI.
    rule: str = "consistent"

    def declared(self) -> bool:
        """True once an estimand is actually specified (outcome + ≥1 predictor)."""
        return bool(self.outcome and self.predictors)

    def targets(self) -> List[str]:
        """Coefficients to certify: the declared subset, or all predictors."""
        return list(self.coefficients_of_interest) or list(self.predictors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "predictors": list(self.predictors),
            "family": self.family,
            "coefficients_of_interest": list(self.coefficients_of_interest),
            "ci_level": self.ci_level,
            "rule": self.rule,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EstimandSpec":
        d = dict(d or {})
        d["predictors"] = list(d.get("predictors") or [])
        d["coefficients_of_interest"] = list(d.get("coefficients_of_interest") or [])
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


# ── The whole use case ────────────────────────────────────────────────────────

@dataclass
class ScenarioSpec:
    columns: Dict[str, ColumnSemantics] = field(default_factory=dict)
    intent: ScenarioIntent = field(default_factory=ScenarioIntent)
    gates: ScenarioGates = field(default_factory=ScenarioGates)
    # The analysis whose estimate the synthetic data must preserve (optional;
    # undeclared → the certificate omits estimand preservation).
    estimand: EstimandSpec = field(default_factory=EstimandSpec)
    notes: str = ""
    # Provenance for non-column fields, e.g. {"intent.label_col": "user"}.
    # Per-column provenance lives on each ColumnSemantics (source/confidence).
    provenance: Dict[str, str] = field(default_factory=dict)
    # The vetting gate's record (G-B rule 7: nothing silent). Empty until a spec
    # has been vetted against data.
    verdicts: List["VettingVerdict"] = field(default_factory=list)
    spec_version: int = 1

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "notes": self.notes,
            "intent": self.intent.to_dict(),
            "gates": self.gates.to_dict(),
            "estimand": self.estimand.to_dict(),
            "provenance": dict(self.provenance),
            "verdicts": [v.to_dict() for v in self.verdicts],
            # column order preserved (dict is insertion-ordered)
            "columns": [c.to_dict() for c in self.columns.values()],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScenarioSpec":
        cols = {c["name"]: ColumnSemantics.from_dict(c) for c in d.get("columns", [])}
        return cls(
            columns=cols,
            intent=ScenarioIntent.from_dict(d.get("intent", {})),
            gates=ScenarioGates.from_dict(d.get("gates", {})),
            estimand=EstimandSpec.from_dict(d.get("estimand", {})),
            notes=d.get("notes", ""),
            provenance=dict(d.get("provenance", {})),
            verdicts=[VettingVerdict.from_dict(v) for v in d.get("verdicts", [])],
            spec_version=int(d.get("spec_version", 1)),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        # sort_keys for a canonical, hashable form; the columns list keeps order.
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=_native)

    @classmethod
    def from_json(cls, s: str) -> "ScenarioSpec":
        return cls.from_dict(json.loads(s))

    def to_yaml(self) -> str:
        import yaml
        return yaml.safe_dump(self.to_dict(), sort_keys=True, default_flow_style=False)

    @classmethod
    def from_yaml(cls, s: str) -> "ScenarioSpec":
        import yaml
        return cls.from_dict(yaml.safe_load(s))

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_yaml())

    @classmethod
    def load_yaml(cls, path: str) -> "ScenarioSpec":
        with open(path) as f:
            return cls.from_yaml(f.read())

    def rare_def(self) -> RareEventDef:
        return self.intent.rare_def()


# ── Source 1: fill a spec's columns from the deterministic schema (structural) ─

def _dtype_for(meta) -> str:
    ft = meta.field_type
    if ft == FieldType.BINARY:
        return "boolean"
    if ft == FieldType.CATEGORICAL:
        return "categorical"
    if ft == FieldType.IDENTIFIER:
        return "integer" if getattr(meta, "is_integer", False) else "categorical"
    return "integer" if getattr(meta, "is_integer", False) else "float"


def _role_for(name: str, meta, label_col: str) -> str:
    if name == label_col:
        return "target"
    if getattr(meta, "is_identifier", False):
        return "identifier"
    return "feature"


def columns_from_field_dict(field_dict: FieldDict, label_col: str) -> Dict[str, ColumnSemantics]:
    """Structural (Source 1) per-column semantics from the ingest field dict.

    Every value is derived deterministically from observed data, so source=
    "structural" and confidence=1.0. Bounds/categories come straight from the
    profiled FieldMeta. (Researcher declarations and model proposals merge on top
    of this via the vetting gate in G-B; here we only lay the baseline.)
    """
    cols: Dict[str, ColumnSemantics] = {}
    for name, meta in field_dict.items():
        cols[name] = ColumnSemantics(
            name=name,
            role=_role_for(name, meta, label_col),
            dtype=_dtype_for(meta),
            min=(float(meta.min_val) if meta.min_val is not None else None),
            max=(float(meta.max_val) if meta.max_val is not None else None),
            integer=bool(getattr(meta, "is_integer", False)),
            categories=list(meta.categories) if getattr(meta, "categories", None) else None,
            confidence=1.0,
            source="structural",
        )
    return cols


def _native(value):
    """JSON default for numpy scalars sneaking into categories/bounds."""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)
