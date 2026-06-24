"""
Data ingestion and schema mapping.

Loads a CSV / JSON / Parquet file and produces an IngestResult:
  - normal_df   : non-rare rows, cleaned (no missing values)
  - rare_df     : isolated rare-event rows
  - schema_graph: relational structure, or empty for a flat table
  - field_dict  : typed metadata for every column
  - label_col   : the resolved target column

Edge cases are handled loudly (fail-loud convention, INVARIANTS.md §9):
  1. No label column        → infer from common names or raise
  2. No rare definition      → default to percentile mode, warn
  3. Fewer than min_rare_rows → raise (the GP cannot fit on too few)
  4. Missing values          → impute (median / mode), log counts
  5. All rows flagged rare    → raise (misconfigured definition)

persist_ingest() writes the on-disk layout the API and CLI consume:
  <base>.normal.parquet, <base>.rare.parquet, <base>.fields.json
"""

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from contracts.types import (
    FieldDict,
    FieldMeta,
    FieldType,
    IngestResult,
    RareEventDef,
    RareMode,
    SchemaGraph,
    TargetDetection,
)

logger = logging.getLogger(__name__)

# Names that hint a column is the classification target. Only a tiebreak bonus
# in structural detection — never the sole basis (a well-named but balanced
# column is a poor rare-event target; an unnamed but imbalanced one is a good one).
_LABEL_CANDIDATES = ("label", "target", "y", "is_fraud", "fraud", "class")


class AmbiguousTargetError(ValueError):
    """Raised when auto-detection cannot confidently choose one target column.

    Two or more columns score within _AMBIGUITY_MARGIN of each other, so guessing
    would be arbitrary. The caller should pass label_col / rare_def explicitly.
    """

    def __init__(self, candidates):
        self.candidates = candidates
        lines = [
            f"    {c.label_col!r}: rare={c.rare_value!r} "
            f"minority_ratio={c.minority_ratio:.3f} n_rare={c.n_rare} score={c.score:.3f}"
            for c in candidates
        ]
        super().__init__(
            "Auto-detection found multiple comparable rare-event targets. "
            "Pass label_col (and rare_def) explicitly to disambiguate. Candidates:\n"
            + "\n".join(lines)
        )


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class IngestConfig:
    rare_percentile: float = 0.05  # bottom 5% of target = rare (percentile mode)
    min_rare_rows: int = 10        # raise if fewer rare rows are found

    # ── Structural auto-detection "useful band" (INVARIANTS.md "What It Cannot Do") ──
    # These bound which columns count as a rare-event target. The defaults track
    # the documented limits (>=10 rare rows; clearly imbalanced; low cardinality),
    # but they are knobs, not law — tune them per dataset if detection misfires.
    max_target_cardinality: int = 20   # more distinct values ⇒ not a class label
    max_minority_ratio: float = 0.35   # above this the column is ~balanced (no lift to gain)
    rare_count_saturation: int = 50    # n_rare at/above this gets full "enough data" credit
    ambiguity_margin: float = 0.05     # top-two scores within this ⇒ AmbiguousTargetError


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest(
    filepath: str,
    label_col: str,
    rare_def: Optional[RareEventDef],
    config: Optional[IngestConfig] = None,
) -> IngestResult:
    """
    Load, clean, and split a tabular dataset into normal and rare subsets.

    Args:
        filepath:  Path to a CSV / JSON / Parquet file.
        label_col: Target column name. Pass "" to auto-detect.
        rare_def:  How to identify rare rows. None → percentile mode.
        config:    Thresholds. Defaults to IngestConfig().

    Returns:
        IngestResult.

    Raises:
        ValueError / FileNotFoundError on bad input (see module docstring).
    """
    config = config or IngestConfig()

    df = _load_file(filepath)
    label_col, rare_def, detection = _resolve_target(df, label_col, rare_def, config)

    df = _impute_missing(df, label_col)

    rare_mask = _build_rare_mask(df, label_col, rare_def, config)
    _validate_split(rare_mask, config)

    normal_df = df[~rare_mask].reset_index(drop=True)
    rare_df = df[rare_mask].reset_index(drop=True)
    field_dict = _build_field_dict(df, label_col)

    logger.info(
        "Ingestion complete: %d normal, %d rare, %d features",
        len(normal_df), len(rare_df), len(field_dict) - 1,
    )
    return IngestResult(
        normal_df=normal_df,
        rare_df=rare_df,
        schema_graph=SchemaGraph(),  # flat table — no relational structure
        field_dict=field_dict,
        label_col=label_col,
        detection=detection,
    )


def persist_ingest(ingest_result: IngestResult, base_path: str) -> str:
    """
    Write the IngestResult to the on-disk layout the stages read.

    Produces:
        <base_path>.normal.parquet
        <base_path>.rare.parquet
        <base_path>.fields.json

    Returns the base_path (the ingest_path the stages take as input).
    """
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)

    ingest_result.normal_df.to_parquet(base_path + ".normal.parquet", index=False)
    ingest_result.rare_df.to_parquet(base_path + ".rare.parquet", index=False)

    fields = {
        name: {
            "name": meta.name,
            "field_type": meta.field_type.value,
            "nullable": meta.nullable,
        }
        for name, meta in ingest_result.field_dict.items()
    }
    with open(base_path + ".fields.json", "w") as f:
        json.dump(fields, f, indent=2)

    logger.info("Persisted ingest to %s.{normal,rare}.parquet + .fields.json", base_path)
    return base_path


# ── File loading ──────────────────────────────────────────────────────────────

def _load_file(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(filepath)
    if suffix in (".json", ".jsonl"):
        return pd.read_json(filepath)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(filepath)
    return pd.read_csv(filepath)  # best-effort fallback


# ── Target resolution (label column + rare value) — edge cases 1 & 2 ──────────

def _resolve_target(
    df: pd.DataFrame,
    label_col: Optional[str],
    rare_def: Optional[RareEventDef],
    config: IngestConfig,
):
    """Resolve (label_col, rare_def, detection), auto-detecting either when asked.

    Manual choice stays optional and authoritative — an explicit label_col or a
    rare_def with a concrete value is honored verbatim. Auto-detection only fires
    for the parts the caller left open:

      * label_col == ""              → structurally detect the target column.
      * LABEL mode, label_value None → take the minority class of the column.

    Detection is purely structural (cardinality + imbalance), so it never depends
    on a downstream model. `detection` is returned for reporting/override (None
    when nothing was auto-detected).
    """
    auto_rare_wanted = (
        rare_def is not None
        and rare_def.mode == RareMode.LABEL
        and rare_def.label_value is None
    )

    # 1. Label column.
    if label_col:
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found. Columns: {list(df.columns)}"
            )
        detection = None
        # 2a. Explicit column, auto rare value → minority class of that column.
        if auto_rare_wanted:
            detection = _candidate_for_column(df, label_col, config)
            detection.auto_rare = True
            rare_def = RareEventDef(mode=RareMode.LABEL, label_value=detection.rare_value)
            logger.info(
                "Auto-selected rare value %r for '%s' (minority class, ratio=%.3f, n=%d)",
                detection.rare_value, label_col, detection.minority_ratio, detection.n_rare,
            )
        else:
            # No rare side at all → historical percentile default (warns).
            rare_def = _resolve_rare_def(rare_def, config)
        return label_col, rare_def, detection

    # 2b. No column given → structural detection picks both column and rare value.
    detection = _detect_target(df, config)
    detection.auto_label = True
    logger.info(
        "Auto-selected target column '%s' (score=%.3f, minority_ratio=%.3f, n_rare=%d)",
        detection.label_col, detection.score, detection.minority_ratio, detection.n_rare,
    )
    # When the caller left the rare side open too (auto LABEL, or no rare_def at
    # all), use the detected minority class. An explicit non-label rare_def
    # (percentile/imbalance) is honored against the detected column.
    if rare_def is None or auto_rare_wanted:
        rare_def = RareEventDef(mode=RareMode.LABEL, label_value=detection.rare_value)
        detection.auto_rare = True
    return detection.label_col, rare_def, detection


def _minority_value(s: pd.Series):
    """The least-frequent value in a series — the rare class under LABEL mode."""
    return s.value_counts().idxmin()


def _candidate_for_column(df: pd.DataFrame, col: str, config: IngestConfig) -> TargetDetection:
    """Build a TargetDetection for a fixed column (no scoring / no ambiguity check)."""
    s = df[col]
    counts = s.value_counts(dropna=True)
    rare_value = counts.idxmin()
    n_rare = int(counts.min())
    n = int(counts.sum())
    return TargetDetection(
        label_col=col,
        rare_value=_as_native(rare_value),
        n_rare=n_rare,
        minority_ratio=(n_rare / n) if n else 0.0,
        cardinality=int(s.nunique(dropna=True)),
    )


def _score_target_columns(df: pd.DataFrame, config: IngestConfig):
    """Score every column for fitness as a rare-event target (descending).

    Fitness is structural, matching REGEN's goal — amplifying a genuine minority
    class to lift detection. A column qualifies only inside the "useful band"
    (config): low cardinality, >= min_rare_rows rare rows, and minority ratio
    below max_minority_ratio (a ~balanced column gives REGEN nothing to amplify).
    Score then favors clearer imbalance, binary targets, and enough rare rows.
    """
    n = len(df)
    candidates = []
    for col in df.columns:
        s = df[col]
        card = int(s.nunique(dropna=True))
        if card < 2 or card > config.max_target_cardinality:
            continue
        counts = s.value_counts(dropna=True)
        n_rare = int(counts.min())
        minority_ratio = n_rare / n if n else 0.0
        if n_rare < config.min_rare_rows or minority_ratio > config.max_minority_ratio:
            continue

        imbalance_score = 1.0 - (minority_ratio / config.max_minority_ratio)   # 0..1, ↑ when rarer
        cardinality_score = 1.0 if card == 2 else 1.0 / (card - 1)             # binary best
        rare_count_score = min(1.0, n_rare / config.rare_count_saturation)    # enough to learn the tail
        name_bonus = 0.15 if str(col).strip().lower() in _LABEL_CANDIDATES else 0.0
        score = (
            0.50 * imbalance_score
            + 0.25 * cardinality_score
            + 0.25 * rare_count_score
            + name_bonus
        )
        candidates.append(
            TargetDetection(
                label_col=str(col),
                rare_value=_as_native(counts.idxmin()),
                n_rare=n_rare,
                minority_ratio=minority_ratio,
                cardinality=card,
                score=score,
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _detect_target(df: pd.DataFrame, config: IngestConfig) -> TargetDetection:
    """Pick the single best rare-event target column, or fail loud if unclear."""
    candidates = _score_target_columns(df, config)
    if not candidates:
        raise ValueError(_no_target_message(df, config))
    best = candidates[0]
    if len(candidates) >= 2 and (best.score - candidates[1].score) < config.ambiguity_margin:
        raise AmbiguousTargetError(candidates[:4])
    best.alternatives = [c.as_dict() for c in candidates[1:4]]
    return best


def _no_target_message(df: pd.DataFrame, config: IngestConfig) -> str:
    """A diagnostic error explaining why each column was rejected as a target.

    The bare "no suitable target" message leaves the caller guessing. Here we list
    the low-cardinality columns (the realistic override candidates) with the
    specific reason each failed the useful band, so they know what to pass as
    label_col / rare_def.
    """
    n = len(df)
    near, high_card = [], 0
    for col in df.columns:
        s = df[col]
        card = int(s.nunique(dropna=True))
        if card < 2:
            continue  # constant column — never a target
        if card > config.max_target_cardinality:
            high_card += 1
            continue
        counts = s.value_counts(dropna=True)
        n_rare = int(counts.min())
        ratio = n_rare / n if n else 0.0
        if n_rare < config.min_rare_rows:
            reason = f"only {n_rare} rare rows (need >= {config.min_rare_rows})"
        elif ratio > config.max_minority_ratio:
            reason = (f"minority class is {ratio:.0%} of rows — too balanced "
                      f"(need <= {config.max_minority_ratio:.0%})")
        else:
            reason = "borderline"
        kind = "binary" if card == 2 else f"{card}-class"
        near.append(f"  • '{col}' ({kind}, rare value={_as_native(counts.idxmin())!r}): {reason}")

    lines = [
        "No column qualified for automatic rare-event detection. REGEN looks for a "
        f"low-cardinality column (<= {config.max_target_cardinality} classes) with "
        f">= {config.min_rare_rows} rare rows and a minority class <= "
        f"{config.max_minority_ratio:.0%} of the data.",
    ]
    if near:
        lines.append("Closest columns and why each was skipped:")
        lines.extend(near[:8])
    if high_card:
        lines.append(f"  ({high_card} other column(s) skipped: too many distinct values — "
                     "likely continuous or an ID.)")
    lines.append(
        "Fix: pass label_col (and a rare value) explicitly. For a continuous target "
        "(e.g. an amount/score), use percentile mode to flag the tail instead of a label."
    )
    return "\n".join(lines)


def _as_native(value):
    """Coerce a NumPy/pandas scalar to a JSON-serialisable Python scalar."""
    if isinstance(value, np.generic):
        return value.item()
    return value


# ── Rare definition resolution (edge case 2) ──────────────────────────────────

def _resolve_rare_def(rare_def: Optional[RareEventDef], config: IngestConfig) -> RareEventDef:
    if rare_def is not None:
        return rare_def
    warnings.warn(
        f"No rare_def given — defaulting to percentile mode "
        f"(bottom {config.rare_percentile * 100:.0f}% of target).",
        UserWarning, stacklevel=3,
    )
    return RareEventDef(mode=RareMode.PERCENTILE, percentile=config.rare_percentile)


# ── Missing-value imputation (edge case 4) ────────────────────────────────────

def _impute_missing(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.copy()

    missing_label = df[label_col].isna()
    if missing_label.any():
        logger.warning("Dropping %d rows with missing label", int(missing_label.sum()))
        df = df[~missing_label].copy()

    counts = {}
    for col in df.columns:
        if col == label_col:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")
        counts[col] = n_missing
    if counts:
        logger.info("Imputed missing values: %s", counts)

    # Fail loud on any column still all-NaN: a numeric column with no observed
    # values has median NaN, so fillna is a no-op and the NaN silently poisons
    # the Prior/GP (NaN scores → NaN residuals → a batch the Auditor passes
    # because `NaN > threshold` is False). Reject it at the door instead.
    still_nan = [c for c in df.columns if df[c].isna().all()]
    if still_nan:
        raise ValueError(
            f"Columns are entirely missing and cannot be imputed: {still_nan}. "
            "Drop them or supply data before ingesting."
        )
    return df


# ── Rare mask ─────────────────────────────────────────────────────────────────

def _build_rare_mask(
    df: pd.DataFrame, label_col: str, rare_def: RareEventDef, config: IngestConfig
) -> pd.Series:
    mode = rare_def.mode

    if mode == RareMode.LABEL:
        # label_value should already be resolved by _resolve_target; auto-pick the
        # minority class as a defensive fallback rather than failing.
        value = rare_def.label_value
        if value is None:
            value = _minority_value(df[label_col])
        return df[label_col] == value

    if mode == RareMode.PERCENTILE:
        pct = rare_def.percentile if rare_def.percentile is not None else config.rare_percentile
        if not pd.api.types.is_numeric_dtype(df[label_col]):
            raise ValueError(
                f"Percentile mode needs a numeric label; '{label_col}' is {df[label_col].dtype}."
            )
        return df[label_col] <= df[label_col].quantile(pct)

    if mode == RareMode.IMBALANCE:
        if rare_def.imbalance_ratio is None:
            raise ValueError("IMBALANCE mode requires rare_def.imbalance_ratio.")
        counts = df[label_col].value_counts()
        total = len(df)
        minority = [c for c, n in counts.items() if n / total <= rare_def.imbalance_ratio]
        if not minority:
            raise ValueError(
                f"No minority class at ratio={rare_def.imbalance_ratio}. "
                f"Distribution: {counts.to_dict()}"
            )
        return df[label_col].isin(minority)

    raise ValueError(f"Unknown RareMode: {mode}")


# ── Split validation (edge cases 3 and 5) ─────────────────────────────────────

def _validate_split(rare_mask: pd.Series, config: IngestConfig) -> None:
    n_rare = int(rare_mask.sum())
    n_normal = len(rare_mask) - n_rare

    if n_rare < config.min_rare_rows:
        raise ValueError(
            f"Only {n_rare} rare rows found; minimum is {config.min_rare_rows}. "
            "Relax the rare definition or supply more data."
        )
    if n_normal == 0:
        raise ValueError(
            "All rows flagged as rare — misconfigured rare_def, not a valid dataset."
        )


# ── Field dictionary ──────────────────────────────────────────────────────────

def _build_field_dict(df: pd.DataFrame, label_col: str) -> FieldDict:
    field_dict: FieldDict = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            ftype = FieldType.BINARY
        elif pd.api.types.is_numeric_dtype(s):
            ftype = FieldType.BINARY if s.nunique() == 2 else FieldType.CONTINUOUS
        else:
            ftype = FieldType.CATEGORICAL
        # A continuous column whose real values are all whole numbers (counts,
        # hour, Time) must come back as integers — the Prior/GP emit floats.
        is_integer = False
        if ftype == FieldType.CONTINUOUS:
            nona = s.dropna()
            is_integer = bool(len(nona)) and bool(np.all(nona == np.floor(nona)))
        # Canonical category order from the FULL dataset (df is normal+rare here),
        # so the Prior's encode and the output decode share one code mapping.
        categories = (
            list(pd.Categorical(s.dropna()).categories)
            if ftype == FieldType.CATEGORICAL else None
        )
        field_dict[col] = FieldMeta(
            name=col,
            field_type=ftype,
            nullable=bool(s.isna().any()),
            cardinality=int(s.nunique()) if ftype == FieldType.CATEGORICAL else None,
            min_val=float(s.min()) if ftype == FieldType.CONTINUOUS else None,
            max_val=float(s.max()) if ftype == FieldType.CONTINUOUS else None,
            is_integer=is_integer,
            categories=categories,
        )
    return field_dict
