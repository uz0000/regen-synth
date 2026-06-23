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
)

logger = logging.getLogger(__name__)

_LABEL_CANDIDATES = ("label", "target", "y", "is_fraud", "fraud", "class")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class IngestConfig:
    rare_percentile: float = 0.05  # bottom 5% of target = rare (percentile mode)
    min_rare_rows: int = 10        # raise if fewer rare rows are found


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
    label_col = _resolve_label_col(df, label_col)
    rare_def = _resolve_rare_def(rare_def, config)

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


# ── Label resolution (edge case 1) ────────────────────────────────────────────

def _resolve_label_col(df: pd.DataFrame, label_col: Optional[str]) -> str:
    if label_col:
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found. Columns: {list(df.columns)}"
            )
        return label_col
    for candidate in _LABEL_CANDIDATES:
        if candidate in df.columns:
            logger.info("No label column given — inferred '%s'", candidate)
            return candidate
    raise ValueError(
        f"No label column given and none of {_LABEL_CANDIDATES} present. "
        f"Columns: {list(df.columns)}"
    )


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
    return df


# ── Rare mask ─────────────────────────────────────────────────────────────────

def _build_rare_mask(
    df: pd.DataFrame, label_col: str, rare_def: RareEventDef, config: IngestConfig
) -> pd.Series:
    mode = rare_def.mode

    if mode == RareMode.LABEL:
        if rare_def.label_value is None:
            raise ValueError("LABEL mode requires rare_def.label_value.")
        return df[label_col] == rare_def.label_value

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
        field_dict[col] = FieldMeta(
            name=col,
            field_type=ftype,
            nullable=bool(s.isna().any()),
            cardinality=int(s.nunique()) if ftype == FieldType.CATEGORICAL else None,
            min_val=float(s.min()) if ftype == FieldType.CONTINUOUS else None,
            max_val=float(s.max()) if ftype == FieldType.CONTINUOUS else None,
        )
    return field_dict
