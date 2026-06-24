"""
Deterministic constraint layer (Semantic Fidelity, M1).

Turns the ad-hoc clamp/round/snap that used to live in regen.api into a real,
testable layer: build a per-column ColumnConstraint spec from the ingest result,
then enforce it on a generated batch.

The Prior (Gaussian) and residual GP both sample on an unbounded real line, so a
synthetic row can land outside what any real row could be — a negative Amount, a
fractional count, a binary flag at 0.7. Enforcement folds that out-of-support
mass back onto valid values. It never invents a value the data never showed
(see docs/SEMANTIC_FIDELITY_PLAN.md §1b, rule 4); it only constrains toward
validity, which tightens fidelity rather than loosening it.

Pure Python (pandas/numpy) — no model, no network. The model-advised version
(M2) fills the same ColumnConstraint spec from a vetted semantic proposal; the
enforcement here is unchanged by that.
"""

from typing import Dict

import numpy as np
import pandas as pd

from contracts.types import ColumnConstraint, FieldType, IngestResult


def build_constraints(ingest: IngestResult) -> Dict[str, ColumnConstraint]:
    """Derive the structural (model-free) constraint spec from the ingest result.

    Continuous → observed [min, max] + integrality. Binary → the two observed
    values (for snapping). The label is marked but not constrained (it is set
    constant to the rare value elsewhere). Categoricals are decoded separately
    (regen.api._decode_categoricals), so they carry no numeric constraint here.
    """
    fd = ingest.field_dict
    rare_df = ingest.rare_df
    constraints: Dict[str, ColumnConstraint] = {}
    for name, meta in fd.items():
        if name == ingest.label_col:
            constraints[name] = ColumnConstraint(name=name, kind="label")
        elif meta.field_type == FieldType.CONTINUOUS:
            constraints[name] = ColumnConstraint(
                name=name, kind="continuous",
                min_val=meta.min_val, max_val=meta.max_val, is_integer=meta.is_integer,
            )
        elif meta.field_type == FieldType.BINARY:
            vals = (sorted(pd.unique(rare_df[name].dropna()))[:2]
                    if name in rare_df.columns and rare_df[name].notna().any()
                    else [0, 1])
            constraints[name] = ColumnConstraint(name=name, kind="binary", binary_values=vals)
        else:  # categorical / other — handled by the decode step, no numeric rule
            constraints[name] = ColumnConstraint(name=name, kind="categorical")
    return constraints


def apply_constraints(df: pd.DataFrame, ingest: IngestResult) -> pd.DataFrame:
    """Enforce the column constraints on a synthetic batch, in place-ish.

    - continuous: clip to [min, max]; round integer-valued columns back to ints.
    - binary: snap to the nearest of the two observed values.
    - label / categorical: untouched here.
    """
    constraints = build_constraints(ingest)
    for col in df.columns:
        c = constraints.get(col)
        if c is None or c.kind in ("label", "categorical"):
            continue
        if c.kind == "continuous":
            if c.min_val is not None and c.max_val is not None:
                df[col] = df[col].clip(c.min_val, c.max_val)
            if c.is_integer:
                df[col] = df[col].round().astype("int64")
        elif c.kind == "binary":
            vals = c.binary_values or [0, 1]
            if len(vals) >= 2:
                lo, hi = float(vals[0]), float(vals[1])
                mid = (lo + hi) / 2.0
                df[col] = np.where(df[col].to_numpy() >= mid, vals[1], vals[0])
            elif len(vals) == 1:
                df[col] = vals[0]
    return df
