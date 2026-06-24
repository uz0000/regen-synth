"""
Deterministic schema profile (Semantic Fidelity, M1).

A compact, model-free summary of each column — dtype, cardinality, observed
bounds, integrality, missingness, and a few sample values. It is independently
useful (inspect what REGEN inferred about a dataset) and is the exact input a
future model-advised layer (M2) would be shown — the model never sees raw rows,
only this profile, and it returns metadata, never values.

Pure Python. No model, no network.
"""

from typing import Any, Dict, List

import pandas as pd

from contracts.types import FieldType, IngestResult


def column_profiles(ingest: IngestResult, n_samples: int = 5) -> List[Dict[str, Any]]:
    """One profile dict per column, in column order.

    Built from the full dataset (normal + rare) so the picture matches what was
    ingested. `role_guess` is a naive structural hint only (the real role call is
    deferred to the model-advised milestone): near-unique columns look like
    identifiers, the label column is flagged, everything else is "feature".
    """
    fd = ingest.field_dict
    df = pd.concat([ingest.normal_df, ingest.rare_df], ignore_index=True)
    n = len(df)
    profiles: List[Dict[str, Any]] = []
    for name, meta in fd.items():
        s = df[name] if name in df.columns else pd.Series(dtype="object")
        card = int(meta.cardinality) if meta.cardinality is not None else int(s.nunique(dropna=True))
        uniq_ratio = (card / n) if n else 0.0
        prof: Dict[str, Any] = {
            "name": name,
            "field_type": meta.field_type.value,
            "cardinality": card,
            "unique_ratio": round(uniq_ratio, 4),
            "nullable": bool(meta.nullable),
            "is_integer": bool(meta.is_integer),
            "role_guess": _role_guess(name, meta, uniq_ratio, ingest.label_col),
            "sample_values": [_native(v) for v in s.dropna().unique()[:n_samples]],
        }
        if meta.field_type == FieldType.CONTINUOUS:
            prof["min"] = meta.min_val
            prof["max"] = meta.max_val
        if meta.field_type in (FieldType.CATEGORICAL, FieldType.BINARY) and not s.empty:
            vc = s.value_counts(dropna=True)
            prof["minority_value"] = _native(vc.idxmin())
            prof["minority_ratio"] = round(float(vc.min()) / n, 4) if n else 0.0
        profiles.append(prof)
    return profiles


def _role_guess(name: str, meta, uniq_ratio: float, label_col: str) -> str:
    if name == label_col:
        return "label"
    if getattr(meta, "is_identifier", False):
        return "identifier"
    return "feature"


def _native(value):
    """JSON-serialisable Python scalar from a NumPy/pandas value."""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value
