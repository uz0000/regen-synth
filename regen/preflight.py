"""
Preflight / generality envelope (G-E) — validate a dataset against what REGEN
actually supports BEFORE generation, and report actionable verdicts instead of
producing a surprising or degenerate batch.

`preflight(path, label_col, rare_def)` returns a list of checks, each with a
level (`ok` / `warn` / `degraded` / `unsupported` / `error`), a message, and a
recommendation. The levels mirror docs/CAPABILITY_MATRIX.md. Every rule here was
observed — the degraded cases came out of the P1-6 privacy sweep (all-categorical
and low-cardinality-integer data), the small-rare cases out of P2-7 / the GP
underdetermination guard.

Pure Python; lives outside engine/. Reuses the deterministic ingest + profile.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from contracts.types import FieldType, RareEventDef

# Envelope thresholds.
MIN_RARE_AMPLIFY = 10        # loader hard-minimum for amplification
MIN_RARE_FOR_LIFT = 14       # ~10 held-out at 30% test split (P2-7 MIN_TEST_RARE)
HIGH_CARD = 50               # top-K TVD path above this many categories
LOW_CARD_INT = 8             # integer feature with fewer distinct values ≈ ordinal
HIGH_NAN_RATE = 0.30
BIG_ROWS = 500_000           # memory watch
FREE_TEXT_AVG_LEN = 40       # avg string length suggesting free text


def _v(check, level, message, recommendation=""):
    return {"check": check, "level": level, "message": message,
            "recommendation": recommendation}


def preflight(
    filepath: str,
    label_col: str = "",
    rare_def: Optional[RareEventDef] = None,
) -> Dict[str, Any]:
    """Validate a dataset against the supported envelope. Never generates."""
    from regen.api import ingest, _auto_rare_def
    from engine.ingest.profile import column_profiles

    checks: List[Dict[str, Any]] = []
    rare_def = rare_def or _auto_rare_def()

    # Ingest may legitimately refuse (too few rare rows, ambiguous target). Report
    # that as a verdict rather than raising — preflight exists to catch it early.
    try:
        result = ingest(filepath, label_col, rare_def)
    except Exception as e:
        msg = str(e)
        if "minimum" in msg or "rare rows" in msg:
            checks.append(_v("rare_count", "unsupported", msg,
                             "Relax the rare definition or supply more rare examples."))
        elif "Ambiguous" in type(e).__name__ or "ambiguous" in msg.lower():
            checks.append(_v("target_detection", "error", msg,
                             "Pass label_col explicitly to disambiguate the target."))
        else:
            checks.append(_v("ingest", "error", msg, "Fix the input and retry."))
        return {"ok_to_generate": False, "checks": checks}

    fd = result.field_dict
    label = result.label_col
    n_rare = len(result.rare_df)
    n_total = len(result.normal_df) + n_rare
    feats = [c for c in fd if c != label]
    numeric = [c for c in feats if fd[c].field_type in (FieldType.CONTINUOUS, FieldType.BINARY)
               and not getattr(fd[c], "is_identifier", False)]
    continuous = [c for c in feats if fd[c].field_type == FieldType.CONTINUOUS
                  and not getattr(fd[c], "is_identifier", False)]

    # 1. Rare count for amplification + a non-degenerate lift estimate.
    if n_rare < MIN_RARE_AMPLIFY:
        checks.append(_v("rare_count", "unsupported",
                         f"{n_rare} rare rows (< {MIN_RARE_AMPLIFY}).",
                         "Amplification needs more rare examples."))
    elif n_rare < MIN_RARE_FOR_LIFT:
        checks.append(_v("rare_count", "warn",
                         f"{n_rare} rare rows: amplification runs, but the held-out "
                         f"lift estimate will report 'insufficient_rare_rows' (P2-7).",
                         "Interpret lift cautiously; add rare examples for a real estimate."))
    else:
        checks.append(_v("rare_count", "ok", f"{n_rare} rare rows."))

    # 2. All-categorical → δ-floor cannot apply (P2-9); floored fidelity may drop (P1-6).
    if not continuous:
        checks.append(_v("privacy_floor", "degraded",
                         "No continuous features: the δ-distance floor cannot apply.",
                         "privacy='floored' falls back to parametric sampling + the "
                         "verbatim guard + k-anonymity; fidelity on high-cardinality "
                         "categoricals may drop (see docs/CAPABILITY_MATRIX.md)."))

    # 3. Low-cardinality integer 'continuous' features behave like ordinals; the
    #    floor can collapse coverage on them (P1-6 solar_flare). Continuous columns
    #    carry no cardinality in the field dict, so count distinct values directly.
    import pandas as pd
    _full = pd.concat([result.normal_df, result.rare_df], ignore_index=True)
    low_card_int = [c for c in continuous
                    if getattr(fd[c], "is_integer", False)
                    and c in _full.columns
                    and _full[c].nunique(dropna=True) <= LOW_CARD_INT]
    if low_card_int and continuous:
        checks.append(_v("low_cardinality_integer", "degraded",
                         f"Integer features with ≤{LOW_CARD_INT} distinct values behave "
                         f"like ordinals: {low_card_int[:5]}.",
                         "privacy='floored' can collapse coverage on these; prefer "
                         "privacy='none' or declare them categorical."))

    # 4. Dimensionality vs rare count — GP underdetermination.
    if len(numeric) > n_rare:
        checks.append(_v("dimensionality", "warn",
                         f"{len(numeric)} numeric features > {n_rare} rare rows: the "
                         "residual GP is underdetermined.",
                         "Set max_features (6–10) or supply more rare rows."))

    # 5. High-cardinality categoricals, NaN rates, constant columns (from the profile).
    for p in column_profiles(result):
        name = p["name"]
        if name == label:
            continue
        if p["field_type"] == FieldType.CATEGORICAL.value and p["cardinality"] > HIGH_CARD:
            checks.append(_v("high_cardinality", "warn",
                             f"'{name}' has {p['cardinality']} categories (> {HIGH_CARD}).",
                             "TVD uses the top-K path; per-category fidelity is approximate."))
            # free-text heuristic: long string values + very high cardinality
            samples = [s for s in p.get("sample_values", []) if isinstance(s, str)]
            if samples and sum(len(s) for s in samples) / len(samples) > FREE_TEXT_AVG_LEN:
                checks.append(_v("free_text", "unsupported",
                                 f"'{name}' looks like free text (long, near-unique values).",
                                 "Free text is not modeled; drop the column or encode it."))
        if p["cardinality"] <= 1:
            checks.append(_v("constant_column", "warn",
                             f"'{name}' is constant — it carries no signal.",
                             "Consider dropping it."))

    # 6. Dataset size vs memory.
    if n_total > BIG_ROWS:
        checks.append(_v("dataset_size", "warn",
                         f"{n_total} rows: ingest loads the full table.",
                         "Expect higher memory/time; the prior subsamples internally."))

    # 7. Out-of-scope shapes named plainly (single-table tabular only).
    ts = [c for c in feats if any(k in c.lower() for k in ("timestamp", "datetime", "_date"))]
    if ts:
        checks.append(_v("time_series", "unsupported",
                         f"Column(s) {ts} look temporal; REGEN does not model time-series "
                         "structure (rows are treated as exchangeable).",
                         "Temporal order is NOT preserved — see docs/CAPABILITY_MATRIX.md."))

    ok = not any(c["level"] in ("unsupported", "error") for c in checks)
    return {"ok_to_generate": ok, "n_rare": n_rare, "n_total": n_total,
            "label_col": label, "checks": checks}
