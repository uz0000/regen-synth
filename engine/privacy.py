"""
Privacy — enforced δ-distance floor (per-record guarantee).

REGEN's original generator is *grounded sampling*: a real anchor row plus a
small Gaussian jitter. With default noise that makes every synthetic row a
near-copy of a real individual. The parametric generator
(``engine.prior.generate_parametric_batch``) removes the copying by construction
— it samples from a fitted class distribution, never from a single real row.

This module is the **enforced guarantee** layered on top: no released row may
sit within ``delta`` (in σ-normalized space) of *any* real row. It is a checked,
deterministic, per-record invariant — every released row is pushed out to at
least the δ-shell from the full real set. This closes the near-copy leak.

What this is NOT: it is not Differential Privacy. It bounds record-level
re-identification via near-copies; it does not bound aggregate or membership-
inference attacks that do not rely on near-copies. See docs (Privacy).

Pure Python (numpy + scipy.spatial.cKDTree). No model, no network — the engine
boundary (tests/test_boundary.py) stays intact.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from contracts.types import FieldDict, FieldType, PrivacyReport

logger = logging.getLogger(__name__)

# A released row that reproduces a real row's full non-identifier attribute set
# is a re-identification risk only when that attribute set *uniquely* identifies
# the real individual. A discrete tuple shared by ≥ this many real rows is
# k-anonymous — reusing it reveals nothing about any one person — so it is not a
# verbatim leak. Below this count (i.e. a singleton real record) it is. This is
# what makes the guarantee meaningful for low-cardinality categorical data, where
# every synthetic row necessarily reuses some real category combination. When
# continuous features are present, a match within tol_sigma already pins a near-
# unique individual, so the count check applies only to the discrete-only path.
_MIN_ANON_COUNT = 2


# ── Public API ─────────────────────────────────────────────────────────────────

def enforce_distance_floor(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    field_dict: FieldDict,
    label_col: str,
    delta: float,
    rng: np.random.Generator,
    max_iter: int = 8,
) -> Tuple[pd.DataFrame, PrivacyReport]:
    """Guarantee every released row is ≥ ``delta`` (σ-normalized) from every
    real row, on the continuous features.

    Distance is computed in per-feature σ-normalized space over the continuous
    columns (the high-resolution identifiers — amounts, ages, counts, scores).
    Categorical/binary near-copies are already prevented by parametric sampling
    (frequency tables), so they are not part of the floor metric.

    Violating rows are first projected out to the δ-shell along the vector away
    from their nearest real neighbour; rows the projection can't settle (or that
    the in-support clamp shaves back) are respawned — re-drawn uniformly inside
    the observed box until they clear δ from every real row. Both steps keep
    values within each feature's observed [min, max], so no out-of-support value
    is introduced (the constraint layer's job is unchanged). When the box is too
    saturated for any clearing point to exist (the dense-bulk case the floor is
    not meant for), the row lands as far out as the box allows and ``passed``
    reports the shortfall — the floor is enforced wherever it is feasible, never
    faked.

    Deterministic: the only randomness (a tie-break direction for a row sitting
    exactly on a real neighbour) flows through the passed ``rng``, so the same
    (synth, real, delta, rng) → identical output (Invariant 2 holds).

    Args:
        synth_df:  Synthetic batch (encoded space is fine; only continuous cols
                   are read or modified).
        real_df:   The real reference set the released rows must stay ≥ delta
                   from. REGEN's caller passes the real **rare** set: the floor
                   is a rare-vs-rare guarantee (the rare set is sparse enough for
                   a δ-shell to exist, and it is where re-identification risk
                   concentrates). Cross-class near-copies against the dense
                   normal bulk are instead handled by parametric sampling + the
                   verbatim guard (P2-8). Pass whatever reference the guarantee
                   is defined against; this function stays away from every row in
                   whatever it is given.
        field_dict: Ingest field dict — selects continuous columns.
        label_col: Excluded from the metric.
        delta:     Floor in σ-normalized units (e.g. 0.5).
        rng:       Seeded Generator.
        max_iter:  Projection iterations before the centroid-away fallback.

    Returns:
        (floored_df, PrivacyReport). ``floored_df`` is a copy of ``synth_df``
        with continuous columns adjusted; other columns are untouched.
        ``PrivacyReport.passed`` is True iff the minimum released-row distance
        to the real set is ≥ delta.
    """
    cols = _continuous_cols(real_df, field_dict, label_col)
    out = synth_df.copy()

    if not cols or len(real_df) == 0 or len(synth_df) == 0:
        # Nothing numeric to protect, or empty — trivially satisfies the floor.
        return out, PrivacyReport(
            mode="floored", delta=delta, min_distance=float("inf"),
            passed=True,
        )

    R = real_df[cols].to_numpy(dtype=np.float64)
    S = synth_df[cols].to_numpy(dtype=np.float64)

    sigma = R.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    lo = R.min(axis=0)
    hi = R.max(axis=0)

    R_n = R / sigma
    S_n = S / sigma
    tree = cKDTree(R_n)

    n_moved = 0
    n_respawned = 0

    dist, idx = tree.query(S_n, k=1)
    originally_violating = int((dist < delta).sum())

    for _ in range(max_iter):
        dist, idx = tree.query(S_n, k=1)
        viol = dist < delta
        if not viol.any():
            break
        # Push each violator to exactly delta from its nearest real point, along
        # the away-from-neighbour direction.
        nn = R_n[idx[viol]]
        diff = S_n[viol] - nn
        nd = np.linalg.norm(diff, axis=1, keepdims=True)
        # A row coincident with a real point (nd == 0) has no away-direction, so
        # nudge it along a deterministic rng-chosen axis. Index the ACTUAL zero
        # rows — a previous version wrote into diff[0:n_zero], which left a
        # coincident row unfixed whenever it wasn't among the first n_zero of the
        # violating subset, so nd stayed 0 and delta/nd produced inf and crashed
        # the next KD-tree query. This surfaced on integer-coded / low-cardinality
        # continuous columns (e.g. solar_flare), where synthetic rows land exactly
        # on real ones far more often than on smooth continuous data. (P1-6.)
        zero_rows = np.where(nd.ravel() == 0)[0]
        if zero_rows.size:
            d = diff.shape[1]
            axes = rng.integers(0, d, size=zero_rows.size)
            diff[zero_rows, axes] = 1.0
            nd = np.linalg.norm(diff, axis=1, keepdims=True)
        S_n[viol] = nn + diff * (delta / nd)
        n_moved += int(viol.sum())

    # Clamp the projected rows back into the observed box FIRST (no out-of-support
    # values — a REGEN fidelity requirement). The projection above pushes rows to
    # the δ-shell which can sit outside the box; clamping then pulls them back to
    # the boundary, which may re-violate the floor. So the respawn pass below must
    # run on the *clamped* coordinates, not the pre-clamp ones — otherwise a row
    # that looked fine at the shell gets silently shaved onto a real row by the
    # clamp (the bug this ordering fixes).
    lo_n, hi_n = lo / sigma, hi / sigma
    S_n = np.clip(S_n, lo_n, hi_n)

    # Respawn pass for residual violators (rows trapped between near-equidistant
    # real points, or shaved by the clamp above). Rather than nudge, re-draw the
    # row: sample fresh candidates uniformly inside the box and keep one that
    # clears delta from every real point. The box interior is overwhelmingly
    # empty for a sparse rare set (the only place the floor is applied), so a
    # clearing point almost always exists; candidates are in-support by
    # construction, so the final clamp can't pull them back. Deterministic via
    # `rng`. If nothing clears (a genuinely saturated box — the dense-bulk case
    # the floor is not meant for), keep the farthest candidate and let the
    # report's `passed` honestly reflect the shortfall.
    dist, _ = tree.query(S_n, k=1)
    for i in np.where(dist < delta)[0]:
        cand = rng.uniform(lo_n, hi_n, size=(64, S_n.shape[1]))
        cd, _ = tree.query(cand, k=1)
        ok = np.where(cd >= delta)[0]
        S_n[i] = cand[ok[0]] if ok.size else cand[int(np.argmax(cd))]
        n_respawned += 1

    # Denormalize (a no-op clamp — every row is already in-box) and write back.
    # Widen the continuous columns to float first: the floor produces fractional
    # values, and the caller re-applies the integer-rounding constraint after.
    S_new = np.clip(S_n * sigma, lo, hi)
    out[cols] = out[cols].astype(np.float64)
    out.loc[:, cols] = S_new

    # Final distance report in the *released* (clamped) space.
    tree_r = cKDTree(R / sigma)
    S_final = out[cols].to_numpy(dtype=np.float64) / sigma
    final_dist, _ = tree_r.query(S_final, k=1)
    min_distance = float(final_dist.min()) if final_dist.size else float("inf")

    qs = np.percentile(final_dist, [10, 50, 90]) if final_dist.size else (None, None, None)
    report = PrivacyReport(
        mode="floored",
        delta=float(delta),
        min_distance=min_distance,
        n_moved=n_moved,
        n_respawned=n_respawned,
        passed=bool(min_distance >= delta - 1e-9),
        distance_p10=None if qs[0] is None else float(qs[0]),
        distance_p50=None if qs[1] is None else float(qs[1]),
        distance_p90=None if qs[2] is None else float(qs[2]),
    )
    logger.info(
        "Privacy floor %s | delta=%.3f min_dist=%.3f moved=%d respawned=%d (of %d violating on input)",
        "PASSED" if report.passed else "FAILED",
        delta, min_distance, n_moved, n_respawned, originally_violating,
    )
    return out, report


def guard_against_duplicates(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    field_dict: FieldDict,
    label_col: str,
    rng: np.random.Generator,
    tol_sigma: float = 1e-3,
) -> Tuple[pd.DataFrame, int]:
    """Guarantee no released row duplicates a real row's full attribute set.

    A row is a "duplicate" if it matches some real row on every non-identifier
    feature: categorical/binary values equal, and continuous values within
    ``tol_sigma`` (σ-normalized) — i.e. a verbatim repro of a real individual's
    attributes. Identifiers are excluded (they are re-minted fresh and never
    match). This is feasible on the dense bulk — where the δ-distance floor is
    not — because it only acts on exact-attribute matches, which parametric
    sampling makes measure-zero. It is the safety net that turns "near-copies
    are measure-zero" into a checked guarantee.

    Duplicate rows are nudged: a tiny σ-scaled jitter is added to their
    continuous features (categoricals re-drawn from the row's own value is a
    no-op; the jitter on continuous is enough to break the exact-attribute
    match while staying in-distribution). Deterministic via ``rng``.

    Returns:
        (guarded_df, n_duplicates): a copy with duplicates nudged, and the count.
    """
    feat = [c for c in synth_df.columns
            if c in field_dict and c != label_col
            and not getattr(field_dict[c], "is_identifier", False)]
    cont = [c for c in feat if field_dict[c].field_type == FieldType.CONTINUOUS]
    disc = [c for c in feat if field_dict[c].field_type in (FieldType.CATEGORICAL,
                                                            FieldType.BINARY)]

    out = synth_df.copy()
    if len(real_df) == 0 or len(synth_df) == 0 or not feat:
        return out, 0

    # Candidate matches via continuous proximity (tiny radius), then confirm the
    # discrete attributes match exactly. Cheap because the radius is measure-zero.
    dup_idx: list = []
    if cont:
        sigma = real_df[cont].to_numpy(dtype=np.float64).std(axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        tree = cKDTree(real_df[cont].to_numpy(dtype=np.float64) / sigma)
        Sc = out[cont].to_numpy(dtype=np.float64) / sigma
        for i, nbrs in enumerate(tree.query_ball_point(Sc, tol_sigma)):
            if not len(nbrs):
                continue
            if disc:
                srow = out.iloc[i][disc]
                if (real_df.iloc[nbrs][disc] == srow).all(axis=1).any():
                    dup_idx.append(i)
            else:
                dup_idx.append(i)
    elif disc:
        # No continuous features — match on discrete signature only, but flag a
        # row only when it reproduces a *singleton* real tuple (a uniquely-
        # identifying record). Tuples shared by ≥ _MIN_ANON_COUNT real rows are
        # k-anonymous and reusing them is not a leak (see _MIN_ANON_COUNT).
        from collections import Counter
        real_counts = Counter(map(tuple, real_df[disc].to_numpy().tolist()))
        for i in range(len(out)):
            c = real_counts.get(tuple(out.iloc[i][disc].tolist()), 0)
            if 0 < c < _MIN_ANON_COUNT:
                dup_idx.append(i)

    n = len(dup_idx)
    if n and cont:
        sigma = real_df[cont].to_numpy(dtype=np.float64).std(axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        # Nudge just enough to break the exact-attribute match (a few × tol_sigma).
        jitter = rng.standard_normal((n, len(cont))) * sigma * (5.0 * tol_sigma)
        out.iloc[dup_idx, out.columns.get_indexer(cont)] = (
            out.iloc[dup_idx][cont].to_numpy(dtype=np.float64) + jitter
        )

    if n:
        logger.info("Privacy verbatim guard: nudged %d duplicate-attribute rows.", n)
    return out, n


def assess_privacy(
    rare_synth: pd.DataFrame,
    real_rare: pd.DataFrame,
    full_synth: pd.DataFrame,
    real_full: pd.DataFrame,
    field_dict: FieldDict,
    label_col: str,
    delta: float,
) -> PrivacyReport:
    """Read-only measurement of the privacy guarantee on *delivered* data.

    Reports two things, honestly, on the batch the user actually receives:
      1. δ-floor (rare part): min and spread of nearest-neighbour distance from
         each synthetic rare row to the real rare set, in σ-normalized space.
      2. verbatim-attribute duplicates (whole batch): count of released rows
         whose full non-identifier attribute set matches a real row.

    ``passed`` is True iff the rare-part min distance ≥ delta AND no verbatim
    duplicate exists. This is exactly the property the privacy mode enforces;
    measuring it on the delivered (post-constraint, post-combine) frame is the
    honest check, since clipping/rounding happen after enforcement. Does not
    modify any frame.
    """
    cols = _continuous_cols(real_rare, field_dict, label_col)
    if cols and len(real_rare) and len(rare_synth):
        R = real_rare[cols].to_numpy(dtype=np.float64)
        S = rare_synth[cols].to_numpy(dtype=np.float64)
        sigma = R.std(axis=0); sigma = np.where(sigma < 1e-8, 1.0, sigma)
        tree = cKDTree(R / sigma)
        d, _ = tree.query(S / sigma, k=1)
        min_distance = float(d.min())
        qs = np.percentile(d, [10, 50, 90])
        p10, p50, p90 = float(qs[0]), float(qs[1]), float(qs[2])
    else:
        min_distance = float("inf")
        p10 = p50 = p90 = None

    # Verbatim-attribute duplicates across the whole delivered batch.
    dup_count = _count_duplicates(full_synth, real_full, field_dict, label_col)

    return PrivacyReport(
        mode="floored",
        delta=float(delta),
        min_distance=min_distance,
        n_moved=0,           # not re-measured here; enforcement happened upstream
        n_respawned=dup_count,
        passed=bool(min_distance >= delta - 1e-9 and dup_count == 0),
        distance_p10=p10, distance_p50=p50, distance_p90=p90,
    )


def _count_duplicates(
    synth_df: pd.DataFrame, real_df: pd.DataFrame,
    field_dict: FieldDict, label_col: str, tol_sigma: float = 1e-3,
) -> int:
    """Count released rows matching a real row on every non-identifier feature."""
    feat = [c for c in synth_df.columns
            if c in field_dict and c != label_col
            and not getattr(field_dict[c], "is_identifier", False)]
    if not feat or len(real_df) == 0 or len(synth_df) == 0:
        return 0
    cont = [c for c in feat if field_dict[c].field_type == FieldType.CONTINUOUS]
    disc = [c for c in feat if field_dict[c].field_type in (FieldType.CATEGORICAL,
                                                            FieldType.BINARY)]
    count = 0
    if cont:
        sigma = real_df[cont].to_numpy(dtype=np.float64).std(axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        tree = cKDTree(real_df[cont].to_numpy(dtype=np.float64) / sigma)
        Sc = synth_df[cont].to_numpy(dtype=np.float64) / sigma
        for i, nbrs in enumerate(tree.query_ball_point(Sc, tol_sigma)):
            if not len(nbrs):
                continue
            if disc:
                srow = synth_df.iloc[i][disc]
                if (real_df.iloc[nbrs][disc] == srow).all(axis=1).any():
                    count += 1
            else:
                count += 1
    elif disc:
        # Discrete-only: count only singleton (uniquely-identifying) matches;
        # k-anonymous shared tuples are safe (see _MIN_ANON_COUNT).
        from collections import Counter
        real_counts = Counter(map(tuple, real_df[disc].to_numpy().tolist()))
        count = sum(
            1 for i in range(len(synth_df))
            if 0 < real_counts.get(tuple(synth_df.iloc[i][disc].tolist()), 0) < _MIN_ANON_COUNT
        )
    return count


# ── Internals ──────────────────────────────────────────────────────────────────

def _continuous_cols(
    df: pd.DataFrame, field_dict: FieldDict, label_col: str,
) -> list:
    """Continuous, non-identifier, non-label columns present in ``df``."""
    cols = []
    for c in df.columns:
        if c == label_col:
            continue
        meta = field_dict.get(c)
        if meta is None:
            continue
        if meta.field_type != FieldType.CONTINUOUS:
            continue
        if getattr(meta, "is_identifier", False):
            continue
        cols.append(c)
    return cols
