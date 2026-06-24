"""
Fidelity Auditor — hard gate.

Checks a synthetic batch against reference statistics and rejects failures.

Metrics:
  TVD (Total Variation Distance): per column, discrete distributions.
    TVD = 0.5 * Σ |P(x) - Q(x)| ∈ [0, 1]. Lower is better.

  Wasserstein-1: earth mover's distance between empirical CDFs of
    continuous columns. Lower is better.

  Rare event coverage rate (PRIMARY METRIC):
    Fraction of real rare events that have at least one synthetic neighbor
    within √D Euclidean distance in σ-normalised space (L2 ball, radius
    = √features). The threshold scales with dimensionality so coverage
    is consistent regardless of feature count. This is checked
    first — a batch can pass bulk column metrics and still fail if it
    doesn't actually cover the tail.

Invariant (test_fidelity.py): a deliberately corrupted batch must be
rejected; a clean batch must be accepted.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from contracts.types import (
    BatchManifest,
    ColumnFidelity,
    FieldDict,
    FieldType,
    FidelityReport,
    IngestResult,
)

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AuditorConfig:
    tvd_threshold: float = 0.15           # max TVD per discrete/binary column
    # Continuous columns are gated on *normalized* Wasserstein (W / ref_std),
    # which is scale-free so one threshold works across features of any unit.
    wasserstein_threshold: float = 0.50   # max normalized Wasserstein per continuous column
    coverage_threshold: float = 0.80      # min rare event coverage rate
    # High-cardinality categorical handling: when a column has more unique values
    # than this threshold, TVD is computed over only the top-K most frequent
    # categories (rest grouped into "other"). This prevents false rejections when
    # the synthetic batch (200 rows) cannot cover 1,000+ categories.
    high_card_threshold: int = 50
    # Cross-column correlation gate: max mean-absolute difference between the real
    # and synthetic correlation matrices. This is the check that stops a batch
    # with correct marginals but broken joint structure (the gate's stated reason
    # for existing). Needs >=2 numeric columns and a few rows to estimate.
    correlation_threshold: float = 0.25


# ── Public API ─────────────────────────────────────────────────────────────────

def audit(
    ingest: IngestResult,
    synthetic_df: pd.DataFrame,
    config: AuditorConfig,
    manifest: Optional[BatchManifest] = None,
    reference_df: Optional[pd.DataFrame] = None,
    check_coverage: bool = True,
) -> FidelityReport:
    """
    Gate the synthetic batch. Returns a FidelityReport.

    If FidelityReport.overall_passed is False, the batch must be discarded.
    The caller (Orchestrator) is responsible for not passing failed batches
    to the Examiner — do not route around this gate.

    Args:
        ingest:        IngestResult (uses normal_df + rare_df for reference).
        synthetic_df:  Generated synthetic records.
        config:        AuditorConfig with thresholds.
        manifest:      BatchManifest to embed in the report (optional).
        reference_df:  Override the reference distribution. Default None →
                       rare_df (the rare-amplification case). Pass normal_df to
                       gate a synthetic *normal* batch (full-synthesis path).
        check_coverage: Coverage answers "did we densify the rare region", which
                       is meaningless for a normal sample — set False there and
                       gate on marginals + correlation only.

    Returns:
        FidelityReport — check .overall_passed before proceeding.
    """
    field_dict = ingest.field_dict
    rare_df = ingest.rare_df

    # REGEN produces rare-event amplification batches: the synthetic batch is
    # deliberately concentrated in the rare region. So the reference for both
    # coverage and per-column marginals is the rare distribution — "do these
    # synthetic rare rows look like real rare events?" Comparing against the
    # full dataset (mostly normal) would reject every valid amplification batch.
    # A caller can override the reference (e.g. the normal half of a full set).
    if reference_df is not None:
        pass
    elif rare_df is not None and len(rare_df) > 0:
        reference_df = rare_df
    else:
        reference_df = pd.concat([ingest.normal_df, ingest.rare_df], ignore_index=True)

    # ── PRIMARY GATE: rare event coverage ────────────────────────────────────
    if check_coverage and rare_df is not None and len(rare_df) > 0:
        coverage_rate = _coverage_rate(reference_df, synthetic_df, field_dict)
    else:
        coverage_rate = 1.0
        if not check_coverage:
            logger.info("Coverage check disabled (non-rare reference).")
        else:
            logger.info("No rare_df — skipping coverage check.")

    coverage_passed = coverage_rate >= config.coverage_threshold
    if not coverage_passed:
        logger.warning(
            "Auditor: coverage FAILED %.3f < %.3f",
            coverage_rate, config.coverage_threshold,
        )

    # ── Per-column TVD / Wasserstein vs the rare reference ────────────────────
    shared = [c for c in reference_df.columns if c in synthetic_df.columns and c in field_dict]
    # Skip the label column — its distribution is metadata (all synthetic rows
    # are rare), not a generated property of the synthetic data. Comparing it
    # against the real rare distribution guarantees a false rejection.
    label_col = ingest.label_col
    if label_col and label_col in shared:
        shared.remove(label_col)
    col_results: List[ColumnFidelity] = []
    for col in shared:
        col_results.append(
            _eval_column(col, reference_df[col], synthetic_df[col],
                         field_dict[col].field_type, config)
        )

    col_failed    = any(not r.passed for r in col_results)

    # ── Cross-column correlation structure ───────────────────────────────────
    # Marginals + coverage can all pass while the joint structure is scrambled
    # (e.g. a column shuffled independently). This is the failure mode the gate
    # exists to stop, so check it explicitly: compare the rare-reference and
    # synthetic correlation matrices.
    correlation_delta = _correlation_delta(reference_df, synthetic_df, field_dict, label_col)
    correlation_passed = (
        correlation_delta is None or correlation_delta <= config.correlation_threshold
    )
    if not correlation_passed:
        logger.warning(
            "Auditor: correlation structure FAILED %.3f > %.3f",
            correlation_delta, config.correlation_threshold,
        )

    overall_passed = coverage_passed and not col_failed and correlation_passed

    report = FidelityReport(
        coverage_rate=coverage_rate,
        coverage_passed=coverage_passed,
        column_results=col_results,
        overall_passed=overall_passed,
        n_real=len(reference_df),
        n_synthetic=len(synthetic_df),
        manifest=manifest,
        correlation_delta=correlation_delta,
        correlation_passed=correlation_passed,
    )
    _log_summary(report, config)
    return report


# ── Correlation structure ──────────────────────────────────────────────────────

def _correlation_delta(
    reference_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    field_dict: FieldDict,
    label_col: str,
) -> Optional[float]:
    """Mean absolute difference between real and synthetic correlation matrices.

    Restricted to numeric (continuous/binary) feature columns, excluding the
    label. Returns None when there are fewer than two such columns or too few
    rows to estimate correlations — in those cases there is no joint structure
    to validate. A NaN result (constant column → undefined correlation) is
    treated as "no signal" for that pair via nan-safe differencing.
    """
    numeric_cols = [
        c for c in reference_df.columns
        if c in synthetic_df.columns and c in field_dict and c != label_col
        and field_dict[c].field_type in (FieldType.CONTINUOUS, FieldType.BINARY)
    ]
    if len(numeric_cols) < 2:
        return None
    if len(reference_df) < 3 or len(synthetic_df) < 3:
        return None

    real_corr = reference_df[numeric_cols].corr().to_numpy()
    synth_corr = synthetic_df[numeric_cols].corr().to_numpy()

    # Compare only the upper triangle (off-diagonal pairs). Where either matrix
    # is NaN (a constant column), skip that pair rather than failing on it.
    iu = np.triu_indices_from(real_corr, k=1)
    diffs = np.abs(real_corr[iu] - synth_corr[iu])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    return float(diffs.mean())


# ── Coverage rate ─────────────────────────────────────────────────────────────

def _coverage_rate(
    rare_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    field_dict: FieldDict,
) -> float:
    """
    Fraction of real rare events covered by the synthetic batch.

    A rare event r is "covered" if there exists a synthetic row s such that
    the Euclidean (L2) distance between r and s, in per-feature σ-normalised
    space, is ≤ √D (where D = number of numeric features).  The threshold
    √D is the expected L2 norm of a D-dimensional standard normal vector,
    so the radius grows naturally with dimensionality — a row that stays
    within ~1σ of the mean along every axis passes the gate regardless of
    how many features there are.
    """
    numeric_cols = [
        c for c in rare_df.columns
        if c in synthetic_df.columns and c in field_dict
        and field_dict[c].field_type in (FieldType.CONTINUOUS, FieldType.BINARY)
    ]
    if not numeric_cols:
        return 1.0

    R = rare_df[numeric_cols].values.astype(np.float64)
    S = synthetic_df[numeric_cols].values.astype(np.float64)

    sigma = R.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    R_n = R / sigma
    S_n = S / sigma

    # Coverage radius adapts to dimensionality: L2 norm / sqrt(D).
    # In D dimensions, the expected L2 norm of a standard normal vector is
    # ~sqrt(D), so a threshold of 1.0 means "within 1 sigma of the mean
    # Euclidean distance" regardless of how many features there are.
    d = R.shape[1]
    radius = np.sqrt(d)

    covered = sum(
        1 for r_row in R_n
        if np.sqrt((((S_n - r_row) ** 2).sum(axis=1))).min() <= radius
    )
    return covered / len(R)


# ── Per-column evaluation ─────────────────────────────────────────────────────

def _eval_column(
    col: str,
    real: pd.Series,
    synth: pd.Series,
    ftype: FieldType,
    config: AuditorConfig,
) -> ColumnFidelity:
    result = ColumnFidelity(col=col, passed=True)

    if ftype == FieldType.CONTINUOUS:
        # Gate on normalized Wasserstein (scale-free, robust to small samples).
        w = _wasserstein(real, synth)
        result.wasserstein = w
        if w > config.wasserstein_threshold:
            result.passed = False
        # Binned TVD is reported for visibility but not gated: with a small
        # rare reference it is too noisy to threshold reliably.
        result.tvd = _tvd_continuous(real, synth)

    elif ftype in (FieldType.CATEGORICAL, FieldType.BINARY):
        t = _tvd_discrete(real, synth, config)
        result.tvd = t
        if t > config.tvd_threshold:
            result.passed = False

    return result


# ── TVD ───────────────────────────────────────────────────────────────────────

def _tvd_discrete(real: pd.Series, synth: pd.Series, config: AuditorConfig) -> float:
    """
    Total Variation Distance between two discrete distributions.

    For high-cardinality columns (> config.high_card_threshold unique values
    in the reference), TVD is computed over only the top-K most frequent
    categories from the reference, with all remaining categories grouped into
    an "other" bucket. This prevents false rejections when a 200-row synthetic
    batch cannot cover 1,000+ categories — the check focuses on whether the
    synthetic data represents the categories that actually matter.
    """
    real_clean = real.dropna()
    synth_clean = synth.dropna()
    nr, ns = len(real_clean), len(synth_clean)
    if nr == 0 or ns == 0:
        return 1.0

    real_vals = real_clean.unique()
    n_unique = len(real_vals)

    if n_unique <= config.high_card_threshold:
        # Low cardinality — compare full distributions.
        all_vals = set(real_vals) | set(synth_clean.unique())
        total = sum(
            abs((real_clean == v).sum() / nr - (synth_clean == v).sum() / ns)
            for v in all_vals
        )
        return 0.5 * total

    # High cardinality — compare top-K categories, group rest into "other".
    # K scales with synthetic batch size: ~5 rows per category for stable
    # proportion estimates.
    k = min(n_unique, max(20, ns // 5))
    top_k = set(real_clean.value_counts().nlargest(k).index)

    real_top_mass = (real_clean.isin(top_k)).sum() / nr
    synth_top_mass = (synth_clean.isin(top_k)).sum() / ns

    total = sum(
        abs((real_clean == v).sum() / nr - (synth_clean == v).sum() / ns)
        for v in top_k
    )
    # Add the "other" bucket contribution.
    real_other = 1.0 - real_top_mass
    synth_other = 1.0 - synth_top_mass
    total += abs(real_other - synth_other)

    return 0.5 * total


def _tvd_continuous(real: pd.Series, synth: pd.Series, n_bins: int = 20) -> float:
    combined = pd.concat([real.dropna(), synth.dropna()])
    mn, mx = combined.min(), combined.max()
    if mx == mn:
        return 0.0
    bins = np.linspace(mn, mx, n_bins + 1)
    p, _ = np.histogram(real.dropna(), bins=bins)
    q, _ = np.histogram(synth.dropna(), bins=bins)
    p = p / (p.sum() + 1e-8)
    q = q / (q.sum() + 1e-8)
    return 0.5 * float(np.abs(p - q).sum())


def _wasserstein(real: pd.Series, synth: pd.Series) -> float:
    """
    Wasserstein-1 distance normalized by the reference (real) std, so the
    result is in units of standard deviations and one threshold works across
    features of any scale. A value of 0.5 means the distributions are shifted
    by half a standard deviation in earth-mover terms.
    """
    r, s = real.dropna().values, synth.dropna().values
    if len(r) == 0 or len(s) == 0:
        return 1.0
    scale = r.std()
    scale = scale if scale > 1e-8 else 1.0
    return float(wasserstein_distance(r, s) / scale)


# ── Summary logging ───────────────────────────────────────────────────────────

def _log_summary(report: FidelityReport, config: AuditorConfig) -> None:
    status = "PASSED" if report.overall_passed else "FAILED"
    n_pass = sum(1 for r in report.column_results if r.passed)
    n_tot  = len(report.column_results)
    corr = "n/a" if report.correlation_delta is None else f"{report.correlation_delta:.3f}"
    logger.info(
        "Auditor %s | coverage=%.3f (thr=%.2f) | %d/%d columns passed | corr_delta=%s (thr=%.2f)",
        status, report.coverage_rate, config.coverage_threshold, n_pass, n_tot,
        corr, config.correlation_threshold,
    )
    worst = sorted(
        (r for r in report.column_results if r.tvd is not None),
        key=lambda r: r.tvd or 0, reverse=True,
    )
    if worst:
        logger.info("Worst TVD: '%s' = %.4f", worst[0].col, worst[0].tvd)
