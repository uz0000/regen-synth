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
    within 1 standard deviation per feature (L∞ ball). This is checked
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


# ── Public API ─────────────────────────────────────────────────────────────────

def audit(
    ingest: IngestResult,
    synthetic_df: pd.DataFrame,
    config: AuditorConfig,
    manifest: Optional[BatchManifest] = None,
) -> FidelityReport:
    """
    Gate the synthetic batch. Returns a FidelityReport.

    If FidelityReport.overall_passed is False, the batch must be discarded.
    The caller (Orchestrator) is responsible for not passing failed batches
    to the Examiner — do not route around this gate.

    Args:
        ingest:       IngestResult (uses normal_df + rare_df for reference).
        synthetic_df: Generated synthetic records.
        config:       AuditorConfig with thresholds.
        manifest:     BatchManifest to embed in the report (optional).

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
    # If there are no rare rows, fall back to the full distribution.
    if rare_df is not None and len(rare_df) > 0:
        reference_df = rare_df
    else:
        reference_df = pd.concat([ingest.normal_df, ingest.rare_df], ignore_index=True)

    # ── PRIMARY GATE: rare event coverage ────────────────────────────────────
    if rare_df is not None and len(rare_df) > 0:
        coverage_rate = _coverage_rate(rare_df, synthetic_df, field_dict)
    else:
        coverage_rate = 1.0
        logger.info("No rare_df — skipping coverage check.")

    coverage_passed = coverage_rate >= config.coverage_threshold
    if not coverage_passed:
        logger.warning(
            "Auditor: coverage FAILED %.3f < %.3f",
            coverage_rate, config.coverage_threshold,
        )

    # ── Per-column TVD / Wasserstein vs the rare reference ────────────────────
    shared = [c for c in reference_df.columns if c in synthetic_df.columns and c in field_dict]
    col_results: List[ColumnFidelity] = []
    for col in shared:
        col_results.append(
            _eval_column(col, reference_df[col], synthetic_df[col],
                         field_dict[col].field_type, config)
        )

    col_failed    = any(not r.passed for r in col_results)
    overall_passed = coverage_passed and not col_failed

    report = FidelityReport(
        coverage_rate=coverage_rate,
        coverage_passed=coverage_passed,
        column_results=col_results,
        overall_passed=overall_passed,
        n_real=len(reference_df),
        n_synthetic=len(synthetic_df),
        manifest=manifest,
    )
    _log_summary(report, config)
    return report


# ── Coverage rate ─────────────────────────────────────────────────────────────

def _coverage_rate(
    rare_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    field_dict: FieldDict,
) -> float:
    """
    Fraction of real rare events covered by the synthetic batch.

    A rare event r is "covered" if there exists a synthetic row s such that
    for all numeric features f:  |r[f] - s[f]| ≤ σ_f  (L∞ ball, radius=1σ).
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

    covered = sum(
        1 for r_row in R_n
        if np.abs(S_n - r_row).max(axis=1).min() <= 1.0
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
        t = _tvd_discrete(real, synth)
        result.tvd = t
        if t > config.tvd_threshold:
            result.passed = False

    return result


# ── TVD ───────────────────────────────────────────────────────────────────────

def _tvd_discrete(real: pd.Series, synth: pd.Series) -> float:
    all_vals = set(real.dropna().unique()) | set(synth.dropna().unique())
    nr, ns = len(real.dropna()), len(synth.dropna())
    if nr == 0 or ns == 0:
        return 1.0
    total = sum(abs((real == v).sum() / nr - (synth == v).sum() / ns) for v in all_vals)
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
    logger.info(
        "Auditor %s | coverage=%.3f (thr=%.2f) | %d/%d columns passed",
        status, report.coverage_rate, config.coverage_threshold, n_pass, n_tot,
    )
    worst = sorted(
        (r for r in report.column_results if r.tvd is not None),
        key=lambda r: r.tvd or 0, reverse=True,
    )
    if worst:
        logger.info("Worst TVD: '%s' = %.4f", worst[0].col, worst[0].tvd)
