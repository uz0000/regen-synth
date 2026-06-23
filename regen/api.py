"""
REGEN API — unified entry point for CLI, UI, and the agent runtime stages.

Pure deterministic Python. No LLM client, no agent runtime, no network lib.
Engine boundary (INVARIANTS.md §1/§4): the API orchestrates the engine but does
not live inside engine/ — it is the public-facing layer above it.

Methods
-------
ingest(path, label_col, rare_def) -> IngestResult
    Load, clean, and split data into normal/rare subsets.

run_campaign(path, label_col, rare_def, ...) -> CampaignResult
    Multi-pass active-learning loop (Scout × Prior × Amplifier × Auditor × Examiner).

screen(path, label_col, rare_def, ...) -> ScreenResult
    Win-boundary predictor: which method (REGEN or SMOTE) is likely to win.

get_results(run_dir) -> CampaignResult
    Read a previously saved campaign result from disk.

load_synthetic(run_dir) -> DataFrame
    Load the best accepted synthetic batch as a DataFrame.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from contracts.types import (
    CampaignResult,
    FieldDict,
    FieldMeta,
    FieldType,
    IngestResult,
    PassDetail,
    RareEventDef,
    RareMode,
    ScreenResult,
)

logger = logging.getLogger(__name__)


# ── Import guard: engine must be pure Python ──────────────────────────────────
# The boundary is enforced by tests/test_boundary.py on engine/ itself.
# This module lives outside engine/ and is allowed to reference engine packages,
# but must not import LLM/agent/network libraries.


# ═══════════════════════════════════════════════════════════════════════════════
# 0. CATEGORICAL DECODING
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_categoricals(df: pd.DataFrame, ingest: IngestResult) -> pd.DataFrame:
    """Restore original categorical values in a synthetic batch.

    The engine encodes categorical columns to integer codes for numerical
    computation (Prior, Amplifier, Examiner). The synthetic batch comes out
    as encoded floats (e.g. 3.0, 7.0). This decodes them back to original
    values (e.g. "management", "technician") so that:

    1. The Auditor compares apples to apples (synthetic strings vs real strings)
    2. The output is human-usable (clients need real values, not integer codes)

    The decoding uses the same pd.Categorical() encoding as _encode_features.
    """
    field_dict = ingest.field_dict
    rare_df = ingest.rare_df
    for col in df.columns:
        if col not in field_dict:
            continue
        ftype = field_dict[col].field_type
        if ftype not in (FieldType.CATEGORICAL,):
            continue
        # Reconstruct the category mapping from the real data
        cats = pd.Categorical(rare_df[col]).categories
        # Round to nearest integer code and clip to valid range
        codes = df[col].round().astype(int).clip(0, len(cats) - 1)
        df[col] = cats[codes]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INGEST
# ═══════════════════════════════════════════════════════════════════════════════

def ingest(
    filepath: str,
    label_col: str,
    rare_def: RareEventDef,
) -> IngestResult:
    """Load and split a dataset into normal and rare subsets.

    Thin wrapper around engine.ingest.loader.ingest. All validation,
    imputation, and rare-definition logic lives in the engine.

    Args:
        filepath: Path to CSV / JSON / Parquet.
        label_col: Label column name (or "" for auto-detect).
        rare_def: How to identify rare events.

    Returns:
        IngestResult with normal_df, rare_df, field_dict, etc.
    """
    from engine.ingest.loader import ingest as _do_ingest
    return _do_ingest(
        filepath=filepath,
        label_col=label_col,
        rare_def=rare_def,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RUN CAMPAIGN
# ═══════════════════════════════════════════════════════════════════════════════

def run_campaign(
    filepath: str,
    label_col: str,
    rare_def: RareEventDef,
    seed: int = 42,
    n_rows: int = 300,
    max_passes: int = 5,
    out_dir: Optional[str] = None,
    coverage_threshold: float = 0.80,
    gp_noise: float = 0.1,
    max_features: int = 0,
    n_estimators: int = 100,
    num_candidates: int = 100,
) -> CampaignResult:
    """Run a full multi-pass REGEN amplification campaign.

    The active-learning loop:
        Ingest → (Prior → Amplifier → Auditor → Examiner → Scout) × max_passes

    Every value is produced by the deterministic engine. The API sequences
    the passes, gates batches, and returns the structured result.

    Args:
        filepath: Path to input data (CSV/JSON/Parquet).
        label_col: Label column name.
        rare_def: Rare event definition.
        seed: Base RNG seed (incremented per pass).
        n_rows: Batch size per pass.
        max_passes: Maximum number of amplification passes.
        out_dir: Output directory for Parquet files. Auto-tempdir if None.
        coverage_threshold: Auditor coverage gate threshold.
        gp_noise: GP noise variance.
        max_features: GP input dim (0 = all). Set 6-10 for high-dim data.
        n_estimators: Number of trees in Examiner's RandomForest.
        num_candidates: Candidate pool size for Scout.

    Returns:
        CampaignResult with best_lift, pass history, output paths, etc.
    """
    from engine.ingest.loader import persist_ingest
    from engine.prior import PriorConfig, fit_prior, generate_base_batch
    from engine.amplifier import AmplifierConfig, fit_residuals, sample_residuals
    from engine.auditor import AuditorConfig, audit
    from engine.examiner import ExaminerConfig, measure_lift
    from engine.scout import ScoutConfig, select_target

    # 1. Ingest
    result = ingest(filepath, label_col, rare_def)

    # 2. Prepare output directory
    if out_dir is None:
        tmpdir = tempfile.mkdtemp(prefix="regen_campaign_")
        out_dir = tmpdir
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Persist ingest so results are self-contained
    ingest_path = str(out_path / "data")
    persist_ingest(result, ingest_path)

    # 3. Configs
    prior_cfg = PriorConfig()
    amp_cfg = AmplifierConfig(
        gp_noise_variance=gp_noise,
        max_features=max_features,
    )
    aud_cfg = AuditorConfig(coverage_threshold=coverage_threshold)
    exam_cfg = ExaminerConfig(n_estimators=n_estimators)
    scout_cfg = ScoutConfig(num_candidates=num_candidates)

    # 4. Multi-pass active-learning loop
    #
    # Architecture (INVARIANTS.md §3):
    #   Scout → Prior → Amplifier → Auditor → Examiner → (loop)
    #
    # Scout runs at the START of every pass. Its explored_points memory
    # accumulates across passes so each pass targets a different region of
    # the rare-event tail. This cross-pass targeting is what makes the loop
    # active-learning rather than independent re-generation.
    passes: list[PassDetail] = []
    best_lift = 0.0
    n_accepted = 0
    n_rejected = 0
    explored_points: list = []
    best_batch_path: Optional[str] = None

    for pass_num in range(max_passes):
        rng = np.random.default_rng(seed + pass_num)

        # Prior + Amplifier (depend only on ingest data, not the generated batch)
        prior = fit_prior(result, prior_cfg, rng)
        residual = fit_residuals(result, prior, amp_cfg)

        # Scout: R-EPIG target selection with cross-pass memory.
        # On pass 1 explored_points is empty so Scout picks the globally
        # highest-scoring region. On passes 2+ the explored penalty
        # down-weights already-mapped anchors so budget goes to new
        # tail structure.
        target_region = select_target(
            residual, prior._feature_cols, rng, scout_cfg,
            explored_points=explored_points or None,
        )
        if target_region.get("candidate_point"):
            explored_points.append(target_region["candidate_point"])

        # Prior Engine: generate base batch in the targeted region
        base = generate_base_batch(prior, n_rows, target_region, rng)

        # Amplifier: ResidualGP tail correction
        rng2 = np.random.default_rng(seed + pass_num)
        _, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng2)
        amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
        if label_col in amp_df.columns:
            amp_df[label_col] = (
                rare_def.label_value if rare_def.mode == RareMode.LABEL else 1
            )

        # Decode categorical columns back to original values so the Auditor
        # compares real values, not encoded integer codes.
        amp_df = _decode_categoricals(amp_df, result)

        # Auditor: fidelity gate
        report = audit(result, amp_df, aud_cfg)

        if not report.overall_passed:
            n_rejected += 1
            passes.append(PassDetail(
                pass_num=pass_num + 1,
                status="rejected",
                coverage=report.coverage_rate,
            ))
            continue

        # Examiner: measure detection lift
        lift = measure_lift(result, amp_df, exam_cfg)
        best_lift = max(best_lift, lift.tail_lift)
        n_accepted += 1

        passes.append(PassDetail(
            pass_num=pass_num + 1,
            status="accepted",
            tail_lift=lift.tail_lift,
            baseline_recall=lift.baseline_recall,
            amplified_recall=lift.amplified_recall,
            baseline_precision=lift.baseline_precision,
            amplified_precision=lift.amplified_precision,
        ))

        # Save batch (features only — label is inherent for rare events)
        batch_path = str(out_path / f"pass_{pass_num + 1}_accepted.parquet")
        amp_df.to_parquet(batch_path, index=False)
        best_batch_path = batch_path  # last accepted is best

    n_features = len(result.field_dict) - 1

    cr = CampaignResult(
        best_lift=best_lift,
        passes=passes,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        n_normal=len(result.normal_df),
        n_rare=len(result.rare_df),
        n_features=n_features,
        n_rows_per_pass=n_rows,
        output_dir=str(out_path),
        best_batch_path=best_batch_path,
    )

    # Persist campaign summary for later retrieval via get_results()
    _save_campaign_summary(cr, out_path)

    return cr


def _save_campaign_summary(cr: CampaignResult, out_path: Path) -> None:
    """Write campaign_summary.json to the output directory."""
    summary = {
        "best_lift": cr.best_lift,
        "n_accepted": cr.n_accepted,
        "n_rejected": cr.n_rejected,
        "n_normal": cr.n_normal,
        "n_rare": cr.n_rare,
        "n_features": cr.n_features,
        "n_rows_per_pass": cr.n_rows_per_pass,
        "best_batch_path": cr.best_batch_path,
        "passes": [
            {
                "pass_num": p.pass_num,
                "status": p.status,
                "tail_lift": p.tail_lift,
                "baseline_recall": p.baseline_recall,
                "amplified_recall": p.amplified_recall,
                "baseline_precision": p.baseline_precision,
                "amplified_precision": p.amplified_precision,
                "coverage": p.coverage,
            }
            for p in cr.passes
        ],
    }
    with open(out_path / "campaign_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SCREEN — win-boundary predictor
# ═══════════════════════════════════════════════════════════════════════════════

# Decision threshold for ARD inverse-lengthscale coefficient of variation.
# Threshold calibrated against benchmark/RESULTS_BREADTH.md (75% accuracy on
# 8 datasets). Datasets with CV >= HETEROGENEITY_THRESHOLD show measurable
# variation in feature informativeness → REGEN recommended.
# CV < threshold → features homogeneous/redundant → SMOTE recommended.
# The two known misclassifications (Satellite, Ozone) are both conservative:
# predicted SMOTE, REGEN actually won.
_HETEROGENEITY_THRESHOLD = 0.3


def screen(
    filepath: str,
    label_col: str,
    rare_def: RareEventDef,
    seed: int = 42,
    quick_campaign: bool = False,
) -> ScreenResult:
    """Predict which method (REGEN or SMOTE) is likely to win on this data.

    The core metric is the coefficient of variation (CV = σ / μ) of the
    fitted ARD kernel inverse-lengthscales from the ResidualGP. This
    measures how much features vary in informativeness:

    - **High CV** → features differ in relevance → REGEN wins (ARD can
      focus on informative features, ignore uninformative ones).
    - **Low CV** → features are homogeneous / redundant → SMOTE wins
      (nearest-neighbor interpolation on equal-weight features is better).

    The decision rule is ~75% accurate across 8 benchmark datasets
    (benchmark/RESULTS_BREADTH.md). Both misclassifications are conservative
    (predicted SMOTE, REGEN actually won).

    Args:
        filepath: Path to input data (CSV/JSON/Parquet).
        label_col: Label column name.
        rare_def: Rare event definition.
        seed: RNG seed for reproducibility.
        quick_campaign: If True, runs a single campaign pass to sharpen
            the estimate with real lift data. Default False (metric only).

    Returns:
        ScreenResult with recommendation, heterogeneity score, etc.
    """
    from engine.ingest.loader import persist_ingest
    from engine.prior import PriorConfig, fit_prior, generate_base_batch
    from engine.amplifier import AmplifierConfig, fit_residuals
    from engine.auditor import AuditorConfig, audit
    from engine.examiner import ExaminerConfig, measure_lift

    # 1. Ingest
    result = ingest(filepath, label_col, rare_def)

    # 2. Fit Prior + Amplifier (ResidualGP with ARD kernel)
    prior_cfg = PriorConfig()
    amp_cfg = AmplifierConfig(
        gp_noise_variance=0.1,
        max_features=min(10, len(result.field_dict) - 1) if (len(result.field_dict) - 1) > 10 else 0,
    )
    rng = np.random.default_rng(seed)
    prior = fit_prior(result, prior_cfg, rng)
    residual = fit_residuals(result, prior, amp_cfg)

    # 3. Extract ARD kernel lengthscales and compute heterogeneity metric
    cv, ls_min, ls_max, n_optimized = _compute_ard_cv(residual)

    heterogeneity_score = round(float(cv), 4)

    # 4. Decision rule
    if heterogeneity_score >= _HETEROGENEITY_THRESHOLD:
        recommended_method = "REGEN"
        rationale = (
            f"ARD inverse-lengthscale CV = {heterogeneity_score:.2f} — features "
            f"vary in informativeness; REGEN can exploit ARD to focus on the "
            f"informative ones."
        )
        predicted_lift_band = "+5% to +25% (typical range across breadth benchmark)"
    else:
        recommended_method = "SMOTE"
        rationale = (
            f"ARD inverse-lengthscale CV = {heterogeneity_score:.2f} — features "
            f"are homogeneous or redundant; SMOTE interpolation is competitive "
            f"or better on this data."
        )
        predicted_lift_band = "SMOTE comparably or better (breadth benchmark)"

    # 5. Confidence: distance from decision boundary, clamped [0, 1]
    distance = abs(heterogeneity_score - _HETEROGENEITY_THRESHOLD)
    confidence = round(min(distance / _HETEROGENEITY_THRESHOLD, 1.0), 4)

    # 6. Optional quick campaign to sharpen estimate
    if quick_campaign:
        try:
            rng2 = np.random.default_rng(seed + 999)
            base = generate_base_batch(prior, 200, {}, rng2)
            from engine.amplifier import sample_residuals
            rng3 = np.random.default_rng(seed + 999)
            _, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng3)
            amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
            if label_col in amp_df.columns:
                amp_df[label_col] = (
                    rare_def.label_value if rare_def.mode == RareMode.LABEL else 1
                )
            amp_df = _decode_categoricals(amp_df, result)
            aud_cfg = AuditorConfig(coverage_threshold=0.50)
            report = audit(result, amp_df, aud_cfg)
            if report.overall_passed:
                exam_cfg = ExaminerConfig(n_estimators=50)
                lift = measure_lift(result, amp_df, exam_cfg)
                # Refine lift band with actual data
                if lift.tail_lift > 0.05:
                    predicted_lift_band = f"+{lift.tail_lift:.1%} to +25% (single-pass sample)"
                elif lift.tail_lift > 0:
                    predicted_lift_band = f"{lift.tail_lift:+.1%} (single-pass sample)"
                else:
                    predicted_lift_band = f"{lift.tail_lift:+.1%} (single-pass sample — check data)"
        except Exception:
            predicted_lift_band += " (quick-campaign sample unavailable)"

    n_features = len(result.field_dict) - 1
    return ScreenResult(
        recommended_method=recommended_method,
        heterogeneity_score=heterogeneity_score,
        confidence=confidence,
        predicted_lift_band=predicted_lift_band,
        rationale=rationale,
        n_rare=len(result.rare_df),
        n_features=n_features,
    )


def _compute_ard_cv(residual) -> tuple[float, float, float, bool]:
    """Compute the coefficient of variation of ARD inverse-lengthscales.

    Returns (cv, ls_min, ls_max, optimized) where:
        cv:       Coefficient of variation of the inverse-lengthscales.
        ls_min:   Minimum fitted lengthscale.
        ls_max:   Maximum fitted lengthscale.
        optimized: Whether the GP was successfully optimized.
    """
    if not residual._gp_optimized:
        # Fallback: use variance-based relevance if GP didn't optimize
        # This still gives a signal, just from the fallback proxy.
        logger.warning(
            "GP optimization did not converge for screen(). "
            "Using variance-based relevance proxy for heterogeneity metric."
        )
        rarity = residual._gp_feature_idx.size
        if rarity > 1:
            # Use the feature variance as a proxy
            X_data = residual._X_train
            if X_data.shape[1] > 1:
                var_proxy = X_data.var(axis=0)
                cv_est = float(np.std(var_proxy) / (np.mean(var_proxy) + 1e-8))
            else:
                cv_est = 0.0
        else:
            cv_est = 0.0
        return cv_est, 0.0, 0.0, False

    try:
        ls = residual._gp.kern.lengthscale.values.copy()
        inv_ls = 1.0 / (ls + 1e-8)
        cv = float(np.std(inv_ls) / (np.mean(inv_ls) + 1e-8))
        return cv, float(ls.min()), float(ls.max()), True
    except Exception as exc:
        logger.warning("Could not extract ARD lengthscales: %s", exc)
        return 0.0, 0.0, 0.0, False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RESULTS LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def get_results(run_dir: str) -> CampaignResult:
    """Read a previously saved campaign result from a run directory.

    Scans the output directory for the campaign summary JSON written by
    run_campaign, and the saved Parquet files.

    Args:
        run_dir: Path to the campaign output directory.

    Returns:
        CampaignResult reconstructed from disk.
    """
    run_path = Path(run_dir)

    # Look for a campaign summary JSON
    summary_path = run_path / "campaign_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        passes = [PassDetail(**p) for p in data.get("passes", [])]
        return CampaignResult(
            best_lift=data.get("best_lift", 0.0),
            passes=passes,
            n_accepted=data.get("n_accepted", 0),
            n_rejected=data.get("n_rejected", 0),
            n_normal=data.get("n_normal", 0),
            n_rare=data.get("n_rare", 0),
            n_features=data.get("n_features", 0),
            n_rows_per_pass=data.get("n_rows_per_pass", 0),
            output_dir=run_dir,
            best_batch_path=data.get("best_batch_path"),
        )

    # Fallback: scan for Parquet files
    parquet_files = sorted(run_path.glob("pass_*_accepted.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No campaign data found in {run_dir}. "
            f"Run a campaign first with regen.api.run_campaign()."
        )

    passes_list: list[PassDetail] = []
    best_lift = 0.0
    best_batch_path: Optional[str] = None

    for pf in parquet_files:
        # Extract pass number from filename
        try:
            pn = int(pf.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        detail = PassDetail(pass_num=pn, status="accepted")
        passes_list.append(detail)
        best_batch_path = str(pf)

    return CampaignResult(
        best_lift=best_lift,
        passes=passes_list,
        n_accepted=len(passes_list),
        output_dir=run_dir,
        best_batch_path=best_batch_path,
    )


def load_synthetic(run_dir: str) -> pd.DataFrame:
    """Load the best accepted synthetic batch from a campaign output dir.

    Looks for the highest pass-number accepted batch, or reads
    best_batch_path from the campaign summary if available.

    Args:
        run_dir: Path to the campaign output directory.

    Returns:
        DataFrame with synthetic rare-event rows.
    """
    run_path = Path(run_dir)

    # First check the summary
    summary_path = run_path / "campaign_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        bbp = data.get("best_batch_path")
        if bbp and Path(bbp).exists():
            return pd.read_parquet(bbp)

    # Scan for accepted Parquet files, pick the last one
    parquet_files = sorted(run_path.glob("pass_*_accepted.parquet"))
    if parquet_files:
        return pd.read_parquet(str(parquet_files[-1]))

    raise FileNotFoundError(
        f"No synthetic batch found in {run_dir}. "
        f"Ensure a campaign completed with at least one accepted pass."
    )