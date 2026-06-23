"""
REGEN API — unified entry point for CLI, server, and demo scripts.

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

def _apply_domain_constraints(df: pd.DataFrame, ingest: IngestResult) -> pd.DataFrame:
    """Coerce synthetic columns back onto their real-world support.

    The Prior (Gaussian) and the residual GP both sample on an unbounded, real-
    valued line, so a synthetic row can land outside what any real row could be.
    Per column type we fix:

      * CONTINUOUS — clip to the ingest-observed [min_val, max_val] (field_dict
        spans the full dataset, so the rare tail's extremes are preserved). Kills
        impossible values like a negative transaction Amount/Time. If the source
        column was integer-valued (counts, hour, Time), round back to whole
        numbers — emitting hour=11.97 or n_prior_txns=25.8 is visibly fake and
        shifts the discrete marginal.
      * BINARY — round to the nearest of the two observed values (the residual GP
        perturbs these too, so a 0/1 column can drift to 0.7 / -0.3).

    Categorical columns are handled in _decode_categoricals; the label is constant
    and skipped. Folding out-of-support mass back in tightens fidelity, not loosens.
    """
    fd = ingest.field_dict
    rare_df = ingest.rare_df
    for col in df.columns:
        meta = fd.get(col)
        if meta is None or col == ingest.label_col:
            continue
        if meta.field_type == FieldType.CONTINUOUS:
            if meta.min_val is not None and meta.max_val is not None:
                df[col] = df[col].clip(meta.min_val, meta.max_val)
            if meta.is_integer:
                df[col] = df[col].round().astype("int64")
        elif meta.field_type == FieldType.BINARY:
            # Snap to the two real values (handles {0,1} and any other binary pair).
            vals = pd.unique(rare_df[col].dropna()) if col in rare_df.columns else np.array([0, 1])
            if len(vals) >= 2:
                lo, hi = sorted(vals[:2])
                mid = (float(lo) + float(hi)) / 2.0
                df[col] = np.where(df[col].to_numpy() >= mid, hi, lo)
            elif len(vals) == 1:
                df[col] = vals[0]
    return df


def _decode_categoricals(df: pd.DataFrame, ingest: IngestResult) -> pd.DataFrame:
    """Restore original categorical values in a synthetic batch.

    The engine encodes categorical columns to integer codes for numerical
    computation (Prior, Amplifier, Examiner). The synthetic batch comes out
    as encoded floats (e.g. 3.0, 7.0). This decodes them back to original
    values (e.g. "management", "technician") so that:

    1. The Auditor compares apples to apples (synthetic strings vs real strings)
    2. The output is human-usable (clients need real values, not integer codes)

    Decoding uses the canonical category order stored in field_dict (computed
    from the FULL dataset at ingest) — the same mapping _encode_features uses —
    so a code always round-trips to the value the Prior encoded it from. Using
    the rare subset's own categories (as before) silently mislabeled values
    whenever the rare rows didn't cover every category.
    """
    field_dict = ingest.field_dict
    rare_df = ingest.rare_df
    for col in df.columns:
        if col not in field_dict:
            continue
        ftype = field_dict[col].field_type
        if ftype not in (FieldType.CATEGORICAL,):
            continue
        cats = field_dict[col].categories
        if cats is None:  # defensive fallback for an externally-built field_dict
            cats = list(pd.Categorical(rare_df[col]).categories)
        # Round to nearest integer code and clip to the valid range.
        codes = df[col].round().astype(int).clip(0, len(cats) - 1)
        df[col] = pd.Categorical.from_codes(codes, categories=cats).astype(object)
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
# 1b. ONE GENERATION PASS  (shared by run_campaign and generate)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_one_pass(
    result: IngestResult,
    prior_cfg,
    amp_cfg,
    aud_cfg,
    exam_cfg,
    scout_cfg,
    seed: int,
    n_rows: int,
    label_col: str,
    rare_def: RareEventDef,
    explored_points: list,
):
    """Run one full generation pass: Prior → Scout → Amplifier → Auditor → Examiner.

    This is the atomic unit of both the multi-pass campaign loop and the
    single-shot generate() path. It is a pure function of (ingest, configs,
    seed) — same inputs → identical batch (INVARIANTS.md Invariant 2).

    The Auditor is a hard gate: the Examiner runs only on a passing batch
    (Invariant 3), so `lift` is None when fidelity fails.

    Args:
        explored_points: mutated in place — Scout appends its target anchor so
            cross-pass memory accumulates for the caller (empty for single-shot).

    Returns:
        (amp_df, fidelity_report, lift_report_or_None, target_region)
    """
    from engine.prior import fit_prior, generate_base_batch
    from engine.amplifier import fit_residuals, sample_residuals
    from engine.auditor import audit
    from engine.examiner import measure_lift
    from engine.scout import select_target

    rng = np.random.default_rng(seed)

    # Prior + Amplifier are fit on the ingest data (not the generated batch)
    prior = fit_prior(result, prior_cfg, rng)
    residual = fit_residuals(result, prior, amp_cfg)

    # Scout: R-EPIG target selection. explored_points lets multi-pass runs
    # down-weight already-mapped anchors so budget goes to new tail structure.
    target_region = select_target(
        residual, prior._feature_cols, rng, scout_cfg,
        explored_points=explored_points or None,
    )
    if target_region.get("candidate_point"):
        explored_points.append(target_region["candidate_point"])

    # Prior Engine: base batch in the targeted region
    base = generate_base_batch(prior, n_rows, target_region, rng,
                               noise_scale=prior_cfg.noise_scale)

    # Amplifier: ResidualGP tail correction
    rng2 = np.random.default_rng(seed)
    _, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng2)
    amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)

    # Constrain to observed support: the Prior + residual GP can sample past the
    # real range, producing impossible values for bounded columns (e.g. negative
    # Amount/Time). Clip every continuous column to its ingest-observed [min,max].
    amp_df = _apply_domain_constraints(amp_df, result)

    # Attach the label. The Prior generates feature columns only (it excludes the
    # label), so the batch arrives unlabeled — but every row is the amplified rare
    # class, so it must carry the rare label to be usable downstream. Source the
    # value from the real rare rows (robust to auto-detected rare values).
    if label_col and label_col in result.rare_df.columns and len(result.rare_df):
        amp_df[label_col] = result.rare_df[label_col].mode().iloc[0]

    # Decode categoricals back to real values so the Auditor compares apples to apples
    amp_df = _decode_categoricals(amp_df, result)

    # Auditor: fidelity gate
    report = audit(result, amp_df, aud_cfg)

    # Examiner: lift, only on a passing batch (Invariant 3)
    lift = measure_lift(result, amp_df, exam_cfg) if report.overall_passed else None
    return amp_df, report, lift, target_region


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
    noise_scale: float = 0.10,
) -> CampaignResult:
    """Run a full multi-pass REGEN amplification campaign.

    The active-learning loop:
        Ingest → (Prior → Amplifier → Auditor → Examiner → Scout) × max_passes

    Every value is produced by the deterministic engine. The API sequences
    the passes, gates batches, and returns the structured result.

    Args:
        filepath: Path to input data (CSV/JSON/Parquet).
        label_col: Label column name.
        rare_def: How to identify rare events.
        seed: Base RNG seed (incremented per pass).
        n_rows: Batch size per pass.
        max_passes: Maximum number of amplification passes.
        out_dir: Output directory for Parquet files. Auto-tempdir if None.
        coverage_threshold: Auditor coverage gate threshold.
        gp_noise: GP noise variance.
        max_features: GP input dim (0 = all). Set 6-10 for high-dim data.
        n_estimators: Number of trees in Examiner's RandomForest.
        num_candidates: Candidate pool size for Scout.
        noise_scale: Prior perturbation scale (fraction of rare std-dev).

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
    prior_cfg = PriorConfig(noise_scale=noise_scale)
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
        amp_df, report, lift, _ = _run_one_pass(
            result, prior_cfg, amp_cfg, aud_cfg, exam_cfg, scout_cfg,
            seed + pass_num, n_rows, label_col, rare_def, explored_points,
        )

        if not report.overall_passed:
            n_rejected += 1
            passes.append(PassDetail(
                pass_num=pass_num + 1,
                status="rejected",
                coverage=report.coverage_rate,
            ))
            continue

        # lift was measured inside _run_one_pass (only on Auditor-passing batches)
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
# 2b. GENERATE — simple primary path with auto-tuning
# ═══════════════════════════════════════════════════════════════════════════════

# Noise candidates the auto-tuner searches. noise_scale is the highest-leverage
# knob (a 0.25→0.10 tune swung results ~23% in the breadth benchmark), so the
# search is concentrated there. Which column/region to amplify is left to Scout,
# which already ranks features by rarity-relevance automatically.
_NOISE_CANDIDATES = (0.05, 0.08, 0.10, 0.13, 0.16, 0.20)

# Objective modes. fidelity = make the closest statistical copy (model-agnostic).
# balanced = maximize detection-lift subject to passing the fidelity gate.
# boost    = same objective, looser gate (more amplification, more lift, more distortion).
_GENERATE_MODES = ("faithful", "balanced", "boost")


def _fidelity_score(report) -> float:
    """Scalar fidelity in [0,1]: fraction of columns whose distribution matches.

    Higher = the synthetic batch preserves more of the real per-column
    distributions. This is the objective for 'faithful' mode and a tie-breaker
    signal otherwise. (Distinct from coverage_rate, which measures rare-region
    coverage, not distributional match.)
    """
    cols = report.column_results
    if not cols:
        return 0.0
    return sum(1 for c in cols if c.passed) / len(cols)


def _autotune(
    result: IngestResult,
    rare_def: RareEventDef,
    mode: str,
    seed: int,
    coverage_threshold: float,
    n_rows: int,
):
    """Search noise_scale against the mode's objective; return (best_noise, trail).

    Each candidate is one generation pass evaluated at the *target* n_rows.
    This is nearly free relative to a tiny eval batch because the GP fit (the
    expensive part) scales with the rare-training-set size, not n_rows — so we
    rank configs at the same size they'll actually run at, avoiding the
    classic tune-cheap/run-full mismatch.

    Scoring:
      faithful → fidelity score (closest statistical copy)
      balanced → detection lift, rejecting any config that fails the gate
      boost    → detection lift, same as balanced (the looser gate is the lever)

    The trail (one entry per candidate) lets the UI draw the fidelity-vs-lift
    frontier so the user can see *why* a config was chosen.

    Returns:
        (best_noise_scale, [{'noise', 'fidelity', 'lift', 'passed'}, ...])
    """
    from engine.prior import PriorConfig
    from engine.amplifier import AmplifierConfig
    from engine.auditor import AuditorConfig
    from engine.examiner import ExaminerConfig
    from engine.scout import ScoutConfig

    amp_cfg = AmplifierConfig(gp_noise_variance=0.1)
    aud_cfg = AuditorConfig(coverage_threshold=coverage_threshold)
    exam_cfg = ExaminerConfig(n_estimators=60)
    scout_cfg = ScoutConfig(num_candidates=80)
    label_col = result.label_col

    trail = []
    best = None
    for i, noise in enumerate(_NOISE_CANDIDATES):
        prior_cfg = PriorConfig(noise_scale=noise)
        try:
            _, report, lift, _ = _run_one_pass(
                result, prior_cfg, amp_cfg, aud_cfg, exam_cfg, scout_cfg,
                seed + 7000 + i, n_rows, label_col, rare_def, [],
            )
        except Exception:
            # A candidate that errors is treated as a failure, not a crash.
            trail.append({"noise": noise, "fidelity": 0.0, "lift": None, "passed": False})
            continue

        fid = round(_fidelity_score(report), 4)
        passed = bool(report.overall_passed)
        lift_v = round(lift.tail_lift, 4) if lift is not None else None
        trail.append({"noise": noise, "fidelity": fid, "lift": lift_v, "passed": passed})

        if mode == "faithful":
            score = fid
        else:  # balanced / boost: lift subject to the fidelity floor
            score = lift.tail_lift if passed else -1.0

        if best is None or score > best[0]:
            best = (score, noise)

    best_noise = best[1] if best else 0.10
    return best_noise, trail


def generate(
    filepath: str,
    label_col: str = "",
    rare_def: Optional[RareEventDef] = None,
    n_rows: int = 300,
    mode: str = "balanced",
    seed: int = 42,
    auto: bool = True,
    noise_scale: Optional[float] = None,
    coverage_threshold: Optional[float] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """The simple primary path: generate a synthetic dataset as a CSV-ready batch.

    This is the user-facing entry point that hides REGEN's technical knobs.
    The user supplies data + how many rows they want (+ an optional intent);
    the system auto-detects the rare class, auto-tunes noise and target region,
    and returns a fidelity-checked synthetic batch plus both quality numbers
    (fidelity + detection lift).

    Two operating points (INVARIANTS.md §4 / design notes):
      * faithful — maximize distributional fidelity. Model-agnostic: the result
        is just a faithful synthetic copy, useful for any downstream model.
      * balanced / boost — maximize detection-lift subject to the fidelity gate.
        The internal Examiner (a Random Forest) is a generic tabular-classifier
        proxy; lift optimized for it transfers well to tree/linear models.

    Manual choice of the target is optional. Leave label_col / rare_def open and
    the ingest layer detects them *structurally* — the most imbalanced
    low-cardinality column, and its minority class as the rare value. Detection
    is model-agnostic on purpose (detection lift depends on the downstream model,
    so it must not decide *what* the rare event is). If two columns are equally
    plausible targets the loader raises AmbiguousTargetError rather than guessing;
    pass label_col explicitly to resolve. What was detected comes back in
    summary["detection"] for display/override.

    Args:
        filepath: Input CSV/JSON/Parquet.
        label_col: Label column. "" → structurally auto-detect the target column.
        rare_def: How rare events are identified. None → auto (minority class of
            the resolved label column). Pass an explicit value to override.
        n_rows: How many synthetic rows to generate. The one knob the user owns.
        mode: "faithful" | "balanced" | "boost". Default "balanced".
        seed: RNG seed (full reproducibility).
        auto: If True, auto-tune noise via a small search. If False, use noise_scale.
        noise_scale: Manual prior-noise (used when auto=False; default 0.10).
        coverage_threshold: Auditor gate strictness. None → 0.30 for boost, else 0.50.
        out_dir: Where to persist the batch + summary. Auto-tempdir if None.

    Returns:
        Dict with: run_id, n_rows, label_col, rare info, detection (what was
        auto-selected, or None if fully manual), fidelity (per-column + score),
        lift (or None), config_used, and the auto-tune candidate trail.
        The batch itself is saved to out_dir/pass_1_accepted.parquet and is
        retrievable via get_results()/load_synthetic().
    """
    import tempfile
    from engine.prior import PriorConfig
    from engine.amplifier import AmplifierConfig
    from engine.auditor import AuditorConfig
    from engine.examiner import ExaminerConfig
    from engine.scout import ScoutConfig

    if mode not in _GENERATE_MODES:
        raise ValueError(f"mode must be one of {_GENERATE_MODES}, got {mode!r}")

    # 1. Ingest + auto-detect the rare class if the caller didn't specify
    rare_def = rare_def or _auto_rare_def()
    result = ingest(filepath, label_col, rare_def)

    if coverage_threshold is None:
        coverage_threshold = 0.30 if mode == "boost" else 0.50

    out_path = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="regen_generate_"))
    out_path.mkdir(parents=True, exist_ok=True)

    # 2. Auto-tune (or use the manual noise)
    if auto:
        best_noise, trail = _autotune(result, rare_def, mode, seed, coverage_threshold, n_rows)
    else:
        best_noise = 0.10 if noise_scale is None else noise_scale
        trail = []

    # 3. Final generation at the requested size with the chosen config
    prior_cfg = PriorConfig(noise_scale=best_noise)
    amp_cfg = AmplifierConfig(gp_noise_variance=0.1)
    aud_cfg = AuditorConfig(coverage_threshold=coverage_threshold)
    exam_cfg = ExaminerConfig(n_estimators=100)
    scout_cfg = ScoutConfig(num_candidates=100)

    amp_df, report, lift, _ = _run_one_pass(
        result, prior_cfg, amp_cfg, aud_cfg, exam_cfg, scout_cfg,
        seed + 9000, n_rows, result.label_col, rare_def, [],
    )

    # 4. Persist the batch (reuses the campaign on-disk layout so the existing
    #    get_results / load_synthetic / download endpoints work unchanged).
    batch_path = str(out_path / "pass_1_accepted.parquet")
    amp_df.to_parquet(batch_path, index=False)

    fidelity = {
        "score": round(_fidelity_score(report), 4),
        "passed": bool(report.overall_passed),
        "coverage": round(report.coverage_rate, 4),
        "columns": [
            {
                "col": c.col,
                "passed": bool(c.passed),
                "metric": "wasserstein" if c.wasserstein is not None else "tvd",
                "value": round(c.wasserstein, 4) if c.wasserstein is not None
                else (round(c.tvd, 4) if c.tvd is not None else None),
            }
            for c in report.column_results
        ],
    }
    lift_out = None
    if lift is not None:
        lift_out = {
            "tail_lift": round(lift.tail_lift, 4),
            "baseline_recall": round(lift.baseline_recall, 4),
            "amplified_recall": round(lift.amplified_recall, 4),
        }

    summary = {
        "run_id": out_path.name,
        "n_rows": len(amp_df),
        "label_col": result.label_col,
        "n_normal": len(result.normal_df),
        "n_rare": len(result.rare_df),
        "n_features": len(result.field_dict) - (1 if result.label_col else 0),
        # What the system chose for you (and what you could override). None when
        # both label_col and rare value were supplied explicitly.
        "detection": result.detection.as_dict() if result.detection else None,
        "fidelity": fidelity,
        "lift": lift_out,
        "config_used": {
            "mode": mode,
            "noise_scale": round(best_noise, 4),
            "coverage_threshold": coverage_threshold,
            "auto": auto,
        },
        "candidates": trail,
        "output_dir": str(out_path),
        "best_batch_path": batch_path,
    }
    # Write a minimal campaign_summary.json so get_results()/download work.
    _save_generate_summary(summary, out_path)
    return summary


def _save_generate_summary(summary: Dict[str, Any], out_path: Path) -> None:
    """Persist a campaign-shaped summary so get_results()/load_synthetic() work."""
    cr_like = {
        "best_lift": (summary["lift"]["tail_lift"] if summary["lift"] else 0.0),
        "passes": [{
            "pass_num": 1,
            "status": "accepted" if summary["fidelity"]["passed"] else "rejected",
            "tail_lift": summary["lift"]["tail_lift"] if summary["lift"] else 0.0,
            "coverage": summary["fidelity"]["coverage"],
        }],
        "n_accepted": 1 if summary["fidelity"]["passed"] else 0,
        "n_rejected": 0 if summary["fidelity"]["passed"] else 1,
        "n_normal": summary["n_normal"],
        "n_rare": summary["n_rare"],
        "n_features": summary["n_features"],
        "n_rows_per_pass": summary["n_rows"],
        "best_batch_path": summary["best_batch_path"],
    }
    with open(out_path / "campaign_summary.json", "w") as f:
        json.dump(cr_like, f, indent=2)


def _auto_rare_def() -> RareEventDef:
    """Default rare definition: label mode with the minority class as rare.

    label_value=None signals the loader to auto-detect the rare class
    structurally (minority class of the resolved label column) rather than
    assuming a fixed encoding like 1. See loader._resolve_target.
    """
    return RareEventDef(mode=RareMode.LABEL, label_value=None)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SCREEN — win-boundary predictor
# ═══════════════════════════════════════════════════════════════════════════════

# Decision threshold for Fisher discriminant CV (supplementary signal only).
# The primary screen runs a quick REGEN vs SMOTE head-to-head.
_HETEROGENEITY_THRESHOLD = 0.5


def screen(
    filepath: str,
    label_col: str,
    rare_def: RareEventDef,
    seed: int = 42,
    quick_campaign: bool = True,
) -> ScreenResult:
    """Predict which method (REGEN or SMOTE) is likely to win on this data.

    Runs a quick head-to-head: one REGEN pass vs one SMOTE pass on the same
    data split, with matched synthetic row budget. The winner is the
    recommendation. Also computes a Fisher discriminant CV as supplementary
    context (how much features vary in informativeness).

    This is slower than a pure metric-based screen (~5-15 seconds) but
    actually reliable — it measures real lift on the user's data rather
    than predicting from proxy statistics.

    Args:
        filepath: Path to input data (CSV/JSON/Parquet).
        label_col: Label column name.
        rare_def: How to identify rare events.
        seed: RNG seed for reproducibility.
        quick_campaign: If True (default), runs the head-to-head comparison.
            If False, uses Fisher CV metric only (less reliable).

    Returns:
        ScreenResult with recommendation, heterogeneity score, etc.
    """
    from engine.prior import PriorConfig, fit_prior, generate_base_batch
    from engine.amplifier import AmplifierConfig, fit_residuals, sample_residuals
    from engine.auditor import AuditorConfig, audit
    from engine.examiner import ExaminerConfig, measure_lift
    from engine.scout import ScoutConfig, select_target

    # 1. Ingest
    result = ingest(filepath, label_col, rare_def)

    # 2. Compute Fisher discriminant CV (supplementary metric)
    heterogeneity_score = _compute_fisher_cv(result)

    # 3. Run quick head-to-head: REGEN 1-pass vs SMOTE 1-pass
    regen_lift = 0.0
    smote_lift = 0.0
    n_synthetic = 200

    if quick_campaign:
        # --- REGEN 1-pass ---
        try:
            prior_cfg = PriorConfig()
            amp_cfg = AmplifierConfig(
                gp_noise_variance=0.1,
                max_features=min(10, len(result.field_dict) - 1) if (len(result.field_dict) - 1) > 10 else 0,
            )
            rng = np.random.default_rng(seed)
            prior = fit_prior(result, prior_cfg, rng)
            residual = fit_residuals(result, prior, amp_cfg)

            target = select_target(
                residual, prior._feature_cols, rng, ScoutConfig(),
                explored_points=None,
            )
            base = generate_base_batch(prior, n_synthetic, target, rng,
                                        noise_scale=prior_cfg.noise_scale)
            rng2 = np.random.default_rng(seed + 1)
            _, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng2)
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
                regen_lift = lift.tail_lift
        except Exception:
            regen_lift = 0.0

        # --- SMOTE 1-pass (matched budget) ---
        try:
            smote_lift = _quick_smote(result, label_col, n_synthetic, seed)
        except Exception:
            smote_lift = 0.0

    # 4. Decision: winner of the head-to-head
    if quick_campaign and (regen_lift != 0.0 or smote_lift != 0.0):
        if regen_lift >= smote_lift:
            recommended_method = "REGEN"
            rationale = (
                f"Quick head-to-head: REGEN +{regen_lift:.1%} vs SMOTE +{smote_lift:.1%} "
                f"(1 pass each). Fisher CV = {heterogeneity_score:.2f}."
            )
            predicted_lift_band = f"+{regen_lift:.1%} (1-pass sample; multi-pass campaigns typically improve on this)"
        else:
            recommended_method = "SMOTE"
            rationale = (
                f"Quick head-to-head: REGEN +{regen_lift:.1%} vs SMOTE +{smote_lift:.1%} "
                f"(1 pass each). Fisher CV = {heterogeneity_score:.2f}."
            )
            predicted_lift_band = f"SMOTE +{smote_lift:.1%} (1-pass sample)"
    else:
        # Fallback: Fisher CV only
        if heterogeneity_score >= _HETEROGENEITY_THRESHOLD:
            recommended_method = "REGEN"
            rationale = (
                f"Fisher discriminant CV = {heterogeneity_score:.2f} — features "
                f"vary in informativeness. (Head-to-head unavailable.)"
            )
            predicted_lift_band = "+5% to +35% (typical range across breadth benchmark)"
        else:
            recommended_method = "SMOTE"
            rationale = (
                f"Fisher discriminant CV = {heterogeneity_score:.2f} — features "
                f"are homogeneous or redundant. (Head-to-head unavailable.)"
            )
            predicted_lift_band = "SMOTE comparably or better (breadth benchmark)"

    # 5. Confidence: margin of victory in the head-to-head
    if quick_campaign and (regen_lift != 0.0 or smote_lift != 0.0):
        margin = abs(regen_lift - smote_lift)
        total = abs(regen_lift) + abs(smote_lift) + 1e-8
        confidence = round(min(margin / total, 1.0), 4)
    else:
        distance = abs(heterogeneity_score - _HETEROGENEITY_THRESHOLD)
        confidence = round(min(distance / max(_HETEROGENEITY_THRESHOLD, 1e-8), 1.0), 4)

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


def _quick_smote(ingest_result: IngestResult, label_col: str,
                 n_synthetic: int, seed: int) -> float:
    """Run a quick SMOTE baseline with matched synthetic budget. Returns lift."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score
    from imblearn.over_sampling import SMOTE

    normal_df = ingest_result.normal_df
    rare_df = ingest_result.rare_df
    feat = [c for c in normal_df.columns if c != label_col]

    def _encode(df):
        out = df[feat].copy()
        for col in out.columns:
            if out[col].dtype == object or str(out[col].dtype) == "category":
                out[col] = pd.Categorical(out[col]).codes.astype(np.float64)
            elif out[col].dtype == bool:
                out[col] = out[col].astype(np.float64)
            else:
                out[col] = out[col].astype(np.float64)
        return out.values

    Xn = _encode(normal_df)
    Xr = _encode(rare_df)
    rng = np.random.RandomState(seed)
    if len(Xn) > 10000:
        Xn = Xn[rng.choice(len(Xn), 10000, replace=False)]

    Xnt, Xnet, _, _ = train_test_split(Xn, np.zeros(len(Xn)), test_size=0.3, random_state=seed)
    Xrt, Xret, _, _ = train_test_split(Xr, np.ones(len(Xr)), test_size=0.3, random_state=seed)

    Xtr = np.vstack([Xnt, Xrt])
    ytr = np.concatenate([np.zeros(len(Xnt)), np.ones(len(Xrt))])
    Xte = np.vstack([Xnet, Xret])
    yte = np.concatenate([np.zeros(len(Xnet)), np.ones(len(Xret))])

    base = RandomForestClassifier(50, class_weight="balanced", random_state=seed)
    base.fit(Xtr, ytr)
    base_r = float(recall_score(yte, base.predict(Xte), zero_division=0))

    rare_c = int((ytr == 1).sum())
    if n_synthetic == 0:
        return 0.0
    sm = SMOTE(random_state=seed, sampling_strategy={1: rare_c + n_synthetic})
    Xsm, ysm = sm.fit_resample(Xtr, ytr)
    clf = RandomForestClassifier(50, class_weight="balanced", random_state=seed)
    clf.fit(Xsm, ysm)
    sm_r = float(recall_score(yte, clf.predict(Xte), zero_division=0))
    return sm_r - base_r


def _compute_fisher_cv(ingest_result: IngestResult) -> float:
    """Compute the coefficient of variation of per-feature Fisher discriminant scores.

    Fisher score for feature x:
        Fisher(x) = (μ_rare - μ_normal)² / (σ_rare² + σ_normal²)

    Returns the CV (σ/μ) of the Fisher scores across all features. Higher CV
    means features vary more in how well they separate rare from normal.
    """
    from engine.prior.rdbpfn import _encode_features

    normal_df = ingest_result.normal_df
    rare_df = ingest_result.rare_df
    label_col = ingest_result.label_col

    feature_cols = [c for c in normal_df.columns if c != label_col]
    if len(feature_cols) == 0:
        return 0.0

    X_normal = _encode_features(normal_df[feature_cols]).astype(np.float64)
    X_rare = _encode_features(rare_df[feature_cols]).astype(np.float64)

    scores = np.zeros(len(feature_cols))
    for i in range(len(feature_cols)):
        n_col = X_normal[:, i]
        r_col = X_rare[:, i]
        mu_n, mu_r = n_col.mean(), r_col.mean()
        var_n, var_r = n_col.var() + 1e-8, r_col.var() + 1e-8
        scores[i] = (mu_r - mu_n) ** 2 / (var_n + var_r)

    mean_score = scores.mean()
    if mean_score < 1e-8:
        return 0.0
    cv = float(np.std(scores) / mean_score)
    return round(cv, 4)


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