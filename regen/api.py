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
from contracts.scenario import (
    ScenarioSpec,
    ScenarioIntent,
    ScenarioGates,
    columns_from_field_dict,
)
# Re-exported so `regen.api.preflight` works (G-E). preflight imports ingest from
# here lazily inside its function, so there is no import cycle.
from regen.preflight import preflight  # noqa: E402,F401

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

    Thin delegate to the deterministic constraint layer (engine.constraints):
    clip continuous columns to observed bounds, round integer-valued columns,
    snap binaries to their two observed values. Categoricals are decoded in
    _decode_categoricals; the label is set constant elsewhere. See
    docs/SEMANTIC_FIDELITY_PLAN.md (M1).
    """
    from engine.constraints import apply_constraints
    return apply_constraints(df, ingest)


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
        meta = field_dict[col]
        if meta.field_type not in (FieldType.CATEGORICAL,):
            continue
        if meta.is_identifier:
            continue  # already replaced with fresh unique values by the constraint layer
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
    privacy: str = "none",
    delta: float = 0.5,
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
    from engine.auditor import audit
    from engine.examiner import measure_lift

    # Deliverable batch: generated from ALL rare rows (the product should use
    # every rare example available).
    amp_df, target_region = _generate_amp_batch(
        result, prior_cfg, amp_cfg, scout_cfg, seed, n_rows, label_col, rare_def,
        explored_points, privacy=privacy, delta=delta,
    )

    # Auditor: fidelity gate
    report = audit(result, amp_df, aud_cfg)

    # Examiner: honest lift, only on a passing batch (Invariant 3). The lift
    # synthetic is regenerated from the rare TRAIN fold inside measure_lift (via
    # this closure) so it never contains perturbations of the held-out rare test
    # rows — otherwise the amplified detector is tested on near-copies of its own
    # training data and the lift is inflated. A fresh seed offset decorrelates it
    # from the deliverable draw.
    def _gen_from_train(train_ingest):
        batch, _ = _generate_amp_batch(
            train_ingest, prior_cfg, amp_cfg, scout_cfg, seed + 12345, n_rows,
            label_col, rare_def, [],
        )
        return batch

    lift = (
        measure_lift(result, exam_cfg, generate_synth_fn=_gen_from_train)
        if report.overall_passed else None
    )
    return amp_df, report, lift, target_region


def _generate_amp_batch(
    result: IngestResult,
    prior_cfg,
    amp_cfg,
    scout_cfg,
    seed: int,
    n_rows: int,
    label_col: str,
    rare_def: RareEventDef,
    explored_points: list,
    privacy: str = "none",
    delta: float = 0.5,
):
    """Generation core: Prior → Scout → Amplifier → constraints → label → decode.

    Returns (amp_df, target_region). Shared by the deliverable path and the
    honest-lift path (which calls it on a train-only IngestResult), so neither
    duplicates the generation logic. Pure function of (result, configs, seed).

    privacy="floored" switches the base from grounded sampling (real anchor +
    jitter) to parametric generation (samples from the fitted rare-class
    distribution — no copying) and enforces the δ-distance floor against the real
    rare set after the GP correction. Campaign/screen leave privacy="none".
    """
    from engine.prior import fit_prior, generate_base_batch, generate_parametric_batch
    from engine.amplifier import fit_residuals, sample_residuals
    from engine.scout import select_target
    from engine.privacy import guard_against_duplicates

    # Three independent RNG substreams from one seed. `rng` drives the prior fit,
    # Scout, and base-batch noise; `rng_res` drives residual sampling; `rng_priv`
    # drives the privacy floor/guard (when active). Spawning via SeedSequence
    # keeps them statistically independent regardless of upstream draw counts.
    rng, rng_res, rng_priv = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )

    # Prior + Amplifier are fit on the ingest data (not the generated batch)
    prior = fit_prior(result, prior_cfg, rng)
    residual = fit_residuals(result, prior, amp_cfg)

    # Scout: R-EPIG target selection. explored_points lets multi-pass runs
    # down-weight already-mapped anchors so budget goes to new tail structure.
    # (In parametric/privacy mode Scout's fine sub-region targeting is carried by
    # the GP rather than anchor selection — the whole rare class is sampled.)
    target_region = select_target(
        residual, prior._feature_cols, rng, scout_cfg,
        explored_points=explored_points or None,
    )
    if target_region.get("candidate_point"):
        explored_points.append(target_region["candidate_point"])

    # Prior: base batch in the targeted region. Privacy mode samples from the
    # fitted rare-class distribution (no real-row copying) instead of grounding
    # on real rare anchors. Falls back to grounded sampling if the class is too
    # small to fit a stable covariance.
    if privacy == "floored":
        try:
            base = generate_parametric_batch(prior, n_rows, rng, which_class="rare")
        except Exception:
            logger.warning("Parametric rare base failed; falling back to grounded.")
            base = generate_base_batch(prior, n_rows, target_region, rng,
                                       noise_scale=prior_cfg.noise_scale)
    else:
        base = generate_base_batch(prior, n_rows, target_region, rng,
                                   noise_scale=prior_cfg.noise_scale)

    # Amplifier: ResidualGP tail correction (independent substream — see above)
    _, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng_res)
    amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)

    # NOTE: the privacy δ-distance floor is intentionally NOT applied here. The
    # constraint layer + the combined-batch re-constrain (see generate) clip
    # continuous values *after* this point, which would shave a floored row back
    # inside δ and silently break the guarantee. The floor is therefore enforced
    # by generate() as the final numeric step on the delivered rare rows, where
    # nothing downstream can re-violate it. rng_priv stays reserved for the
    # per-part verbatim guard below so RNG consumption is unchanged.

    # Constrain to observed support + dtype (clip continuous, round integers, snap
    # binaries); see _apply_domain_constraints.
    amp_df = _apply_domain_constraints(amp_df, result)

    # Attach the label. The Prior generates feature columns only (it excludes the
    # label), so the batch arrives unlabeled — but every row is the amplified rare
    # class, so it must carry the rare label to be usable downstream. Source the
    # value from the real rare rows (robust to auto-detected rare values).
    if label_col and label_col in result.rare_df.columns and len(result.rare_df):
        amp_df[label_col] = result.rare_df[label_col].mode().iloc[0]

    # Decode categoricals back to real values so the Auditor compares apples to apples
    amp_df = _decode_categoricals(amp_df, result)

    # Verbatim-attribute guard: no released row duplicates a real row's full
    # (non-identifier) attribute set. Catches the measure-zero accidental copy
    # that parametric sampling can still produce. Guard against the FULL real set
    # (normal + rare), not just the rare set: a synthetic rare row could verbatim-
    # match a real *normal* row (a cross-class copy), and assess_privacy counts
    # duplicates against the full set — enforcement must be at least as wide as
    # measurement, or a batch could fail its own privacy check with no upstream
    # step that prevents it (P2-8b).
    if privacy == "floored":
        real_full = pd.concat([result.normal_df, result.rare_df], ignore_index=True)
        amp_df, _ = guard_against_duplicates(
            amp_df, real_full, result.field_dict, label_col, rng_priv,
        )
    return amp_df, target_region


def _generate_normal_batch(
    result: IngestResult,
    prior_cfg,
    seed: int,
    n_rows: int,
    label_col: str,
    privacy: str = "none",
    delta: float = 0.5,
) -> pd.DataFrame:
    """Generate n_rows synthetic *normal*-class rows for the full-dataset path.

    Mirror of the generation core, but for the majority class: fit the Prior
    (anchored on the normal covariate support, ``_X_train``) and sample from it
    — no Scout, no Amplifier (those exist to densify the rare tail, which the
    normal class does not have). Then apply the same domain constraints +
    categorical decode as the rare path, and attach the normal label so the row
    is usable downstream. Pure function of (result, config, seed).

    privacy="floored" samples parametrically (no real-row copying) and applies
    the verbatim-attribute guard. The δ-distance floor is intentionally NOT
    applied here: the normal/bulk set is dense (real rows sit ~0.3σ apart), so a
    δ-shell is infeasible and would push rows out of the distribution, destroying
    the marginal. The bulk is protected by crowd anonymity + parametric sampling
    + the duplicate guard rather than by isolation.
    """
    from engine.prior import fit_prior, generate_normal_batch, generate_parametric_batch
    from engine.privacy import guard_against_duplicates

    rng = np.random.default_rng(seed)
    prior = fit_prior(result, prior_cfg, rng)
    if privacy == "floored":
        try:
            normal_df = generate_parametric_batch(prior, n_rows, rng, which_class="normal")
        except Exception:
            logger.warning("Parametric normal base failed; falling back to grounded.")
            normal_df = generate_normal_batch(
                prior, n_rows, rng, noise_scale=prior_cfg.noise_scale,
            )
    else:
        normal_df = generate_normal_batch(
            prior, n_rows, rng, noise_scale=prior_cfg.noise_scale,
        )

    # Clip continuous, round integers, snap binaries (shared constraint layer).
    normal_df = _apply_domain_constraints(normal_df, result)

    # Attach the label. Like the rare path, the Prior emits feature columns only;
    # every row here is the normal class, so it carries the normal label sourced
    # from the real normal rows (robust to whatever encoding the rare value has).
    if label_col and label_col in result.normal_df.columns and len(result.normal_df):
        normal_df[label_col] = result.normal_df[label_col].mode().iloc[0]

    normal_df = _decode_categoricals(normal_df, result)

    # Guard against the FULL real set (normal + rare), matching the rare path and
    # assess_privacy's measurement scope — a synthetic normal row could verbatim-
    # match a real rare row just as easily as a real normal one (P2-8b).
    if privacy == "floored":
        real_full = pd.concat([result.normal_df, result.rare_df], ignore_index=True)
        normal_df, _ = guard_against_duplicates(
            normal_df, real_full, result.field_dict, label_col, rng,
        )
    return normal_df


def _enforce_rare_floor(
    full_df: pd.DataFrame,
    result: IngestResult,
    delta: float,
    seed: int,
) -> tuple[pd.DataFrame, bool, Optional[str]]:
    """Enforce the privacy δ-distance floor on the rare rows of a delivered batch.

    This is the FINAL numeric mutation before a batch is persisted, so the
    guarantee holds on the data the caller actually receives — applying it
    earlier lets the constraint layer clip floored rows back inside δ. The floor
    moves only continuous columns and clamps to the observed [min,max], so
    binary/categorical snaps stay valid; integer-valued continuous columns still
    need re-rounding afterward (which nudges a row by ≤0.5 raw units), so we
    enforce to δ plus that worst-case rounding margin and the delivered distance
    clears δ even after the round. Deterministic: a dedicated substream off
    ``seed`` (Invariant 2).

    Shared by generate() (full-dataset path) and run_campaign() (rare-only
    diagnostic path) so the floor is enforced identically — one implementation,
    no drift (P1-5). Mutates and returns ``full_df``.

    Returns (full_df, floor_applied, floor_skip_reason). The floor is skipped
    (and said so, never silently — P2-9) when the data can't support a δ-shell:
    ``no_label`` / ``no_continuous_features`` / ``no_rare_rows``.
    """
    from engine.privacy import enforce_distance_floor, _continuous_cols
    rare_val = (result.rare_df[result.label_col].mode().iloc[0]
                if result.label_col and len(result.rare_df) else None)
    cont_cols = _continuous_cols(full_df, result.field_dict, result.label_col)
    if not (result.label_col and rare_val is not None):
        return full_df, False, "no_label"
    if not cont_cols:
        return full_df, False, "no_continuous_features"
    rare_mask = full_df[result.label_col] == rare_val
    if not rare_mask.any():
        return full_df, False, "no_rare_rows"

    # Worst-case L2 displacement (σ-normalized) from re-rounding the integer-
    # valued continuous columns: ≤0.5 raw unit each.
    sig = result.rare_df[cont_cols].to_numpy(dtype=float).std(axis=0)
    sig = np.where(sig < 1e-8, 1.0, sig)
    int_mask = np.array(
        [bool(getattr(result.field_dict[c], "is_integer", False)) for c in cont_cols]
    )
    margin = float(np.sqrt(np.sum(((0.5 / sig) * int_mask) ** 2)))
    rng_floor = np.random.default_rng(seed + 4242)
    floored, _ = enforce_distance_floor(
        full_df.loc[rare_mask], result.rare_df, result.field_dict,
        result.label_col, delta + margin, rng_floor,
    )
    # Write back ONLY the continuous columns the floor adjusts (not identifiers/
    # label), widening integer ones to float first — the re-round below restores
    # int64. Avoids a pandas incompatible-dtype assignment into int columns.
    full_df[cont_cols] = full_df[cont_cols].astype(float)
    full_df.loc[rare_mask, cont_cols] = floored[cont_cols].values
    for c in cont_cols:
        meta = result.field_dict[c]
        if getattr(meta, "is_integer", False):
            col = full_df[c].clip(meta.min_val, meta.max_val).round()
            full_df[c] = col.astype("int64")
    return full_df, True, None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RUN CAMPAIGN
# ═══════════════════════════════════════════════════════════════════════════════

def run_campaign(
    filepath: str,
    label_col: str = "",
    rare_def: Optional[RareEventDef] = None,
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
    privacy: str = "none",
    delta: float = 0.5,
    scenario: Optional[ScenarioSpec] = None,
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
        privacy: "none" (default) or "floored". The campaign is a multi-pass
            *diagnostic* path (it maps the lift trajectory across rare regions),
            so it defaults to "none" — but when a persisted pass batch is meant
            for release, pass "floored" and each accepted batch gets the same
            parametric generation + verbatim guard + δ-distance floor that
            generate() applies (P1-5: same guarantee, one implementation).
        delta: δ-distance floor in σ-units (only meaningful when privacy="floored").

    Returns:
        CampaignResult with best_lift, pass history, output paths, etc. The
        privacy regime is recorded in campaign_summary.json and the manifest.
    """
    # A ScenarioSpec is authoritative for the use-case fields (G-A). Loose params
    # remain for direct callers.
    if scenario is not None:
        label_col = scenario.intent.label_col
        rare_def = scenario.intent.rare_def()
        seed = scenario.intent.seed
        n_rows = scenario.intent.n_rows
        privacy = scenario.gates.privacy
        delta = scenario.gates.delta
    if rare_def is None:
        rare_def = _auto_rare_def()
    if privacy not in ("none", "floored"):
        raise ValueError(f"privacy must be 'none' or 'floored', got {privacy!r}")
    if not (0.0 < delta <= 2.0):
        raise ValueError(f"delta must be in (0, 2] σ-units, got {delta!r}")
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
    best_seed: Optional[int] = None
    best_target: Dict[str, Any] = {}

    floor_applied_any = False
    floor_skip_reason: Optional[str] = None
    for pass_num in range(max_passes):
        amp_df, report, lift, target_region = _run_one_pass(
            result, prior_cfg, amp_cfg, aud_cfg, exam_cfg, scout_cfg,
            seed + pass_num, n_rows, label_col, rare_def, explored_points,
            privacy=privacy, delta=delta,
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

        # Under privacy="floored", enforce the δ-distance floor on the accepted
        # batch as the final numeric step before persistence — the same helper
        # generate() uses, so a released campaign batch carries the identical
        # guarantee (P1-5). The batch is rare-only (all rows are the rare class),
        # so the floor applies to the whole frame.
        if privacy == "floored":
            amp_df, fa, fr = _enforce_rare_floor(amp_df, result, delta, seed + pass_num)
            floor_applied_any = floor_applied_any or fa
            if not fa:
                floor_skip_reason = fr

        # Save the accepted batch (carries the rare label, see _generate_amp_batch).
        batch_path = str(out_path / f"pass_{pass_num + 1}_accepted.parquet")
        amp_df.to_parquet(batch_path, index=False)
        best_batch_path = batch_path  # last accepted is best
        best_seed = seed + pass_num
        best_target = target_region

    # Persist the manifest for the best (last accepted) batch so it is
    # reproducible from disk (Invariant 2).
    if best_batch_path is not None:
        _write_manifest(out_path, best_seed, result, prior_cfg, amp_cfg, best_target,
                        n_rows, privacy=privacy, delta=delta)

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

    # Privacy regime — always visible in the summary so the diagnostic-vs-private
    # distinction is never ambiguous (P1-5).
    if privacy == "floored":
        privacy_block = {
            "mode": "floored",
            "delta": delta,
            "floor_applied": floor_applied_any,
            "floor_skip_reason": None if floor_applied_any else floor_skip_reason,
            "note": ("Each accepted pass batch carries parametric generation + "
                     "verbatim guard + δ-distance floor (same as generate()). "
                     "NOT differential privacy — see docs/PRIVACY.md."),
        }
    else:
        privacy_block = {
            "mode": "none",
            "note": ("Diagnostic campaign path — batches are grounded-sampled and "
                     "may contain near-copies of real rows. Use privacy='floored' "
                     "(or generate(privacy='floored')) for a released dataset."),
        }

    # Persist campaign summary for later retrieval via get_results()
    _save_campaign_summary(cr, out_path, privacy_block)

    return cr


def _save_campaign_summary(cr: CampaignResult, out_path: Path,
                           privacy_block: Optional[Dict[str, Any]] = None) -> None:
    """Write campaign_summary.json to the output directory."""
    summary = {
        "best_lift": cr.best_lift,
        "privacy": privacy_block,
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

# Full-synthesis auto rare:normal ratio. The deliverable is a full dataset =
# amplified rare part + synthetic normal part. The rare fraction reflects the
# amplification: amplify any true minority up to at least this floor so the
# detector always gets strong rare signal, but never push a well-represented
# class (prevalence above the floor) past its natural rate. The loader always
# picks the minority class, so natural prevalence is ≤ 0.5 — the resolved auto
# ratio therefore lands in [DEFAULT_MIN_RARE_FRAC, 0.5].
DEFAULT_MIN_RARE_FRAC = 0.25


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
    rare_ratio: Optional[float] = None,
    privacy: str = "floored",
    delta: float = 0.5,
    out_dir: Optional[str] = None,
    scenario: Optional["ScenarioSpec"] = None,
) -> Dict[str, Any]:
    """The simple primary path: generate a synthetic *dataset* as a CSV-ready batch.

    This is the user-facing entry point that hides REGEN's technical knobs.
    The user supplies data + how many rows they want (+ an optional intent);
    the system auto-detects the rare class, auto-tunes noise and target region,
    and returns a fidelity-checked synthetic **full dataset** — an amplified
    rare part concatenated with a synthetic normal part — plus both quality
    numbers (fidelity + detection lift).

    The deliverable is a full dataset, not just the rare rows. The rare part is
    the amplified batch (Prior → Amplifier, gated against the rare reference);
    the normal part is grounded-sampled from the normal covariate support (gated
    against the normal reference). They are concatenated at ``rare_ratio``, the
    fraction of the batch that is rare — the ratio reflects the amplification.

    Three operating points (INVARIANTS.md §4 / design notes):
      * faithful — maximize distributional fidelity. Model-agnostic copy.
      * balanced / boost — maximize detection-lift subject to the fidelity gate.
        The internal Examiner (a Random Forest) is a generic tabular-classifier
        proxy; lift optimized for it transfers well to tree/linear models.
      All three return the combined full dataset; the mode only controls how the
      rare part is generated/tuned, not whether the normal part is included.

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
        n_rows: Size of the full synthetic dataset (normal + rare combined).
        mode: "faithful" | "balanced" | "boost". Default "balanced".
        seed: RNG seed (full reproducibility).
        auto: If True, auto-tune noise via a small search. If False, use noise_scale.
        noise_scale: Manual prior-noise (used when auto=False; default 0.10).
        coverage_threshold: Auditor gate strictness. None → 0.30 for boost, else 0.50.
        rare_ratio: Target fraction of the batch that is the rare class
            (0 < rare_ratio < 1). None → auto: ``max(natural_prevalence,
            DEFAULT_MIN_RARE_FRAC)`` — amplifies any true minority to at least 25%
            so the detector always gets strong rare signal, without de-amplifying
            a class that is already well represented.
        privacy: "floored" (default) or "none". "floored" generates parametrically
            (no copying of real rows) and enforces a per-record δ-distance floor
            on the rare part plus a verbatim-attribute guard on both parts. This
            is a real, checked guarantee but NOT differential privacy — see the
            summary's "privacy" block for exactly what is and isn't guaranteed.
        delta: The δ-distance floor for the rare part, in σ-normalized units
            (default 0.5). Only meaningful when privacy="floored".
        out_dir: Where to persist the batch + summary. Auto-tempdir if None.

    Returns:
        Dict with: run_id, n_rows, label_col, rare info, detection (what was
        auto-selected, or None if fully manual), rare_ratio + the normal/rare
        split, fidelity (per-column + score, rare-half), normal_fidelity,
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

    # A ScenarioSpec, when supplied, is the authoritative statement of the use
    # case (G-A): its intent + gates drive generation, overriding the loose
    # convenience params. Loose params remain the no-spec path (which builds a
    # spec below), so there is no API break.
    if scenario is not None:
        _in, _g = scenario.intent, scenario.gates
        label_col = _in.label_col
        rare_def = _in.rare_def()
        n_rows = _in.n_rows
        mode = _in.mode
        seed = _in.seed
        if _in.rare_ratio is not None:
            rare_ratio = _in.rare_ratio
        privacy = _g.privacy
        delta = _g.delta
        if _g.coverage_threshold is not None:
            coverage_threshold = _g.coverage_threshold

    if mode not in _GENERATE_MODES:
        raise ValueError(f"mode must be one of {_GENERATE_MODES}, got {mode!r}")
    if privacy not in ("none", "floored"):
        raise ValueError(f"privacy must be 'none' or 'floored', got {privacy!r}")
    if not (0.0 < delta <= 2.0):
        raise ValueError(f"delta must be in (0, 2] σ-units, got {delta!r}")

    # 1. Ingest + auto-detect the rare class if the caller didn't specify
    rare_def = rare_def or _auto_rare_def()
    result = ingest(filepath, label_col, rare_def)

    if coverage_threshold is None:
        coverage_threshold = 0.30 if mode == "boost" else 0.50

    # Resolve the rare:normal split. n_rows is the FULL dataset size; the rare
    # part is amplified to `rare_ratio` of it and the normal part fills the rest.
    natural_prevalence = (
        len(result.rare_df) / (len(result.normal_df) + len(result.rare_df))
        if (len(result.normal_df) + len(result.rare_df)) > 0 else 0.0
    )
    if rare_ratio is None:
        rare_ratio_resolved = max(natural_prevalence, DEFAULT_MIN_RARE_FRAC)
    else:
        if not (0.0 < rare_ratio < 1.0):
            raise ValueError(f"rare_ratio must be in (0, 1), got {rare_ratio!r}")
        rare_ratio_resolved = rare_ratio
    n_rare = max(1, int(round(n_rows * rare_ratio_resolved)))
    n_rare = min(n_rare, n_rows - 1) if n_rows > 1 else n_rare
    n_normal = n_rows - n_rare

    out_path = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="regen_generate_"))
    out_path.mkdir(parents=True, exist_ok=True)

    # 2. Auto-tune (or use the manual noise). Tuning runs at the RARE batch size:
    #    the rare part is what the objective (fidelity / lift) scores against.
    if auto:
        best_noise, trail = _autotune(result, rare_def, mode, seed, coverage_threshold, n_rare)
    else:
        best_noise = 0.10 if noise_scale is None else noise_scale
        trail = []

    # 3. Final generation with the chosen config.
    prior_cfg = PriorConfig(noise_scale=best_noise)
    amp_cfg = AmplifierConfig(gp_noise_variance=0.1)
    aud_cfg = AuditorConfig(coverage_threshold=coverage_threshold)
    exam_cfg = ExaminerConfig(n_estimators=100)
    scout_cfg = ScoutConfig(num_candidates=100)

    final_seed = seed + 9000

    # 3a. Rare part: Prior → Scout → Amplifier, gated against the rare reference.
    #     privacy threads into _generate_amp_batch (parametric base + δ-floor on
    #     the rare set + verbatim guard) via _run_one_pass.
    rare_df_synth, rare_report, lift, target_region = _run_one_pass(
        result, prior_cfg, amp_cfg, aud_cfg, exam_cfg, scout_cfg,
        final_seed, n_rare, result.label_col, rare_def, [],
        privacy=privacy, delta=delta,
    )

    # 3b. Normal part: grounded sampling on the normal covariate support, gated
    #     against the normal reference. Coverage is a rare-region question
    #     ("did we densify the tail?") and is meaningless for a normal sample, so
    #     it is turned off — the normal part is judged on marginals + correlation.
    #     A decorrelated seed offset keeps the normal draw independent of the rare
    #     pipeline's RNG consumption.
    normal_df_synth = _generate_normal_batch(
        result, prior_cfg, final_seed + 7777, n_normal, result.label_col,
        privacy=privacy, delta=delta,
    )
    from engine.auditor import audit
    normal_report = audit(
        result, normal_df_synth, aud_cfg,
        reference_df=result.normal_df, check_coverage=False,
    )

    # 4. Combine + persist (reuses the campaign on-disk layout so the existing
    #    get_results / load_synthetic / download endpoints work unchanged).
    full_df = pd.concat([normal_df_synth, rare_df_synth], ignore_index=True)
    # Each part minted its own fresh identifiers starting past the observed max,
    # so the two parts now collide on keys. Re-run the constraint layer on the
    # combined frame to re-mint identifiers across the whole batch (unique again);
    # the clip/snap steps are idempotent on the already-constrained columns.
    full_df = _apply_domain_constraints(full_df, result)

    # Privacy δ-distance floor (rare part), enforced as the final numeric mutation
    # before persistence via the shared helper (also used by run_campaign), so the
    # guarantee holds on the delivered data and there is one floor implementation
    # (P1-5). floor_applied/reason feed the loud-skip reporting (P2-9).
    floor_applied = False
    floor_skip_reason: Optional[str] = None
    if privacy == "floored":
        full_df, floor_applied, floor_skip_reason = _enforce_rare_floor(
            full_df, result, delta, final_seed,
        )

    batch_path = str(out_path / "pass_1_accepted.parquet")
    full_df.to_parquet(batch_path, index=False)

    # The Auditor gate (Invariant 3) — purely a fidelity verdict on both halves.
    # Privacy is a separate guarantee folded into the top-level `passed` below, so
    # a privacy miss never masquerades as a fidelity failure.
    auditor_passed = bool(rare_report.overall_passed and normal_report.overall_passed)

    # Privacy assessment on the DELIVERED data (honest post-constraint measure).
    # Isolate the rare rows of the delivered batch to check the δ-floor against
    # the real rare set; check verbatim-attribute duplicates across the whole
    # batch against the full real set. Skipped (None) when privacy is off.
    if privacy == "floored":
        from engine.privacy import assess_privacy
        rare_val = (result.rare_df[result.label_col].mode().iloc[0]
                    if result.label_col and len(result.rare_df) else None)
        rare_delivered = (
            full_df[full_df[result.label_col] == rare_val]
            if result.label_col and rare_val is not None else full_df.iloc[:0]
        )
        real_full = pd.concat([result.normal_df, result.rare_df], ignore_index=True)
        privacy_report = assess_privacy(
            rare_delivered, result.rare_df, full_df, real_full,
            result.field_dict, result.label_col, delta,
        )
        # Record whether the δ-floor was actually enforced (P2-9). When it was
        # skipped, min_distance is inf (no rare-continuous distance to measure)
        # and `passed` reflects only the verbatim guard — which is what protects
        # an all-categorical / no-label batch, and must be stated, not implied.
        privacy_report.floor_applied = floor_applied
        privacy_report.floor_skip_reason = floor_skip_reason
        if floor_applied:
            note = ("Per-record δ-distance floor on the rare class + "
                    "verbatim-attribute guard on the whole batch (vs the full "
                    "real set). NOT differential privacy — see docs/PRIVACY.md.")
        else:
            note = (f"δ-distance floor NOT applied ({floor_skip_reason}); "
                    "protection is parametric sampling + the verbatim-attribute "
                    "guard against the full real set. NOT differential privacy "
                    "— see docs/PRIVACY.md.")
        privacy_out = {
            "mode": "floored",
            "delta": delta,
            "floor_applied": floor_applied,
            "floor_skip_reason": floor_skip_reason,
            "min_distance": round(privacy_report.min_distance, 4),
            "distance_p50": (round(privacy_report.distance_p50, 4)
                             if privacy_report.distance_p50 is not None else None),
            "n_verbatim_duplicates": int(privacy_report.n_respawned),
            "passed": bool(privacy_report.passed),
            "note": note,
        }
    else:
        privacy_out = None
        privacy_report = None

    # Build the *vetted* ScenarioSpec this batch was generated under (G-A/G-B).
    # The vetting gate merges Source 1 (structural) + Source 2 (researcher
    # declaration in `scenario`) under the 10 rules, dropping any proposal that
    # contradicts the data and logging every decision (rule 7). Intent + gates
    # record the *resolved* values actually used, so a re-run reproduces the batch.
    from regen.vetting import vet_scenario
    vetted_cols, verdicts = vet_scenario(scenario, result)
    vetted_spec = ScenarioSpec(
        columns=vetted_cols,
        intent=ScenarioIntent(
            task=(scenario.intent.task if scenario is not None else "detector_training"),
            label_col=result.label_col,
            rare_mode=rare_def.mode.value,
            rare_value=rare_def.label_value,
            percentile=rare_def.percentile,
            tail=rare_def.tail,
            imbalance_ratio=rare_def.imbalance_ratio,
            rare_ratio=round(rare_ratio_resolved, 6),
            focus_features=(scenario.intent.focus_features if scenario is not None else []),
            n_rows=n_rows,
            seed=seed,
            mode=mode,
        ),
        gates=ScenarioGates(
            coverage_threshold=coverage_threshold,
            privacy=privacy,
            delta=delta,
            min_tail_lift=(scenario.gates.min_tail_lift if scenario is not None else None),
        ),
        notes=(scenario.notes if scenario is not None else ""),
        provenance=(dict(scenario.provenance) if scenario is not None else {}),
        verdicts=verdicts,
    )

    # Conformance audit (G-B rule 9): the delivered batch must obey every vetted
    # constraint. A conformance failure fails the batch exactly like a fidelity
    # failure (Invariant 3 extended to the contract).
    from engine.auditor import check_conformance
    conformance = check_conformance(full_df, vetted_spec, result.label_col)
    conformance_out = conformance.to_dict()

    # Top-level verdict: fidelity AND conformance AND (privacy, when enforced).
    # The fidelity block keeps its own Auditor-only `passed` (Invariant 3).
    overall_passed = bool(
        auditor_passed
        and conformance.passed
        and (privacy_report.passed if privacy_report is not None else True)
    )

    scenario_dict = vetted_spec.to_dict()
    # Save the spec next to the batch — the unit a researcher saves/shares/re-runs.
    try:
        vetted_spec.save_yaml(str(out_path / "scenario.yaml"))
    except Exception:  # yaml optional; the manifest is the source of truth
        pass

    # Explainability (G-C): every batch explains itself from computed numbers.
    # Built AFTER the privacy block so its numbers match exactly, and written
    # BEFORE the manifest so the manifest can hash it (G-G).
    from regen.explain import build_explanation
    explanation = build_explanation(
        result=result, vetted_spec=vetted_spec, rare_report=rare_report,
        normal_report=normal_report, conformance=conformance,
        privacy_out=privacy_out, lift=lift, target_region=target_region,
        aud_cfg=aud_cfg, coverage_threshold=coverage_threshold,
    )
    (out_path / "explanation.json").write_text(json.dumps(explanation, indent=2, default=str))

    # Audit bundle (G-G): reference aggregates of the REAL data (disclosure-
    # bounded), then the manifest carrying the SHA-256 of every artifact + the
    # metric versions, so a third party can `regen verify` the batch and detect
    # tampering. The manifest is written LAST because it hashes the others.
    from regen.audit_bundle import (
        build_reference_aggregates, sha256_file, BATCH_NAME, EXPLAIN_NAME, AGG_NAME,
    )
    from regen.metrics import metric_versions
    agg = build_reference_aggregates(result, n_normal, n_rare)
    (out_path / AGG_NAME).write_text(json.dumps(agg, indent=2, default=str))
    artifact_sha256 = {
        BATCH_NAME: sha256_file(out_path / BATCH_NAME),
        EXPLAIN_NAME: sha256_file(out_path / EXPLAIN_NAME),
        AGG_NAME: sha256_file(out_path / AGG_NAME),
    }

    # Persist the manifest so the batch is reproducible from disk (Invariant 2):
    # seed + configs + schema hash + rare split + privacy regime + the vetted
    # ScenarioSpec + artifact hashes + metric versions fully determine + attest it.
    manifest_path = _write_manifest(
        out_path, final_seed, result, prior_cfg, amp_cfg, target_region,
        len(full_df), rare_ratio_resolved, privacy=privacy, delta=delta,
        scenario=scenario_dict, artifact_sha256=artifact_sha256,
        metric_versions=metric_versions(),
    )

    fidelity = {
        "score": round(_fidelity_score(rare_report), 4),
        "passed": auditor_passed,
        "coverage": round(rare_report.coverage_rate, 4),
        # Cross-column correlation-structure gate (B2): None when too few numeric
        # columns/rows to estimate.
        "correlation": {
            "delta": round(rare_report.correlation_delta, 4) if rare_report.correlation_delta is not None else None,
            "passed": bool(rare_report.correlation_passed),
        },
        "columns": [
            {
                "col": c.col,
                "passed": bool(c.passed),
                "metric": "wasserstein" if c.wasserstein is not None else "tvd",
                "value": round(c.wasserstein, 4) if c.wasserstein is not None
                else (round(c.tvd, 4) if c.tvd is not None else None),
            }
            for c in rare_report.column_results
        ],
    }
    normal_fidelity = {
        "score": round(_fidelity_score(normal_report), 4),
        "passed": bool(normal_report.overall_passed),
        "correlation": {
            "delta": round(normal_report.correlation_delta, 4) if normal_report.correlation_delta is not None else None,
            "passed": bool(normal_report.correlation_passed),
        },
    }
    lift_out = None
    if lift is not None:
        lift_out = {
            "status": lift.status,
            "n_test_rare": lift.n_test_rare,
            "baseline_recall": round(lift.baseline_recall, 4),
            "amplified_recall": round(lift.amplified_recall, 4),
            # tail_lift is None (not a bare 0.0) when the held-out rare fold is
            # too small to trust the estimate (P2-7). The recall numbers are kept
            # for context but flagged by status.
            "tail_lift": (round(lift.tail_lift, 4)
                          if lift.status == "ok" else None),
        }

    summary = {
        "run_id": out_path.name,
        "n_rows": len(full_df),
        "label_col": result.label_col,
        # Real (source) class counts — get_results() reads these as the original
        # dataset's normal/rare sizes, so they stay about the input, not the output.
        "n_normal": len(result.normal_df),
        "n_rare": len(result.rare_df),
        # Synthetic split of the delivered full dataset.
        "n_synthetic_normal": n_normal,
        "n_synthetic_rare": n_rare,
        "rare_ratio": round(rare_ratio_resolved, 4),
        "natural_prevalence": round(natural_prevalence, 4),
        "n_features": len(result.field_dict) - (1 if result.label_col else 0),
        # Shippable-batch verdict: fidelity gate AND privacy guarantee (when on).
        "passed": overall_passed,
        # What the system chose for you (and what you could override). None when
        # both label_col and rare value were supplied explicitly.
        "detection": result.detection.as_dict() if result.detection else None,
        "fidelity": fidelity,
        "normal_fidelity": normal_fidelity,
        "privacy": privacy_out,
        "lift": lift_out,
        "config_used": {
            "mode": mode,
            "noise_scale": round(best_noise, 4),
            "coverage_threshold": coverage_threshold,
            "rare_ratio": round(rare_ratio_resolved, 4),
            "auto": auto,
        },
        "candidates": trail,
        "output_dir": str(out_path),
        "best_batch_path": batch_path,
        "manifest_path": manifest_path,
        "scenario": scenario_dict,
        "conformance": conformance_out,
        "explain": explanation,
        "seed": final_seed,
    }
    # Write a minimal campaign_summary.json so get_results()/download work.
    _save_generate_summary(summary, out_path)
    return summary


def _write_manifest(
    out_path: Path,
    seed: int,
    result: IngestResult,
    prior_cfg,
    amp_cfg,
    target_region: Dict[str, Any],
    n_rows: int,
    rare_ratio: float = 0.0,
    privacy: str = "none",
    delta: float = 0.0,
    scenario: Optional[Dict[str, Any]] = None,
    artifact_sha256: Optional[Dict[str, str]] = None,
    metric_versions: Optional[Dict[str, int]] = None,
) -> str:
    """Build and persist the batch manifest (Invariant 2).

    Captures everything that determines the output — seed, schema hash, prior
    and amplifier configs, the Scout target, n_rows, the rare:normal split, the
    privacy regime, and the code version — so the batch can be regenerated
    bit-for-bit from what is saved on disk. Returns the manifest file path.
    """
    from dataclasses import asdict, is_dataclass
    from engine.manifest import build_manifest

    def _cfg(c):
        return asdict(c) if is_dataclass(c) else dict(getattr(c, "__dict__", {}))

    manifest = build_manifest(
        seed=seed,
        schema=result.schema_graph,
        prior_config=_cfg(prior_cfg),
        target_region={k: v for k, v in (target_region or {}).items()
                       if isinstance(v, (int, float, str, bool, type(None)))},
        amplifier_params=_cfg(amp_cfg),
        n_rows=n_rows,
        rare_ratio=rare_ratio,
        privacy=privacy,
        delta=delta,
        scenario=scenario,
        artifact_sha256=artifact_sha256,
        metric_versions=metric_versions,
    )
    path = out_path / "manifest.json"
    path.write_text(manifest.to_json())
    return str(path)


def _save_generate_summary(summary: Dict[str, Any], out_path: Path) -> None:
    """Persist a campaign-shaped summary so get_results()/load_synthetic() work."""
    _tl = (summary["lift"] or {}).get("tail_lift")
    _tl = _tl if _tl is not None else 0.0   # None (insufficient rare fold, P2-7) → 0.0 on disk
    cr_like = {
        "best_lift": _tl,
        "passes": [{
            "pass_num": 1,
            "status": "accepted" if summary["fidelity"]["passed"] else "rejected",
            "tail_lift": _tl,
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
    label_col: str = "",
    rare_def: Optional[RareEventDef] = None,
    seed: int = 42,
    quick_campaign: bool = True,
    scenario: Optional[ScenarioSpec] = None,
) -> ScreenResult:
    """Predict which method (REGEN or SMOTE) is likely to win on this data.

    Runs a quick head-to-head: one REGEN pass vs one SMOTE pass on the same
    data split, with matched synthetic row budget. The winner is the
    recommendation. Also computes a Fisher discriminant CV as supplementary
    context (how much features vary in informativeness).

    This is slower than a pure metric-based screen (~5-15 seconds) but
    actually reliable — it measures real lift on the user's data rather
    than predicting from proxy statistics.

    Privacy (P1-5): screen is a **non-private diagnostic** and takes no privacy
    parameter by design. It returns a method recommendation, not a dataset — no
    synthetic rows are persisted or handed back — so output re-identification is
    not in play. Its REGEN arm is generated non-privately on purpose, so the
    REGEN-vs-SMOTE lift comparison is apples-to-apples (SMOTE has no privacy
    mode; enforcing the δ-floor on only one arm would bias the comparison). For a
    private *deliverable*, use generate(privacy="floored").

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

    # ScenarioSpec supplies the target when given (screen stays non-private, P1-5).
    if scenario is not None:
        label_col = scenario.intent.label_col
        rare_def = scenario.intent.rare_def()
        seed = scenario.intent.seed
    if rare_def is None:
        rare_def = _auto_rare_def()

    # 1. Ingest
    result = ingest(filepath, label_col, rare_def)

    # 2. Compute Fisher discriminant CV (supplementary metric)
    heterogeneity_score = _compute_fisher_cv(result)

    # 3. Run quick head-to-head: REGEN 1-pass vs SMOTE 1-pass
    regen_lift = 0.0
    smote_lift = 0.0
    n_synthetic = 200

    if quick_campaign:
        # --- REGEN 1-pass --- (uses the shared generation core + leakage-free lift)
        try:
            prior_cfg = PriorConfig()
            amp_cfg = AmplifierConfig(
                gp_noise_variance=0.1,
                max_features=min(10, len(result.field_dict) - 1) if (len(result.field_dict) - 1) > 10 else 0,
            )
            scout_cfg = ScoutConfig()
            amp_df, _ = _generate_amp_batch(
                result, prior_cfg, amp_cfg, scout_cfg, seed, n_synthetic,
                label_col, rare_def, [],
            )
            aud_cfg = AuditorConfig(coverage_threshold=0.50)
            report = audit(result, amp_df, aud_cfg)
            if report.overall_passed:
                exam_cfg = ExaminerConfig(n_estimators=50)
                lift = measure_lift(
                    result, exam_cfg,
                    generate_synth_fn=lambda ti: _generate_amp_batch(
                        ti, prior_cfg, amp_cfg, scout_cfg, seed + 12345,
                        n_synthetic, label_col, rare_def, [],
                    )[0],
                )
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
    from engine.prior.grounded import _encode_features

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