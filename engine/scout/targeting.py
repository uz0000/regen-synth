"""
Scout targeting acquisition function (Gao et al., R-Design).

Scores candidate covariate regions by their expected information gain about
the GP's prediction over the rare-event target population.

Closed-form for Gaussians (R-Design, Eq. 40):
    α(x) = E_{x* ~ p_rare} [ I(r; τ_δ(x*) | x, H) ]
          = (1/|targets|) Σ_{x*} 0.5 · log(1 / (1 - ρ²(x, x*)))

where ρ²(x, x*) = Cov(y_x, τ_δ(x*))² / (Var(y_x) · Var(τ_δ(x*)))

The covariance is the *posterior* cross-covariance conditioned on observed
rare events — so it shrinks as more rare events are observed (exploration
that avoids already-seen regions automatically).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from engine.amplifier.tail_corrector import TailCorrector

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ScoutConfig:
    num_candidates: int = 100      # candidate pool size
    explored_penalty: float = 0.7  # max score discount on an already-explored anchor (0–1)


# ── Public API ─────────────────────────────────────────────────────────────────

def score_candidates(
    residual_model: TailCorrector,
    candidates: np.ndarray,
) -> np.ndarray:
    """
    Score each candidate point by Scout targeting.

    Args:
        residual_model: Fitted TailCorrector from the Amplifier.
        candidates:     (C, D) array of candidate covariate points.

    Returns:
        (C,) array — higher score = more informative for the rare tail.
    """
    gp      = residual_model._gp
    targets = residual_model._X_train   # observed rare events = target population

    return _repig(gp, candidates, targets)


def select_target(
    residual_model: TailCorrector,
    feature_cols: List[str],
    rng: np.random.Generator,
    config: ScoutConfig,
    explored_points: Optional[List[List[float]]] = None,
) -> Dict:
    """
    Build a candidate pool from the rare-event distribution and select the
    highest-scoring target region for the next amplification pass.

    Args:
        explored_points: Anchor points of regions already explored in prior
            passes (and prior runs, via persistent memory). Candidates near
            these are down-weighted so Scout spends budget on new tail
            structure rather than re-mapping what it already knows. Scout targeting's
            posterior already shrinks near observed data within a single fit;
            this term carries that intent ACROSS runs, where the GP cannot.

    Returns a target_region dict interpretable by engine.prior.generate_base_batch.
    """
    X_rare = residual_model._X_train  # (n, D)
    std = X_rare.std(axis=0)

    # Candidate pool: perturb rare events into slightly unexplored directions
    idx = rng.choice(len(X_rare), size=config.num_candidates, replace=True)
    candidates = X_rare[idx].copy()
    candidates += rng.standard_normal(candidates.shape) * std * 0.5

    scores = score_candidates(residual_model, candidates)

    # Diversity penalty: discount candidates close to already-explored anchors.
    scores = _apply_explored_penalty(scores, candidates, explored_points, std, config)

    best = int(np.argmax(scores))

    # Identify which feature is most informative for this candidate
    relevance = residual_model._feature_relevance
    top_feat  = int(np.argmax(relevance))

    # Target the upper half of the most-relevant feature, not just the extreme
    # top decile. Too narrow a band collapses coverage and the Auditor rejects
    # every targeted batch; this keeps enough of the rare region covered while
    # still concentrating generation where Scout targeting says the tail is informative.
    target_region = {
        "feature_idx":    top_feat,
        "feature_name":   feature_cols[top_feat] if top_feat < len(feature_cols) else "",
        "percentile_low":  0.50,
        "percentile_high": 1.00,
        "repig_score":     float(scores[best]),
        "candidate_point": candidates[best].tolist(),
    }

    logger.info(
        "Scout selected target: feature '%s' (idx %d), Scout targeting=%.4f",
        target_region["feature_name"], top_feat, scores[best],
    )
    return target_region


def _apply_explored_penalty(
    scores: np.ndarray,
    candidates: np.ndarray,
    explored_points: Optional[List[List[float]]],
    std: np.ndarray,
    config: ScoutConfig,
) -> np.ndarray:
    """
    Multiply each candidate's Scout targeting score by a factor in (0, 1] that shrinks
    as the candidate approaches an already-explored anchor.

    Distance is measured in standard-deviation units (scale-free). A candidate
    sitting exactly on an explored anchor is suppressed by `explored_penalty`;
    one many sigmas away is essentially untouched. This is the cross-run
    persistence of exploration — within a fit the GP posterior handles it, but
    across runs the GP is refit from scratch and would otherwise forget.
    """
    if not explored_points:
        return scores
    E = np.asarray(explored_points, dtype=np.float64)
    if E.ndim != 2 or E.shape[1] != candidates.shape[1]:
        return scores  # shape mismatch (e.g. schema changed) — skip safely

    safe_std = np.where(std < 1e-8, 1.0, std)
    penalty_strength = config.explored_penalty

    factors = np.ones(len(candidates))
    for i, c in enumerate(candidates):
        # Nearest explored anchor, in sigma units
        d = np.abs((E - c) / safe_std).mean(axis=1).min()
        # Gaussian falloff: near (d→0) → max penalty; far → factor→1
        factors[i] = 1.0 - penalty_strength * np.exp(-0.5 * d ** 2)
    return scores * factors


# ── Scout targeting core ───────────────────────────────────────────────────────────────

def _repig(gp, candidates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute Scout targeting scores for each candidate against the target population.

    For a Gaussian process, the mutual information between a candidate
    observation y_x and the posterior at target x* is:
        MI = 0.5 · log(1 / (1 - ρ²))
    averaged over the target population.

    ρ² = Cov(y_x, τ_δ(x*))² / (Var(y_x) · Var(τ_δ(x*)))

    The posterior cross-covariance is:
        K_post(x, x*) = K(x, x*) - K(x, X) · (K(X,X) + σ²I)⁻¹ · K(X, x*)
    """
    C = len(candidates)
    T = len(targets)

    # Predictive variance at candidates (includes noise)
    _, var_cand = gp.predict(candidates)                      # (C, 1)
    sigma_y_sq  = var_cand.ravel() + gp.Gaussian_noise.variance.values.item()  # (C,)

    # Posterior variance at targets
    _, var_target = gp.predict(targets)                       # (T, 1)
    var_target = var_target.ravel()                           # (T,)

    # Prior cross-covariance
    K_cross = gp.kern.K(candidates, targets)                  # (C, T)

    # Subtract posterior correction using Woodbury inverse
    if gp.X.shape[0] > 0:
        K_cX  = gp.kern.K(candidates, gp.X)                  # (C, n)
        K_Xt  = gp.kern.K(gp.X, targets)                     # (n, T)
        K_cross = K_cross - K_cX @ (gp.posterior.woodbury_inv @ K_Xt)

    # ρ² = Cov² / (Var_cand · Var_target)
    rho_sq = K_cross ** 2 / (
        sigma_y_sq[:, None] * var_target[None, :] + 1e-8
    )
    rho_sq = np.clip(rho_sq, 0.0, 0.9999)

    # MI = 0.5 · log(1 / (1 - ρ²)), averaged over targets
    mi = 0.5 * np.log(1.0 / (1.0 - rho_sq + 1e-8))          # (C, T)
    return mi.mean(axis=1)                                    # (C,)
