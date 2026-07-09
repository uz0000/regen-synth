"""
TailCorrector — active residual learning (R-Design).

Core identity (Gao et al., R-Design):
    τ(x) = τ_o(x) + τ_δ(x)
    truth  =  prior  +  correction

The GP learns τ_δ — the residual between what the prior predicts for rare
rows and what those rows actually look like. Because residuals are smoother
than raw outcomes (Lemma 1 of R-Design), the GP converges far faster.

Kernel: GPy RBF with ARD. Each feature gets its own lengthscale.
Short lengthscale → feature drives rare-event deviation.
Long lengthscale → feature is mostly irrelevant to the tail.

Rolling buffer: observations are capped at gp_max_obs (Cholesky stability).

High-dim feature selection: when max_features > 0, only the top-K columns
(by variance in the rare data) are used as GP inputs. This keeps the ARD
kernel fitting fast — GP hyperparameter optimization scales as O(d³) where
d = input dimension.
"""

import logging
import signal
from dataclasses import dataclass
from typing import List, Optional, Tuple

import GPy
import numpy as np
import pandas as pd

from contracts.types import IngestResult
from engine.prior.grounded import PriorModel, _encode_features

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AmplifierConfig:
    gp_max_obs: int = 300          # rolling buffer cap (Cholesky stays stable)
    gp_noise_variance: float = 0.1 # observation noise σ²
    max_features: int = 0          # 0 = all features; >0 = top-K by variance (speeds GP on high-D data)
    gp_optimize_iters: int = 500   # max L-BFGS iterations for GP hyperparameter optimisation
    min_obs_per_dim: float = 2.0   # warn if rare obs per ARD lengthscale falls below this
                                   # (the GP is underdetermined → tail fit is unreliable)


# ── Fitted residual model ──────────────────────────────────────────────────────

@dataclass
class TailCorrector:
    _gp: GPy.models.GPRegression
    _feature_cols: List[str]
    _X_train: np.ndarray          # rare-event feature points (training set)
    _feature_relevance: np.ndarray  # normalized ARD relevance per feature [0,1]
    _gp_feature_idx: np.ndarray   # indices of features used as GP inputs
    _n_total_features: int        # total feature count before selection
    _is_continuous: np.ndarray    # bool mask: which features are continuous
    _gp_optimized: bool           # whether the GP was successfully optimized

    def posterior(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GP posterior mean and variance at X. Returns (mean, var) as (n,) arrays."""
        # If the GP was fit on a subset of features, select those columns
        X_gp = X[:, self._gp_feature_idx] if len(self._gp_feature_idx) < self._n_total_features else X
        mean, var = self._gp.predict(X_gp)
        return mean.ravel(), var.ravel()


# ── Public API ─────────────────────────────────────────────────────────────────

def fit_correction(
    ingest: IngestResult,
    prior: PriorModel,
    config: AmplifierConfig,
) -> TailCorrector:
    """
    Compute residuals on rare events and fit a GP over them.

    Residual definition: how far does each rare event deviate from the prior's
    normal prediction? A prior score near 1.0 means "looks very normal" — the
    residual (1 - score) captures how anomalous the rare event actually is.

    The GP uses an ARD (Automatic Relevance Determination) RBF kernel: each
    feature gets its own lengthscale. After fitting, the inverse-lengthscales
    give per-feature relevance — shorter lengthscale = more relevant to
    rare-event deviation. This is the R-Design TailCorrector approach.

    When config.max_features > 0, only the top-K features (by variance in the
    rare data) are used as GP inputs. This speeds the ARD kernel significantly
    on high-dimensional data without losing informativeness — the GP only
    needs the most varying dimensions to learn the residual structure.

    Args:
        ingest: IngestResult (uses rare_df).
        prior:  Fitted PriorModel from engine.prior.
        config: AmplifierConfig.

    Returns:
        TailCorrector exposing .posterior(X).
    """
    rare_df = ingest.rare_df
    feature_cols = prior._feature_cols
    # Use the prior's canonical category encoding so codes match the scored space.
    X_rare = _encode_features(rare_df[feature_cols], prior._field_dict).astype(np.float64)

    prior_scores = prior.score(rare_df).values  # P(normal) for each rare row
    residuals = 1.0 - prior_scores              # high = very un-normal

    # Feature selection: when max_features > 0, keep only the top-K by variance
    n_total = X_rare.shape[1]
    if 0 < config.max_features < n_total:
        variances = X_rare.var(axis=0)
        gp_feature_idx = np.argsort(variances)[::-1][:config.max_features]
        X_gp = X_rare[:, gp_feature_idx].copy()
        logger.info(
            "GP input dim: %d → %d (top %d by variance)",
            n_total, config.max_features, config.max_features,
        )
    else:
        gp_feature_idx = np.arange(n_total)
        X_gp = X_rare

    # Enforce rolling buffer so Cholesky stays stable
    if len(X_gp) > config.gp_max_obs:
        logger.info("Rare event buffer capped: %d → %d", len(X_gp), config.gp_max_obs)
        X_gp = X_gp[-config.gp_max_obs:]
        residuals = residuals[-config.gp_max_obs:]

    # Underdetermination guard: the ARD kernel fits one lengthscale per input
    # dimension, so with few rare rows relative to dimensions the tail fit is
    # unreliable (overfit lengthscales, posterior variance masked by the 1e-6
    # clip downstream). Warn loudly rather than silently producing a confident-
    # looking but ungrounded correction; the Auditor still gates the result.
    n_obs, n_dim = X_gp.shape
    if n_dim > 0 and n_obs < config.min_obs_per_dim * n_dim:
        logger.warning(
            "GP underdetermined: %d rare obs for %d feature dims (< %.1f per dim). "
            "Tail correction is unreliable — consider more rare data or "
            "AmplifierConfig.max_features to reduce GP input dimensions.",
            n_obs, n_dim, config.min_obs_per_dim,
        )

    gp, optimized = _fit_gp(X_gp, residuals, config)

    # Per-feature relevance: derive from fitted ARD inverse-lengthscales.
    # Shorter lengthscale → more relevant (feature drives rare-event deviation).
    # Falls back to variance proxy if optimization failed or was skipped.
    if optimized:
        try:
            ls = gp.kern.lengthscale.values.copy()          # (D,) for ARD
            relevance_gp = 1.0 / (ls + 1e-8)
            relevance_gp = relevance_gp / (relevance_gp.max() + 1e-8)
            logger.info(
                "ARD relevance from fitted lengthscales: min=%.3f, max=%.3f",
                relevance_gp.min(), relevance_gp.max(),
            )
        except Exception as exc:
            logger.warning(
                "Could not extract ARD lengthscales (%s) — falling back to variance proxy.",
                type(exc).__name__,
            )
            rarity = X_gp.var(axis=0)
            relevance_gp = rarity / (rarity.max() + 1e-8)
            optimized = False
    else:
        # Optimization failed — use variance proxy with a loud log
        logger.warning(
            "GP optimization did not converge. Using variance-based feature relevance "
            "as fallback — per-feature ARD lengthscales not available."
        )
        rarity = X_gp.var(axis=0)
        relevance_gp = rarity / (rarity.max() + 1e-8)

    # Expand relevance back to full feature space: selected features get
    # their relevance, unselected features get near-zero relevance.
    # Non-continuous features (binary, categorical) get zero relevance
    # because perturbing them produces meaningless intermediate values
    # (e.g. on_thyroxine=0.35 instead of 0 or 1).
    relevance = np.zeros(n_total, dtype=np.float64)
    relevance[gp_feature_idx] = relevance_gp * prior._is_continuous[gp_feature_idx]

    top5 = np.argsort(relevance)[::-1][:5]
    logger.info(
        "TailCorrector fitted on %d rare rows, %d GP dims; top-5 feature indices: %s",
        len(X_gp), X_gp.shape[1], top5.tolist(),
    )

    return TailCorrector(
        _gp=gp,
        _feature_cols=feature_cols,
        _X_train=X_gp,
        _feature_relevance=relevance,
        _gp_feature_idx=gp_feature_idx,
        _n_total_features=n_total,
        _is_continuous=prior._is_continuous,
        _gp_optimized=optimized,
    )


def sample_correction(
    residual_model: TailCorrector,
    X_base: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw residual correction vectors from the GP posterior at X_base.

    For each base row, the posterior at that point is N(μ*, σ²*). We sample
    independently (diagonal covariance) — tractable at scale. The scalar
    residual is broadcast to all features via ARD relevance weights so only
    the features that matter for rare events are shifted.

    If the GP was fit on a subset of features (max_features > 0), the
    correction is computed on the GP's input dimensions and then expanded
    to the full feature space via relevance weighting — unselected features
    get negligible correction.

    Returns:
        (gp_mean, gp_var, X_residuals) — all shape (n, D) or (n,)
    """
    gp = residual_model._gp
    n  = len(X_base)
    D  = X_base.shape[1]

    # Select the GP input columns from the base features
    idx = residual_model._gp_feature_idx
    X_gp = X_base[:, idx].astype(np.float64) if len(idx) < D else X_base.astype(np.float64)

    mean, var = gp.predict(X_gp)
    mean = mean.ravel()
    var  = var.ravel().clip(min=1e-6)

    residuals = mean + np.sqrt(var) * rng.standard_normal(n)

    relevance = residual_model._feature_relevance  # (D,)
    if len(relevance) == D:
        X_residuals = residuals[:, None] * relevance[None, :]
    else:
        X_residuals = np.tile(residuals[:, None], (1, D)) * 0.1

    return mean, var, X_residuals


# ── GP fitting ────────────────────────────────────────────────────────────────

class _TimeoutError(Exception):
    """Raised when GP optimization exceeds the time budget."""


def _timeout_handler(*_args):
    raise _TimeoutError("GP optimization timed out")


def _fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    config: AmplifierConfig,
) -> Tuple[GPy.models.GPRegression, bool]:
    """
    Fit a GPy RBF+ARD model on the residuals and optimize hyperparameters.

    Kernel: RBF with ARD=True — each feature gets its own lengthscale.
    Optimization: L-BFGS-B with up to gp_optimize_iters iterations.
    Timeout: 120-second watchdog (SIGALRM on Unix) to prevent hangs.

    Returns:
        (gp, optimized) — optimized is True iff ARD optimization succeeded.
        When optimized is False the caller falls back to variance-based
        relevance rather than using the unoptimized lengthscales.
    """
    D = X.shape[1]
    kernel = GPy.kern.RBF(
        input_dim=D,
        variance=1.0,
        lengthscale=np.ones(D),
        ARD=True,
    )
    gp = GPy.models.GPRegression(
        X,
        y.reshape(-1, 1),
        kernel=kernel,
        noise_var=config.gp_noise_variance,
    )
    # warning=False: the parameter already carries GPy's default positive
    # constraint, and re-constraining it otherwise prints "reconstraining
    # parameters ..." to stdout on every fit — junk for a library/server caller.
    gp.Gaussian_noise.variance.constrain_bounded(1e-4, 1.0, warning=False)

    # Optimize with timeout guard
    optimized = True
    old_handler = None
    try:
        # SIGALRM is Unix-only; on other platforms we skip the timeout guard.
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(120)  # 2-minute hard limit
        gp.optimize(optimizer="lbfgsb", max_iters=config.gp_optimize_iters, messages=False)
    except _TimeoutError:
        logger.warning(
            "GP optimization timed out after 120s (%d dims, %d obs). "
            "Lengthscales are unoptimized; relevance will use variance proxy.",
            D, len(X),
        )
        optimized = False
    except Exception as exc:
        logger.warning(
            "GP optimization failed (%s). "
            "Lengthscales are unoptimized; relevance will use variance proxy.",
            type(exc).__name__,
        )
        optimized = False
    finally:
        if hasattr(signal, "SIGALRM") and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    if optimized:
        ls = gp.kern.lengthscale.values
        logger.info(
            "GP optimized (%d iters): lengthscales min=%.3f, max=%.3f, "
            "noise=%.4f",
            config.gp_optimize_iters,
            ls.min(), ls.max(),
            float(gp.Gaussian_noise.variance.values[0]),
        )

    return gp, optimized
