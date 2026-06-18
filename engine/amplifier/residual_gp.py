"""
ResidualGP — active residual learning (R-Design).

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
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import GPy
import numpy as np
import pandas as pd

from contracts.types import IngestResult
from engine.prior.rdbpfn import PriorModel, _encode_features

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AmplifierConfig:
    gp_max_obs: int = 300          # rolling buffer cap (Cholesky stays stable)
    gp_noise_variance: float = 0.1 # observation noise σ²


# ── Fitted residual model ──────────────────────────────────────────────────────

@dataclass
class ResidualModel:
    _gp: GPy.models.GPRegression
    _feature_cols: List[str]
    _X_train: np.ndarray          # rare-event feature points (training set)
    _feature_relevance: np.ndarray  # normalized ARD relevance per feature [0,1]

    def posterior(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GP posterior mean and variance at X. Returns (mean, var) as (n,) arrays."""
        mean, var = self._gp.predict(X)
        return mean.ravel(), var.ravel()


# ── Public API ─────────────────────────────────────────────────────────────────

def fit_residuals(
    ingest: IngestResult,
    prior: PriorModel,
    config: AmplifierConfig,
) -> ResidualModel:
    """
    Compute residuals on rare events and fit a GP over them.

    Residual definition: how far does each rare event deviate from the prior's
    normal prediction? A prior score near 1.0 means "looks very normal" — the
    residual (1 - score) captures how anomalous the rare event actually is.

    Args:
        ingest: IngestResult (uses rare_df).
        prior:  Fitted PriorModel from engine.prior.
        config: AmplifierConfig.

    Returns:
        ResidualModel exposing .posterior(X).
    """
    rare_df = ingest.rare_df
    feature_cols = prior._feature_cols
    X_rare = _encode_features(rare_df[feature_cols]).astype(np.float64)

    prior_scores = prior.score(rare_df).values  # P(normal) for each rare row
    residuals = 1.0 - prior_scores              # high = very un-normal

    # Enforce rolling buffer so Cholesky stays stable
    if len(X_rare) > config.gp_max_obs:
        logger.info("Rare event buffer capped: %d → %d", len(X_rare), config.gp_max_obs)
        X_rare    = X_rare[-config.gp_max_obs:]
        residuals = residuals[-config.gp_max_obs:]

    gp = _fit_gp(X_rare, residuals, config)

    # Per-feature relevance from ARD lengthscales: shorter → more relevant
    ls = gp.kern.lengthscale.values.copy()
    relevance = 1.0 / (ls + 1e-8)
    relevance = relevance / (relevance.max() + 1e-8)

    logger.info(
        "ResidualGP fitted on %d rare rows; top-5 feature indices: %s",
        len(X_rare),
        np.argsort(relevance)[::-1][:5].tolist(),
    )

    return ResidualModel(
        _gp=gp,
        _feature_cols=feature_cols,
        _X_train=X_rare,
        _feature_relevance=relevance,
    )


def sample_residuals(
    residual_model: ResidualModel,
    X_base: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw residual correction vectors from the GP posterior at X_base.

    For each base row, the posterior at that point is N(μ*, σ²*). We sample
    independently (diagonal covariance) — tractable at scale. The scalar
    residual is broadcast to all features via ARD relevance weights so only
    the features that matter for rare events are shifted.

    Returns:
        (gp_mean, gp_var, X_residuals) — all shape (n, D) or (n,)
    """
    gp = residual_model._gp
    n  = len(X_base)

    mean, var = gp.predict(X_base.astype(np.float64))
    mean = mean.ravel()
    var  = var.ravel().clip(min=1e-6)

    residuals = mean + np.sqrt(var) * rng.standard_normal(n)

    relevance = residual_model._feature_relevance  # (D,)
    D = X_base.shape[1]
    if len(relevance) == D:
        X_residuals = residuals[:, None] * relevance[None, :]
    else:
        X_residuals = np.tile(residuals[:, None], (1, D)) * 0.1

    return mean, var, X_residuals


# ── GP fitting ────────────────────────────────────────────────────────────────

def _fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    config: AmplifierConfig,
) -> GPy.models.GPRegression:
    """Fit a GPy RBF+ARD model on the residuals."""
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
    # Prevent noise from collapsing to zero (numerical instability)
    gp.Gaussian_noise.variance.constrain_bounded(1e-4, 1.0)
    gp.optimize(messages=False)
    return gp
