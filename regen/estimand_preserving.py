"""
Estimand-preserving generation (v2) — synthetic data that keeps the conclusion.

The certifier showed that marginals-plus-linear-correlation generators (Gaussian
copula, SMOTE, REGEN's amplifier) silently distort regression coefficients, and
that *perturbing* real rows for privacy destroys them regardless of noise shape
(docs/KNOWN_ISSUES #6). The construction that actually preserves a declared
estimand has two parts, both validated empirically:

  R1  sample predictors from a **Gaussian-mixture model of the real predictor
      joint** — novel rows, not perturbed real rows. A single Gaussian / copula is
      too weak and distorts *partial* coefficients (which depend on cov(x)); a
      richer mixture preserves them.
  R2  draw the outcome from a **calibrated model of the real conditional P(y|x)**
      (gradient-boosted — deliberately NOT the declared model form, so the
      coefficient is never injected, only recovered from a preserved conditional).

Deterministic given ``seed``. Every value comes from a statistical model that was
fit on real data and sampled — never from an LLM (Invariant 1) and never from the
declared coefficient (Invariant 4). It lives outside ``engine/``.

**Honest tradeoff:** this preserves inference by staying faithful to the real
joint, which costs privacy *distance* — it generates novel rows but sits nearer the
real data than a strong δ-floor would allow. More mixture components → better
estimand fidelity → less privacy. There is no free lunch; the certifier + a
nearest-neighbour distance report tell you where you are on that frontier
(docs/KNOWN_ISSUES #6). numpy + scikit-learn only; no new dependency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.mixture import GaussianMixture

from contracts.scenario import EstimandSpec


def generate_estimand_preserving(
    real_df: pd.DataFrame,
    estimand: EstimandSpec,
    n_rows: int = 6000,
    n_components: int = 20,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate synthetic ``[outcome, *predictors]`` that preserves ``estimand``.

    Returns a DataFrame with the estimand's outcome + predictor columns. Raises
    ``ValueError`` if the estimand is undeclared or its columns are missing.
    """
    if not estimand.declared():
        raise ValueError("estimand must be declared (outcome + >=1 predictor)")
    preds = list(estimand.predictors)
    cols = [estimand.outcome, *preds]
    missing = [c for c in cols if c not in real_df.columns]
    if missing:
        raise ValueError(f"columns absent from real data: {missing}")

    sub = real_df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    X = sub[preds].to_numpy(dtype=float)
    y = sub[estimand.outcome].to_numpy(dtype=float)

    # R1 — novel predictors from a rich model of the real joint. Standardise first:
    # a full-covariance GMM on raw columns is dominated by large-scale features and
    # models the small-scale joint (hence its partial coefficients) poorly.
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    k = max(1, min(n_components, max(1, len(sub) // 50)))
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=seed, reg_covar=1e-6).fit((X - mu) / sd)
    Xs, _ = gmm.sample(n_rows)
    Xs = Xs * sd + mu
    rng = np.random.default_rng(seed)
    rng.shuffle(Xs)  # GMM.sample() returns component-ordered rows

    # R2 — outcome from a calibrated model of the real conditional P(y|x).
    binary = set(np.unique(y).tolist()) <= {0.0, 1.0}
    if binary or estimand.family == "logit":
        clf = GradientBoostingClassifier(random_state=seed, max_depth=3,
                                         n_estimators=200).fit(X, y.astype(int))
        p = clf.predict_proba(Xs)[:, 1]
        ys = (rng.uniform(size=n_rows) < p).astype(int)
    else:
        reg = GradientBoostingRegressor(random_state=seed, max_depth=3,
                                        n_estimators=200).fit(X, y)
        resid_sd = float((y - reg.predict(X)).std())
        ys = reg.predict(Xs) + rng.normal(0, resid_sd, n_rows)

    out = pd.DataFrame(Xs, columns=preds)
    out[estimand.outcome] = ys
    return out[cols]
