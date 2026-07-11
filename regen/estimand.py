"""
Estimand preservation — regression-coefficient recovery (the differentiator).

A researcher declares an analysis (``EstimandSpec``: ``outcome ~ predictors``,
family ols|logit). We fit it on the **real** reference to get θ_real ± CI, fit the
*same* spec on the **delivered synthetic** to get θ_synth, and CERTIFY that each
coefficient-of-interest's θ_synth lands within θ_real's confidence interval. This
is a guarantee distinct from fidelity (marginals/correlations) and TSTR
(prediction): a batch can pass both while a coefficient silently shifts.

Self-contained (numpy + scipy.stats) — deliberately **no statsmodels and no
sklearn solver** — so the recomputation in ``regen verify`` is deterministic to a
fixed tolerance on any machine (Invariants 2/7). OLS is the closed-form normal
equations with a t-interval; logit is Newton/IRLS with a Wald interval from the
inverse Fisher information.

Lives outside engine/ (Invariant 1): a coefficient is a *metric*, never a
synthetic value.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from contracts.scenario import EstimandSpec

# Newton/IRLS controls for logit — fixed so the fit is reproducible.
_IRLS_MAX_ITER = 100
_IRLS_TOL = 1e-10
_RIDGE = 1e-10  # tiny jitter so a singular design doesn't blow up the solve


class EstimandError(ValueError):
    """A declared estimand cannot be fit as specified (bad columns, degenerate)."""


def _design(df, spec: EstimandSpec) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Build (X_with_intercept, y, coefficient_names, n) from a frame.

    v1 requires numeric outcome + predictors; rows with any NaN in the used
    columns are dropped (listwise). Coefficient names are ["(Intercept)", *predictors].
    """
    cols = [spec.outcome, *spec.predictors]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise EstimandError(f"estimand columns absent from data: {missing}")

    sub = df[cols].apply(lambda s: __import__("pandas").to_numeric(s, errors="coerce"))
    nonnum = [c for c in cols if sub[c].isna().all() and not df[c].isna().all()]
    if nonnum:
        raise EstimandError(
            f"estimand v1 supports numeric columns only; non-numeric: {nonnum}")
    sub = sub.dropna()
    n = len(sub)
    p = len(spec.predictors) + 1  # + intercept
    if n <= p:
        raise EstimandError(
            f"too few complete rows ({n}) to fit {p} parameters")

    y = sub[spec.outcome].to_numpy(dtype=np.float64)
    Xp = sub[list(spec.predictors)].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(n), Xp])  # intercept first
    names = ["(Intercept)", *spec.predictors]
    return X, y, names, n


def _fit_ols(X, y, names, ci_level) -> Dict[str, Any]:
    n, p = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX + _RIDGE * np.eye(p))
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = n - p
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    tcrit = float(stats.t.ppf(0.5 + ci_level / 2.0, dof))
    return _pack(names, beta, se, tcrit, n, dof, family="ols")


def _fit_logit(X, y, names, ci_level) -> Dict[str, Any]:
    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({0.0, 1.0}):
        raise EstimandError(
            f"logit outcome must be binary 0/1; saw {sorted(uniq)[:5]}")
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(_IRLS_MAX_ITER):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-12, None)  # working weights
        XtWX = (X * w[:, None]).T @ X
        grad = X.T @ (y - mu)
        step = np.linalg.solve(XtWX + _RIDGE * np.eye(p), grad)
        beta_new = beta + step
        if np.max(np.abs(step)) < _IRLS_TOL:
            beta = beta_new
            break
        beta = beta_new
    # Wald covariance from the inverse Fisher information at the MLE.
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1.0 - mu), 1e-12, None)
    cov = np.linalg.inv((X * w[:, None]).T @ X + _RIDGE * np.eye(p))
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    zcrit = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    return _pack(names, beta, se, zcrit, n, n - p, family="logit")


def _pack(names, beta, se, crit, n, dof, family) -> Dict[str, Any]:
    coefs = {}
    for name, b, s in zip(names, beta, se):
        b, s = float(b), float(s)
        coefs[name] = {
            "coef": b, "se": s,
            "ci_low": b - crit * s, "ci_high": b + crit * s,
        }
    return {"family": family, "n": int(n), "dof": int(dof), "coefficients": coefs}


def fit_estimand(df, spec: EstimandSpec) -> Dict[str, Any]:
    """Fit the declared estimand on ``df`` → {family, n, dof, coefficients:{name:{coef,se,ci_low,ci_high}}}.

    Deterministic given the data. Raises ``EstimandError`` if the spec cannot be
    fit (missing/non-numeric columns, too few rows, non-binary logit outcome).
    """
    if not spec.declared():
        raise EstimandError("no estimand declared (need outcome + >=1 predictor)")
    if spec.family not in ("ols", "logit"):
        raise EstimandError(f"unsupported family: {spec.family!r}")
    X, y, names, n = _design(df, spec)
    if spec.family == "ols":
        return _fit_ols(X, y, names, spec.ci_level)
    return _fit_logit(X, y, names, spec.ci_level)


def certify(real_fit: Dict[str, Any], synth_fit: Dict[str, Any],
            spec: EstimandSpec) -> Dict[str, Any]:
    """Decide, per coefficient of interest, whether θ_synth preserves θ_real.

    Rule ``consistent`` (default): preserved iff the two estimates are not
    distinguishable beyond their combined standard error — a two-sample Wald test,
    ``|θ_real − θ_synth| ≤ z · √(se_real² + se_synth²)`` at level ``ci_level``. This
    accounts for BOTH samples' uncertainty, so it does not false-fail on genuinely
    preserved data (unlike a bare "point in the other's CI" check), and it reduces
    to that check as the synthetic set grows (se_synth → 0).

    Rule ``within_ci`` (stricter): preserved iff θ_synth ∈ θ_real's CI.

    The estimand is ``certified`` iff every declared target is preserved. Returns a
    self-describing verdict — every field recomputable by ``regen verify``.
    """
    real_c = real_fit.get("coefficients", {})
    synth_c = synth_fit.get("coefficients", {})
    zcrit = float(stats.norm.ppf(0.5 + spec.ci_level / 2.0))
    per: List[Dict[str, Any]] = []
    for name in spec.targets():
        r, s = real_c.get(name), synth_c.get(name)
        if r is None or s is None:
            per.append({"coefficient": name, "preserved": None,
                        "note": "coefficient absent from a fit"})
            continue
        delta = abs(r["coef"] - s["coef"])
        if spec.rule == "within_ci":
            preserved = bool(r["ci_low"] <= s["coef"] <= r["ci_high"])
            se_delta = None
            z = None
        else:  # "consistent" — two-sample Wald test
            se_delta = float(np.hypot(r["se"], s["se"]))
            z = delta / se_delta if se_delta > 0 else np.inf
            preserved = bool(z <= zcrit)
        # Honesty flag: if θ_real's own CI includes 0, there is no real signal to
        # preserve — "preservation" of a null effect is vacuous. Surfaced, not
        # (yet) failed; power-aware certification is a documented v2.
        real_significant = not (r["ci_low"] <= 0.0 <= r["ci_high"])
        per.append({
            "coefficient": name,
            "theta_real": r["coef"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "theta_synth": s["coef"], "delta": delta,
            "se_delta": se_delta, "z": z, "z_crit": zcrit,
            "real_significant": real_significant,
            "preserved": preserved,
        })
    checkable = [p for p in per if p["preserved"] is not None]
    certified = bool(checkable) and all(p["preserved"] for p in checkable)
    return {
        "rule": spec.rule, "ci_level": spec.ci_level, "family": spec.family,
        "certified": certified,
        "targets": per,
        "n_real": real_fit.get("n"), "n_synth": synth_fit.get("n"),
    }


def reference_aggregate(real_fit: Dict[str, Any], spec: EstimandSpec) -> Dict[str, Any]:
    """The disclosed θ_real ± SE aggregate embedded in reference_aggregates.json.

    A coefficient vector + its standard errors is an aggregate (no per-row values),
    so it obeys the same disclosure policy as the published correlation matrix. It
    is what lets ``regen verify`` re-run certification without the raw real rows.
    """
    return {
        "family": spec.family, "outcome": spec.outcome,
        "predictors": list(spec.predictors), "rule": spec.rule,
        "ci_level": spec.ci_level,
        "n": real_fit.get("n"), "dof": real_fit.get("dof"),
        "coefficients": real_fit.get("coefficients", {}),
    }


def evaluate(real_df, synth_df, spec: EstimandSpec):
    """Full estimand assessment for a generate() run. **Never raises** — a spec
    that cannot be fit becomes a status, not an exception (fail loud, not crash).

    Returns ``(assessment, real_fit)`` where ``assessment`` is the JSON-ready block
    for explanation.json and ``real_fit`` (or None) is handed to
    ``reference_aggregate`` for the bundle. ``assessment['status']`` is one of:
    ``not_declared`` | ``uncertifiable`` (fit failed / too little data) |
    ``certified`` | ``not_preserved``.
    """
    base = {"declared": bool(spec.declared()), "outcome": spec.outcome,
            "predictors": list(spec.predictors), "family": spec.family}
    if not spec.declared():
        return {**base, "status": "not_declared", "certified": None}, None
    try:
        real_fit = fit_estimand(real_df, spec)
        synth_fit = fit_estimand(synth_df, spec)
    except EstimandError as e:
        # No credible θ_real (or θ_synth) → refuse to certify, never fake it.
        # This is the estimand's readiness floor: the verification gap is never
        # filled with synthetic data.
        return {**base, "status": "uncertifiable", "certified": False,
                "reason": str(e)}, None
    verdict = certify(real_fit, synth_fit, spec)
    status = "certified" if verdict["certified"] else "not_preserved"
    assessment = {**base, "status": status, **verdict}
    return assessment, real_fit
