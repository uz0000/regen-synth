"""
Is the coefficient failure specific to one dataset and one model family?

The credit demo is a logistic regression on a discrete ordinal predictor. If the
failure only shows up there, it is a quirk of that table. This script re-runs the
same comparison on a different dataset and a different estimand family: ordinary
least squares on California housing (20,640 census block groups, sklearn's copy
of the 1990 census extract).

Same structure as the credit demo. Fit the declared regression on the real data,
fit it again on each synthetic source, compare coefficient by coefficient with the
same two-sample test.

Run from the repo root:  python examples/certifier_demo/generality_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.datasets import fetch_california_housing

from contracts.scenario import EstimandSpec
from regen.certifier import certify_many

HERE = Path(__file__).resolve().parent
PREDICTORS = ["MedInc", "HouseAge", "AveRooms", "Latitude"]
OUTCOME = "MedHouseVal"
COLS = [OUTCOME] + PREDICTORS
N = 6000
SEED = 7


def load_real() -> pd.DataFrame:
    frame = fetch_california_housing(as_frame=True).frame[COLS].copy()
    # Trim the AveRooms tail. A handful of block groups report averages in the
    # hundreds, which is a reporting artifact of very small populations and
    # dominates any covariance-based sampler.
    hi = frame["AveRooms"].quantile(0.995)
    return frame[frame["AveRooms"] <= hi].reset_index(drop=True)


def g_bootstrap(real: pd.DataFrame) -> pd.DataFrame:
    """Positive control: a resample of the real rows. Must certify."""
    return real.sample(N, replace=True, random_state=SEED).reset_index(drop=True)


def g_independent(real: pd.DataFrame) -> pd.DataFrame:
    """Negative control: every marginal preserved, every association destroyed."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame({c: rng.choice(real[c].to_numpy(), N, replace=True)
                         for c in COLS})


def g_gaussian_copula(real: pd.DataFrame) -> pd.DataFrame:
    """Marginals plus a single linear rank correlation, which is the common baseline."""
    rng = np.random.default_rng(SEED)
    z = np.column_stack([norm.ppf((real[c].rank(method="average") - 0.5) / len(real))
                         for c in COLS])
    z = np.clip(z, -5, 5)
    chol = np.linalg.cholesky(np.corrcoef(z, rowvar=False) + 1e-8 * np.eye(len(COLS)))
    u = norm.cdf(rng.standard_normal((N, len(COLS))) @ chol.T)
    return pd.DataFrame({c: np.quantile(real[c].to_numpy(), u[:, j])
                         for j, c in enumerate(COLS)})


def g_noised(real: pd.DataFrame) -> pd.DataFrame:
    """Additive Gaussian perturbation of real rows, the usual anonymisation move."""
    rng = np.random.default_rng(SEED)
    out = real.sample(N, replace=True, random_state=SEED).reset_index(drop=True).copy()
    for c in PREDICTORS:
        out[c] = out[c] + rng.normal(0, 0.5 * real[c].std(), size=N)
    return out


def g_conditional_on_real_x(real: pd.DataFrame) -> pd.DataFrame:
    """Real predictor joint, outcome redrawn from a model of the real conditional.

    This isolates the two requirements. The predictor joint is exactly right
    because it is resampled from real rows, so anything that still breaks must
    come from the conditional model rather than the covariance structure.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(SEED)
    model = HistGradientBoostingRegressor(random_state=SEED)
    model.fit(real[PREDICTORS].to_numpy(), real[OUTCOME].to_numpy())
    resid = real[OUTCOME].to_numpy() - model.predict(real[PREDICTORS].to_numpy())
    scale = float(resid.std())

    out = real[PREDICTORS].sample(N, replace=True, random_state=SEED).reset_index(drop=True)
    out[OUTCOME] = model.predict(out[PREDICTORS].to_numpy()) + rng.normal(0, scale, N)
    return out[COLS]


def g_conditional_on_copula_x(real: pd.DataFrame) -> pd.DataFrame:
    """Copula predictor joint, same conditional model for the outcome.

    Differs from the row above only in where the predictors came from, so the
    gap between the two rows is attributable to the predictor joint alone.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(SEED)
    model = HistGradientBoostingRegressor(random_state=SEED)
    model.fit(real[PREDICTORS].to_numpy(), real[OUTCOME].to_numpy())
    resid = real[OUTCOME].to_numpy() - model.predict(real[PREDICTORS].to_numpy())
    scale = float(resid.std())

    out = g_gaussian_copula(real)[PREDICTORS].copy()
    out[OUTCOME] = model.predict(out[PREDICTORS].to_numpy()) + rng.normal(0, scale, N)
    return out[COLS]


def main():
    real = load_real()
    spec = EstimandSpec(outcome=OUTCOME, predictors=PREDICTORS, family="ols")

    producers = {
        "bootstrap_real      (positive control)": g_bootstrap,
        "independent_cols    (negative control)": g_independent,
        "noised_real         (0.5s anonymise)":   g_noised,
        "gaussian_copula     (marginals+corr)":   g_gaussian_copula,
        "conditional | real x    (joint right)":  g_conditional_on_real_x,
        "conditional | copula x  (joint wrong)":  g_conditional_on_copula_x,
    }
    synthetics = {name: fn(real) for name, fn in producers.items()}
    certs = certify_many(real, synthetics, spec)

    any_cert = next(c for c in certs.values() if "theta_real_disclosed" in c)
    tr = any_cert["theta_real_disclosed"]["coefficients"]

    print(f"California housing: {len(real):,} rows")
    print(f"Analysis: OLS  {OUTCOME} ~ {' + '.join(PREDICTORS)}\n")
    print("Real coefficients:")
    for p in PREDICTORS:
        print(f"    {p:>12}: {tr[p]['coef']:+.4f}")

    head = f"\n{'source':<40}{'certified':<11}" + "".join(f"{p:>13}" for p in PREDICTORS)
    print(head)
    print("-" * (len(head) - 1))
    rows = []
    for name, cert in certs.items():
        tgt = {t["coefficient"]: t for t in cert.get("targets", [])}
        cells, marks = "", {}
        for p in PREDICTORS:
            t = tgt.get(p)
            ok = bool(t and t["preserved"])
            marks[p] = (t["theta_synth"] if t else float("nan"), ok)
            cells += (f"{t['theta_synth']:+.3f} {'OK' if ok else 'x'}".rjust(13)
                      if t else "n/a".rjust(13))
        verdict = "CERTIFIED" if cert["certified"] else "refused"
        print(f"{name:<40}{verdict:<11}{cells}")
        rows.append((name, cert["certified"], marks))

    md = ["<!-- Generated by examples/certifier_demo/generality_check.py. Do not edit. -->",
          "",
          "# Does it hold on another dataset and another model family?",
          "",
          f"OLS on California housing, {len(real):,} block groups, "
          f"`{OUTCOME} ~ {' + '.join(PREDICTORS)}`. Same certifier, same test as the "
          f"credit demo, which was a logistic regression.",
          "",
          "| coefficient | &theta;_real |",
          "|---|---|"]
    for p in PREDICTORS:
        md.append(f"| `{p}` | {tr[p]['coef']:+.4f} |")
    md += ["",
           "| source | certified | " + " | ".join(f"`{p}`" for p in PREDICTORS) + " |",
           "|---|---|" + "---|" * len(PREDICTORS)]
    for name, ok, marks in rows:
        cells = " | ".join(
            f"{v:+.3f} {'&#10003;' if good else '&#10007;'}"
            for v, good in (marks[p] for p in PREDICTORS))
        md.append(f"| `{name.strip()}` | "
                  f"{'**certified**' if ok else 'refused'} | {cells} |")
    md.append("")
    (HERE / "GENERALITY.md").write_text("\n".join(md))
    print(f"\nTable → {(HERE / 'GENERALITY.md').name}")


if __name__ == "__main__":
    main()
