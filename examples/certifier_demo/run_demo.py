"""
Generator-agnostic certifier demo — does synthetic data preserve the CONCLUSION?

One real dataset (UCI credit-card default), one declared analysis (a logistic
regression a credit analyst would actually run), and several synthetic producers.
The certifier fits the analysis on the real data ONCE (θ_real) and on each
synthetic set (θ_synth), then reports — per coefficient — whether the estimate is
preserved. It does not care who made the data; only whether the conclusion holds.

The point: fidelity and prediction can pass while the coefficients you'd act on
shift. The certificate is the only check watching that axis — and it discriminates
(a faithful source certifies; a structure-destroying one does not).

Run from the repo root:  python examples/certifier_demo/run_demo.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from contracts.scenario import (ScenarioSpec, ScenarioIntent, ScenarioGates,
                                EstimandSpec)
from regen.api import generate
from regen.certifier import certify_many

HERE = Path(__file__).resolve().parent
CSV = HERE / "credit_default.csv"
PREDICTORS = ["pay_delay_1", "utilization", "log_limit", "age"]
COLS = ["default"] + PREDICTORS
N = 6000
SEED = 7


# ── Synthetic producers (baselines are home-grown; REGEN is the real pipeline) ──

def g_bootstrap(real):
    """Positive control: resample the real rows. Preserves everything → should certify."""
    return real[COLS].sample(N, replace=True, random_state=SEED).reset_index(drop=True)

def g_independent(real):
    """Negative control: sample each column independently → destroys all joint structure."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame({c: rng.choice(real[c].to_numpy(), size=N) for c in COLS})

def g_noised(real):
    """Naive anonymisation: real rows + 0.5σ Gaussian noise on predictors (regression dilution)."""
    rng = np.random.default_rng(SEED)
    d = real[COLS].sample(N, replace=True, random_state=SEED).reset_index(drop=True).copy()
    for c in PREDICTORS:
        d[c] = d[c] + rng.normal(0, 0.5 * real[c].std(), size=N)
    return d

def g_gaussian_copula(real):
    """Gaussian copula: preserves each marginal + linear (rank) correlation, not conditional structure."""
    rng = np.random.default_rng(SEED)
    R = real[COLS]
    Z = np.column_stack([norm.ppf((R[c].rank(method="average") - 0.5) / len(R)) for c in COLS])
    Z = np.clip(Z, -5, 5)
    L = np.linalg.cholesky(np.corrcoef(Z, rowvar=False) + 1e-8 * np.eye(len(COLS)))
    U = norm.cdf(rng.standard_normal((N, len(COLS))) @ L.T)
    d = pd.DataFrame({c: np.quantile(R[c].to_numpy(), U[:, j]) for j, c in enumerate(COLS)})
    d["default"] = (d["default"] >= 0.5).astype(int)  # binarise the outcome back
    return d

def g_smote(real):
    """SMOTE: interpolate synthetic minority (default=1) samples between nearest neighbours."""
    from imblearn.over_sampling import SMOTE
    Xr, yr = SMOTE(random_state=SEED).fit_resample(real[PREDICTORS].to_numpy(),
                                                   real["default"].to_numpy())
    d = pd.DataFrame(Xr, columns=PREDICTORS); d["default"] = yr
    return d.sample(min(N, len(d)), random_state=SEED).reset_index(drop=True)

def g_regen(real):
    """REGEN's own pipeline (grounded sampling + copula + tail amplifier)."""
    spec = ScenarioSpec(
        intent=ScenarioIntent(label_col="default", rare_mode="label", rare_value=1,
                              n_rows=N, seed=SEED),
        gates=ScenarioGates(privacy="none"))
    with tempfile.TemporaryDirectory() as d:
        s = generate(str(CSV), scenario=spec, out_dir=d)
        return pd.read_parquet(s["best_batch_path"])[COLS]

def g_estimand_preserving(real):
    """v2: GMM model of the predictor joint + calibrated real conditional P(y|x).
    Preserves the declared analysis where the marginals+correlation methods fail."""
    from regen.estimand_preserving import generate_estimand_preserving
    return generate_estimand_preserving(
        real, EstimandSpec(outcome="default", predictors=PREDICTORS, family="logit"),
        n_rows=N, seed=SEED)


def main():
    real = pd.read_csv(CSV)
    spec = EstimandSpec(outcome="default", predictors=PREDICTORS, family="logit")
    print(f"Real data: {len(real):,} rows, default rate {real['default'].mean():.1%}")
    print(f"Analysis:  logit  default ~ {' + '.join(PREDICTORS)}\n")

    producers = {
        "bootstrap_real  (positive control)": g_bootstrap,
        "independent_cols(negative control)": g_independent,
        "noised_real     (0.5σ anonymise)":   g_noised,
        "gaussian_copula (marginals+corr)":   g_gaussian_copula,
        "SMOTE           (imblearn)":         g_smote,
        "REGEN           (this repo)":        g_regen,
        "estimand_preserving (GMM+cond,v2)":  g_estimand_preserving,
    }
    synthetics = {name: fn(real) for name, fn in producers.items()}
    certs = certify_many(real, synthetics, spec)

    # θ_real is identical across sources — pull it from any certificate.
    any_cert = next(c for c in certs.values() if "theta_real_disclosed" in c)
    tr = any_cert["theta_real_disclosed"]["coefficients"]
    print("θ_real (the conclusion to preserve):")
    for p in PREDICTORS:
        print(f"    {p:>12}: {tr[p]['coef']:+.4f}  CI[{tr[p]['ci_low']:+.4f}, {tr[p]['ci_high']:+.4f}]")

    print("\nPer-source coefficient preservation ( ✓ preserved / ✗ shifted ):\n")
    head = f"{'source':<36}{'certified':<11}" + "".join(f"{p:>14}" for p in PREDICTORS)
    print(head); print("-" * len(head))
    for name, cert in certs.items():
        tgt = {t["coefficient"]: t for t in cert.get("targets", [])}
        cells = ""
        for p in PREDICTORS:
            t = tgt.get(p)
            mark = "✓" if t and t["preserved"] else "✗"
            cells += f"{t['theta_synth']:+.3f} {mark}".rjust(14) if t else "n/a".rjust(14)
        verdict = "CERTIFIED" if cert["certified"] else "refused"
        print(f"{name:<36}{verdict:<11}{cells}")

    out = HERE / "certificates.json"
    out.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "theta_real_disclosed"}
         for k, v in certs.items()}, indent=2, default=str))
    n_cert = sum(1 for c in certs.values() if c["certified"])
    print(f"\n{n_cert}/{len(certs)} sources certified. Certificates → {out.name}")
    print("Same analysis, same θ_real — only the source's fidelity to the conclusion differs.")


if __name__ == "__main__":
    main()
