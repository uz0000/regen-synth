"""
Do the standard quality checks catch what the certifier catches?

The claim this repo rests on is that a synthetic table can pass the checks the
field normally applies and still move the regression coefficient someone acts on.
That claim is only worth making if the standard checks are actually run, so this
script runs them on the same sources as `run_demo.py` and prints the two verdicts
side by side.

Three checks, all standard:

  marginal agreement   per-column Kolmogorov-Smirnov distance between the real
                       and synthetic empirical CDFs, reported as the maximum
                       across columns. This is the "does each column look right"
                       check.

  dependence agreement maximum absolute difference between the real and
                       synthetic Pearson correlation matrices. This is the "do
                       the columns move together" check.

  TSTR                 train a gradient-boosted classifier on the synthetic
                       table, score it on a held-out real test split, and divide
                       by the ROC-AUC of the same learner trained on real data.
                       1.0 means the synthetic table is a full stand-in for
                       training. This is the "can you model on it" check.

Thresholds are the conventional ones (KS <= 0.10, |Δρ| <= 0.10, TSTR >= 0.95).
They are stated here rather than imported so the pass/fail rule is visible at the
point of use.

Run from the repo root:  python examples/certifier_demo/fidelity_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from contracts.scenario import EstimandSpec
from regen.certifier import certify_many

import run_demo as demo

HERE = Path(__file__).resolve().parent
KS_MAX = 0.10
RHO_MAX = 0.10
TSTR_MIN = 0.95


def marginal_ks(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Largest per-column KS distance over the PREDICTORS.

    The outcome is excluded on purpose. SMOTE and the rare-event pipeline
    rebalance the class deliberately, so their outcome marginal differs by
    design and nobody applying a fidelity check would score them on it. The
    class rate is reported separately instead.
    """
    return max(float(ks_2samp(real[c].to_numpy(), synth[c].to_numpy()).statistic)
               for c in demo.PREDICTORS)


def correlation_delta(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Largest absolute shift in the correlation matrix."""
    a = real[demo.PREDICTORS].corr().to_numpy()
    b = synth[demo.PREDICTORS].corr().to_numpy()
    return float(np.nanmax(np.abs(a - b)))


def tstr(real: pd.DataFrame, synth: pd.DataFrame, seed: int = 0) -> float:
    """ROC-AUC of a model trained on synthetic, over one trained on real.

    Both are scored on the same held-out real split, so the comparison isolates
    the training source.
    """
    tr, te = train_test_split(real, test_size=0.3, random_state=seed,
                              stratify=real["default"])
    x_te, y_te = te[demo.PREDICTORS].to_numpy(), te["default"].to_numpy()

    def auc(frame):
        y = frame["default"].to_numpy()
        if len(np.unique(y)) < 2:
            return float("nan")
        m = HistGradientBoostingClassifier(random_state=seed)
        m.fit(frame[demo.PREDICTORS].to_numpy(), y)
        return roc_auc_score(y_te, m.predict_proba(x_te)[:, 1])

    base = auc(tr)
    return float(auc(synth) / base) if base and not np.isnan(base) else float("nan")


def main():
    real = pd.read_csv(demo.CSV)
    spec = EstimandSpec(outcome="default", predictors=demo.PREDICTORS,
                        family="logit")

    producers = {
        "bootstrap_real  (positive control)": demo.g_bootstrap,
        "independent_cols(negative control)": demo.g_independent,
        "noised_real     (0.5s anonymise)":   demo.g_noised,
        "gaussian_copula (marginals+corr)":   demo.g_gaussian_copula,
        "SMOTE           (imblearn)":         demo.g_smote,
        "REGEN           (this repo)":        demo.g_regen,
        "estimand_preserving (GMM+cond,v2)":  demo.g_estimand_preserving,
    }
    synthetics = {name: fn(real) for name, fn in producers.items()}
    certs = certify_many(real, synthetics, spec)

    rows = []
    for name, syn in synthetics.items():
        ks, rho, ts = marginal_ks(real, syn), correlation_delta(real, syn), tstr(real, syn)
        passes = (ks <= KS_MAX) and (rho <= RHO_MAX) and (ts >= TSTR_MIN)
        rows.append({
            "source": name,
            "KS": ks,
            "corr": rho,
            "TSTR": ts,
            "rate": float(syn["default"].mean()),
            "standard": "pass" if passes else "FAIL",
            "certified": "yes" if certs[name]["certified"] else "no",
        })

    print(f"Real data: {len(real):,} rows\n")
    print(f"Standard checks: KS <= {KS_MAX}, |dcorr| <= {RHO_MAX}, TSTR >= {TSTR_MIN}\n")
    head = (f"{'source':<36}{'KS':>7}{'dcorr':>8}{'TSTR':>7}{'rate':>7}"
            f"{'standard':>10}{'certified':>11}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['source']:<36}{r['KS']:>7.3f}{r['corr']:>8.3f}{r['TSTR']:>7.3f}"
              f"{r['rate']:>7.3f}{r['standard']:>10}{r['certified']:>11}")
    print(f"\n(real class rate {real['default'].mean():.3f}; the outcome marginal is "
          f"excluded from KS because some methods rebalance it by design)")

    gap = [r for r in rows if r["standard"] == "pass" and r["certified"] == "no"]
    print(f"\n{len(gap)} of {len(rows)} sources pass every standard check and "
          f"still fail certification.")
    for r in gap:
        print(f"    {r['source'].strip()}")

    # The field does not agree on where these thresholds sit, and the answer
    # above depends on them. Report that dependence rather than hiding it.
    grid = [(0.10, 0.10, 0.95), (0.15, 0.10, 0.95), (0.15, 0.15, 0.90),
            (0.20, 0.15, 0.90), (0.20, 0.20, 0.85)]
    print("\nHow many silent failures, as a function of where the lines are drawn:\n")
    print(f"{'KS <=':>7}{'|dcorr| <=':>12}{'TSTR >=':>9}   silent failures")
    print("-" * 55)
    sens = []
    for ks_m, rho_m, ts_m in grid:
        hits = [r for r in rows
                if r["certified"] == "no" and r["KS"] <= ks_m
                and r["corr"] <= rho_m and r["TSTR"] >= ts_m]
        names = ", ".join(r["source"].split()[0] for r in hits) or "none"
        sens.append((ks_m, rho_m, ts_m, len(hits), names))
        print(f"{ks_m:>7.2f}{rho_m:>12.2f}{ts_m:>9.2f}   {len(hits)}  ({names})")

    out = HERE / "FIDELITY.md"
    lines = [
        "<!-- Generated by examples/certifier_demo/fidelity_check.py. Do not edit. -->",
        "",
        "# Do the standard checks catch it?",
        "",
        f"Thresholds: KS &le; {KS_MAX}, |&Delta;&rho;| &le; {RHO_MAX}, "
        f"TSTR &ge; {TSTR_MIN}.",
        "",
        "KS and &Delta;&rho; are over the predictors only. Some methods rebalance the "
        "outcome by design, so the class rate is reported separately rather than scored.",
        "",
        "| source | KS | &Delta;&rho; | TSTR | class rate | standard checks | certified |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['source'].strip()}` | {r['KS']:.3f} | {r['corr']:.3f} | "
                     f"{r['TSTR']:.3f} | {r['rate']:.3f} | {r['standard']} | "
                     f"{r['certified']} |")
    lines += ["",
              f"**{len(gap)} of {len(rows)} sources pass every standard check and still "
              f"fail certification** at the thresholds above.",
              "",
              "## The answer depends on where the lines are drawn",
              "",
              "The field does not standardise these cut-offs, and the count moves with "
              "them. Every row below is the same seven sources and the same "
              "certification verdicts; only the pass marks change.",
              "",
              "| KS &le; | &#124;&Delta;&rho;&#124; &le; | TSTR &ge; | silent failures | which |",
              "|---|---|---|---|---|"]
    for ks_m, rho_m, ts_m, n, names in sens:
        lines.append(f"| {ks_m:.2f} | {rho_m:.2f} | {ts_m:.2f} | {n} | {names} |")
    lines.append("")
    out.write_text("\n".join(lines))
    print(f"\nWrote {out.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
