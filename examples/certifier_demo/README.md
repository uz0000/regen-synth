# Does your synthetic data preserve the *conclusion*? — a generator-agnostic demo

Synthetic data is usually judged on two things: does it **look real** (fidelity)
and does a model **trained on it predict** (TSTR). Neither tells you whether the
*conclusion you'd draw from an analysis* survives. This demo shows the third axis
— **estimand preservation** — and that the certifier that measures it works on
**any** synthetic data, regardless of who produced it.

## The setup

- **Real data:** UCI *Default of Credit Card Clients* — 30,000 accounts, 22.1%
  default rate (`prepare_data.py` documents provenance).
- **The analysis a credit analyst would run:** a logistic regression
  `default ~ pay_delay_1 + utilization + log_limit + age`. They care about the
  **coefficients** — which factors drive default, and how much.
- **θ_real (the conclusion to preserve):**

  | coefficient | θ_real | 95% CI |
  |---|---|---|
  | pay_delay_1 | **+0.714** | [+0.685, +0.743] |
  | utilization | **−0.369** | [−0.449, −0.290] |
  | log_limit   | **−0.315** | [−0.348, −0.281] |
  | age         | **+0.010** | [+0.007, +0.013] |

## The result — certify the *same* analysis across six producers

Each producer makes a synthetic dataset; the certifier fits the analysis on each
and compares to θ_real (`✓` preserved / `✗` shifted, two-sample Wald test):

```
source                              certified     pay_delay_1   utilization     log_limit           age
-------------------------------------------------------------------------------------------------------
bootstrap_real  (positive control)  CERTIFIED        +0.649 ✓      -0.395 ✓      -0.369 ✓      +0.010 ✓
independent_cols(negative control)  refused          -0.007 ✗      -0.010 ✗      -0.010 ✗      -0.001 ✗
noised_real     (0.5σ anonymise)    refused          +0.521 ✗      -0.140 ✗      -0.294 ✓      +0.007 ✓
gaussian_copula (marginals+corr)    refused          +0.464 ✗      -0.314 ✓      -0.225 ✗      +0.011 ✓
SMOTE           (imblearn)          refused          +0.608 ✗      -0.469 ✓      -0.353 ✓      +0.015 ✓
REGEN           (this repo)         refused          +0.932 ✗      -0.194 ✓      -0.339 ✓      +0.009 ✓
estimand_preserving (GMM+cond,v2)   CERTIFIED        +0.731 ✓      -0.483 ✓      -0.330 ✓      +0.009 ✓
```

(Reproduce: `python examples/certifier_demo/run_demo.py` from the repo root.)

## What it shows

1. **The certifier discriminates.** A faithful source (bootstrap) certifies; a
   structure-destroying one (independent columns) fails every coefficient. It is
   not an always-fail rubber stamp — it passes what should pass.
2. **Every *marginals-based* method silently breaks the key coefficient.**
   Noised-real, a proper Gaussian copula, SMOTE, and REGEN **all** distort
   `pay_delay_1` — the strongest, discrete, non-linear predictor — while smooth
   continuous factors (`log_limit`, `age`) mostly survive. An analyst using any of
   these would over- or under-weight the single most important risk driver, and
   **no fidelity or prediction check would flag it.**
3. **The v2 generator preserves it — and certifies.** `estimand_preserving` models
   the predictor *joint* (a Gaussian mixture, not marginals+correlation) and draws
   the outcome from a calibrated model of the *real conditional* P(y|x); it recovers
   all four coefficients where every marginals-based method failed. That is the
   "here's what actually works" — the fix, verified by the same certificate.
4. **It is generator-agnostic.** θ_real is identical across all rows; only θ_synth
   differs. The certificate is about whether the conclusion survives, not about
   provenance — so it can certify data you did **not** generate.

## Why this is the product

The certificate is portable and recomputable: it carries θ_real ± SE, so a third
party can recompute θ_synth from the synthetic data alone and re-check the verdict
**without the real rows** — the "attach a trust certificate to synthetic data you
share" model. Prediction and fidelity are crowded, commoditising axes; *certified
inferential validity* is not, and this shows it working on real data across every
generator we tried.

## From finding to fix (v2)

REGEN's own generator is refused here — and that was the point, not an
embarrassment: the certifier caught its reference generator distorting a
coefficient, exactly as it would catch anyone else's. The pattern (copula, REGEN,
and — via interpolation — SMOTE all failing on the discrete, high-signal
`pay_delay_1`) has a diagnosed cause: they preserve marginals + linear correlation
but not the **conditional** structure a coefficient depends on.

The `estimand_preserving` row is the fix, built against that diagnosis: model the
predictor **joint** with a Gaussian mixture (novel rows, not perturbed real ones)
and draw the outcome from a calibrated model of the **real conditional** P(y|x) —
never the declared coefficient, so nothing is injected. It certifies.

**Honest limit — it is not free.** Preserving inference means staying faithful to
the real joint, which costs privacy *distance*: the estimand-preserving rows are
novel (no verbatim copies) but sit nearer the real data than a strong δ-floor would
allow, and perturbing them for more privacy re-breaks the coefficients. So the
achievement is a *navigable* frontier — you can now certify with novel synthetic
data, which perturbation never allowed — not a defeated one. The mechanism,
fix-validation, generality (OLS + a second dataset), and the measured
privacy↔inference frontier are in `docs/KNOWN_ISSUES.md` (#6).
