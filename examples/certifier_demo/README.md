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
```

(Reproduce: `python examples/certifier_demo/run_demo.py` from the repo root.)

## What it shows

1. **The certifier discriminates.** A faithful source (bootstrap) certifies; a
   structure-destroying one (independent columns) fails every coefficient. It is
   not an always-fail rubber stamp — it passes what should pass.
2. **Every practical synthetic method silently breaks the key coefficient.**
   Noised-real, a proper Gaussian copula, SMOTE, and REGEN **all** distort
   `pay_delay_1` — the strongest, discrete, non-linear predictor — while smooth
   continuous factors (`log_limit`, `age`) mostly survive. An analyst using any of
   these would over- or under-weight the single most important risk driver, and
   **no fidelity or prediction check would flag it.**
3. **It is generator-agnostic.** θ_real is identical across all six rows; only
   θ_synth differs. The certificate is about whether the conclusion survives, not
   about provenance — so it can certify data you did **not** generate.

## Why this is the product

The certificate is portable and recomputable: it carries θ_real ± SE, so a third
party can recompute θ_synth from the synthetic data alone and re-check the verdict
**without the real rows** — the "attach a trust certificate to synthetic data you
share" model. Prediction and fidelity are crowded, commoditising axes; *certified
inferential validity* is not, and this shows it working on real data across every
generator we tried.

## The honest finding (→ v2)

REGEN's own generator is refused here — and that is the point, not an
embarrassment: the certifier caught its reference generator distorting a
coefficient, exactly as it would catch anyone else's. The pattern across **all**
Gaussian-dependence methods (copula, REGEN, and — via interpolation — SMOTE)
failing on the discrete, high-signal `pay_delay_1` is the concrete target for the
v2 generator investigation: *why do marginals-plus-linear-correlation methods lose
the conditional structure of discrete, non-linear predictors, and what generation
change preserves it?* See `docs/KNOWN_ISSUES.md`.
