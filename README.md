# REGEN — does your synthetic data preserve the *conclusion*?

**Synthetic data can pass every fidelity and prediction check while silently
breaking the conclusions you'd draw from it. REGEN certifies whether a declared
analysis survives — for *any* synthetic dataset, whoever produced it — with a
recomputable certificate.**

Fidelity ("does it look real") and TSTR ("does a model trained on it predict")
are the usual quality bars. Neither tells you whether a **regression coefficient**
— the thing a risk model, a policy, or a published finding actually acts on — is
preserved. It often isn't, and the failure is **silent**: the data passes every
check while the coefficient shifts, attenuates, or flips sign. An analyst would
never know. This is the gap REGEN measures and certifies.

Everything is deterministic and recomputable — no LLM in the value or verification
path. Single-table, cross-sectional tabular only (not time-series, relational,
text, or images).

---

## The certifier (the core)

```python
from regen.certifier import certify_dataset
from contracts.scenario import EstimandSpec

# Declare the analysis whose conclusion must survive.
estimand = EstimandSpec(outcome="default",
                        predictors=["pay_delay_1", "utilization", "log_limit", "age"],
                        family="logit")

cert = certify_dataset(real_df, synthetic_df, estimand)
cert["certified"]     # True iff every declared coefficient is preserved
cert["targets"]       # per-coefficient: θ_real vs θ_synth, the two-sample test, preserved?
```

The certificate is **generator-agnostic** (it never asks who made the data),
**per-coefficient** (it tells you *which* conclusions survive, not a blunt
pass/fail), and **portable** — it carries θ_real ± SE, so a third party can
recompute θ_synth from the synthetic data alone and re-check the verdict *without
the real rows*. This is the "attach a trust certificate to synthetic data you
share" model. The estimand must be declared: you certify what you tell it matters,
which is what makes the guarantee precise and honest.

## What it shows — the demo ([`examples/certifier_demo/`](examples/certifier_demo/))

One logistic regression, certified across six synthetic producers on **real UCI
credit-default data** (`python examples/certifier_demo/run_demo.py`):

```
source                              certified     pay_delay_1   utilization     log_limit           age
bootstrap_real  (positive control)  CERTIFIED        +0.649 ✓      -0.395 ✓      -0.369 ✓      +0.010 ✓
independent_cols(negative control)  refused          -0.007 ✗      -0.010 ✗      -0.010 ✗      -0.001 ✗
gaussian_copula (marginals+corr)    refused          +0.464 ✗      -0.314 ✓      -0.225 ✗      +0.011 ✓
SMOTE           (imblearn)          refused          +0.608 ✗      -0.469 ✓      -0.353 ✓      +0.015 ✓
REGEN           (this repo)         refused          +0.932 ✗      -0.194 ✓      -0.339 ✓      +0.009 ✓
```

A faithful source (bootstrap) certifies; **every practical method silently breaks
the strongest, discrete predictor** (`pay_delay_1`) while fidelity and prediction
flag none of it. Same θ_real across all rows — it's about whether the conclusion
survives, not provenance. Full write-up: [`examples/certifier_demo/README.md`](examples/certifier_demo/README.md).

## The honest limit — the privacy↔inference frontier

Preserving inference and protecting privacy pull against each other. Measured on
the credit data, full certification collapses at ~0.1σ of privacy noise — *before*
you gain meaningful privacy — because a regression coefficient depends on the
predictor joint that perturbation distorts. The certifier's value is telling you,
**per conclusion**, exactly where you are on that frontier instead of shipping data
everyone falsely believes preserves their analysis.

**Preserving inference — the v2 generator** ([`regen/estimand_preserving.py`](regen/estimand_preserving.py)).
`generate_estimand_preserving(real_df, estimand)` produces synthetic data a declared
analysis *survives* — it models the predictor **joint** with a Gaussian mixture (novel
rows, not perturbed real ones) and draws the outcome from a calibrated model of the
**real conditional** P(y|x) (never the declared coefficient, so nothing is injected).
On the demo it **certifies** where the copula, SMOTE, REGEN, and every perturbation
method are refused — the `estimand_preserving` row above. The honest cost: it stays
faithful to the real joint, so it trades away privacy *distance* (novel rows, but
nearer real than a strong δ-floor). The frontier is navigable, not free. Mechanism,
fix-validation, and generality (OLS + a second dataset): [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) (#4–#6).

---

## The reference generator (REGEN)

REGEN also *is* a deterministic rare-event synthetic-data generator — the reference
implementation the certifier was built against, and the source of the finding that
motivated it (**its own output is refused above** — the certifier catches its
reference generator, exactly as it would anyone else's). It runs an active-learning
campaign — Scout (targeting) → Prior (grounded sampling) → Amplifier (tail
correction) → Auditor (fidelity gate) → Examiner (detection lift) — and ships each
batch with a `ScenarioSpec`, an `explanation.json`, and an audit bundle you can
re-check with `regen verify`. Privacy is on by default (δ-distance floor + verbatim
guard; **not** differential privacy — see [`docs/PRIVACY.md`](docs/PRIVACY.md)).

```bash
regen generate my_data.csv --label is_fraud     # generate a synthetic dataset
regen doctor   my_data.csv --label is_fraud     # preflight: fits the envelope?
regen verify   regen-output/                    # independently recompute a batch's stats
regen screen   my_data.csv --label is_fraud     # REGEN vs SMOTE for this dataset
```

Honest generation numbers (leakage-free TSTR + conditional lift, including where
REGEN doesn't help) live in [`benchmark/RESULTS_TSTR.md`](benchmark/RESULTS_TSTR.md)
and [`docs/BUILDLOG.md`](docs/BUILDLOG.md); the amplifier helps only when the
baseline is genuinely starved of rare examples.

## Scope & honesty

- **Single-table, cross-sectional** tabular only (not time-series/relational/text/images).
- **Estimand certification v1** covers **numeric** predictors and OLS/logit; a declared
  analysis is required. Power-aware certification, categorical predictors, and ATE are
  on the roadmap ([`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)).
- Deterministic and recomputable throughout; **not** differential privacy.

## Documentation

- Known limits, the v2 investigation + frontier: [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)
- Build log (every change, before/after observed): [`docs/BUILDLOG.md`](docs/BUILDLOG.md)
- Full system reference: [`docs/REGEN_DOCUMENTATION.md`](docs/REGEN_DOCUMENTATION.md)
- Privacy (what's guaranteed and what isn't): [`docs/PRIVACY.md`](docs/PRIVACY.md)
- Statistical methods + verification: [`docs/METHODS.md`](docs/METHODS.md)

## License

MIT
