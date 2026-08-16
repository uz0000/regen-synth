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

## Install

```bash
git clone https://github.com/uz0000/regen-synth.git && cd regen-synth
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                              # installs `regen`, `contracts`, `engine`, `cli`, `server`

pytest tests/ -q                               # 216 tests
python examples/certifier_demo/run_demo.py     # reproduces the table below
```

Requires Python 3.10+. `GPy` (the Amplifier's Gaussian-process backend) can be
slow to build on some platforms — if `pip install -r requirements.txt` fails there,
install `numpy pandas scipy scikit-learn pyarrow` on their own to run the certifier
and CLI without the tail-correction amplifier.

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
`generate_estimand_preserving(real_df, estimand)` models the predictor **joint** with a
Gaussian mixture (novel rows, not perturbed real ones) and draws the outcome from a
calibrated model of the **real conditional** P(y|x) (never the declared coefficient, so
nothing is injected). Measured across a 30-seed sweep on the credit demo
(`python examples/certifier_demo/seed_sweep.py`), two of the four coefficients recover
essentially unbiased — something no other generator here achieves — and the full
analysis certifies on **11/30 seeds (37%)**, against 0–1/4 coefficients for every other
generator tested. The other two coefficients carry a real, systematic bias, not noise
(`utilization` attenuated ~46%, `log_limit` ~19%) — a partial-correlation-sensitivity
gap in how the Gaussian mixture approximates the real joint, quantified and explained
in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md). It also trades away privacy
*distance* (novel rows, but nearer real than a strong δ-floor). The frontier is
navigable, not free, and not fully closed — an honest, partial fix, not a solved
problem.

---

## The reference generator (REGEN)

REGEN also *is* a synthetic-data generator in its own right — the system the
certifier was originally built to check, and the source of the finding that
motivated it (**its own output is refused above** — the certifier catches its
own reference generator, exactly as it would anyone else's). It works in five
plain steps, run in a loop that focuses effort on the rare cases:

1. **Scout** picks which rare region of the data most needs more synthetic
   examples.
2. **Prior** draws base synthetic rows grounded in the real data's statistics
   (not copies of real rows).
3. **Amplifier** densifies and corrects that rare region specifically, since a
   generic sampler under-represents it.
4. **Auditor** checks the delivered batch against the real data and rejects it
   if the structure is broken — a hard gate, not a warning.
5. **Examiner** measures whether adding the synthetic data actually improves a
   downstream detection model, honestly (only claims a lift when there is one).

Each batch ships with a `ScenarioSpec` (what was asked for), an
`explanation.json` (why the batch passed, in computed numbers), and an audit
bundle you can independently re-check with `regen verify`. Privacy is on by
default (δ-distance floor + verbatim guard; **not** differential privacy — see
[`docs/PRIVACY.md`](docs/PRIVACY.md)).

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

- Known limits + the v2 investigation and its correction: [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)
- How each mechanism actually works: [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)
- System layout — how the components connect and why: [`docs/PRODUCT_SPEC.md`](docs/PRODUCT_SPEC.md)
- Method ↔ file ↔ why, plus FAQ: [`docs/COMPONENT_GUIDE.md`](docs/COMPONENT_GUIDE.md)
- Privacy (what's guaranteed and what isn't): [`docs/PRIVACY.md`](docs/PRIVACY.md)
- Statistical methods + verification: [`docs/METHODS.md`](docs/METHODS.md)
- What's supported, degraded, or out of scope: [`docs/CAPABILITY_MATRIX.md`](docs/CAPABILITY_MATRIX.md)
- Build log (every change, with before/after numbers): [`docs/BUILDLOG.md`](docs/BUILDLOG.md)

## License

MIT
