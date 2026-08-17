# REGEN — does synthetic data preserve the *conclusion*?

**The finding: every synthetic-data generator tested — a Gaussian copula, SMOTE,
and this repo's own generator — passes the standard quality checks while breaking
the regression coefficients someone would actually act on. Nothing in the normal
workflow catches it.**

Fidelity ("does it look real") and TSTR ("does a model trained on it predict")
are the usual quality bars. Neither tells you whether a **regression coefficient**
— the thing a risk model, a policy, or a published finding actually acts on — is
preserved. It often isn't, and the failure is **silent**: the data passes every
check while the coefficient shifts, attenuates, or flips sign. An analyst would
never know.

To measure that, this repo builds a certifier: declare the analysis you care
about, and it fits that analysis on the real and synthetic data and reports
per-coefficient agreement. It is generator-agnostic, so the finding above covers
other tools' output, not just this one's.

**What it does not solve.** Certification requires the real data — `certify_dataset`
takes `real_df` — so it addresses *sharing* (you hold the truth, someone else
doesn't) and not *scarcity* (nobody has enough data). If you can compute the real
answer to check against, you did not need the synthetic data. That limit is
structural, not an implementation gap.

Everything is deterministic and recomputable — no LLM in the value or verification
path. Single-table, cross-sectional tabular only (not time-series, relational,
text, or images).

**Relationship to [`regen-basic`](https://github.com/uz0000/regen-basic).** That
repo is the compact version of the same question: one simulator, one check, three
dependencies, no model call anywhere. This repo is the larger system — rare-event
amplification, a privacy floor, an audit bundle, and a second generator built
specifically to preserve a declared analysis.

Install them into **separate environments**. Both claim the top-level package
names `engine`, `contracts` and `cli`, so a shared environment resolves those
imports to whichever was installed last and the `regen` and `synth` commands
shadow each other.

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

One logistic regression, run against seven sources of the same data on **real UCI
credit-default data** — 30,000 rows, default rate 22.1%
(`python examples/certifier_demo/run_demo.py`):

```
θ_real (the conclusion that has to survive):
     pay_delay_1: +0.7141  CI[+0.6850, +0.7432]
     utilization: -0.3693  CI[-0.4489, -0.2897]
       log_limit: -0.3145  CI[-0.3476, -0.2814]
             age: +0.0100  CI[+0.0069, +0.0131]

source                              certified     pay_delay_1   utilization     log_limit           age
bootstrap_real  (positive control)  CERTIFIED        +0.649 ✓      -0.395 ✓      -0.369 ✓      +0.010 ✓
independent_cols(negative control)  refused          -0.007 ✗      -0.010 ✗      -0.010 ✗      -0.001 ✗
noised_real     (0.5σ anonymise)    refused          +0.521 ✗      -0.140 ✗      -0.294 ✓      +0.007 ✓
gaussian_copula (marginals+corr)    refused          +0.464 ✗      -0.314 ✓      -0.225 ✗      +0.011 ✓
SMOTE           (imblearn)          refused          +0.608 ✗      -0.469 ✓      -0.353 ✓      +0.015 ✓
REGEN           (this repo)         refused          +0.932 ✗      -0.194 ✓      -0.339 ✓      +0.009 ✓
estimand_preserving (GMM+cond,v2)   refused          +0.691 ✓      -0.304 ✓      -0.230 ✗      +0.005 ✓
```

**1 of 7 sources certified — and it was the one that isn't synthetic.** A plain
resample of the real data certifies, as it must; if that failed, the certifier
would be broken. Every practical method breaks `pay_delay_1`, the strongest
predictor, while fidelity and prediction checks flag none of it. The v2 generator
recovers that coefficient and still misses on `log_limit` — a partial fix, not a
solved problem.

Every row is scored against the same θ_real above: this is about whether the
conclusion survives, not who produced the data. Full write-up:
[`examples/certifier_demo/README.md`](examples/certifier_demo/README.md).

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
analysis certifies on **11/30 seeds (37%)**, which no other generator here manages
even once. The other two coefficients carry a real, systematic bias, not noise
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

The certifier has a command-line surface too, and it works on **any** generator's
output — nothing about it is specific to this repo:

```bash
regen certify real.csv synthetic.csv \
      --outcome default --predictors pay_delay_1,utilization,log_limit,age \
      --family logit
```

Exit `0` means every declared coefficient survived, `1` means one shifted (the
per-coefficient table says which), and `2` means the check could not run at all —
so a pipeline can tell a failed check from a check that never happened. Add
`--json` for the full certificate.

Honest generation numbers (leakage-free TSTR + conditional lift, including where
REGEN doesn't help) live in [`benchmark/RESULTS.md`](benchmark/RESULTS.md) and
[`docs/BUILDLOG.md`](docs/BUILDLOG.md); the amplifier helps only when the
baseline is genuinely starved of rare examples.

## Scope & honesty

- **Single-table, cross-sectional** tabular only (not time-series/relational/text/images).
- **Estimand certification v1** covers **numeric** predictors and OLS/logit; a declared
  analysis is required. Power-aware certification, categorical predictors, and ATE are
  on the roadmap ([`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)).
- Deterministic and recomputable throughout; **not** differential privacy.

## Documentation

Read in this order:

1. [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) — how the certifier and the
   generator actually work, mechanism by mechanism. Start here.
2. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the parts connect, and
   the design rules the system refuses to break.
3. [`docs/COMPONENT_GUIDE.md`](docs/COMPONENT_GUIDE.md) — the lookup table:
   component, method, file, why.

Reference:

- [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) — what is open, and what was
  fixed (including this project's own corrected results)
- [`docs/METHODS.md`](docs/METHODS.md) — formal metric definitions and
  verification tolerances
- [`docs/PRIVACY.md`](docs/PRIVACY.md) — what the privacy floor guarantees, and
  what it does not
- [`docs/CAPABILITY_MATRIX.md`](docs/CAPABILITY_MATRIX.md) — supported, degraded,
  or out of scope, by data shape
- [`docs/EXPLAINABILITY.md`](docs/EXPLAINABILITY.md) — the `explanation.json`
  field reference
- [`docs/SERVER_API.md`](docs/SERVER_API.md) — HTTP endpoints
- [`docs/BUILDLOG.md`](docs/BUILDLOG.md) — every change, with before/after numbers
- [`benchmark/RESULTS.md`](benchmark/RESULTS.md) — index of benchmark runs
- [`INVARIANTS.md`](INVARIANTS.md) — the rules that hold across the whole repo

**File conventions.** `README.md` is the entry point of any directory a reader
lands in — the repo root and each example. All other prose lives in `docs/`, one
subject per file, named `UPPER_SNAKE_CASE.md` for its subject.
[`INVARIANTS.md`](INVARIANTS.md) sits at the root because it is a repo-wide
contract rather than a document about one part. Benchmark tables are written by
the scripts that produce them (`benchmark/RESULTS_<TOPIC>.md`), never
hand-edited; superseded runs are frozen and date-stamped under
`benchmark/superseded/`.

## License

MIT
