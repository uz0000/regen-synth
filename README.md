# REGEN

**Can synthetic data do the job real data does?**

This repository is a study of that question. Synthetic data is offered as a
stand-in for records that cannot be shared, and the claim attached to it is that
you can work with the stand-in and end up in the same place. We tested that claim
against the generation methods people actually use. None of them held it
reliably.

**What "the job" means here.** Tabular data is mostly used for one thing:
someone fits a regression and acts on a coefficient. The coefficient says how
much the outcome moves when one factor changes and the rest stays put — a lender
uses it to decide which warning signs matter and by how much. That number is the
utility. It is what the decision is actually made from. So the test is direct:
fit the same declared analysis on the real table and on the synthetic one, and
compare the estimates coefficient by coefficient. The properties synthetic data
is normally graded on — each column having the right spread, the columns moving
together correctly, a model trained on it predicting well — are proxies for this.
The result below is that the proxies pass while the thing they stand for fails.

**The result.** Five of five real generation methods failed to preserve the
declared analysis: noise injection, a Gaussian copula, SMOTE, this repo's
rare-event amplifier, and a generator built specifically to preserve the
coefficient. The only source that passed is a resample of the real rows, which is
the positive control and is not synthetic data at all. The companion repo
[`regen-basic`](https://github.com/uz0000/regen-basic) runs the same test against
SDV — the most widely used synthetic-tabular library — using both its Gaussian
copula and CTGAN. Both fail it too. Seven methods, two independent codebases, one
outcome.

**"Reliably" is the operative word.** The one generator built on purpose to
survive this test does recover the coefficient the others break — on 11 of 30
seeds. It certifies 37% of the time, and nothing about a given run tells you in
advance which kind you got. A method that works on a third of attempts is not
something you can hand to someone who has to act on the answer.

Every method tested, and whether the declared analysis survived it:

| source | what it is | conclusion preserved |
|---|---|---|
| resample of real rows | positive control, not synthetic | **yes** |
| column-wise shuffle | negative control, must fail | no |
| noise injection (0.5σ) | common anonymisation approach | no |
| Gaussian copula | marginals + correlation structure | no |
| SMOTE | standard oversampler (`imblearn`) | no |
| rare-event amplifier | this repo's generator | no |
| coefficient-preserving generator | built for this test specifically | no (37% of seeds) |
| SDV Gaussian copula | industry standard, in `regen-basic` | no |
| SDV CTGAN | deep generative, in `regen-basic` | no |

Read [**FINDINGS.md**](FINDINGS.md) for the full result, the mechanism, the
replication on a second dataset and model family, and the limits.
[**CORRECTIONS.md**](CORRECTIONS.md) records the three published claims that did
not survive re-measurement.

---

## Install

```bash
git clone https://github.com/uz0000/regen-synth.git && cd regen-synth
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

pytest tests/ -q                                    # 222 tests
python examples/certifier_demo/run_demo.py          # reproduces the headline table
```

Requires Python 3.10 or later. `GPy`, the Gaussian-process backend used by the
tail correction, can be slow to build on some platforms. If the full install
fails there, `pip install numpy pandas scipy scikit-learn pyarrow
imbalanced-learn` is enough to run the certifier, the CLI, and every demo.

## How the claim is tested

Declare the analysis whose conclusion has to survive. The certifier fits it on
the real data and on the synthetic data and compares the estimates coefficient by
coefficient.

```python
from regen.certifier import certify_dataset
from contracts.scenario import EstimandSpec

estimand = EstimandSpec(outcome="default",
                        predictors=["pay_delay_1", "utilization", "log_limit", "age"],
                        family="logit")

cert = certify_dataset(real_df, synthetic_df, estimand)
cert["certified"]     # True only if every declared coefficient is preserved
cert["targets"]       # per coefficient: theta_real, theta_synth, the test, preserved?
```

Preservation is decided by a two-sample Wald test on the difference, not by
asking whether the synthetic estimate lands inside the real confidence interval.
The naive rule rewards imprecision, since a noisy synthetic estimate is more
likely to overlap.

Three properties make the certificate useful:

- **Generator-agnostic.** It never asks who produced the data, so it scores any
  tool's output on equal terms, including this repository's own.
- **Per coefficient.** It reports which conclusions survived rather than a single
  pass or fail.
- **Portable.** It carries theta_real and its standard error, so a third party
  can recompute theta_synth from the synthetic table alone and re-check the
  verdict without ever seeing the real rows.

The analysis has to be declared up front. No generator can guess which
relationship in a table you plan to act on, and "check everything" is not a
question anyone can answer, since a table supports an unlimited number of
analyses.

There is a command-line equivalent that works on any generator's output:

```bash
regen certify real.csv synthetic.csv \
      --outcome default --predictors pay_delay_1,utilization,log_limit,age \
      --family logit
```

Exit code `0` means every declared coefficient survived, `1` means at least one
shifted, and `2` means the check could not run, so a pipeline can distinguish a
failed check from a check that never happened.

## The generators tested here

REGEN is also a synthetic data generator. It exists in this repository as the
system the certifier was built to check, and the certifier refuses its output.

It generates rows in a loop, concentrating effort on the rare cases a general
sampler under-produces. Each batch is scored against the real data on four
things: how much of the real range it reaches, how far off the categorical
columns are, how far off the numeric columns are, and how much the relationships
between columns have shifted. A batch that fails is thrown away rather than
shipped with a warning attached.

Base rows come from a Gaussian copula, which learns each column's own
distribution and the pattern of how columns move together as two separate things,
then reproduces both. A separate step models the sparse tail of the data and
fills it in, since that is where a general sampler is weakest.

Generation quality is reported in [`benchmark/RESULTS.md`](benchmark/RESULTS.md),
including the datasets where amplification does not help.

A second generator, `regen/estimand_preserving.py`, is built to satisfy the two
conditions a coefficient actually depends on: the predictor joint and the
conditional distribution of the outcome. It certifies on 37% of seeds, which is
more than any other method here achieves and less than a solution.

## Scope

- Single-table cross-sectional tabular data. Not time series, relational, text,
  or images.
- Checking covers numeric predictors, for straight-line regression and for
  yes-or-no outcomes. A declared analysis is required. Categorical predictors, interaction terms, and
  average treatment effects are not supported yet.
- The same inputs always give the same outputs, and every reported number can be
  recomputed from the saved artifacts. No language model produces or checks any
  value.
- The privacy floor is a distance constraint plus a verbatim guard. It is not
  differential privacy. See [`docs/PRIVACY.md`](docs/PRIVACY.md).
- Certification requires the real data, so it addresses sharing rather than
  scarcity. [`FINDINGS.md`](FINDINGS.md) section 7 explains why that limit is
  structural.

## Where to find things

| You want | Go to |
|---|---|
| What was found, and how far it holds | [`FINDINGS.md`](FINDINGS.md) |
| The same result as a visual walkthrough | [`docs/inference-explainer.html`](docs/inference-explainer.html) |
| Claims that were revised, and why | [`CORRECTIONS.md`](CORRECTIONS.md) |
| How the certifier and generator work | [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) |
| How the parts fit together | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Which file implements which method | [`docs/COMPONENT_GUIDE.md`](docs/COMPONENT_GUIDE.md) |
| Exact metric definitions and tolerances | [`docs/METHODS.md`](docs/METHODS.md) |
| What is currently broken | [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) |
| What the privacy floor does and does not guarantee | [`docs/PRIVACY.md`](docs/PRIVACY.md) |
| Whether your data shape is supported | [`docs/CAPABILITY_MATRIX.md`](docs/CAPABILITY_MATRIX.md) |
| Generator benchmark numbers | [`benchmark/RESULTS.md`](benchmark/RESULTS.md) |
| Rules the codebase holds itself to | [`INVARIANTS.md`](INVARIANTS.md) |
| Change history with before and after numbers | [`docs/BUILDLOG.md`](docs/BUILDLOG.md) |

**Related repositories.** [`regen-basic`](https://github.com/uz0000/regen-basic)
is the compact version of the same question: one simulator, one check, three
dependencies. This repository is the larger system, adding rare-event
amplification, a privacy floor, an audit bundle, and the estimand-preserving
generator. Install them in separate environments, since both claim the top-level
package names `engine`, `contracts`, and `cli`.

**File conventions.** Every result table under `examples/` and `benchmark/` is
written by the script that produced it and carries a generated-file header. Prose
files link to those tables rather than restating their numbers, so a document
cannot drift from a run. `README.md` is the entry point of any directory a reader
lands in; all other prose lives in `docs/`, one subject per file.

## License

MIT
