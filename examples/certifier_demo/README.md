# The certifier demo

Four scripts. Each one answers a question, prints its result, and writes a
generated table that the prose elsewhere links to instead of restating.

The interpretation of all four lives in [`../../FINDINGS.md`](../../FINDINGS.md).
This file is about running them and reading the output.

Run from the repository root, with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` set for bit-identical results.

| Script | Question | Writes |
|---|---|---|
| `run_demo.py` | Does a declared regression survive synthetic data? | [`RESULTS.md`](RESULTS.md), `certificates.json` |
| `fidelity_check.py` | Do the standard quality checks catch what the certifier catches? | [`FIDELITY.md`](FIDELITY.md) |
| `generality_check.py` | Does the failure replicate on another dataset and model family? | [`GENERALITY.md`](GENERALITY.md) |
| `seed_sweep.py` | Is the estimand-preserving generator's result a seed accident? | prints a bias and spread table |

## The data

`credit_default.csv` is the UCI *Default of Credit Card Clients* table, 30,000
accounts with a 22.1% default rate. `prepare_data.py` documents where it came
from and how the derived columns were built.

`generality_check.py` uses California housing, which sklearn downloads on first
run.

## The analysis under test

```
logit    default ~ pay_delay_1 + utilization + log_limit + age
```

This is the estimand. It is declared up front, because a certifier cannot infer
which relationship you intend to act on, and every source is graded against the
same real fit.

## The sources

Seven, in a deliberate order.

**Two controls, which exist to show the certifier discriminates.**
`bootstrap_real` is a resample of the real rows and must certify; if it ever
fails, the certifier is broken. `independent_cols` shuffles every column
independently, preserving all marginals and destroying every association, and
must fail everything.

**Five generators.** `noised_real` adds 0.5σ Gaussian noise to real rows, the
common anonymisation move. `gaussian_copula` reproduces the marginals and a
single linear rank correlation. `SMOTE` interpolates between nearest neighbours
in the minority class. `REGEN` is this repository's pipeline. `estimand_preserving`
is the second generator, built to satisfy the two conditions a coefficient
depends on.

## Reading the output

`run_demo.py` prints one row per source and one column per coefficient. A check
mark means the synthetic estimate is statistically consistent with the real one
under a two-sample Wald test on the difference. A source is certified only when
every declared coefficient is preserved, so a row can be mostly check marks and
still be refused. That is the intended behaviour: a conclusion is not partly
true.

`fidelity_check.py` prints two verdicts per source, the standard checks and the
certificate, so the two axes can be compared directly. It also prints how the
count of silent failures moves as the thresholds move, because it does move, and
the thresholds are not standardised.

`generality_check.py` includes two rows that differ only in where the predictors
came from, both using the same model of the real conditional. The gap between
them isolates the contribution of the predictor joint.

## One thing to expect

The `estimand_preserving` row can differ between runs at the same seed unless
BLAS threading is pinned. Its fit involves operations whose floating-point
summation order depends on thread count, which is occasionally enough to flip a
borderline coefficient. A single run is one draw. `seed_sweep.py` is the honest
summary, and it reports 11 of 30 seeds.
