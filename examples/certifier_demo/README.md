# The certifier demo

Five scripts. Each one answers a question, prints its result, and writes a
generated table that the prose elsewhere links to instead of restating.

The interpretation of all five lives in [`../../FINDINGS.md`](../../FINDINGS.md).
This file is about running them and reading the output.

Run from the repository root, with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` set for bit-identical results.

| Script | Question | Writes | FINDINGS |
|---|---|---|---|
| `run_demo.py` | Does a declared regression survive synthetic data? | [`RESULTS.md`](RESULTS.md), `certificates.json` | §2 |
| `fidelity_check.py` | Do the standard quality checks catch what the certifier catches? | [`FIDELITY.md`](FIDELITY.md) | §3 |
| `mechanism_check.py` | Is the explanation for *why* it fails actually true? | [`MECHANISM.md`](MECHANISM.md) | §4 |
| `generality_check.py` | Does the failure replicate on another dataset and model family? | [`GENERALITY.md`](GENERALITY.md) | §5 |
| `seed_sweep.py` | Is the estimand-preserving generator's result a seed accident? | prints a bias and spread table | §6 |

A sixth script, `prepare_data.py`, is not part of that sequence: it documents where
`credit_default.csv` came from and how the derived columns were built. It does not
need to be re-run.

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
`bootstrap_real` is a resample of the real rows and should certify.
`independent_cols` shuffles every column independently, preserving all marginals
and destroying every association, and must fail.

Neither control is absolute, and it is worth knowing why before you re-run with a
different seed. Certification needs all four coefficients to agree at once and
each is a 95% test, so `bootstrap_real` is refused on about **12% of seeds** —
roughly one run in eight, from chance alone. Seeds 9, 26, 27 and 35 refuse; the
committed table uses seed 7, which certifies. A refusal there is not a broken
certifier.

`independent_cols` never certifies across 200 seeds, but it does not fail on
every coefficient: `age` survives about 20% of the time. Its real effect is
+0.010, and a coefficient the real data barely established is one almost any
table can match — the power limit in
[`../../docs/KNOWN_ISSUES.md`](../../docs/KNOWN_ISSUES.md) issue 4, visible in a
control. Both rates are pinned in `tests/test_control_rates.py`.

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

`mechanism_check.py` prints each factual claim FINDINGS section 4 rests on, the
measurement behind it, and whether it holds. It is there so the explanation is
checked rather than argued: if a claim stopped holding, this is the script that
would say so.

`generality_check.py` includes two rows that differ only in where the predictors
came from, both using the same model of the real conditional. The gap between
them isolates the contribution of the predictor joint.

## One thing to expect

The `estimand_preserving` row can differ between runs at the same seed unless
BLAS threading is pinned. Its fit involves operations whose floating-point
summation order depends on thread count, which is occasionally enough to flip a
borderline coefficient. A single run is one draw. `seed_sweep.py` is the honest
summary, and it reports 11 of 30 seeds.
