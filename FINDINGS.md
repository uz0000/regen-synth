# Synthetic data can match every distribution you check and still move the coefficient you act on

Synthetic data is normally judged on two things: whether the marginal and joint
distributions match the real data, and whether a model trained on it predicts
well on real data. Both are reasonable. Neither is a statement about the quantity
most tabular data is actually used to produce, which is a regression coefficient
someone reads as an effect size and acts on.

Those are different properties, and this repository measures the gap between
them.

Every number below is produced by a script named beside it. Nothing here is
hand-copied.

---

## 1. The setup

Real data: the UCI *Default of Credit Card Clients* table, 30,000 accounts, a
22.1% default rate. The analysis is the one a credit risk team actually runs:

```
logit    default ~ pay_delay_1 + utilization + log_limit + age
```

The estimand is the coefficient vector. `pay_delay_1`, how far behind an account
already is, is by far the strongest predictor at +0.714.

Seven sources of the same table are compared: a bootstrap resample of the real
data as a positive control, independently shuffled columns as a negative
control, and five synthetic generators. Each is fit with the identical
specification, and each coefficient is compared to the real one with a
two-sample Wald test on the difference.

## 2. What happens

**Six of seven sources fail. The only one that passes is the resample of real
data, which is the control and is not synthetic.**

Full per-coefficient table: [`examples/certifier_demo/RESULTS.md`](examples/certifier_demo/RESULTS.md)
(`python examples/certifier_demo/run_demo.py`).

The pattern is specific rather than general. Smooth, roughly linear predictors
(`log_limit`, `age`) survive almost everywhere. `pay_delay_1`, a discrete ordinal
with a threshold effect, breaks under every method built from marginals:

| source | `pay_delay_1` |
|---|---|
| real | **+0.714** |
| Gaussian copula | +0.464 |
| SMOTE | +0.608 |
| additive noise (0.5σ) | +0.521 |
| REGEN, this repo's generator | +0.932 |

Note the direction. Three methods attenuate the coefficient toward zero and one
inflates it. A practitioner using any of them misjudges the strongest risk driver
in the model, and the sign of the error depends on which tool they picked.

## 3. Do the standard checks catch it?

This is the claim the repository previously asserted without measuring, so it is
now measured: [`examples/certifier_demo/FIDELITY.md`](examples/certifier_demo/FIDELITY.md)
(`python examples/certifier_demo/fidelity_check.py`). Three conventional checks
run on the same seven sources: maximum per-predictor Kolmogorov-Smirnov distance,
maximum shift in the Pearson correlation matrix, and train-on-synthetic
test-on-real ROC-AUC ratio.

**The honest answer is that it depends on where the thresholds sit, and the
field does not standardise them.**

At strict cut-offs (KS ≤ 0.10, |Δρ| ≤ 0.10, TSTR ≥ 0.95) every failing source is
caught by at least one check, so there is no silent failure. But two of them are
caught by margins of nothing: SMOTE fails on a KS of 0.107 and REGEN on 0.112.
Loosen KS to 0.15, which is well inside the range in ordinary use, and both pass
every standard check while still moving the coefficient. Loosen further and the
Gaussian copula joins them, despite a near-perfect KS of 0.010 and a correlation
shift of 0.078 that passes the strict threshold outright.

| KS ≤ | \|Δρ\| ≤ | TSTR ≥ | sources that pass everything and still fail |
|---|---|---|---|
| 0.10 | 0.10 | 0.95 | none |
| 0.15 | 0.10 | 0.95 | SMOTE, REGEN |
| 0.20 | 0.20 | 0.85 | Gaussian copula, SMOTE, REGEN |

The finding is therefore narrower and more useful than "the standard checks miss
it." The standard checks are not measuring the coefficient, and whether they
happen to flag a coefficient failure is decided by threshold choices unrelated to
it. A distributional distance of 0.11 tells you nothing about whether an effect
size moved by 30%. Sometimes the two coincide. Nothing makes them coincide.

## 4. Why it happens

A regression coefficient is a partial effect. Recovering it requires two distinct
things from the data, and generators tend to get one without the other.

**R1, the predictor joint.** A coefficient on `utilization` is its effect holding
the other predictors fixed, so it depends on the covariance structure among
predictors, not just on each predictor's own distribution. A Gaussian copula
reproduces every marginal exactly and imposes a single linear rank correlation.
Where the real dependence is a threshold rather than a monotone trend, that
linear summary cannot represent it and the coefficient flattens.

**R2, the conditional distribution.** The coefficient is a property of
P(y | x). A method can leave the predictor joint intact and still distort the
outcome given the predictors. Oversampling a rare region does exactly this: it
raises the local default rate where `pay_delay_1` is high, which steepens the
gradient and inflates the coefficient. This is why REGEN overshoots to +0.932
while the copula undershoots to +0.464. Opposite mechanisms, the same failed
estimand.

The two requirements are separable, and section 5 separates them.

## 5. Does it generalise beyond one table and one model family?

Yes. [`examples/certifier_demo/GENERALITY.md`](examples/certifier_demo/GENERALITY.md)
(`python examples/certifier_demo/generality_check.py`) re-runs the comparison as
ordinary least squares on California housing, 20,536 block groups,
`MedHouseVal ~ MedInc + HouseAge + AveRooms + Latitude`. Different dataset,
different estimand family, continuous outcome.

The controls behave (bootstrap certifies, shuffled columns fail everything) and
the Gaussian copula distorts `AveRooms` from −0.140 to −0.172.

The useful part is the last two rows, which isolate R1 from R2. Both draw the
outcome from the same gradient-boosted model of the real conditional. They differ
only in where the predictors came from:

| predictor source | conditional model | certified | `MedInc` | `AveRooms` |
|---|---|---|---|---|
| real (resampled) | real P(y\|x) | **yes** | +0.490 | −0.140 |
| Gaussian copula | real P(y\|x) | no | +0.445 | −0.092 |

Real coefficients are +0.489 and −0.140. Getting the conditional right is not
enough on its own: with the same correct conditional and a copula predictor
joint, `AveRooms` loses a third of its magnitude. Both requirements bind.

The distortion is milder here than on the credit data, with no sign flips. That
fits the mechanism: these predictors are smoother and closer to linear, and the
size of the error scales with how non-linear the predictor-outcome relationship
is. The mechanism is general, the severity is dataset-dependent.

## 6. Can it be fixed?

Partly, and the honest number is smaller than the one this project first
published.

`regen/estimand_preserving.py` builds synthetic data to satisfy both
requirements: predictors are drawn from a Gaussian mixture fit to the real
predictor joint, producing novel rows rather than perturbed real ones, and the
outcome is drawn from a calibrated model of the real conditional. It never reads
the declared coefficient, so nothing is injected into the quantity being graded.

Across 30 seeds (`python examples/certifier_demo/seed_sweep.py`):

```
predictor        theta_real   mean bias       std
pay_delay_1         +0.7141     -0.0133    0.0382
utilization         -0.3693     +0.1689    0.1018
log_limit           -0.3145     +0.0587    0.0342
age                 +0.0100     -0.0010    0.0039
```

**11 of 30 seeds certify the full analysis (37%).** No other generator here
certifies on any seed. `pay_delay_1` and `age` recover with bias small relative
to spread, and fixing `pay_delay_1` is something no other method manages.
`utilization` and `log_limit` carry bias several times their standard deviation,
which makes them systematic distortion rather than seed noise, traceable to how
well a Gaussian mixture approximates the real predictor joint.

This is a partial fix. It is reported as one.

## 7. The limit that does not go away

Certification requires the real data, because `certify_dataset` takes `real_df`
and refits the analysis on it. So this addresses sharing, where one party holds
the truth and another does not, and it does not address scarcity, where nobody
has enough data. If you can compute the real answer to check against, you did not
need the synthetic data for that answer. The limit is structural.

Underneath that sits a three-way tension that no generator here escapes:

- Preserving the estimand pushes the predictor joint toward the real one.
- Privacy pushes it away, since a distribution close enough to reproduce partial
  effects is close to the records themselves.
- Rare-event amplification reshapes P(y | x) by construction, which is exactly
  what R2 forbids.

Measured on the credit data, full certification collapses at roughly 0.1σ of
added privacy noise, before the noise buys meaningful privacy. The bootstrap
control wins on estimand and has no privacy at all. The v2 generator produces
novel rows but sits at modest distance from the real ones, and perturbing them
further to gain privacy re-breaks the coefficients.

The contribution is not resolving that tension. It is making the position on it
measurable per coefficient, so an operating point is chosen deliberately instead
of assumed.

## 8. What is not settled

- Certification is not power-aware. When the real data is scarce, its standard
  error is wide and the two-sample test becomes lenient, so a coefficient the
  real data never pinned down can pass. This is currently surfaced rather than
  failed. See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).
- Numeric predictors and OLS or logistic families only. Categorical predictors,
  interaction terms, and average treatment effects are not certifiable yet.
- Single-table cross-sectional data only. Not time series, relational, text, or
  images.
- The residual bias in the v2 generator on `utilization` and `log_limit` is
  diagnosed but not closed.
- The threshold sensitivity in section 3 is measured on one dataset. How often
  distributional checks and estimand failures diverge in general is an open
  empirical question, and this repo shows only that they can.

## 9. What was corrected along the way

Three published claims from this project did not survive re-measurement,
including the certification rate in section 6 and the framing in section 3. Each
is recorded with what was claimed, what the re-run showed, and what changed, in
[`CORRECTIONS.md`](CORRECTIONS.md).

## 10. Prior art

The concern is not new. Whether synthetic data supports valid inference, rather
than merely resembling the source, is a long-standing question in the statistical
disclosure control and multiple imputation literature, where the analogous
quantity is whether an analyst's estimates and standard errors remain valid. What
is built here is a small, re-runnable, generator-agnostic check for that property
on a declared regression, plus a measurement of how current tabular generators
behave under it.

## A visual version

[`docs/inference-explainer.html`](docs/inference-explainer.html) walks through
sections 1 and 2 for a reader who does not work with regressions: the three
questions you can ask of synthetic data, the four coefficients, and the table of
which sources break which. Open it in a browser.

## Reproducing everything

```bash
pip install -r requirements.txt
pip install -e .

python examples/certifier_demo/run_demo.py          # section 2
python examples/certifier_demo/fidelity_check.py    # section 3
python examples/certifier_demo/generality_check.py  # section 5
python examples/certifier_demo/seed_sweep.py        # section 6
```

Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for
bit-identical output. Without it, BLAS summation order varies with thread count
and can flip a borderline coefficient.
