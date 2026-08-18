# Synthetic data can pass every check you run and still change the number you act on

Synthetic data is invented records that stand in for real ones when the real ones
cannot be shared. It is normally judged two ways. Does it look right, meaning
each column has roughly the right mix of values and the columns move together the
way they should? And can you train a working model on it?

Both are reasonable. Neither asks the question most tables are actually used to
answer, which is what happens when someone runs a regression on the data and acts
on the result.

The number they act on is a coefficient. It says how much the outcome moves when
one factor changes and everything else stays put. A lender uses it to decide
which warning signs matter and how much. That number can move while every check
above still passes, and this repository measures how far apart the two can get.

Every number below is produced by a script named beside it. Nothing here is
hand-copied.

---

## 1. The setup

Real data: the UCI *Default of Credit Card Clients* table, 30,000 accounts, a
22.1% default rate. The analysis is the one a credit risk team actually runs:

```
logit    default ~ pay_delay_1 + utilization + log_limit + age
```

The four coefficients are what has to survive. `pay_delay_1`, how many periods
behind the account already is, is by far the strongest warning sign at +0.714.

Seven versions of this table are compared. Two are controls, included to show the
check itself works. The first is a plain resample of the real rows, which must
come back matching, and if it ever fails the check is broken rather than the
data. The second shuffles each column on its own, which leaves every column
looking untouched while destroying how they line up, and must fail.

The other five are real synthetic-data methods.

Each version gets the identical regression, and each coefficient is compared with
the real one using a standard test that asks whether the two estimates are far
enough apart, given how precisely each was measured, to count as different
answers rather than the same answer measured twice.

## 2. What happens

**Six of seven sources fail. The only one that passes is the resample of real
data, which is the control and is not synthetic.**

Full per-coefficient table: [`examples/certifier_demo/RESULTS.md`](examples/certifier_demo/RESULTS.md)
(`python examples/certifier_demo/run_demo.py`).

The pattern is specific rather than general. Smooth, roughly linear predictors
(`log_limit`, `age`) survive almost everywhere. `pay_delay_1` takes a small number of
whole-number steps and its effect arrives as a jump rather than a gradual climb,
and it breaks under every method that builds a table one column at a time:

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
run on the same seven sources.

**Does each column look right?** The largest gap between the real and synthetic
version of any single column, measured as the widest vertical distance between
their two cumulative curves. Zero means identical, one means no overlap. Written
KS below, after the two statisticians it is named for.

**Do the columns move together correctly?** The largest change in how strongly
any pair of columns tracks each other. Written as a change in correlation, and
zero means the relationships are unchanged.

**Can you still model on it?** Train a classifier on the synthetic table, score it
on real data held back for the purpose, and divide by the score of the same
classifier trained on real data. One means the synthetic table is a full stand-in
for training. This is usually called TSTR, for train on synthetic, test on real.

**The honest answer is that it depends on where the thresholds sit, and the
field does not standardise them.**

At strict cut-offs (column gap 0.10, correlation shift 0.10, model score 0.95) every failing source is
caught by at least one check, so there is no silent failure. But two of them are
caught by margins of nothing: SMOTE fails on a KS of 0.107 and REGEN on 0.112.
Loosen KS to 0.15, which is well inside the range in ordinary use, and both pass
every standard check while still moving the coefficient. Loosen further and the
Gaussian copula joins them. A copula is a way of building a table that gets each
column's own distribution exactly right and then imposes a single number for how
strongly the columns move together. It has a near-perfect column gap of 0.010 and
a correlation shift of 0.078, which passes the strict threshold outright.

| column gap at most | correlation shift at most | model score at least | sources that pass everything and still fail |
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

A coefficient answers a narrow question: if this one factor changes and
everything else stays put, how much does the outcome move? Getting that number
back out of synthetic data takes two separate things, and the methods here get
one without the other.

**The predictors have to relate to each other the way they really do.** Holding
everything else fixed only means something if the generator knows what "everything
else" was doing. In the real table, accounts using more of their
limit also tend to be further behind on payments, at a correlation of +0.387, so
the two carry overlapping information and splitting the credit between them
depends on how much they overlap. A Gaussian copula, as above, gets each column
separately right and then ties the columns together with a single number. That
works when the relationship is a steady
trend. Being behind on payments is not a steady trend. Measured on the real
table, the share of accounts that default runs 0.128 for those not behind, 0.339
one period behind, and 0.691 two periods behind. It climbs and then jumps. A
single number describing how two columns move together cannot express a jump, so
the copula renders it as a gentle slope: its correlation with the outcome drops
from +0.325 to +0.218, and the coefficient flattens from +0.714 to +0.464.

**The outcome has to depend on the predictors the way it really does.** A
generator can place the rows correctly and still get the default rate wrong at
each spot. REGEN does this. It deliberately manufactures extra rare
cases, and rare cases sit where accounts are furthest behind, so that region
fills with defaults at a higher rate than reality. Its correlation with the
outcome rises from +0.325 to +0.417 and the jump gets bigger than the real one,
so the coefficient overshoots from +0.714 to +0.932.

Those are opposite errors from opposite causes, and the certifier catches both,
because it never asks how the data was made. Only whether the answer came back
the same.

Every claim in this section is measured rather than reasoned:
[`examples/certifier_demo/MECHANISM.md`](examples/certifier_demo/MECHANISM.md)
(`python examples/certifier_demo/mechanism_check.py`) states each one, measures
it, and reports whether it holds.

Section 5 pulls the two apart and shows each one failing on its own.

## 5. Does it generalise beyond one table and one model family?

Yes. [`examples/certifier_demo/GENERALITY.md`](examples/certifier_demo/GENERALITY.md)
(`python examples/certifier_demo/generality_check.py`) re-runs the comparison as
ordinary least squares on California housing, 20,536 block groups,
`MedHouseVal ~ MedInc + HouseAge + AveRooms + Latitude`. Different dataset,
different model family, and an outcome that is a
quantity rather than a yes or no.

The controls behave (bootstrap certifies, shuffled columns fail everything) and
the Gaussian copula distorts `AveRooms` from −0.140 to −0.172.

The useful part is the last two rows, which separate the two requirements. Both
get the outcome right by construction: each one learns the real relationship
between the predictors and the outcome, and uses it. The only difference is where
the predictor rows came from.

| predictors taken from | outcome rule | certified | `MedInc` | `AveRooms` |
|---|---|---|---|---|
| the real rows | learned from real | **yes** | +0.490 | −0.140 |
| a Gaussian copula | learned from real | no | +0.445 | −0.092 |

The real coefficients are +0.489 and −0.140. Both rows use the correct outcome
rule, so the second row fails for one reason only: its predictor rows sit wrong
relative to each other. That alone costs `AveRooms` a third of its size. Getting
the outcome right does not rescue you from getting the predictors wrong.

The distortion is milder here than on the credit data, with no sign flips. That
fits the mechanism: these predictors are smoother and closer to linear, and the
size of the error scales with how non-linear the predictor-outcome relationship
is. The mechanism is general, the severity is dataset-dependent.

## 6. Can it be fixed?

Partly, and the honest number is smaller than the one this project first
published.

A second generator, `regen/estimand_preserving.py`, sets out to satisfy both
conditions from section 4 at once. For the first, it learns the shape of the real predictor rows as a
blend of several overlapping clouds, then draws new rows from that shape. The
rows are new, not real rows with noise added. For the second, it learns how the
real outcome depends on where a row sits, and uses that to decide each new row's
outcome. It never looks at the coefficient it is being graded on, so the answer
is not planted in the data.

Across 30 seeds (`python examples/certifier_demo/seed_sweep.py`):

```
predictor        theta_real   mean bias       std
pay_delay_1         +0.7141     -0.0133    0.0382
utilization         -0.3693     +0.1689    0.1018
log_limit           -0.3145     +0.0587    0.0342
age                 +0.0100     -0.0010    0.0039
```

**11 of 30 runs certify the full analysis (37%).** Each run starts the random
number generator from a different point, so the spread across runs shows how much
of any single result is luck. No other generator here certifies on even one run.

Read the table by comparing its two right-hand columns. Bias is how far the
average run lands from the truth. Std is how much the runs scatter around that
average. When the scatter is bigger than the miss, the miss is luck and more runs
would wash it out. When the miss is bigger than the scatter, it is a real error
that repeats every time in the same direction.

`pay_delay_1` and `age` are the first case, and recovering `pay_delay_1` is
something no other method here manages. `utilization` and `log_limit` are the
second: they miss by more than they scatter, on every run. That is a genuine
limit on how well a blend of overlapping clouds can describe the real arrangement
of the predictors, not noise that would average away.

This is a partial fix. It is reported as one.

## 7. The limit that does not go away

Certification requires the real data, because `certify_dataset` takes `real_df`
and refits the analysis on it. So this addresses sharing, where one party holds
the truth and another does not, and it does not address scarcity, where nobody
has enough data. If you can compute the real answer to check against, you did not
need the synthetic data for that answer. The limit is structural.

Underneath that sits a three-way tension that no generator here escapes:

- Keeping the coefficient right pushes the synthetic rows toward sitting exactly
  where the real ones sit.
- Privacy pushes them away, because rows arranged closely enough to reproduce a
  partial effect are close to the records themselves.
- Manufacturing extra rare cases changes how often the outcome occurs in the
  region it fills, which is the second condition being broken on purpose.

Measured on the credit data, full certification collapses at roughly 0.1σ of
added privacy noise, before the noise buys meaningful privacy. The bootstrap
control keeps the coefficients and offers no privacy at all, since it is the real
rows. The second generator makes genuinely new rows, but they sit close to the
real ones, and pushing them further away for privacy breaks the coefficients
again.

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
- The repeating error in the second generator on `utilization` and `log_limit`
  is diagnosed but not fixed.
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
