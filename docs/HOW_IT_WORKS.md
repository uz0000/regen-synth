# How it works

This repo checks whether synthetic data that looks like the real data is still
usable for analysis. It turns out that looking right and supporting the same
conclusion are different properties, and only the second one matters once
somebody acts on the result.

There are two parts, and they argue with each other:

- **The certifier** decides whether a conclusion survived. It is the core.
- **The generator** makes synthetic data. It exists mainly as the thing the
  certifier was built to check — and the certifier refuses its output.

This document explains how each part works. Every number and rule below is in
the code, with file references.

---

## 1. The certifier

You tell it the analysis you care about. It runs that analysis twice — once on
the real data, once on the synthetic — and compares the answers one number at a
time.

### Step 1: you declare the analysis

```python
estimand = EstimandSpec(outcome="default",
                        predictors=["pay_delay_1", "utilization", "log_limit", "age"],
                        family="logit")
```

This is you saying: *this is the conclusion that has to survive*. It has to be
declared, because no generator can guess which relationship in a table you plan
to act on. Certifying "everything" is not a coherent request; certifying a
stated analysis is. (`contracts/scenario.py`)

### Step 2: fit it on the real data

You get the real answer and how uncertain that answer is — a coefficient and its
standard error, per predictor. This is `θ_real`, the thing to be preserved. The
fits are OLS (closed-form) and logistic regression (IRLS), written directly
against numpy and scipy so the check does not depend on a solver whose behaviour
could drift between library versions. (`regen/estimand.py`)

### Step 3: fit the same analysis on the synthetic data

Same specification, same procedure, different rows. That gives `θ_synth`, which
also carries its own standard error — it is an estimate too, not a known
quantity.

### Step 4: compare them, coefficient by coefficient

The default rule is a two-sample consistency test: the coefficient is preserved
when

```
|θ_real − θ_synth|  ≤  z · √(SE_real² + SE_synth²)
```

Both uncertainties are on the right-hand side, and that detail matters. The
obvious alternative — "is θ_synth inside θ_real's confidence interval?" — treats
the synthetic estimate as exact, so it fails data that is actually fine whenever
the synthetic sample is small. A test caught that during development, which is
why the naive rule is not the default. (It remains available as the stricter
`within_ci` rule.) (`certify`, `regen/estimand.py:140`)

The whole estimand is **certified only if every declared coefficient is
preserved**. A coefficient that cannot be fit at all becomes an honest status —
`uncertifiable` — rather than an exception or a silent pass.

### What comes out: the certificate

`certify_dataset(real_df, synthetic_df, estimand)` returns a certificate with
three properties worth naming (`regen/certifier.py`). The same thing is available
from the command line, on any generator's output:

```bash
regen certify real.csv synthetic.csv \
      --outcome default --predictors pay_delay_1,utilization --family logit
```

Exit `0` certified, `1` a coefficient shifted, `2` the check could not run —
distinct codes, because a pipeline needs to tell a failed check from one that
never happened.

**Generator-agnostic.** It never asks who made the data. It works on output from
SMOTE, from a GAN, from a commercial tool, from this repo — the comparison is
identical. That is what makes the finding below apply to the field rather than
to one implementation.

**Per-coefficient.** It reports which conclusions survived, not one blended
score. A dataset can preserve three relationships and break the fourth, and if
the broken one is the one driving your decision, an aggregate "87% similar"
would have hidden exactly the thing you needed to know.

**Portable.** The certificate carries `θ_real ± SE` — an aggregate, not rows —
so someone holding only the synthetic data can refit `θ_synth` themselves and
re-check the verdict without ever seeing a real record. This is the "attach a
certificate to synthetic data you share" model.

### The limit, stated plainly

Certification needs the real data to check against: `certify_dataset` takes
`real_df`. So this addresses **sharing** — you hold the truth and someone else
does not — and it does not address **scarcity**, where nobody has enough data.
If you can compute the real answer to compare with, you did not need the
synthetic data. That limit is structural, not an implementation gap.

---

## 2. What it found

The results are not repeated here, so that they cannot drift from the runs that
produced them. [`../FINDINGS.md`](../FINDINGS.md) carries the full account: which
sources fail, which coefficients they break, whether the standard quality checks
catch it, whether the failure replicates on a second dataset and model family,
and how far a generator built to preserve coefficients gets.

The one-line version is that six of seven sources fail to preserve a declared
logistic regression, and the only one that passes is a resample of the real data.

---

## 3. The generator this repo ships

REGEN is also a synthetic-data generator — the system the certifier was
originally built to check, and the source of the finding above. It runs in five
steps, in a loop that concentrates effort on the rare cases.

| Stage | What it is, statistically | Where |
|---|---|---|
| targeting | selects the region of the feature space that is under-represented relative to the rare class, and directs sampling effort there | `engine/scout/targeting.py` |
| base sampler | mixed-data Gaussian copula: marginal quantile functions composed with a Gaussian dependence structure, so novel rows are drawn rather than real rows copied | `engine/prior/grounded.py` |
| tail correction | Gaussian-process model of the rare region, used to densify where a copula under-produces | `engine/amplifier/tail_corrector.py` |
| rejection gate | scores the batch against the real data on four statistics and discards it if any fails | `engine/auditor/` |
| utility evaluation | train-on-synthetic, test-on-real performance of a downstream classifier | `engine/examiner/` |

Three of those deserve detail.

Three of those deserve detail.

**The base sampler is a mixed-data Gaussian copula.** The easy way to build synthetic
data is column by column: learn what values `age` takes, learn what values
`income` takes, draw each independently. Every column comes out with a perfect
distribution and the table is nonsense, because in the real data those columns
move together and nothing in that procedure recorded that they did. A copula
separates the two things being learned — what each column looks like alone, and
how the columns move together — and reproduces both. Fixing this dropped the
correlation-structure error from **0.331 to 0.101** on the transactions set.

**The rejection gate is four named statistics**, each with a stated tolerance: coverage radius,
total variation distance (categorical marginals), Wasserstein-1 (continuous
marginals), and Pearson correlation delta. A batch that breaks the real
correlation structure is rejected rather than shipped with a warning.

**Utility evaluation only claims a gain when there is one.** Amplification helps when
a detector is genuinely starved of rare examples and reports approximately zero
when the baseline is already strong. This is where the early overclaiming in
this project was caught: a headline of +39% collapsed to +4.4% once evaluation
was made leakage-free.

### The v2 generator

`generate_estimand_preserving(real_df, estimand)` tries to fix the problem
instead of only measuring it (`regen/estimand_preserving.py`). It models the
predictor **joint** with a Gaussian mixture — producing novel rows rather than
perturbed real ones — and draws the outcome from a calibrated model of the real
conditional P(y|x). It never reads the declared coefficient, so nothing is
injected into the answer it is being graded on.

Measured across a 30-seed sweep (`examples/certifier_demo/seed_sweep.py`), the
full analysis certifies on **11 of 30 seeds (37%)** — something no other
generator here manages even once. Two of the four coefficients recover
essentially unbiased, which is also unique to it: notably it is the only
generator that fixes `pay_delay_1`, the coefficient every other method breaks. The other two carry real, systematic bias rather
than noise — `utilization` attenuated by ~46%, `log_limit` by ~19% — traced to
how the Gaussian mixture approximates the real joint and documented in
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md). It is an honest partial fix.

---

## 4. Choosing which column to amplify

The generator amplifies one column's minority class, so getting that column
right is load-bearing. There are three paths, in strict authority order.

**You name it.** An explicit `label_col` short-circuits all detection and always
wins. (`engine/ingest/loader.py:243`)

**Rules score the candidates.** With no label given, every column is scored. A
column is disqualified unless it sits in a useful band: cardinality between 2 and
20, at least 10 rare rows, and a minority class no larger than 35% of the data —
a roughly balanced column offers nothing to amplify. Survivors are ranked by a
weighted formula favouring rarer, binary, adequately-populated columns, with a
small bonus for recognisable names (`label`, `target`, `is_fraud`, …).
(`_score_target_columns`, `engine/ingest/loader.py:300`)

**If the top two scores are within 0.05 of each other, it refuses to guess** and
raises `AmbiguousTargetError` listing the tied candidates. Refusing is the
feature: a confidently-wrong target corrupts everything downstream.
(`loader.py:352`)

**A model may break that tie, and only that tie.** When the rules tie, a language
model is shown your plain-language goal, the tied candidates with their
statistics, a few example values each, and the *names* of the other columns as
domain context. It never sees raw rows. Its pick must be one of the tied
candidates or it is discarded. Offline, with no key, or on a declined pick, the
honest `AmbiguousTargetError` comes straight back and a human chooses.
(`resolve_ambiguous_target`, `regen/semantics.py`)

This is the only place a model touches anything. It can change *which column
gets amplified*; it can never write a value. Once the target is fixed, the rare
mask, the generation, and every metric are deterministic.

---

## 5. How the utility numbers are produced

The headline generator metric is TSTR: how much of real-data performance does a
model trained on the synthetic set recover? (`measure_tstr`,
`engine/examiner/surrogate.py`)

The task is defined rather than assumed. The label is the rare-vs-rest
binarisation of the target column from §4; the features are every other column,
encoded through a shared dictionary so real and synthetic live in the exact same
feature space.

Three models are used rather than one — logistic regression, random forest, and
gradient boosting — so the number is not a fluke of a single learner. Each is
trained twice, once on synthetic and once on real, and both copies are graded on
the **same held-out real test set**:

```
recovered = (model trained on SYNTHETIC) / (model trained on REAL)
```

Both ROC-AUC and PR-AUC are computed, and the median across the three models is
the headline. Two guardrails keep it honest: fewer than 10 held-out rare rows
produces the status `insufficient_real_test` rather than a fabricated number,
and a recovered score above 1.05 is **flagged**, because synthetic data beating
real data almost always means leakage rather than magic.

---

## 6. What "recomputable" means

Every batch ships a self-contained bundle: the delivered data, a manifest
(seed, config, the vetted `ScenarioSpec`, a SHA-256 of every artifact, metric
version IDs), the statistics the run reported, and aggregate statistics of the
real reference each gate was checked against — bucketed, with no per-row values,
and only for classes with at least 10 real rows.

`regen verify <bundle>` then recomputes from those files alone. It never reads a
cached number, so it would catch a system that lied. It checks integrity
(hashes match the manifest), refuses to compare across metric-definition
changes, and re-derives each statistic with a PASS/FAIL inside a fixed tolerance.

The part that makes it a standard rather than a formality: **it is explicit
about what it cannot check.** Statistics that would need the raw real rows —
coverage radius, the privacy nearest-neighbour distance, downstream lift — are
reported as `UNCHECKABLE` rather than quietly passed.

```
  Statistics (recomputed from delivered data + reference aggregates):
    ✓ correlation_delta: reported=0.101 recomputed=0.101
    ✓ class_counts: reported=40 recomputed=40
    – coverage_rate: UNCHECKABLE (needs raw real rare rows)
    – privacy_min_distance: UNCHECKABLE (needs raw real rare rows)
```

---

## 7. What is standard here, and what is not

The mathematics is standard and should be named as such: Gaussian copulas,
Gaussian-process regression with an ARD kernel, total variation distance,
Wasserstein-1, Pearson correlation, nearest-neighbour distance, TSTR/TRTR,
k-anonymity, and ordinary OLS and logistic regression. None of it was invented
here.

What the repo contributes is a standard of evidence: a deterministic engine that
produces every value, a model confined to exactly one judgement call that never
touches a number, and end-to-end recomputable assurance where a stranger can
re-derive every reported statistic.

The empirical result is the strongest part. Of seven sources put through the
same analysis, only the non-synthetic one certified. The practical generators, a
Gaussian copula, SMOTE, noised real data, and this repo's own, each preserve two
or three of the four coefficients, which is not a pass, since certification
requires all four. Whether the standard distributional checks also catch these
failures depends on where their thresholds are set, which is measured in
[`../FINDINGS.md`](../FINDINGS.md) section 3 rather than assumed.

Alongside it sit cleanly measured re-confirmations of known effects:
amplification lift is conditional on baseline recall, leakage-free evaluation
collapses inflated headlines (+39% to +4.4%), and synthetic utility is strongly
dataset-dependent, with train-on-synthetic recovery ranging from about 1.0 to
about 0.65.

---

*Related: [`COMPONENT_GUIDE.md`](COMPONENT_GUIDE.md) maps each component to the
method it uses and where it lives. [`METHODS.md`](METHODS.md) gives the formal
metric definitions. [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) records what is broken
and what was fixed.*
