# Known issues

What is currently open, with the effect on results and the workaround where one
exists. Issues are numbered for citation and the numbers are stable: a resolved
issue keeps its number and moves to the resolved list at the bottom rather than
being renumbered.

Claims that were published and later revised are not issues. Those are in
[`../CORRECTIONS.md`](../CORRECTIONS.md).

| # | Issue | Severity | Effect |
|---|---|---|---|
| [1](#1-the-distance-floor-degrades-low-cardinality-integer-data) | Distance floor degrades low-cardinality integer data | Medium | coverage collapses, batch rejected |
| [2](#2-the-distance-floor-costs-fidelity-on-high-cardinality-categorical-data) | Distance floor costs fidelity on high-cardinality categorical data | Medium | fidelity roughly halves |
| [3](#3-pandas-futurewarning-on-integer-write-back) | pandas `FutureWarning` on integer write-back | Low | cosmetic today, an error in pandas 3.0 |
| [4](#4-the-check-does-not-account-for-how-precise-the-real-estimate-is) | The check does not account for how precise the real estimate is | Medium | a coefficient the real data never pinned down can pass |
| [5](#5-numeric-predictors-only) | Numeric predictors only | Low | categorical predictors and treatment effects are out of scope |
| [6](#6-a-repeating-error-in-the-coefficient-preserving-generator) | A repeating error in the coefficient-preserving generator | Medium | two of four coefficients are wrong the same way every run |
| [7](#7-pandas-3x-breaks-numeric-coercion) | pandas 3.x breaks numeric coercion | Medium | dependencies are pinned as a result |
| [8](#8-keeping-coefficients-privacy-and-rare-case-amplification-cannot-all-be-maximised) | Keeping coefficients, privacy, and rare-case amplification cannot all be maximised | Structural | no generator can maximise all three |
| [9](#9-refusal-gets-more-likely-as-an-estimand-declares-more-coefficients) | Refusal gets more likely as an estimand declares more coefficients | Medium | a faithful generator is refused more often on a larger estimand |
| [10](#10-the-high-cardinality-tvd-gate-sits-inside-its-own-noise-band) | The high-cardinality TVD gate sits inside its own noise band | Medium | a faithful high-cardinality batch is rejected on some seeds |

---

## 1. The distance floor degrades low-cardinality integer data

**Severity:** Medium. A documented limit, not a correctness bug.

With `privacy="floored"`, enforcing a minimum distance between synthetic and real
records assumes the feature space is continuous enough to move a point without
leaving its own region. On features that are 3 to 6 valued integer codes, it is
not. On `solar_flare`, coverage falls from 1.00 to 0.039 and the batch fails the
fidelity gate, because the floor plus the integer-rounding margin pushes the rare
cluster off the small integer grid it lives on.

**Workaround.** Use `privacy="none"` for such data, or declare those columns
categorical so the floor is skipped rather than misapplied.

## 2. The distance floor costs fidelity on high-cardinality categorical data

**Severity:** Medium.

A distance floor needs a metric. On an all-categorical table there is no
continuous feature to enforce it on, so the floor is correctly skipped and
reports that it was skipped. The fidelity loss comes from the fallback path:
sampling high-cardinality categoricals from parametric frequency tables and a
copula, rather than anchoring on real values. On `open_payments`, fidelity falls
from 0.80 to 0.40. The verbatim guard and the k-anonymity constraint still hold,
so the privacy properties that do not depend on a metric are unaffected.

**Workaround.** Use `privacy="none"` when fidelity on high-cardinality
categoricals matters more than near-copy protection.

## 3. pandas `FutureWarning` on integer write-back

**Severity:** Low. Behaviour is correct today and will become an error.

Enforcing the floor on integer-valued continuous columns assigns float values
into an `int64` column before a re-round restores the dtype. pandas warns about
this and will raise in 3.0. The assignment path should cast to float up front.

## 4. The check does not account for how precise the real estimate is

**Severity:** Medium. A scope limit, not a correctness bug.

Whether a coefficient survived is decided by a standard test that asks if the
real and synthetic estimates are far enough apart, given how precisely each was
measured, to count as different answers. When the real data is scarce, its own estimate is
imprecise, so the range it could plausibly occupy is wide and the test becomes
easy to pass. It can then certify
a coefficient the real data never established in the first place, which is
preservation of a null result rather than preservation of a finding.

This is surfaced rather than failed. Every target reports `real_significant`,
which is whether the real coefficient's interval excludes zero, so a vacuous
certification is visible in `explanation.json` and in `regen verify`.

The hard floor already holds: if the real coefficient cannot be fit at all, the
status is `uncertifiable` and certification is refused rather than faked.

**Planned.** Couple certification to a precision target on the real estimate, so
that an underpowered real fit downgrades to `uncertifiable` instead of passing.

## 5. Numeric predictors only

**Severity:** Low. Scope, stated up front.

`fit_estimand` requires numeric outcome and predictors. A non-numeric predictor
raises `EstimandError`, which `evaluate` turns into `uncertifiable` rather than a
crash. Categorical and one-hot predictors, interaction terms, and average
treatment effects over a declared adjustment set are not supported. The
recompute-and-compare machinery is expected to extend to them without changing
the shape of the certificate.

## 6. A repeating error in the coefficient-preserving generator

**Severity:** Medium. A real generation-quality gap, correctly flagged by the
certifier.

`regen/estimand_preserving.py` satisfies both conditions a coefficient depends
on, and it gets the full analysis right on 11 of 30 runs. The shortfall is not
luck. Across 30 runs, `utilization` misses the truth by +0.169 on
average while scattering only 0.102 around that average, and `log_limit` misses
by +0.059 while scattering 0.034. When the miss is larger than the scatter, the
error repeats every run in the same direction rather than averaging away.

The cause is how the method describes the arrangement of the real predictor rows,
using a blend of overlapping clouds. A coefficient measures one factor's effect
with the others held still, which depends on exactly how the factors overlap, so
a description that looks right overall can still get that overlap wrong.

`pay_delay_1` and `age` do not have this problem: they scatter more than they
miss, which is ordinary noise.

Two implementation facts that matter for anyone extending it. The predictors must be put on a common scale before
fitting, or columns with large numbers dominate and the ones with small numbers
are described badly. And a more detailed description keeps the coefficients better
while sitting closer to the real records, which is the trade in issue 8 rather
than a free improvement.

## 7. pandas 3.x breaks numeric coercion

**Severity:** Medium. Contained by pinning.

pandas 3.x defaults to Arrow-backed string storage, which breaks the numeric
coercion of categorical columns in the engine. `requirements.txt` pins pandas
below 3.0 for this reason. The coercion path needs fixing before the pin is
lifted.

## 8. Keeping coefficients, privacy, and rare-case amplification cannot all be maximised

**Severity:** Structural. Not fixable, only measurable.

Keeping a coefficient right pushes the synthetic rows toward sitting where the
real rows sit. Privacy pushes them away. Manufacturing extra rare cases changes
how often the outcome occurs in the region being filled, which is exactly what
keeping the coefficient forbids.

Measured on the credit data, the coefficients stop surviving once about 0.1
standard deviations of noise is added, which is before that noise buys any
meaningful privacy.

No generator here escapes this. The control that resamples real rows keeps the
coefficients and provides no privacy at all, because it is the real rows. A batch generated with the distance floor on and rare cases
amplified provides privacy and loses the coefficients. The contribution is making the position measurable for each
coefficient, so the trade is chosen deliberately rather than assumed. See
[`../FINDINGS.md`](../FINDINGS.md) section 7.

## 9. Refusal gets more likely as an estimand declares more coefficients

**Severity:** Medium. A property of the decision rule, not a bug.

An estimand is certified only when every declared coefficient is preserved, and
each is compared at 95%. So a generator that is genuinely faithful still gets
refused whenever any one comparison lands in its 5% tail, and that chance
compounds with the number of coefficients declared. Four independent comparisons
refuse about 18.5% of the time; the certifier is milder than that in practice
because it treats the two fits as independent when they are correlated, which
inflates the combined standard error.

Measured on the positive control under the demo's configuration:

```
positive control refused   36 / 300 seeds  = 12.0%   (95% CI 8.3% - 15.7%)
```

Two consequences worth knowing:

- **A refused control is not a broken checker.** This was previously documented
  the other way round, and is [`../CORRECTIONS.md`](../CORRECTIONS.md) entry 4.
- **Certification is not comparable across estimands of different sizes.** A
  two-predictor estimand is easier to certify than a ten-predictor one, at the
  same underlying fidelity, so a certification rate only means something against
  a fixed declared analysis.

**Direction of the error.** The compounding makes the check refuse good
generators; the independence assumption makes it pass bad ones. On the reported
results the second dominates — the headline failures sit at z = 3.4 to 7.3
against a 1.96 cutoff — so no published verdict turns on either.

**Planned.** An equivalence test with a declared margin of practical equivalence
would replace "we could not tell them apart" with "they agree to within delta",
and composes across coefficients without this compounding, since the claim is an
intersection of rejections rather than of failures to reject. Not built. Related
to issue 4, which is the same weakness seen from the other side.

Rates are pinned in `tests/test_control_rates.py`.

## 10. The high-cardinality TVD gate sits inside its own noise band

**Severity:** Medium. A threshold that does not separate what it is meant to
separate, on this data shape.

For a column with more categories than the batch can cover, the auditor compares
only the top-K categories by real frequency. That fixed the arithmetic problem in
the resolved list below. It did not make the gate reliable.

Measured on the Open Payments-shaped fixture (500 power-law categories, 1,000
real rows, 200 synthetic), five seeds:

```
matching distributions      TVD 0.137 mean, range 0.106 - 0.174
genuinely mismatched        TVD 0.587 mean
gate                        0.15
```

Discrimination is not in doubt — matching and mismatched are four-fold apart. The
problem is that the *gate* sits inside the matching case's spread, so a faithful
batch is rejected on a substantial share of seeds while a corrupted one is always
caught. The check is directionally right and its cut-off is not earned.

**Why it looked fine before.** The top-K set was selected with
`value_counts().nlargest(k)`, which breaks ties arbitrarily; 17 categories tie at
the boundary on this fixture. The score therefore varied by platform, and the
knife-edge stayed invisible until the tie-break was made deterministic and the
score settled at a reproducible 0.152. That reproducibility bug is fixed
(`engine/auditor/fidelity.py`); the gate question it was hiding is this issue.

**Not fixed here on purpose.** Loosening the threshold changes what the auditor
ships, which is a product decision rather than a bug fix, and would move
accepted-batch counts in the benchmark. The options are a wider cut-off for
high-cardinality columns specifically, a K that does not grow with batch size
(TVD *worsens* as K grows, because a larger K pulls in categories the real sample
itself estimates badly), or a gate calibrated from the sampling noise at the
observed scale rather than set as a constant.

**Workaround.** Treat a high-cardinality TVD near the gate as inconclusive rather
than as a verdict, and read it beside the mismatched reference of roughly 0.6.

---

## Resolved

Kept for the record, with the numbers that showed the fix worked.

**Categorical decode (resolved 2026-06-22).** Synthetic batches carried encoded
integer codes where the real data had category labels, so the rejection gate
compared integers against strings, produced a total variation distance of 1.0,
and rejected every batch. `_decode_categoricals()` in `regen/api.py` reconstructs
the mapping and inverts it before scoring. Bank Marketing went from 0 of 5 batches
accepted to 5 of 5.

**High-cardinality total variation distance (resolved 2026-06-22).** The
discrete-marginal check compared the full distribution over every category. With
1,115 distinct drug names and 200 synthetic rows per batch, a high distance was
arithmetically guaranteed, since most categories could not appear at all. This
was a sampling limit being read as data corruption. `_tvd_discrete()` now
compares the top K categories by real frequency, with the remainder pooled into a
single bucket, where `K = min(n_unique, max(20, n_synthetic // 5))` keeps roughly
five synthetic rows per compared category. At Open Payments scale, matching
distributions score about 0.14 against roughly 0.60 for genuinely mismatched
ones, so the two are cleanly separated.

**Partially superseded by issue 10.** This entry originally read "score about
0.14 and pass". The mean is right; the "and pass" was not. The spread across
seeds is 0.106 to 0.174 against a 0.15 gate, so a faithful batch is rejected on a
substantial share of runs. The arithmetic problem described here is genuinely
resolved; the threshold question it exposed is open as issue 10.

Implementation: `engine/auditor/fidelity.py`, tests in `tests/test_fidelity.py`.
