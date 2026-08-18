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
| [4](#4-certification-is-not-power-aware) | Certification is not power-aware | Medium | a coefficient the real data never pinned down can pass |
| [5](#5-numeric-predictors-only) | Numeric predictors only | Low | categorical predictors and treatment effects are out of scope |
| [6](#6-residual-bias-in-the-estimand-preserving-generator) | Residual bias in the estimand-preserving generator | Medium | two of four coefficients carry systematic bias |
| [7](#7-pandas-3x-breaks-numeric-coercion) | pandas 3.x breaks numeric coercion | Medium | dependencies are pinned as a result |

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

## 4. Certification is not power-aware

**Severity:** Medium. A scope limit, not a correctness bug.

Preservation is decided by a two-sample Wald test that uses the real
coefficient's standard error. When the real data is scarce, that standard error
is large, the interval is wide, and the test becomes lenient. It can then certify
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

## 6. Residual bias in the estimand-preserving generator

**Severity:** Medium. A real generation-quality gap, correctly flagged by the
certifier.

`regen/estimand_preserving.py` satisfies both conditions a coefficient depends
on, and it certifies the full analysis on 11 of 30 seeds. The shortfall is
systematic rather than random. Across 30 seeds, `utilization` carries a mean bias
of +0.169 against a spread of 0.102, and `log_limit` +0.059 against 0.034. Bias
larger than spread is distortion, not noise.

The source is the approximation gap in modelling the real predictor joint with a
Gaussian mixture. Partial coefficients are sensitive to that joint, so an
approximation that is adequate in aggregate can still misplace the partial
effects. `pay_delay_1` and `age` recover with bias small relative to spread.

Two implementation facts that matter for anyone extending it. Predictors must be
standardised before fitting the mixture, since a full-covariance mixture on raw
columns is dominated by large-scale features and models the small-scale structure
poorly. And richer mixtures preserve coefficients better while sitting closer to
the real records, which is the tension in issue 8 below rather than a free
improvement.

## 7. pandas 3.x breaks numeric coercion

**Severity:** Medium. Contained by pinning.

pandas 3.x defaults to Arrow-backed string storage, which breaks the numeric
coercion of categorical columns in the engine. `requirements.txt` pins pandas
below 3.0 for this reason. The coercion path needs fixing before the pin is
lifted.

## 8. Estimand preservation, privacy, and amplification cannot all be maximised

**Severity:** Structural. Not fixable, only measurable.

Preserving a coefficient pushes the synthetic predictor joint toward the real
one. Privacy pushes it away. Rare-event amplification reshapes the conditional
distribution of the outcome by construction, which is what preservation forbids.
Measured on the credit data, full certification collapses at roughly 0.1σ of
added noise, before that noise buys meaningful privacy.

No generator here escapes this. The bootstrap control preserves the estimand and
provides no privacy at all. A floored, amplified batch provides privacy and loses
the estimand. The contribution is making the position measurable per coefficient
so an operating point is chosen deliberately. See
[`../FINDINGS.md`](../FINDINGS.md) section 7.

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
distributions now score about 0.14 and pass, while genuinely mismatched
distributions score about 0.60 and are still rejected.

Implementation: `engine/auditor/fidelity.py`, tests in `tests/test_fidelity.py`.
