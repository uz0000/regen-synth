# Known Issues

> This file mixes **historical, resolved** entries (kept as a record of what was
> fixed and how) with a **current** section at the bottom, added 2026-07-06 during
> the privacy/capability build (see `docs/BUILDLOG.md`). The first entry below is
> historical (resolved 2026-06-22).

---

# KNOWN ISSUE: High-Cardinality Categorical TVD Failures

**Status:** Resolved — top-K TVD comparison implemented for high-cardinality columns
**Discovered:** 2026-06-22 breadth benchmark re-run
**Severity:** Was Medium — blocked REGEN on datasets with extreme categorical cardinality (>500 unique values per column)

---

## What Was Fixed (Categorical Decode)

The synthetic batch was coming out with encoded integer codes (3.0, 7.0) instead
of original categorical values ("management", "technician"). The Auditor compared
encoded integers against real strings → TVD=1.0 → every batch rejected.

**Root cause:** The Prior and Amplifier encode categoricals to integers for
numerical computation, but nothing decoded them back before audit.

**Fix:** Added `_decode_categoricals()` in `regen/api.py` that reconstructs the
original category mapping from `pd.Categorical(rare_df[col])` and maps the
synthetic integer codes back to real values. Applied in three places:
- `run_campaign()` main loop
- `screen()` quick-campaign path
- `benchmark/run_breadth.py`

**Result:** Bank Marketing (16 features, 9 categorical) went from 0/5 passes to
5/5 accepted. REGEN now produces +1.12% lift (SMOTE wins at +1.52%, but REGEN
runs and produces valid synthetic data).

---

## What Was Fixed (High-Cardinality TVD)

Open Payments has 5 features, all categorical with extreme cardinality:

| Column | Unique values | TVD (old) | TVD (top-K) | Passes? |
|--------|--------------|-----------|-------------|---------|
| Drug/Biological Name | 1,115 | 0.47 | ~0.14 | Yes |
| Manufacturer Name | 612 | 0.45 | ~0.12 | Yes |
| Device/Medical Supply | 691 | 0.23 | ~0.10 | Yes |
| Physician Specialty | 199 | 0.23 | ~0.08 | Yes |
| Dispute Status | 2 | 0.004 | 0.004 | Yes |

**Root cause:** The Auditor's TVD check compared the full distribution over all
unique categories. With 1,115 unique drug names and only 200 synthetic rows per
pass, the TVD was mathematically guaranteed to be high — most of the 1,115
categories had zero synthetic representation. This was a sampling limitation,
not a data corruption bug.

**Fix:** Implemented top-K TVD comparison in `_tvd_discrete()`:
- When a column has >50 unique values (configurable via `AuditorConfig.high_card_threshold`),
  TVD is computed over only the top-K most frequent categories from the reference
  distribution, with all remaining categories grouped into an "other" bucket.
- K scales with the synthetic batch size: `K = min(n_unique, max(20, n_synthetic // 5))`.
  This ensures ~5 synthetic rows per compared category for stable proportion estimates.
- The "other" bucket captures whether the synthetic data covers the long tail at
  the right rate, without requiring it to represent every individual category.

**Result:** At Open Payments scale (1000 real rows, 200 synthetic), matching
distributions now produce TVD ≈ 0.14 (passes the 0.15 threshold), while
mismatched distributions produce TVD ≈ 0.60 (correctly rejected).

---

## Related Files

- `regen/api.py` — `_decode_categoricals()` (categorical decode fix)
- `engine/auditor/fidelity.py` — `_tvd_discrete()` (top-K TVD), `AuditorConfig.high_card_threshold`
- `tests/test_fidelity.py` — `test_auditor_high_cardinality_tvd_topk`, `test_auditor_high_cardinality_rejects_mismatched`
- `benchmark/run_breadth.py` — verification script

---

# CURRENT KNOWN ISSUES (2026-07-06 privacy/capability build)

These are open, characterized limitations surfaced by the P1-6 privacy sweep.
None are crashes (the one crash found — the coincident-row `inf` in
`enforce_distance_floor` — was fixed). Each has a repro in `docs/BUILDLOG.md`
and is slated for the G-E capability matrix.

## 1. `privacy="floored"` degrades low-cardinality integer/ordinal data
**Severity:** Medium (documented limit, not a correctness bug).
`solar_flare` (features are 3–6-value integer codes) under `privacy="floored"`:
coverage collapses 1.00→0.039 and the batch fails the fidelity gate. The
δ-distance floor (plus the integer-rounding margin) pushes the tiny integer-grid
rare cluster off its own region. **Workaround:** use `privacy="none"` for such
data, or treat those columns as categorical. Preflight (`regen doctor`, G-E)
should warn.

## 2. `privacy="floored"` costs fidelity on all-categorical high-cardinality data
**Severity:** Medium.
`open_payments` (all-categorical, high-cardinality) under `floored`: fidelity
0.80→0.40. The δ-floor is correctly skipped (`floor_applied=false`,
`no_continuous_features`); the loss comes from parametric frequency-table +
copula sampling of high-cardinality categoricals vs grounded anchoring. The
verbatim guard + k-anonymity still hold. **Workaround:** `privacy="none"` if
fidelity on high-card categoricals matters more than near-copy protection.

## 3. Pandas FutureWarning on integer-column write-back
**Severity:** Low (cosmetic; correct today, will error in pandas 3.0).
Enforcing the floor on integer-valued continuous columns emits a pandas
FutureWarning about assigning float values into an int64 column before the
re-round restores the dtype. Behaviour is correct; the assignment path should be
made dtype-clean (cast column to float up front) before pandas 3.0.

---

# ESTIMAND PRESERVATION (2026-07-11 G-H build, regression-coefficient v1)

## 4. Estimand certification is not yet power-aware
**Severity:** Medium (documented scope limit, not a correctness bug).
The `consistent` rule certifies preservation via a two-sample Wald test using
θ_real's standard error. When the *real* data is scarce, se_real is large, θ_real's
CI is wide, and the test becomes lenient — it can "certify" a coefficient the real
data never pinned down. Today this is **surfaced, not failed**: each target reports
`real_significant` (does θ_real's CI exclude 0?), so a vacuous "preservation of a
null effect" is visible in `explanation.json` / `regen verify`. **Planned v2:** a
readiness floor that refuses (or down-grades to `uncertifiable`) when θ_real itself
is not credibly estimated — i.e. couple estimand certification to a power/precision
target on θ_real, the estimand analogue of the rare-count readiness gate. The hard
floor already holds: if θ_real cannot be *fit at all* (too few rows), certification
is refused (`status="uncertifiable"`), never faked.

## 5. Estimand v1 supports numeric predictors only
**Severity:** Low (scope, stated up front).
`fit_estimand` requires the outcome and predictors to be numeric; a non-numeric
predictor raises `EstimandError` (caught by `evaluate` → `uncertifiable`, never a
crash). Categorical/one-hot predictors and interaction terms are a v2 extension.
The same recompute-and-certify machinery is intended to extend from coefficients
to an ATE (declared treatment/outcome/adjustment set) without changing the
certificate's shape.

## 6. REGEN's generator does not preserve estimands on discrete non-linear predictors (V2 TARGET)
**Severity:** Medium (a real generation-quality gap; the certifier correctly flags it).
On real UCI credit-default data (`examples/certifier_demo/`), REGEN's synthetic
passes every fidelity check but shifts the logit coefficient of `pay_delay_1` (a
discrete ordinal, the strongest predictor) from +0.71 to ~+0.93, consistently
across all modes/ratios (so it is **not** class-rebalancing). The multi-generator
demo shows this is **not specific to REGEN**: a plain Gaussian copula (+0.46) and
SMOTE (+0.61) fail the same coefficient. **Diagnosis (CONFIRMED by univariate-vs-multivariate diagnostic, 2026-07-11):**
the marginal distribution of `pay_delay_1` is preserved everywhere (why fidelity
passes), but the **conditional relationship `P(default | pay_delay)` is distorted**,
and it is a marginal-level distortion — the univariate and multivariate coefficients
move *together* per generator (real 0.74/0.71, REGEN 0.97/0.93, copula 0.47/0.46),
which **rules out partial-effect redistribution** between correlated predictors.
Two opposite mechanisms, same failure: (a) the **Gaussian copula linearises** a
threshold dependence — `pay_delay` has a non-linear jump (P(default) 0.70 at pay≥2
vs lower below), which a single linear rank-correlation cannot represent → it flattens
it → coefficient collapses (corr 0.325→0.218); (b) **REGEN's amplifier over-represents
the tail** — densifying the rare/default region (where pay_delay is high) steepens the
gradient (P(default|pay≥2) 0.696→0.766) → coefficient inflates (corr 0.325→0.417).
**Key tension:** REGEN's rare-event amplification (its detection-lift value) reshapes
P(y|x) BY CONSTRUCTION — you cannot maximise amplification and preserve the estimand
at once. **v2 fix direction:** estimand preservation needs generation from the real
**conditional** P(y|x), not marginals+dependence — e.g. draw x from the joint, then
draw y from a flexible calibrated model of the *real* conditional (preserves whatever
the real conditional is, coefficients included, with NO coefficient injection —
Invariant 1 holds; values come from a statistical model sampled, not a declared
number). For REGEN: an explicit "estimand-preserving mode" trading amplification for
conditional fidelity. Needs its own demo. See `docs/BUILDLOG.md` session 2026-07-11 (b).

**FIX-VALIDATION experiment (2026-07-11) — the fix has TWO requirements, not one.**
Reconstructing `y` from a flexible real conditional P(y|x) (gradient-boosted, not a
logit → no form injection) over several x-sources:
- With **faithful x** (bootstrap of real predictors), `pay_delay_1` recovers almost
  exactly (+0.711 vs +0.714 real; raw REGEN was +0.93). The primary distortion IS in
  P(y|x), and conditional resampling fixes it. ✔ mechanism validated.
- With **copula-x / independent-x**, `utilization` stays wrong and **flips sign**
  (+0.086 / +0.202 vs −0.369 real) despite a correct P(y|x). A logit coefficient is a
  *partial* effect depending on **cov(x)**, which those x-sources distort.
- Even bootstrap-x overshoots `utilization` (−0.481 vs −0.369): a conditional-*model*-
  fidelity gap (GB's logit projection ≠ real logit). Addressable.

**Refined v2 requirements:** estimand-preserving generation needs BOTH (R1) the
predictor **joint/dependence** structure right (beyond marginals + linear correlation
— utilization is sensitive to this) AND (R2) `y` drawn from a well-calibrated real
**conditional** P(y|x) (pay_delay is sensitive to this). A naive "resample y only" fix
satisfies R2 alone and would silently ship a sign-flipped utilization — which is why
this was validated before building. Bootstrap satisfies both (it is real data) → the
target behaviour. Smooth predictors (`log_limit`, `age`) are insensitive to both.

**GENERALITY confirmed (2026-07-11):** the same R1/R2 structure replicates on a
different dataset AND estimand family — **OLS `MedHouseVal ~ MedInc + HouseAge +
AveRooms + Latitude` on California housing** (20,640 rows): bootstrap preserves;
Gaussian copula distorts (AveRooms −0.14→−0.19); conditional-resample over
bootstrap-x recovers all four; conditional-resample over copula-x still distorts the
partial coefficients (MedInc 0.49→0.43, AveRooms 0.14→0.10). Nuance: the distortion
is **milder** here (no sign flips) because these predictors are smoother/more linear
than the discrete-threshold `pay_delay_1` — **estimand-loss magnitude scales with how
non-linear/discrete the predictor↔outcome relationship is.** Mechanism general;
severity data-dependent. **Also uncovered — a three-way tension:** R1 (preserve the
predictor joint) collides with **privacy** (keeping x close to real = copying records;
the verbatim guard forbids it), and with **amplification** (reshapes the tail). No
generator maximises estimand + privacy + amplification at once (bootstrap wins
estimand, loses privacy; floored+amplified REGEN wins privacy, loses estimand). The
certifier's role is to make this surface **measurable** so an operating point is
chosen deliberately — that reframes v2 from "fix the generator" to "expose and price
the tradeoff."
