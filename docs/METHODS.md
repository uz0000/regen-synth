# METHODS — statistical methods reference (G-G)

The document a model-risk auditor reads. It formally defines every metric REGEN
scores, so a reported number can be **independently recomputed** and checked
(`regen verify`). Every metric has an **ID** and a **version** (in `regen/metrics.py`);
`explanation.json` and the manifest cite them, and the regression harness refuses
to compare results across metric versions silently. Changing a definition bumps
its version.

**Tolerance policy.** Cross-machine floating-point / BLAS differences are
expected, so verification compares within a per-metric numeric tolerance (the
`tol` field in the registry), never for exact bitwise equality. Tolerances live
in the registry, not in anyone's head.

---

## `correlation_delta` (v1, tol 1e-6; verify tol 5e-4)
**Formula.** Over the numeric (continuous + binary) feature columns, take the
Pearson correlation matrices of the real-rare reference `Cᵣ` and the synthetic
rare rows `Cₛ`; report `mean(|Cᵣ − Cₛ|)` over the upper triangle (off-diagonal),
skipping pairs where either is undefined (a constant column).
**Detects.** Broken joint structure — right marginals but scrambled dependence.
**Cannot detect.** Non-linear dependence the Pearson coefficient misses; per-row
issues.
**Threshold.** 0.25 (AuditorConfig). Rationale: on the benchmark datasets, honest
synthetic batches sit well under 0.15 while a column-shuffled batch exceeds 0.4;
0.25 separates them with margin (see `tests/test_fidelity.py`).
**Recomputable from aggregates?** Yes — the real correlation matrix is disclosed.
When the δ-floor was applied the *reported* value was measured pre-floor (not in
the bundle), so verify reports the delivered post-floor value as informational
rather than PASS/FAIL.

## `coverage_rate` (v1, tol 1e-6)
**Formula.** Fraction of real rare rows `r` for which some synthetic row `s` lies
within `√D` in per-feature σ-normalised L2 space (`D` = numeric feature count).
`√D` is the expected L2 norm of a `D`-dim standard normal, so the radius scales
with dimensionality.
**Detects.** A batch that fails to densify the actual rare region.
**Cannot detect.** Over-concentration; marginal errors.
**Threshold.** 0.50 default (0.30 for `boost`). **Recomputable from aggregates?**
No — needs the raw real rare rows; verify marks it UNCHECKABLE at aggregate
disclosure.

## `fisher_separation` (v1, tol 1e-4)
**Formula.** Per feature (encoded), `(μ_rare − μ_normal)² / (σ²_rare + σ²_normal)`.
Ranks features by how strongly they separate rare from normal — the
"features worth observing" signal.
**Detects.** Where the rare-class signal concentrates. **Cannot detect.**
Interactions between features (it is univariate). **Recomputable from aggregates?**
Yes — from the per-class per-column moments disclosed in the bundle.

## `privacy_min_distance` (v1, tol 1e-4)
**Formula.** Minimum σ-normalised L2 distance from any delivered rare row to any
real rare row, over continuous columns. When the floor holds, `≥ δ`.
**Detects.** Near-copy re-identification of a real rare individual. **Cannot
detect.** Membership inference / aggregate attacks (this is NOT differential
privacy — see `docs/PRIVACY.md`). **Recomputable from aggregates?** No — needs the
raw real rare rows; UNCHECKABLE at aggregate disclosure.

## `class_counts` (v1, tol 0)
**Formula.** Delivered per-class row counts. **Recomputable from aggregates?** Yes
(the synthetic split is disclosed). Verify checks the delivered rare count equals
the recorded split.

## `tail_lift` (v1, tol 1e-4)
**Formula.** `amplified_recall − baseline_recall`, both measured on a held-out
fold of real rare rows; the amplified detector's synthetic training data is
generated from the train fold only (leakage-free). Reported with `n_test_rare`
and a `status` (`insufficient_rare_rows` below `MIN_TEST_RARE = 10`).
**Detects.** Whether amplification improves rare-event recall. **Cannot detect.**
Transfer to a model class very unlike the RandomForest proxy. **Recomputable from
aggregates?** No — needs the full held-out detector protocol; UNCHECKABLE from the
bundle.

---

## Disclosure policy (auditability vs privacy)

Reference aggregates reveal information about the real data, so they are bounded:
histogram/quantile buckets are published only for a class with **≥
`min_bucket_count`** rows (default 10); correlation matrices and per-class column
moments are allowed; **no per-row values, ever**; identifier columns are
summarised as counts only. A deployment needing stricter disclosure dials
`min_bucket_count` up via the ScenarioSpec gates; `regen verify` then honestly
reports which statistics became UNCHECKABLE at that level rather than pretending
to verify them. The bucket floor is tested (`tests/test_audit.py`).
