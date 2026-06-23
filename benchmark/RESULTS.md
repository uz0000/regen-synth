# REGEN Benchmark Results

> **⚠ SUPERSEDED** — These are single-pass results from 2026-06-18.
> Current results: [`RESULTS_BREADTH.md`](RESULTS_BREADTH.md) (11 datasets, multi-pass, 5 seeds each).

**Date:** 2026-06-18
**Engine:** GaussianPrior (default), GP with ARD RBF kernel + L-BFGS optimization
**Seeds per dataset:** 5 (42–46)

---

## Summary Table

| Dataset | N (normal) | N (rare) | REGEN lift μ±σ | SMOTE lift μ±σ | REGEN wins? | Auditor pass |
|---|---|---|---|---|---|---|
| Credit Card Fraud | 284,315 | 492 | **+0.68% ± 0.48%** | +3.24% ± 1.30% | ❌ SMOTE | 100% |
| Hypothyroid | 3,012 | 151 | **+5.65% ± 1.94%** | +7.83% ± 4.51% | ❌ SMOTE | 0% |
| Satellite | 5,025 | 75 | **+20.00% ± 3.89%** | +12.17% ± 7.78% | ✅ REGEN | 100% |

---

## What Actually Ran

### Backend
All three benchmarks used the **GaussianPrior** backend (`backend='gaussian'`). The PFN backend (TabPFN) was verified separately and works end-to-end, but its `fit()` does not scale past ~1000 training rows, so the GaussianPrior is the practical default.

### GP Learning
The GP uses an **ARD RBF kernel with L-BFGS optimization** (max 500 iterations, 120s timeout). On all three datasets the optimization converged:
- Lengthscale ranges vary from ~1.0 (highly relevant) to ~11,000 (irrelevant) — the ARD kernel correctly identifies which features drive tail deviation.
- Per-feature relevance is derived from fitted inverse-lengthscales, not a variance proxy.

### Pass Configuration
- Batch size: 200 rows per pass
- GP max features: 0 (all features used; no dimensionality reduction)
- Coverage threshold: 0.50 (lenient — allows more passes through)
- One pass only (no active-learning loop; the Scout was not used in these single-pass benchmarks)

---

## Honest Assessment

### Where REGEN Wins: Satellite (+20% lift, beats SMOTE on all 5 seeds)

On this anomaly detection dataset (75 anomalies in 5,100 samples), REGEN significantly outperforms SMOTE. The residual GP correction meaningfully shifts generated data toward the tail distribution, and the ARD kernel identifies the features most relevant to the anomaly class.

This is the case where REGEN's architecture justifies itself: the data has enough rare rows (75) for the GP to learn a useful residual surface, and the features are informative enough that the ARD lengthscales differentiate them.

### Where REGEN Underperforms: Credit Card Fraud (-2.6% vs SMOTE)

On the largest dataset, REGEN produces small positive lift (+0.68%) but SMOTE does better (+3.24%). Likely causes:
- **The base batch generator (jittered resampling) is conceptually close to SMOTE.** Both generate new rows by perturbing real rare rows. REGEN's advantage should come from the GP correction, but with 30 PCA features and only 300 rolling-buffer observations, the GP's residual surface may be too smooth to meaningfully shift the generated distribution beyond what simple jittering already does.
- **The Auditor gate is lenient** (coverage=0.50 threshold), so batches that don't match the rare-event distribution well still pass.

### Where REGEN Needs Work: Hypothyroid (0% Auditor pass rate)

All 5 REGEN passes failed the Auditor — likely the coverage check with 25 features and only 151 rare rows. The coverage metric (L2/sqrt(D) adaptive radius) is stringent enough that with 25 dimensions, few synthetic rows fall within 5 standard deviations of each real rare row. The lifts reported are from batches that *would* have been rejected in production.

### Comparison to Baseline

| Dataset | Baseline recall | REGEN amplified recall | SMOTE amplified recall |
|---------|----------------|----------------------|----------------------|
| Credit Card | ~0.838 | ~0.845 | ~0.870 |
| Hypothyroid | ~0.650 | ~0.706 | ~0.728 |
| Satellite | ~0.580 | ~0.780 | ~0.702 |

---

## Recommendations

1. **Single-pass mode is not the product.** The active-learning loop (Scout → multiple passes) is where REGEN's advantage compound. The single-pass results above understate the system's potential — each pass should target a different tail region and build on the last. Running the full campaign (5 passes, with Scout selecting targets) produced +2.03% on the earlier credit card benchmark.

2. **Hypothyroid's 0% audit rate needs investigation.** The coverage radius may need to scale differently with dimensionality. Alternatively, the 151 rare rows may simply be too few for meaningful amplification on 25-dimensional data.

3. **SMOTE is a stronger baseline than expected.** On two of three datasets, SMOTE beats or ties REGEN on single-pass recall lift. REGEN needs to demonstrate value-add through either (a) the multi-pass active-learning loop, or (b) the ARD-driven feature relevance that SMOTE cannot provide.

---

## Raw Data

Full per-seed results in `benchmark/RESULTS.json`.
