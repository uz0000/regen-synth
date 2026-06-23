# REGEN Breadth Benchmark — Testing the Heterogeneity Hypothesis

**Date:** 2026-06-22
**Engine:** GaussianPrior, GP ARD RBF + L-BFGS, Scout R-EPIG (5 passes, cross-pass memory)
**Protocol:** Baseline / SMOTE (matched budget) / REGEN multi-pass, 5 seeds each
**Prediction rule:** Heterogeneous features → REGEN wins; Homogeneous → SMOTE wins
**Loop fix:** Scout now runs at the START of every pass with accumulated explored_points memory (was blind on pass 1, no cross-pass memory before)
**Categorical fix:** Synthetic batches now decode categorical columns back to original values before audit (was comparing encoded integers against real strings)

---

## Master Results (10 of 11 datasets valid; 1 high-cardinality failure)

| Dataset | Regime | Predicted | Actual | REGEN μ±σ | SMOTE μ±σ | Ratio | Match? | Passes acc. |
|---|---|---|---|---|---|---|---|---|
| Hypothyroid | heterogeneous | REGEN | **REGEN** | +10.87% ± 0.00% | +6.09% ± 3.19% | **1.79×** | ✅ | 5.0/5 |
| Satellite | homogeneous | SMOTE | **REGEN** | +33.04% ± 2.13% | +10.43% ± 6.51% | **3.17×** | ❌ | 5.0/5 |
| Churn (Telecom) | heterogeneous | REGEN | **REGEN** | +9.77% ± 0.46% | +6.48% ± 2.21% | **1.51×** | ✅ | 5.0/5 |
| Wilt (Remote Sensing) | heterogeneous | REGEN | **SMOTE** | +11.14% ± 1.86% | +13.16% ± 6.53% | 0.85× | ❌ | 5.0/5 |
| Amazon Employee Access | heterogeneous | REGEN | **REGEN** | +0.10% ± 0.00% | +0.02% ± 0.03% | **4.50×** | ✅ | 5.0/5 |
| Credit Card Fraud | homogeneous | SMOTE | **SMOTE** | +1.35% ± 0.00% | +1.76% ± 1.52% | 0.77× | ✅ | 5.0/5 |
| CreditCard Subset | homogeneous | SMOTE | **SMOTE** | +14.29% ± 0.00% | +31.43% ± 21.00% | 0.45× | ✅ | 5.0/5 |
| Ozone Level (8hr) | homogeneous | SMOTE | **REGEN** | +0.70% ± 0.00% | +0.31% ± 0.41% | **2.27×** | ❌ | 5.0/5 |
| Solar Flare | heterogeneous | REGEN | **REGEN** | ~0.00% ± 0.00% | –8.00% ± 3.56% | — | ✅ | 4.2/5 |
| Bank Marketing | heterogeneous | REGEN | **SMOTE** | +1.12% ± 0.12% | +1.52% ± 0.28% | 0.74× | ❌ | 5.0/5 |
| Open Payments | heterogeneous | REGEN | **FAILED** | — | — | — | — | 0.0/5 |

**Open Payments** (1 dataset) failed: all 5 features are categorical with extreme cardinality (612–1,115 unique values). The Auditor's TVD check mathematically cannot pass with 200 synthetic rows vs 1,115 categories. See `docs/KNOWN_ISSUES.md`.

---

## REGEN Win Rate: 6/10 (60%)

Across 10 datasets with valid results, REGEN beats SMOTE on 6. The 4 SMOTE wins:
- **Credit Card Fraud** and **CreditCard Subset** — both PCA-compressed (28 PCA components + Amount + Time). SMOTE's nearest-neighbor interpolation is better on redundant features.
- **Wilt** — 5 mixed features (satellite + soil). REGEN gets +11.1% but SMOTE edges it out at +13.2%.
- **Bank Marketing** (new, categorical-heavy) — REGEN gets +1.12% but SMOTE does +1.52%. Both methods produce marginal lift on this data.

---

## Impact of the Active-Learning Loop Fix

The fix (Scout targeting with cross-pass memory) improved REGEN's performance on 5 of 8 previously-benchmarked datasets:

| Dataset | Old (broken loop) | New (fixed loop) | Change |
|---|---|---|---|
| Satellite | +26.09% | **+33.04%** | **+6.95%** |
| CreditCard Subset | +8.57% | **+14.29%** | **+5.72%** |
| Credit Card Fraud | +0.81% | **+1.35%** | +0.54% |
| Hypothyroid | +10.00% | **+10.87%** | +0.87% |
| Amazon | +0.08% | **+0.10%** | +0.02% |
| Ozone | +0.81% | +0.70% | -0.11% |
| Churn | +11.17% | +9.77% | -1.40% |
| Solar Flare | ~0% | ~0% | — |

Satellite saw the largest gain (+7%), confirming that cross-pass targeting compounds coverage across the campaign.

---

## Impact of the Categorical Decode Fix

Bank Marketing went from 0/5 passes (total failure) to 5/5 accepted with +1.12% lift. The synthetic output now contains real categorical values ("management", "technician") instead of encoded integers (3.0, 9.0).

---

## When the Lift Is Meaningful vs Negligible

| Dataset | REGEN lift | Practical value |
|---------|-----------|----------------|
| Satellite | +33% | **High** — catching 1 in 3 more anomalies |
| Hypothyroid | +10.9% | **High** — meaningful medical diagnosis gain |
| Churn | +9.8% | **High** — meaningful customer retention gain |
| Wilt | +11.1% | Moderate — but SMOTE does +13.2% |
| CreditCard Subset | +14.3% | Moderate — but SMOTE does +31.4% |
| Bank Marketing | +1.1% | Low — SMOTE does +1.5% |
| Credit Card Fraud | +1.4% | Low — SMOTE does +1.8% |
| Ozone | +0.7% | Low — small absolute lift |
| Amazon | +0.10% | Negligible — both methods near ceiling |
| Solar Flare | ~0% | Negligible — REGEN doesn't hurt but doesn't help |

---

## Auditor Pass Rate

9 of 10 valid datasets achieved 5/5 accepted across all seeds. Solar Flare averaged 4.2/5 (one rejection per seed). Open Payments achieved 0/5 (high-cardinality TVD — see KNOWN_ISSUES.md).

---

## Screening rule for prospect data

Before quoting a prospect:
1. Check if features are PCA-compressed → recommend SMOTE, caveat REGEN
2. Check if features have extreme cardinality (>500 unique values per column) → Auditor may reject; fix pending
3. Check if features are native/raw numeric → REGEN likely wins
4. Run `regen screen` on the dataset → ARD heterogeneity metric predicts winner at ~56% accuracy
5. Run one seed on a sample → confirms within ~30s
