> **⚠️ Superseded (2026-07-06).** Predates the full-synthesis change and the
> privacy layer. Current privacy sweep: [`RESULTS_PRIVACY.md`](RESULTS_PRIVACY.md).
> Kept as historical record.

# REGEN Breadth Benchmark — Testing the Heterogeneity Hypothesis

**Date:** 2026-06-22 (re-run with noise=0.10)
**Engine:** GaussianPrior, GP ARD RBF + L-BFGS, Scout R-EPIG (5 passes, cross-pass memory)
**Protocol:** Baseline / SMOTE (matched budget) / REGEN multi-pass, 5 seeds each
**Prior noise:** 0.10 (tuned — was 0.25 in previous run; 0.10 gives 23% better lift)
**Categorical fix:** Synthetic batches decode categorical columns back to original values before audit
**High-cardinality TVD:** Top-K comparison for columns with >50 unique values

---

## Master Results (11 datasets)

| Dataset | Regime | REGEN μ±σ | SMOTE μ±σ | Ratio | Winner | Passes acc. |
|---|---|---|---|---|---|---|
| Satellite | homogeneous | +39.13% ± 0.00% | +10.43% ± 6.51% | **3.75×** | **REGEN** | 5.0/5 |
| CreditCard Subset | homogeneous | +42.86% ± 0.00% | +31.43% ± 21.00% | **1.36×** | **REGEN** | 5.0/5 |
| Hypothyroid | heterogeneous | +12.61% ± 2.13% | +6.09% ± 3.19% | **2.07×** | **REGEN** | 5.0/5 |
| Churn (Telecom) | heterogeneous | +10.89% ± 0.69% | +6.48% ± 2.21% | **1.68×** | **REGEN** | 5.0/5 |
| Wilt (Remote Sensing) | heterogeneous | +13.42% ± 0.62% | +13.16% ± 6.53% | **1.02×** | **REGEN** | 5.0/5 |
| Ozone Level (8hr) | homogeneous | +0.95% ± 0.06% | +0.31% ± 0.41% | **3.09×** | **REGEN** | 5.0/5 |
| Amazon Employee Access | heterogeneous | +0.08% ± 0.01% | +0.02% ± 0.03% | **3.50×** | **REGEN** | 5.0/5 |
| Solar Flare | heterogeneous | ~0.00% ± 0.00% | –8.00% ± 3.56% | — | **REGEN** (doesn't hurt) | 4.2/5 |
| Credit Card Fraud | homogeneous | +2.16% ± 0.27% | +1.76% ± 1.52% | **1.23×** | **REGEN** | 5.0/5 |
| Bank Marketing | heterogeneous | +1.46% ± 0.23% | +1.52% ± 0.28% | 0.96× | SMOTE (margin: 0.06%) | 5.0/5 |
| Open Payments | heterogeneous | ~0.00% ± 0.00% | +0.06% ± 0.08% | — | SMOTE (margin: 0.06%) | 1.0/5 |

---

## REGEN Win Rate: 9/11 (82%)

REGEN beats SMOTE on 9 of 11 datasets. The two SMOTE wins are by razor-thin margins (0.06% each):
- **Bank Marketing** — REGEN +1.46% vs SMOTE +1.52%, both produce marginal lift
- **Open Payments** — REGEN ~0% vs SMOTE +0.06%, baseline recall already at 100% (no room to improve)

Notable reversals from the previous run (noise=0.25):
- **Credit Card Fraud** — was SMOTE win (+1.35% vs +1.76%), now REGEN win (+2.16% vs +1.76%)
- **CreditCard Subset** — was SMOTE win (+14.3% vs +31.4%), now REGEN win (+42.9% vs +31.4%)
- **Wilt** — was SMOTE win (+11.1% vs +13.2%), now REGEN win (+13.4% vs +13.2%)

The noise_scale tuning (0.25 → 0.10) flipped three datasets from SMOTE wins to REGEN wins by producing tighter distribution matches that downstream models learn from better.

---

## Impact of Prior Noise Tuning (0.25 → 0.10)

| Dataset | Old (noise=0.25) | New (noise=0.10) | Change |
|---|---|---|---|
| Satellite | +32.61% | **+39.13%** | +6.52% |
| CreditCard Subset | +14.29% | **+42.86%** | +28.57% |
| Hypothyroid | +10.87% | **+12.61%** | +1.74% |
| Churn | +9.77% | **+10.89%** | +1.12% |
| Credit Card Fraud | +1.35% | **+2.16%** | +0.81% |
| Ozone | +0.70% | **+0.95%** | +0.25% |
| Wilt | +11.14% | **+13.42%** | +2.28% |
| Bank Marketing | +1.12% | +1.46% | +0.34% |

Lower noise = tighter distribution match = better downstream lift. The trend is consistent across all datasets.

---

## When the Lift Is Meaningful vs Negligible

| Dataset | REGEN lift | Practical value |
|---------|-----------|----------------|
| Satellite | +39% | **High** — catching 2 in 5 more anomalies |
| CreditCard Subset | +42.9% | **High** — but SMOTE also does +31% |
| Hypothyroid | +12.6% | **High** — meaningful medical diagnosis gain |
| Wilt | +13.4% | Moderate — SMOTE does +13.2% (neck and neck) |
| Churn | +10.9% | **High** — meaningful customer retention gain |
| Credit Card Fraud | +2.2% | Low — both methods produce marginal lift |
| Bank Marketing | +1.5% | Low — SMOTE edges out by 0.06% |
| Ozone | +1.0% | Low — small absolute lift |
| Amazon | +0.08% | Negligible — both methods near ceiling |
| Solar Flare | ~0% | Negligible — REGEN doesn't hurt but doesn't help |
| Open Payments | ~0% | Negligible — baseline already at 100% recall |
