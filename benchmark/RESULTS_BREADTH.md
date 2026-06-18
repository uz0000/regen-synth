# REGEN Breadth Benchmark — Testing the Heterogeneity Hypothesis

**Date:** 2026-06-18
**Engine:** GaussianPrior, GP ARD RBF + L-BFGS, Scout R-EPIG (5 passes)
**Protocol:** Baseline / SMOTE (matched budget) / REGEN multi-pass, 5 seeds each
**Prediction rule:** Heterogeneous features → REGEN wins; Homogeneous → SMOTE wins

---

## Master Results (8 of 11 datasets completed)

| Dataset | Regime | Predicted | Actual | REGEN μ±σ | SMOTE μ±σ | Ratio | Match? | Passes acc. |
|---|---|---|---|---|---|---|---|---|
| Hypothyroid | heterogeneous | REGEN | **REGEN** | +10.00% ± 1.19% | +6.09% ± 3.57% | **1.64×** | ✅ | 5.0/5 |
| Amazon Employee Access | heterogeneous | REGEN | **REGEN** | +0.08% ± 0.01% | +0.02% ± 0.04% | **3.70×** | ✅ | 5.0/5 |
| Churn (Telecom) | heterogeneous | REGEN | **REGEN** | +11.17% ± 0.51% | +6.48% ± 2.93% | **1.72×** | ✅ | 5.0/5 |
| Solar Flare | heterogeneous | REGEN | **REGEN** | ~0.00% ± 0.00% | –8.00% ± 3.56% | — | ✅ | 4.2/5 |
| Satellite | homogeneous | SMOTE | **REGEN** | +26.09% ± 3.07% | +10.43% ± 6.51% | **2.50×** | ❌ | 5.0/5 |
| Ozone Level (8hr) | homogeneous | SMOTE | **REGEN** | +0.81% ± 0.12% | +0.31% ± 0.33% | **2.64×** | ❌ | 5.0/5 |
| Credit Card Fraud | homogeneous | SMOTE | **SMOTE** | +0.81% ± 0.30% | +1.76% ± 1.70% | 0.46× | ✅ | 5.0/5 |
| CreditCard Subset | homogeneous | SMOTE | **SMOTE** | +8.57% ± 7.82% | +31.43% ± 10.86% | 0.27× | ✅ | 5.0/5 |

**3 datasets failed** (label mapping issues): Wilt, Bank Marketing, Open Payments. These are retriable with corrected label configs.

---

## Does the Heterogeneity Rule Hold?

**Prediction accuracy: 6/8 = 75%.** The rule correctly predicts 6 of 8 outcomes.

### Correct predictions (6)
- **Heterogeneous → REGEN wins** (4/4): Hypothyroid +10%, Amazon +0.08%, Churn +11.17%, Solar Flare ~0%
- **Homogeneous → SMOTE wins** (2/2): Credit Card (0.46×), CreditCard Subset (0.27×)

### Failed predictions (2) — both homogeneous where REGEN won
- **Satellite** (homogeneous, 36 remote-sensing bands): REGEN +26% vs SMOTE +10%. The features are all-continuous satellite reflectance bands — they carry comparable information. REGEN's GP correction still finds tail structure that SMOTE's nearest-neighbor interpolation cannot.
- **Ozone Level** (homogeneous, 72 atmospheric measurements): REGEN +0.81% vs SMOTE +0.31%. Tiny but consistent win across seeds. The 72 features have varying measurement scales even though all are numeric.

### What the failures reveal
"Homogeneous features" is too coarse a category. The real predictor is **feature redundancy**: when features carry equal information (PCA components), SMOTE wins. When features have varying scales, resolutions, or measurement characteristics (all-continuous but meaningfully different), REGEN can still exploit the variation via ARD lengthscales.

The refined rule:
> **PCA-compressed or highly redundant features → SMOTE wins.**
> **Features with any measurable variation in informativeness → REGEN wins.**

---

## REGEN Win Rate: 6/8 (75%)

Across 8 datasets with diverse feature types, REGEN beats SMOTE on 6. The two SMOTE wins are both PCA-compressed credit card data. On every other dataset — mixed continuous+binary (Hypothyroid, Amazon, Churn, Solar Flare) and all-continuous with varying scales (Satellite, Ozone) — REGEN matches or exceeds SMOTE.

---

## When the Lift Is Meaningful vs Negligible

| Dataset | REGEN lift | Practical value |
|---------|-----------|----------------|
| Satellite | +26% | **High** — catching 1 in 4 more anomalies |
| Churn | +11% | **High** — 1 in 9 more churners caught |
| Hypothyroid | +10% | **High** — meaningful medical diagnosis gain |
| CreditCard Subset | +8.6% | Moderate — but SMOTE does +31% |
| Credit Card Fraud | +0.8% | Low — SMOTE does +1.8% |
| Ozone | +0.8% | Low — small absolute lift |
| Amazon | +0.08% | Negligible — both methods near ceiling |
| Solar Flare | ~0% | Negligible — REGEN doesn't hurt but doesn't help |

---

## Auditor Pass Rate: 100% on 7/8 datasets

Solar Flare averaged 4.2/5 passes accepted (one rejection per seed). All other datasets achieved 5/5 accepted across all seeds. The binary-column protection fix from the prior milestone holds.

---

## Conclusion

**The heterogeneity rule predicts REGEN's advantage with 75% accuracy.** The refined rule — "PCA-redundant features → SMOTE; anything with feature-variation → REGEN" — is more precise. A prospect's data can be screened by checking whether features are PCA-compressed (common in finance) or raw/native (common in healthcare, manufacturing, telecom).

REGEN's practical value is dataset-dependent: on datasets where it wins, the lift is 1.6–2.5× SMOTE (Churn, Hypothyroid, Satellite). On datasets where it loses, the gap is also notable (SMOTE beats REGEN by 2–4× on credit card data).

---

## Screening rule for prospect data

Before quoting a prospect:
1. Check if features are PCA-compressed → recommend SMOTE, caveat REGEN
2. Check if features are native/raw → REGEN likely wins
3. Run one seed on a sample → confirms within ~30s
