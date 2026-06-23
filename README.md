# REGEN — Rare-Event Generation & Noise amplification

**Statistically grounded synthetic data that improves ML model performance on rare events.**

REGEN runs a closed-loop active-learning campaign that generates synthetic rare events
(fraud, intrusions, defects) to improve detection models. Every number comes from a
deterministic statistical engine — no LLM hallucination, no black box.

## Quick Start

```bash
pip install regen-synth

# Run a campaign
regen run my_data.csv --label is_fraud --rare-mode label --rare-value 1

# Inspect a dataset first
regen ingest my_data.csv

# Screen: predict whether REGEN or SMOTE will win on your data
regen screen my_data.csv --label is_fraud --rare-mode label --rare-value 1

# Run tests
regen test
```

## Benchmark

On 11 datasets across diverse feature types (5 seeds each, 5 passes per seed, matched synthetic row budget between REGEN and SMOTE):

**REGEN wins 9/11 datasets (82%).** Where it wins, the lift is 1.02–3.75× SMOTE:

| Dataset | REGEN lift | SMOTE lift | Ratio |
|---------|-----------|-----------|-------|
| Satellite | +39.1% | +10.4% | **3.75×** |
| CreditCard Subset | +42.9% | +31.4% | **1.36×** |
| Hypothyroid | +12.6% | +6.1% | **2.07×** |
| Churn | +10.9% | +6.5% | **1.68×** |
| Credit Card Fraud | +2.2% | +1.8% | **1.23×** |
| Ozone | +1.0% | +0.3% | **3.09×** |
| Wilt | +13.4% | +13.2% | **1.02×** |
| Amazon | +0.1% | +0.0% | **3.50×** |
| Solar Flare | ~0% | -8.0% | REGEN doesn't hurt |

SMOTE edges out REGEN by 0.06% on 2 datasets where both methods produce marginal lift:
- Bank Marketing: REGEN +1.46% vs SMOTE +1.52%
- Open Payments: REGEN ~0% vs SMOTE +0.06% (baseline already at 100% recall)

A noise-scale tuning (0.25 → 0.10) flipped 3 datasets from SMOTE wins to REGEN wins (Credit Card Fraud, CreditCard Subset, Wilt) — REGEN now wins on PCA-compressed data too.

Full results: [`benchmark/RESULTS_BREADTH.md`](benchmark/RESULTS_BREADTH.md)

## How It Works

```
Scout (R-EPIG) → Prior → Amplifier (ResidualGP) → Auditor → Examiner
     ↑                                                     |
     └───────────── lift signal + explored memory ─────────┘
```

1. **Scout** picks the most informative tail region (cross-pass memory avoids re-exploration)
2. **Prior** generates a base batch anchored on real rare rows
3. **Amplifier** corrects the tail via ResidualGP with ARD kernel
4. **Auditor** gates the batch against real data statistics (hard reject on failure)
5. **Examiner** measures the lift and feeds back to Scout

## Documentation

Full system documentation: [`docs/REGEN_DOCUMENTATION.md`](docs/REGEN_DOCUMENTATION.md)

## Dependencies

```bash
# Core (default, air-gapped):
pip install regen-synth

# With PFN backend for relational data:
pip install regen-synth[pfn]

# Everything:
pip install regen-synth[all]
```

## License

MIT
