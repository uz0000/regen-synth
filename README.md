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

On 10 datasets across diverse feature types (5 seeds each, multi-pass active-learning):

**REGEN wins 6/10 datasets.** Where it wins, the lift is 1.5–4.5× SMOTE:

| Dataset | REGEN lift | SMOTE lift | Ratio |
|---------|-----------|-----------|-------|
| Satellite | +33.0% | +10.4% | **3.17×** |
| Hypothyroid | +10.9% | +6.1% | **1.79×** |
| Churn | +9.8% | +6.5% | **1.51×** |
| Ozone | +0.7% | +0.3% | **2.27×** |
| Amazon | +0.1% | +0.0% | **4.50×** |
| Solar Flare | ~0% | -8.0% | REGEN doesn't hurt |

**SMOTE wins on PCA-compressed data** (Credit Card Fraud, CreditCard Subset) and on 2 mixed-type datasets (Wilt, Bank Marketing). SMOTE's nearest-neighbor interpolation is competitive when features are redundant or homogeneous.

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
