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

# Run tests
regen test
```

## Demo

```bash
pip install streamlit
streamlit run demo/app.py
```

## Benchmark

On the Kaggle Credit Card Fraud Detection dataset (284,807 rows, 492 fraud):

| Metric | Value |
|--------|-------|
| Best tail lift | **+2.03%** recall |
| Baseline recall | 83.8% |
| Amplified recall | 85.8% |
| Passes accepted | 5/5 |
| Campaign time | ~21s |

## How It Works

```
Scout (R-EPIG) → Prior → Amplifier (ResidualGP) → Auditor → Examiner
     ↑                                                     |
     └───────────── lift signal ────────────────────────────┘
```

1. **Scout** picks the most informative tail region
2. **Prior** generates a base batch anchored on real rare rows
3. **Amplifier** corrects the tail via ResidualGP (faster on residuals than raw outcomes)
4. **Auditor** gates the batch against real data statistics
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
