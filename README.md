# REGEN — Rare-Event Generation & Noise amplification

**Statistically grounded synthetic data for rare-event problems — every batch ships a certificate a third party can independently re-check.**

REGEN runs a deterministic active-learning campaign that generates synthetic rare events
(fraud, intrusions, defects) to help detection models when real rare examples are scarce.
Every number comes from a deterministic statistical engine — no LLM hallucination, no black
box. It is single-table, cross-sectional tabular only (not time-series, relational, text, or
images), and it reports honestly where it helps and where it doesn't.

## Quick Start

```bash
pip install regen-synth

# Generate a synthetic dataset (primary path) — privacy on by default
regen generate my_data.csv --label is_fraud

# Preflight: does this dataset fit the supported envelope?
regen doctor my_data.csv --label is_fraud

# Independently verify a produced batch (recompute its reported stats)
regen verify regen-output/

# Run a multi-pass campaign
regen run my_data.csv --label is_fraud --rare-mode label --rare-value 1

# Inspect a dataset / screen REGEN vs SMOTE
regen ingest my_data.csv
regen screen my_data.csv --label is_fraud --rare-mode label --rare-value 1

# Run tests
regen test
```

Every generated batch ships a **ScenarioSpec** (the use case, in the manifest),
an **`explanation.json`** (per-gate stats, feature informativeness, provenance —
computed, never narrated), and an audit bundle you can re-check with `regen
verify`. Output privacy is **on by default** (`--privacy floored`): parametric
generation + a δ-distance floor + a verbatim guard. This prevents near-copy
re-identification but is **not differential privacy** — see
[`docs/PRIVACY.md`](docs/PRIVACY.md).

## What the numbers actually say

Two things are measured, **leakage-free** (train/test split before any generation), and reported honestly — including where REGEN doesn't help.

### 1. Surrogate quality (TSTR) — the headline

Train a detector panel (Logistic Regression / Random Forest / Gradient Boosting) on the **synthetic** data, then grade it on a **held-out real** test set the models never saw. `recovered = (trained on synthetic) / (trained on real)`, reported as the median across the panel — ROC-AUC and PR-AUC. `1.0` = the surrogate stands in fully; a gap below 1.0 is expected, and with a healthy privacy min-distance it's the price of *not* memorizing real records (a perfect 1.0 would be a red flag).

| Dataset | held-out rare | recovered ROC-AUC | recovered PR-AUC |
|---|---|---|---|
| satellite | 23 | 1.02 | 1.01 |
| creditcard (full) | 148 | 1.03 | 0.98 |
| wilt | 78 | 1.00 | 0.98 |
| ozone | 48 | 0.97 | 1.00 |
| hypothyroid | 45 | 0.99 | 0.90 |
| churn | 212 | 0.65 | 0.39 |
| creditcard_subset | 7 | _insufficient_ | _insufficient_ |

Read this honestly: on most sets the synthetic surrogate recovers ~0.97–1.03 of real performance; **churn is a genuine weak spot** (~0.65 ROC / 0.39 PR — the synthetic features don't carry churn's class signal well); and `creditcard_subset` has too few held-out rare rows to estimate, so REGEN **refuses to report a number** rather than fake one. Single-config, indicative — re-run to reproduce. Provenance + full table: [`benchmark/RESULTS_TSTR.md`](benchmark/RESULTS_TSTR.md).

### 2. Detection lift is conditional — not a guaranteed win

REGEN's rare-event amplification improves a detector **only when the baseline is genuinely starved of rare examples**. When the baseline detector is already strong, the lift is ≈ 0 — amplification can't add signal that recall already captures. Earlier headline "wins" (e.g. Satellite *+39%*) were inflated by **evaluation leakage**; under leakage-free measurement they shrink to small, real numbers (Satellite → ~+4%). So the honest claim is *"conditional lift where rare data is scarce and the baseline is weak,"* not a blanket improvement. Use `regen screen` to check whether a given dataset is a REGEN-vs-SMOTE win before committing.

_(The older lift/SMOTE sweep in [`benchmark/RESULTS_BREADTH.md`](benchmark/RESULTS_BREADTH.md) predates the leakage-free harness; treat the TSTR table above as the current measure.)_

## How It Works

```
Scout (targeting) → Prior → Amplifier (TailCorrector) → Auditor → Examiner
     ↑                                                     |
     └───────────── lift signal + explored memory ─────────┘
```

1. **Scout** picks the most informative tail region (cross-pass memory avoids re-exploration)
2. **Prior** generates a base batch anchored on real rare rows
3. **Amplifier** corrects the tail via TailCorrector with ARD kernel
4. **Auditor** gates the batch against real data statistics (hard reject on failure)
5. **Examiner** measures the lift and feeds back to Scout

## Documentation

- Full system reference: [`docs/REGEN_DOCUMENTATION.md`](docs/REGEN_DOCUMENTATION.md)
- Privacy (what's guaranteed and what isn't): [`docs/PRIVACY.md`](docs/PRIVACY.md)
- Explainability (`explanation.json` fields): [`docs/EXPLAINABILITY.md`](docs/EXPLAINABILITY.md)
- Statistical methods + verification: [`docs/METHODS.md`](docs/METHODS.md)
- Supported / degraded / unsupported data shapes: [`docs/CAPABILITY_MATRIX.md`](docs/CAPABILITY_MATRIX.md)
- Build log (every change, before/after observed): [`docs/BUILDLOG.md`](docs/BUILDLOG.md)

## Dependencies

```bash
# Core (air-gapped, CPU-only):
pip install regen-synth

# Everything:
pip install regen-synth[all]
```

## License

MIT
