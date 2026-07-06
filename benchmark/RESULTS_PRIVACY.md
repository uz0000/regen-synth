# Privacy sweep — floored vs none

**Run date:** 2026-07-06  |  **Code version:** `c2ec51a`  |  **Command:** `python benchmark/run_privacy_sweep.py`

Fixed config: n_rows=400, seed=7, auto-tune OFF (so floored-vs-none isolates the privacy cost). Rare class auto-detected. Privacy is NOT differential privacy (near-copy re-identification floor + verbatim guard).

Per dataset, `none` (grounded sampling) vs `floored` (parametric + δ-floor + verbatim guard). ΔX = floored − none.

| Dataset | mode | fid | cov | corrΔ | gate | lift | floor | minDist | verbatim | t(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| creditcard_subset.csv | none | 1.0 | 0.9565 | 0.1129 | PASS | 0.0 | — | — | — | 3.11 |
| creditcard_subset.csv | floored | 1.0 | 0.913 | 0.1415 | PASS | 0.0 | True | 2.1684 | 0 | 1.93 |
| satellite.csv | none | 1.0 | 1.0 | 0.0897 | PASS | 0.0 | — | — | — | 1.31 |
| satellite.csv | floored | 1.0 | 1.0 | 0.0649 | PASS | 0.0 | True | 1.2625 | 0 | 1.55 |
| hypothyroid.csv | none | 1.0 | 0.9801 | 0.0516 | PASS | 0.0652 | — | — | — | 1.62 |
| hypothyroid.csv | floored | 1.0 | 0.947 | 0.1026 | PASS | 0.0652 | True | 1.4398 | 0 | 1.44 |
| wilt.csv | none | 1.0 | 0.9962 | 0.0774 | PASS | 0.0127 | — | — | — | 2.01 |
| wilt.csv | floored | 1.0 | 0.9962 | 0.0517 | PASS | 0.0127 | True | 0.5 | 0 | 2.31 |
| ozone.csv | none | 1.0 | 0.9437 | 0.079 | PASS | 0.125 | — | — | — | 2.18 |
| ozone.csv | floored | 1.0 | 0.9313 | 0.0622 | PASS | 0.125 | True | 4.0855 | 0 | 3.36 |
| bank_marketing.csv | none | 0.9375 | 0.979 | 0.0877 | FAIL | — | — | — | — | 1.5 |
| bank_marketing.csv | floored | 0.9375 | 0.9743 | 0.0718 | FAIL | — | True | 0.6782 | 0 | 1.46 |
| churn.csv | none | 0.95 | 0.8289 | 0.0854 | FAIL | — | — | — | — | 1.0 |
| churn.csv | floored | 0.95 | 0.802 | 0.0796 | FAIL | — | True | 1.3401 | 0 | 1.14 |
| solar_flare.csv | none | 1.0 | 1.0 | 0.071 | PASS | 0.0182 | — | — | — | 1.57 |
| solar_flare.csv | floored | 0.7 | 0.0385 | 0.21 | FAIL | — | True | 1.6332 | 31 | 1.25 |
| open_payments.csv | none | 0.8 | 1.0 | — | FAIL | — | — | — | — | 1.54 |
| open_payments.csv | floored | 0.4 | 1.0 | — | FAIL | — | False | inf | 11 | 1.78 |
| amazon.csv | none | 1.0 | 0.9763 | 0.0671 | PASS | -0.007 | — | — | — | 3.74 |
| amazon.csv | floored | 1.0 | 0.9462 | 0.1282 | PASS | -0.007 | True | 0.5001 | 0 | 3.91 |
| creditcard.csv | none | 1.0 | 0.9797 | 0.1039 | PASS | 0.0 | — | — | — | 11.19 |
| creditcard.csv | floored | 1.0 | 0.9187 | 0.1181 | PASS | 0.0 | True | 1.9378 | 0 | 13.67 |

## Reading this

- **gate** is the Auditor fidelity verdict (coverage + per-column + correlation). A floored row that flips PASS→FAIL is a privacy cost worth flagging.
- **floor** = whether the δ-distance floor was enforced. `False` on all-categorical datasets (no continuous features) is expected and honest (P2-9); the verbatim guard still applies.
- **minDist** ≥ delta (0.5) when the floor is applied and passes; `inf` when no continuous features exist.
- **lift** `—` means the batch failed the gate (no lift measured) or the held-out rare fold was too small for a lift estimate.
