# TSTR sweep — how well does a REGEN surrogate stand in for real data?

**Run:** 2026-07-09 · `585ae51` · privacy=none, auto-tune off, 3 seeds, 30% held-out real test.

`recovered = (model trained on synthetic) / (model trained on real)`, both scored on the **held-out real** test set. 1.0 = the surrogate stands in fully; the gap is expected (and, with a healthy privacy min-distance, is the price of privacy — a perfect match would signal memorization).

| Dataset | status | held-out rare | recovered ROC-AUC | recovered PR-AUC |
|---|---|---|---|---|
| creditcard_subset.csv | insufficient_real_test | 7 | None | None |
| satellite.csv | ok | 23 | 1.0209 | 1.0124 |
| hypothyroid.csv | ok | 45 | 0.9879 | 0.9024 |
| wilt.csv | ok | 78 | 0.9975 | 0.9785 |
| ozone.csv | ok | 48 | 0.974 | 0.9972 |
| churn.csv | ok | 212 | 0.6462 | 0.3888 |
| creditcard.csv | ok | 148 | 1.0339 | 0.981 |

Notes: `insufficient_real_test` = too few held-out rare rows for a trustworthy estimate (not a failure — an honest refusal). Numbers are single-config, indicative; re-run this script to reproduce.
