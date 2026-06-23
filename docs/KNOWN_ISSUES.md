# KNOWN ISSUE: High-Cardinality Categorical TVD Failures

**Status:** Resolved — top-K TVD comparison implemented for high-cardinality columns
**Discovered:** 2026-06-22 breadth benchmark re-run
**Severity:** Was Medium — blocked REGEN on datasets with extreme categorical cardinality (>500 unique values per column)

---

## What Was Fixed (Categorical Decode)

The synthetic batch was coming out with encoded integer codes (3.0, 7.0) instead
of original categorical values ("management", "technician"). The Auditor compared
encoded integers against real strings → TVD=1.0 → every batch rejected.

**Root cause:** The Prior and Amplifier encode categoricals to integers for
numerical computation, but nothing decoded them back before audit.

**Fix:** Added `_decode_categoricals()` in `regen/api.py` that reconstructs the
original category mapping from `pd.Categorical(rare_df[col])` and maps the
synthetic integer codes back to real values. Applied in three places:
- `run_campaign()` main loop
- `screen()` quick-campaign path
- `benchmark/run_breadth.py`

**Result:** Bank Marketing (16 features, 9 categorical) went from 0/5 passes to
5/5 accepted. REGEN now produces +1.12% lift (SMOTE wins at +1.52%, but REGEN
runs and produces valid synthetic data).

---

## What Was Fixed (High-Cardinality TVD)

Open Payments has 5 features, all categorical with extreme cardinality:

| Column | Unique values | TVD (old) | TVD (top-K) | Passes? |
|--------|--------------|-----------|-------------|---------|
| Drug/Biological Name | 1,115 | 0.47 | ~0.14 | Yes |
| Manufacturer Name | 612 | 0.45 | ~0.12 | Yes |
| Device/Medical Supply | 691 | 0.23 | ~0.10 | Yes |
| Physician Specialty | 199 | 0.23 | ~0.08 | Yes |
| Dispute Status | 2 | 0.004 | 0.004 | Yes |

**Root cause:** The Auditor's TVD check compared the full distribution over all
unique categories. With 1,115 unique drug names and only 200 synthetic rows per
pass, the TVD was mathematically guaranteed to be high — most of the 1,115
categories had zero synthetic representation. This was a sampling limitation,
not a data corruption bug.

**Fix:** Implemented top-K TVD comparison in `_tvd_discrete()`:
- When a column has >50 unique values (configurable via `AuditorConfig.high_card_threshold`),
  TVD is computed over only the top-K most frequent categories from the reference
  distribution, with all remaining categories grouped into an "other" bucket.
- K scales with the synthetic batch size: `K = min(n_unique, max(20, n_synthetic // 5))`.
  This ensures ~5 synthetic rows per compared category for stable proportion estimates.
- The "other" bucket captures whether the synthetic data covers the long tail at
  the right rate, without requiring it to represent every individual category.

**Result:** At Open Payments scale (1000 real rows, 200 synthetic), matching
distributions now produce TVD ≈ 0.14 (passes the 0.15 threshold), while
mismatched distributions produce TVD ≈ 0.60 (correctly rejected).

---

## Related Files

- `regen/api.py` — `_decode_categoricals()` (categorical decode fix)
- `engine/auditor/fidelity.py` — `_tvd_discrete()` (top-K TVD), `AuditorConfig.high_card_threshold`
- `tests/test_fidelity.py` — `test_auditor_high_cardinality_tvd_topk`, `test_auditor_high_cardinality_rejects_mismatched`
- `benchmark/run_breadth.py` — verification script
