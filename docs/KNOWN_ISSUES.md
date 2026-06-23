# KNOWN ISSUE: High-Cardinality Categorical TVD Failures

**Status:** Partially resolved — categorical decode fixed (Bank Marketing now works);
Open Payments still blocked by high-cardinality TVD
**Discovered:** 2026-06-22 breadth benchmark re-run
**Severity:** Medium — blocks REGEN on datasets with extreme categorical cardinality (>500 unique values per column)

---

## What Was Fixed

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

## What Still Fails (Open Payments)

Open Payments has 5 features, all categorical with extreme cardinality:

| Column | Unique values | TVD | Passes? |
|--------|--------------|-----|---------|
| Drug/Biological Name | 1,115 | 0.47 | No |
| Manufacturer Name | 612 | 0.45 | No |
| Device/Medical Supply | 691 | 0.23 | No |
| Physician Specialty | 199 | 0.23 | No |
| Dispute Status | 2 | 0.004 | Yes |

The Auditor's TVD threshold for categorical columns is 0.15. With 1,115 unique
drug names and only 200 synthetic rows per pass, the TVD is mathematically
guaranteed to be high — most of the 1,115 categories will have zero synthetic
representation. This is a sampling limitation, not a data corruption bug.

---

## Proposed Fix for High-Cardinality TVD

Two options, not mutually exclusive:

### Option A: Relax TVD for high-cardinality columns
Scale the TVD threshold inversely with cardinality. A column with 1000+ unique
values should not be held to the same TVD standard as one with 5.

```python
# In AuditorConfig or _eval_column:
effective_threshold = config.tvd_threshold * max(1.0, card / 50)
# e.g. card=1000 → threshold becomes 3.0 (effectively passes)
```

### Option B: Compare top-K categories only
Instead of comparing the full distribution, compare only the top-K most frequent
categories. This focuses the fidelity check on the categories that actually matter
and ignores the long tail that no 200-row batch can cover.

---

## Related Files

- `regen/api.py` — `_decode_categoricals()` (the fix, now applied)
- `engine/auditor/fidelity.py` — `_eval_column()`, `_tvd_discrete()` (where TVD is computed)
- `benchmark/run_breadth.py` — verification script
