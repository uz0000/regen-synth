# KNOWN ISSUE: Categorical Feature Corruption in Amplifier

**Status:** Open — blocks REGEN on categorical-heavy datasets
**Discovered:** 2026-06-22 breadth benchmark re-run
**Severity:** High — excludes an entire class of real-world datasets

---

## The Problem

REGEN currently cannot amplify datasets where the majority of features are
categorical. On such datasets the Auditor rejects 100% of batches (0/5 passes),
producing zero synthetic data and zero lift.

This was discovered on two new benchmark datasets:

| Dataset | Features | Categorical | Passes accepted | Result |
|---------|----------|-------------|-----------------|--------|
| Bank Marketing | 16 | 9 (56%) | 0/5 | Total failure |
| Open Payments | 5 | 5 (100%) | 0/5 | Total failure |

Both datasets have correct label distributions and ingest cleanly. The failure
is in the generation pipeline, not ingestion.

---

## Root Cause

The Amplifier's ResidualGP applies continuous residual corrections to ALL
columns, including categorical ones that were label-encoded to integers during
feature encoding.

Pipeline trace:

```
1. Ingest: categorical strings → integer codes (0, 1, 2, ...)
2. Prior: generate_base_batch() correctly skips perturbing categorical
   columns (the _is_continuous mask in rdbpfn.py line 292-294)
3. Amplifier: sample_residuals() adds GP residual corrections to ALL
   columns — NO continuous/categorical mask applied
4. Result: category code 2 becomes 2.7 or 1.3 (invalid intermediate values)
5. Auditor: TVD between real categorical distribution and corrupted
   synthetic = 1.0 on every categorical column → hard reject
```

The Prior Engine already solved this problem (see `engine/prior/rdbpfn.py`,
`generate_base_batch`, lines 288-295):

```python
# Only perturb continuous features. Binary/categorical features keep
# their anchor values — perturbing a binary column (e.g. on_thyroxine)
# produces meaningless intermediate values.
continuous = prior._is_continuous
noise = np.zeros_like(X_base)
noise[:, continuous] = rng.standard_normal(...) * anchor_std[continuous] * 0.25
X_base += noise
```

The Amplifier needs the same treatment: residual corrections should only apply
to continuous features. Categorical features should keep their Prior-generated
values (which are already grounded in real rare rows).

---

## Why This Matters for Adoption

This is not a niche edge case. Categorical-heavy data is common in the exact
domains REGEN targets:

- **Healthcare** — ICD codes, treatment flags, patient demographics
- **Manufacturing** — fault codes, machine IDs, operator shifts
- **Security** — protocol types, service categories, alert classifications
- **Finance** — transaction types, merchant categories, geographic flags

A prospect in any of these verticals is likely to have mixed-type data. If
REGEN silently produces 0% accepted passes, the prospect's first experience
is a total failure — worse than if it had never run.

The datasets where REGEN currently wins are predominantly numeric/continuous
(Satellite, Ozone, Churn, Hypothyroid's continuous subset). Expanding to
categorical data doubles the addressable market.

---

## Proposed Fix

Apply the same `_is_continuous` mask from the Prior Engine to the Amplifier's
residual sampling. Concretely, in `engine/amplifier/residual_gp.py` (or
wherever `sample_residuals` is implemented):

1. Accept the `is_continuous` mask (already stored on `PriorModel`)
2. After sampling residuals, zero out corrections on non-continuous columns
3. Return the masked residual array

Pseudocode:

```python
def sample_residuals(residual_model, X_base, rng):
    X_res = gp_sample(...)           # existing GP residual sampling
    X_res[:, ~is_continuous] = 0.0   # preserve categorical values
    return mu, std, X_res
```

The `is_continuous` mask is already computed during `fit_prior()` and stored
on `PriorModel._is_continuous`. It just needs to flow through to the Amplifier.

**Estimated effort:** Small — the mask exists, the pattern exists in the Prior,
the fix is applying the same mask in one more place. Likely 1-2 hours including
testing.

---

## Verification After Fix

Re-run the breadth benchmark on Bank Marketing and Open Payments. Success
criteria:
- Auditor pass rate > 0 (at least some batches accepted)
- REGEN produces a non-zero lift number
- If REGEN beats SMOTE, the categorical gap is closed
- If SMOTE still wins, that's an honest result — but at least REGEN ran

---

## Related Files

- `engine/amplifier/residual_gp.py` — fix location (sample_residuals)
- `engine/prior/rdbpfn.py` — reference implementation (generate_base_batch, line 288)
- `benchmark/run_breadth.py` — verification script
- `benchmark/RESULTS_BREADTH.md` — current results (0/5 on both datasets)
