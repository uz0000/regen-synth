# Privacy in REGEN

REGEN's default output mode (`privacy="floored"`) is a **real, checked,
per-record guarantee against near-copy re-identification**. It is emphatically
**NOT differential privacy**. This document states the threat model, the
mechanism, the exact guarantee, and — just as important — what it does *not*
promise.

## Threat model

The original generator was *grounded sampling*: take a real anchor row and add a
small Gaussian jitter. With realistic noise, every synthetic row is a near-copy
of a real individual — trivially re-identifiable. The threat we close is
**record-level re-identification via near-copies**: an attacker who has (part of)
a real individual's attributes should not find a released row that is
recognizably that person.

## Mechanism (three layers)

1. **Parametric generation (no copying).** Under `floored`, rows are drawn from a
   **mixed-data Gaussian copula** fit to the class distribution — continuous
   columns via empirical-quantile inverse, categorical/binary via inverse-CDF of
   per-class frequency tables — plus the residual-GP tail correction. No real row
   is ever the seed of a synthetic row. (This also fixed the P0-2 correlation
   defect: the copula is joint over *all* features, so discrete↔continuous
   correlation is preserved.)
2. **δ-distance floor (rare part).** Every released **rare** row is pushed to at
   least `δ` (σ-normalised, over continuous features) from **every real rare
   row**, by projection + in-support respawn, enforced as the *final* numeric
   step so nothing downstream can pull a row back inside δ. Default `δ = 0.5σ`.
3. **Verbatim-attribute guard (whole batch).** No released row may reproduce the
   full non-identifier attribute set of a **uniquely-identifying** real record.
   Discrete tuples shared by ≥2 real rows are k-anonymous and allowed (reusing a
   common category combination reveals nothing about one person); a singleton
   match is flagged. Runs against the **full** real set (normal + rare).

Identifier columns are re-minted as fresh unique values and never carried over.

## The exact guarantee

When the summary's `privacy` block reports `passed: true` **and**
`floor_applied: true`: every released rare row is ≥ `δ` (σ-normalised, continuous
features) from every real rare row, and no released row verbatim-duplicates a
uniquely-identifying real record. Both are **measured on the delivered data**
(post-constraint, post-rounding) — the honest check.

## What it does NOT guarantee (read this)

- **It is not differential privacy.** There is no ε/δ-DP bound. It does not bound
  membership-inference or aggregate/reconstruction attacks that do not rely on
  near-copies.
- **The dense normal bulk is not δ-floored.** Real normal rows sit ~0.3σ apart, so
  a 0.5σ shell is geometrically infeasible there and would destroy the marginal.
  The bulk is protected by crowd anonymity + parametric sampling + the verbatim
  guard, not by isolation. This is deliberate.
- **All-categorical / no-continuous data:** the δ-floor cannot apply. The summary
  says so — `floor_applied: false`, `floor_skip_reason: "no_continuous_features"`
  — and protection reduces to parametric sampling + the verbatim guard +
  k-anonymity. `regen doctor` flags this; see `docs/CAPABILITY_MATRIX.md`.

## Reading the `privacy` block

```
"privacy": {
  "mode": "floored",           # or "none"
  "delta": 0.5,                # the σ-floor requested
  "floor_applied": true,       # false + floor_skip_reason when it can't apply
  "min_distance": 0.53,        # nearest released-rare → real-rare distance (≥ δ when passed)
  "n_verbatim_duplicates": 0,  # uniquely-identifying verbatim copies (must be 0)
  "passed": true               # min_distance ≥ δ AND no verbatim duplicate
}
```
The top-level `passed` is **fidelity AND conformance AND privacy** — a batch is
shippable only if all hold.

## The δ ↔ fidelity trade-off (observed)

Larger `δ` pushes rare rows farther from real ones — more privacy headroom, some
fidelity cost. Measured on the bundled demo and the benchmark sweep
(`benchmark/RESULTS_PRIVACY.md`, run `benchmark/run_privacy_sweep.py`):

- **Demo (`examples/transactions.csv`, δ=0.5):** fidelity PASS, nearest real row
  ~0.52–0.53σ ≥ floor, 0 verbatim, detection lift ≈ +0.28 — privacy on, unchanged
  utility.
- **creditcard (δ=0.5):** coverage 0.980 → 0.919, correlation Δ ≈ 0.10–0.12, both
  under gate — small cost.
- **hypothyroid (δ=0.5):** coverage 0.980 → 0.947; correlation often *improves*
  under the copula (e.g. ozone 0.079 → 0.062).
- **Degraded cases** (low-cardinality integer like solar_flare; all-categorical
  high-cardinality like open_payments) are documented in
  `docs/CAPABILITY_MATRIX.md` and flagged by `regen doctor` — prefer
  `privacy="none"` there.

## Auditability vs privacy

The audit bundle (`regen verify`, `docs/METHODS.md`) publishes **aggregate**
reference statistics only — histogram/quantile buckets above a minimum count, no
per-row values, identifiers as counts. Stricter disclosure (a higher
`min_bucket_count`) makes some statistics UNCHECKABLE, and `regen verify` says so
rather than pretending to verify them.
