# How it works — the mechanisms, spelled out

**In plain terms first:** REGEN takes a real dataset that has a rare category
you care about (fraud, default, disease) and makes more synthetic examples of
it, so a detection model has more to learn from. Separately, it makes sure
whatever gets shipped is checked against the real thing rather than just
assumed to be good — a hard gate rejects a batch that doesn't hold up, and
every check is independently re-computable by someone who doesn't trust you.

Everything below is the deep-dive reference to speak from when someone asks
*"but how does it actually decide/measure that?"* — every number and rule is
in the code, with file references so you can pull it up live.

---

## 0. Scope — what REGEN models, and what it does not

The single most important thing to say out loud before anything else. `regen doctor` checks a dataset
against this envelope up front and refuses (or warns) rather than producing a misleading batch.

| ✅ In scope | ❌ Out of scope |
|---|---|
| **Single-table** data (one flat table) | **Relational / multi-table** (foreign keys across tables) |
| **Cross-sectional** rows — *exchangeable*, order carries no signal | **Temporal / time-series** — trends, seasonality, autocorrelation, event sequences |
| **Mixed columns:** continuous + categorical/binary | **Free text** and **images** |
| A **rare minority class** to amplify (≥10 rare rows) | A ~balanced target (nothing to amplify) |
| Numeric-privacy **distance floor** | **Differential privacy** (it is *not* DP) |

**The escape hatch for "temporal" problems:** if time matters, engineer it into **per-row features**
first (`txns_last_24h`, `avg_amount_7d`, `days_since_last_visit`). Once the time dimension lives in
columns instead of row order, each row is exchangeable again and REGEN applies fully — which is how
most production fraud/risk tabular ML already works. REGEN models the *feature-engineered snapshot*,
not the raw sequence.

**One-line scope statement:** *"REGEN is for single-table, cross-sectional tabular data with a rare
class to amplify. Not time-series, not relational, not text/images — and it tells you up front when a
dataset is out of bounds."*

---

## 1. How REGEN chooses the rare-event target

The whole pipeline amplifies **one column's minority class** (the "rare event"). Getting that
column right is load-bearing, so there are three resolution paths, in strict authority order.

### Path A — you name it (authoritative)
`ingest(path, label_col="is_fraud", rare_def={mode:"label", value:1})`. An explicit `label_col`
short-circuits all detection. This always wins. (`engine/ingest/loader.py:243`)

### Path B — rule-based scoring (no model, fully reproducible)
Pass `label_col=""` and every column is scored for fitness as a rare-event target
(`_score_target_columns`, `engine/ingest/loader.py:300`). A column is first **disqualified** unless
it sits in the "useful band":

- cardinality between 2 and **20** (more distinct values ⇒ not a class label),
- at least **10** rare rows (`min_rare_rows`),
- minority class **≤ 35%** of the data (`max_minority_ratio` — a ~balanced column gives nothing to amplify).

Survivors are scored on a weighted structural formula:

```
score = 0.50 · imbalance_score      # 1 − minority_ratio/0.35   → higher when rarer
      + 0.25 · cardinality_score    # 1.0 if binary, else 1/(card−1)  → binary preferred
      + 0.25 · rare_count_score     # min(1, n_rare/50)         → enough rows to learn the tail
      + 0.15 · name_bonus           # +0.15 if the name is one of
                                    #   {label, target, y, is_fraud, fraud, class}
```

The highest score wins. **But if the top two scores are within 0.05 of each other
(`ambiguity_margin`), REGEN refuses to guess** and raises `AmbiguousTargetError` listing the tied
candidates (`loader.py:352`). Refusing is the feature — a confidently-wrong target corrupts
everything downstream.

### Path C — the LLM breaks a tie from your goal (advisory, gated)
This is the one place an LLM is genuinely needed, and it's scoped to exactly that need. When Path B
ties, the model (`resolve_ambiguous_target`, `regen/semantics.py`) is shown a small, redacted
**semantic context** and returns which candidate the goal favors:

- your plain-language **goal**,
- the **tied candidates** — names + tie statistics (rare value, minority ratio, count, score) + up
  to `REGEN_SEMANTICS_SAMPLES` (default 3) **example values** each,
- **`other_columns`** — the *names* of the rest of the schema (no values), as domain context, so the
  model can tell a churn table (`tenure`, `monthly_charges`, …) from a credit table even when the
  target column is opaquely named (`y`, `flag_9`).

It never sees raw rows; example values leave only for the ≤4 tied candidates, are capped, and are
suppressed entirely when `REGEN_SEMANTICS_SAMPLES=0`. It returns which candidate the goal favors.

Why a model earns its place here (and nowhere in the number-making):

- **It reads intent the rules can't.** Two columns tie structurally; only "*I want a churn model*"
  tells you it's `churned`, not `defaulted`.
- **It recognizes names the dictionary misses** — a target called `y`, `outcome`, or `flag_9` earns
  no keyword bonus from the rules, but the model recognizes it from the goal + profile.
- **It maps a fuzzy goal to a continuous-tail target** — "flag the top 1% of amounts" → percentile
  mode on `amount`, which the rules (which only look for low-cardinality classes) never propose.

The guardrails that keep this honest:

- The model's pick **must be one of the tied candidates**, or it's discarded (`semantics.py`).
- **Offline, no key, a bad key, or a declined pick → the honest `AmbiguousTargetError` comes right
  back** and a human chooses. The system never invents a target and never crashes on a missing key.
- The choice, the goal, and the candidate list are recorded in the spec's provenance
  (`target_tiebreak`), so the audit trail says *who* broke the tie and *why*.
- Once the target is fixed, the rare mask, generation, and every metric are 100% deterministic. A
  wrong LLM guess changes *which column is amplified* — it never fabricates a value (Invariant 4).

**In short:** rules pick the target when it's unambiguous — reproducible, no model call. When two
columns tie, a model breaks the tie from your stated goal, but only from a shortlist the rules
already validated. Offline, it raises the honest error instead.

This is the "model decides, engine grounds" boundary made concrete: the model touches exactly one
judgment call — which of two statistically tied columns your goal means — and nothing else. A wrong
model guess changes *which column gets amplified*, never a value.

---

## 2. How the ROC-AUC / TSTR numbers are produced

The headline utility number ("how much of real-data performance does the synthetic set recover?")
comes from a real classifier benchmark, not a hand-wave. (`measure_tstr`, `engine/examiner/surrogate.py`)

### The task is defined, not assumed
- **Target (label):** `y = (row[label_col] == rare_value)` — the same rare-event column chosen in §1,
  binarized to **rare-vs-rest**. `rare_value` defaults to the minority class if not given.
- **Features:** every other column, encoded to a numeric matrix through the shared `field_dict`
  (categoricals → identical codes across synthetic and real; NaNs → 0), so both sides live in the
  exact same feature space.

### Three models, not one (a panel, so the number isn't a fluke of one learner)
| Model | What it is | What it stresses |
|---|---|---|
| **Logistic Regression** | scaled features, `class_weight="balanced"` | linear separability — does the synthetic set preserve linear structure? |
| **Random Forest** | 100 trees, `class_weight="balanced"` | non-linear feature interactions and thresholds |
| **Gradient Boosting** | sequential boosted trees | subtle, high-order structure a single tree misses |

### TSTR vs TRTR, and the "recovered" ratio
For each model, two copies are trained and both graded on the **same held-out real test set**:

- **TSTR** — Train on Synthetic, Test on Real → `M_synth`
- **TRTR** — Train on Real, Test on Real → `M_real` (the ceiling)

Each is scored with **ROC-AUC** (ranking quality across all thresholds) and **PR-AUC / average
precision** (precision–recall on the rare class — the honest metric under heavy imbalance), both on
`predict_proba`.

```
recovered = (model trained on SYNTHETIC) / (model trained on REAL)   # on real held-out data
```

The **median across the three models** is the headline. `1.0` = the synthetic set stands in fully;
a gap is expected (and with a healthy privacy min-distance, is the *price of privacy* — a perfect
1.0 would suggest memorization). Guardrails: fewer than **10** held-out rare rows → status
`insufficient_real_test` (an honest refusal, not a fabricated number); a `recovered > 1.05` is
**flagged**, because synthetic beating real usually means leakage, not magic.

**In short:** the same three detectors are trained twice — once on synthetic, once on real — and both
are graded on real data neither saw. The number is how much of the real detector's ranking and
precision the synthetic-trained one recovers, as a median across the three, with an explicit refusal
when there's too little rare data to trust.

A single accuracy number is easy to fool yourself with; a three-learner panel graded on held-out real
data is what turns "looks good" into "measured, and here's the ceiling." The gap below 1.0 is
expected — paired with a healthy privacy distance, it's the price of not memorizing real records.
This discipline is what caught the early overclaiming here: leakage-inflated wins collapsed to small,
real ones the moment evaluation was done this way.

---

## 3. Other mechanisms worth naming (the same specificity, briefly)

If you want more "here's exactly what happens" material, these are the strongest:

- **Correlation preservation (the copula).** The base batch is drawn through a **mixed-data Gaussian
  copula** so discrete↔continuous correlations survive — not independent per-column sampling. Concrete
  proof point: fixing this dropped the correlation-structure error from **0.331 → 0.101** on the
  transactions set. (`engine/prior/grounded.py`)
- **The privacy floor is a specific measurement, not a promise.** Every released rare row is checked
  to be **≥ δ** away (σ-normalized nearest-neighbor distance via a `scipy` cKDTree) from every real
  rare row, plus a verbatim/k-anonymity guard. State plainly: **it is not differential privacy.**
  (`engine/privacy.py`, `docs/PRIVACY.md`)
- **The Auditor gate is four named statistics.** Coverage radius, **TVD** (categorical marginals),
  **Wasserstein-1** (continuous marginals), and **Pearson correlation-delta** — a batch that breaks
  the real correlation structure is rejected, not shipped. (`engine/auditor/`)
- **Bit-reproducibility.** Same manifest (seed + vetted ScenarioSpec + code version) → identical
  rows, down to a row-hash, with zero model calls on replay. (Invariant 2/7)
- **`regen verify` re-derives the reported statistics.** A stranger can take the self-contained
  bundle and independently recompute the fidelity numbers — the assurance isn't "trust me," it's
  "re-run it." Spelled out just below. (`regen/audit_bundle.py`, `docs/METHODS.md`)

---

## 3b. What "a recomputable standard" concretely means

This is the load-bearing phrase, so here's *exactly* what it is — not a gloss.

**Every batch ships a self-contained bundle** (the run directory) with four things:

| File | What it holds |
|---|---|
| `pass_1_accepted.parquet` | the delivered synthetic data |
| `manifest.json` | seed + config + the vetted ScenarioSpec, a **SHA-256 of every artifact**, and the metric-version IDs |
| `explanation.json` | the statistics REGEN *reported* |
| `reference_aggregates.json` | aggregate stats of the **real** reference each gate was checked against — under a disclosure policy: **no per-row values**, histogram/quantile buckets only for a class with ≥ 10 real rows |

**`regen verify <bundle>` then recomputes from those files alone — it never reads a cached number**,
so it would catch a system that lied. It checks three things, in order:

1. **Integrity** — recompute each artifact's SHA-256; it must match the manifest (catches any edit to
   the data or the reported numbers after the fact).
2. **Metric-version guard** — refuse to compare across metric-definition changes.
3. **Value recomputation, stat-by-stat, each PASS/FAIL within a fixed tolerance** — the correlation
   structure, the per-feature class separation (Fisher), and the delivered class counts, all
   re-derived from the delivered data + the disclosed real aggregates.

Crucially, **it's honest about what it *can't* check.** Statistics that would need the raw real rows
(coverage radius, the privacy nearest-neighbour distance, the downstream lift) are reported as
**UNCHECKABLE** — not silently passed. That honesty *is* the standard.

A real run prints exactly this shape:

```
============================================================
  REGEN — AUDIT VERIFY  (runs/2026-07-09-fraud)
============================================================
  Integrity (artifact hashes vs manifest):
    ✓ pass_1_accepted.parquet
    ✓ explanation.json
    ✓ reference_aggregates.json
  Statistics (recomputed from delivered data + reference aggregates):
    ✓ correlation_delta: reported=0.101 recomputed=0.101
    ✓ fisher_separation: reported=ranked scores recomputed=max abs diff 3.1e-06
    ✓ class_counts: reported=40 recomputed=40
    – coverage_rate: UNCHECKABLE (needs raw real rare rows)
    – privacy_min_distance: UNCHECKABLE (needs raw real rare rows)
    – tail_lift: UNCHECKABLE (needs the full held-out detector protocol)
------------------------------------------------------------
  RESULT: VERIFIED
============================================================
```

In short: every batch ships a certificate, and one command (`regen verify`) re-derives the reported
numbers from the data itself and reports PASS/FAIL per number — without trusting anything the
generator claimed. Where a statistic can't be checked without the private raw data, it says so
rather than pretending.

---

## 4. What is and isn't new here

The contribution isn't a novel algorithm — it's a *standard of evidence*. Three properties, held
together, are uncommon in synthetic-data tooling:

1. **A provable boundary.** The deterministic engine produces every value; a model is allowed at
   exactly one seam (target tie-break) and never touches a number.
2. **Independent recomputability.** Every batch ships a certificate a stranger can recompute from
   scratch (`regen verify`) — assurance by re-execution, not by trust.
3. **Measurement that can catch you being wrong.** Leakage-free splits, a model panel, honest
   refusals, and a privacy floor that treats a *too-perfect* result as a red flag.

Precisely:

- **New mathematics / algorithm?** No. Copulas, Gaussian processes, TSTR, nearest-neighbour distance,
  TVD/Wasserstein are all standard and pre-existing.
- **New engineering contribution?** Modestly. The composition — a deterministic core + a narrow
  audited LLM seam + end-to-end recomputable assurance — is a stance most tools don't take.
- **New empirical findings?** The strongest is the estimand result (see the root `README.md` and
  `KNOWN_ISSUES.md`): every generator tested, including this one, preserves at most 1 of 4 regression
  coefficients on the credit demo while passing every standard fidelity check. Alongside that,
  cleanly-measured re-confirmations of known phenomena: **amplification lift is conditional on
  baseline recall** (a strong baseline leaves ~0 to gain); **leakage-free evaluation collapses
  inflated headlines** (+39% → +4.4%); and **synthetic utility is strongly dataset-dependent** (TSTR
  recovery ranged ~1.0 to ~0.65).

---

*Pair this with `COMPONENT_GUIDE.md` (what technique lives where) and `PRODUCT_SPEC.md` (the
pictures). This file is the "how exactly" layer under both.*
