# Semantic Fidelity — Plan

**Status:** planning (no code yet). Owner sign-off needed on the two open decisions in §7 before M2.

**Goal.** Give REGEN a *semantic* understanding of the input schema so it recreates the
starting dataset more faithfully — valid per-column values, realistic shapes, and consistent
relationships between columns. The understanding is advisory **metadata**; it never produces a
synthetic value.

---

## 1. Hard constraints (must not break)

These are the existing invariants (INVARIANTS.md §1, §4, §8). The plan is designed around them:

1. **No model output becomes a data value.** The model emits column *meanings* and *constraints*
   (decisions/metadata). The deterministic engine still produces every number and then *enforces*
   the constraints. (Invariants 1 & 4.)
2. **`engine/` imports no LLM/network library.** The model call lives in a new module
   **outside** `engine/` (`regen/semantics.py`). Enforcement logic lives in `engine/` but is pure
   Python operating on a plain constraint spec. (`test_boundary.py` stays green.)
3. **Reproducible.** A model call is not bit-reproducible across providers/time, so the raw
   semantic proposal is persisted in the manifest and treated as a *recorded input*. Re-running
   from a manifest replays the saved proposal instead of calling the model. (Invariant 2, §4 rule.)
4. **Optional & offline-safe.** No API key / no network (cron, CI, air-gapped) → the system falls
   back to the deterministic structural inference we already have. Never crash, never block. (§5.)
5. **Cost-bounded.** One model call per dataset, cached by schema hash. Never inside a loop. (§4.)

---

## 2. Architecture

```
regen/semantics.py        NEW. Outside engine/. One model call. Builds a SemanticProfile
                          from the deterministic schema profile. Cached by schema hash.
                          Offline → returns the structural-only profile.

engine/ingest/profile.py  NEW. Deterministic schema profile (no model): per-column dtype,
                          cardinality, sample values, imbalance, naive role guess, observed
                          bounds. Reusable; this is what the model is *shown*.

contracts/types.py        NEW dataclasses: ColumnSemantics, SemanticProfile, ColumnConstraint,
                          CrossColumnConstraint. The shared spec the model fills and the engine
                          reads.

engine/constraints.py     NEW. Pure enforcement: given a generated batch + constraint spec,
                          clamp/round/snap/transform/derive/repair-or-reject. Extends the
                          current _apply_domain_constraints into a full constraint layer.

engine/prior, amplifier   Transforms (L2) applied here: fit on transformed space, invert on output.
```

Data flow: `ingest → schema profile → (semantics: model or structural) → SemanticProfile →
generate → enforce constraints → audit`. The Auditor gains a constraint-violation check.

---

## 3. The semantic contract (what the model returns)

A single structured object, validated on receipt (reject + retry on malformed). Sketch:

```python
ColumnSemantics:
  name: str
  role: "feature" | "identifier" | "timestamp" | "free_text" | "target"
  dtype: "integer" | "float" | "categorical" | "boolean" | "datetime"
  unit: str | None              # "currency", "ratio[0,1]", "count", "percentage", ...
  min/max: float | None         # semantic bounds (not just observed)
  distribution_hint: "normal" | "lognormal" | "count" | "bounded01" | None   # L2
  notes: str

CrossColumnConstraint:                                                       # L3
  kind: "equation" | "inequality" | "implication"
  expr: str                     # e.g. "total == price * quantity", "end >= start"
  columns: [str]
```

The model is **shown** the deterministic profile (names, dtypes, cardinality, sample rows, stats)
and **asked** to fill this in + flag anything it's unsure about. Low confidence → that field is
dropped and we fall back to structural inference for it. Everything is enforced deterministically,
so a wrong proposal degrades fidelity but can never inject a fabricated value.

---

## 4. Milestones (value-ordered, each shippable)

| # | Deliverable | Level | Risk |
|---|-------------|-------|------|
| **M1** | Deterministic schema profile + `ColumnConstraint` spec + `engine/constraints.py` enforcing **hand/structural** constraints. No model yet. Refactor current clamp/round/snap into it. | infra | low |
| **M2** | **L1**: model-advised type / unit / bounds + **identifier handling** (IDs regenerated as unique or passed through, not Gaussian noise). Enforced via M1 layer. | L1 | low–med |
| **M3** | **L2**: distribution-family transforms (log / logit) — fit prior on transformed space, invert on output. Model advises the family; structural skew test is the fallback. | L2 | med |
| **M4** | **L3**: cross-column constraints. Derived columns computed (not generated); inequalities/implications enforced by repair-or-reject; new Auditor constraint gate. | L3 | med–high |
| **M5** | Advisory wiring: API + UI show the SemanticProfile and let the user edit/confirm; manifest persistence; offline fallback path; cost cache; full test pass. | glue | med |

**M1 is valuable on its own** even if we never call a model — it turns today's ad-hoc constraint
code into a real, testable layer and is the foundation everything else plugs into. Recommend
building M1 first regardless of the §7 decisions.

---

## 5. Enforcement mechanics (the hard part = L3)

- **L1 bounds/type/unit:** clamp to semantic min/max, round integers, snap booleans/categoricals,
  format units (e.g. round currency to 2 dp). Extends current behavior.
- **Identifiers:** never run through the prior; regenerate as fresh unique values (or pass through
  a sampled real value if referential integrity matters).
- **L2 transforms:** `x → log1p(x)` (or logit) before fit; inverse after generation + before
  constraint enforcement. Round-trip must be exact for untouched values (tested).
- **L3:**
  - *equation* (`total = price*qty`): compute the derived column from its inputs — don't generate it.
  - *inequality* (`end >= start`): repair (sort/clip the pair) or reject-and-regenerate the row.
  - *implication* (`city ⇒ state`): map via a lookup learned from the real data.
  - A new Auditor gate reports residual violations; a batch over threshold is rejected (Invariant 3).

---

## 6. Reproducibility, offline, cost

- Persist the raw `SemanticProfile` (and the prompt + model id) in the manifest. Re-runs replay it.
- `REGEN_SEMANTICS=off` (or no key) → structural-only profile; identical code path, no network.
- One call per dataset, cached by `schema_hash`. Surfaced cost in logs.

---

## 7. Open decisions (need owner sign-off — INVARIANTS.md §7)

1. **Provider** — GLM 5.2 (guide's example), Claude, or any OpenAI-compatible endpoint?
2. **Authority** — *advisory* (recommends + explains; user confirms in API/UI) vs *authoritative*
   (auto-applies). Recommendation: **advisory**, for human-in-the-loop + graceful offline.

(Scope is decided: all three levels L1–L3 are in.)

---

## 8. Risks

- First network/LLM dependency in the product path → must stay optional + offline-safe (mitigated §6).
- L3 enforcement can over-constrain and shrink diversity → make cross-column constraints opt-in per
  dataset and report how many rows were repaired/rejected (no silent truncation).
- Model mis-labels a column → bounded blast radius (enforced deterministically; low-confidence
  fields fall back to structural). Always show the proposal for review (advisory mode).

---

## 9. Is all three needed? — Verdict

**No.** Value is not evenly distributed across the levels:

- **L1 is the must-have (~70% of the value).** It's what makes output pass the sniff test:
  no negative amounts, no fractional counts, no garbage in ID columns. Without L1 the data looks
  fake regardless of statistics. Notably, *most* of L1 is achievable with **no model at all**
  (we already infer integer-ness and observed bounds); the model only adds units, true semantic
  bounds, and identifier detection. Cheapest level, highest floor.
- **L2 is a real but conditional bump.** It matters specifically for **skewed / heavy-tailed**
  columns — which are common in REGEN's target domains (fraud amounts, latencies). Worth doing,
  but the system delivers value without it, and a structural skew test covers the common case.
- **L3 is highest-ceiling, highest-cost, and dataset-dependent.** It only earns its keep when
  inter-column logic *is* the credibility bar (financial records, date ranges, hierarchies). For
  many flat tabular sets it's optional, and it's the one most able to over-constrain and shrink
  diversity. Treat it as opt-in per dataset, not a default.

**Recommendation:** ship **M1 + L1** first and evaluate on real data. Add **L2** only if you see
skew problems; add **L3** only for datasets whose value depends on cross-column rules. Building all
three up front is over-engineering until a real dataset demands L2/L3.

## 10. Suggested first step

Build **M1** (deterministic constraint layer + schema profile) now — pure Python, no model, no new
dependency. It improves fidelity immediately, de-risks everything above it, and is worth doing even
if L2/L3 are never built. Hold M2+ until the §7 decisions are signed off.
