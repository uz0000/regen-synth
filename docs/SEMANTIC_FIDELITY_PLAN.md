# Semantic Fidelity (L1) — Plan

**Status:** planning (no code yet). Owner sign-off needed on the two open decisions in §7 before the
model-backed milestone (M2).

**Goal.** Give REGEN a *semantic* understanding of each column so it recreates the starting dataset
more faithfully — **valid per-column values**: correct type, units, bounds, and proper handling of
identifier columns. The understanding is advisory **metadata**; it never produces a synthetic value.

**Scope: L1 only.** Two further levels were considered and **explicitly descoped** as not needed:
distribution-family transforms (L2) and cross-column logical constraints (L3). This plan does not
build them; if a future dataset demands them, revisit.

---

## 1. Hard constraints (must not break)

Existing invariants (INVARIANTS.md §1, §4, §8). The plan is designed around them:

1. **No model output becomes a data value.** The model emits column *meaning* and *constraints*
   (metadata). The deterministic engine produces every number and then *enforces* the constraints.
2. **`engine/` imports no LLM/network library.** The model call lives **outside** `engine/`
   (`regen/semantics.py`). Enforcement lives in `engine/` as pure Python over a plain spec.
3. **Reproducible.** The raw semantic proposal is persisted in the manifest and replayed on re-run;
   no fresh model call is needed to reproduce a batch.
4. **Optional & offline-safe.** No key / no network → fall back to deterministic structural
   inference. Never crash, never block.
5. **Cost-bounded.** One model call per dataset, cached by schema hash. Never in a loop.

---

## 1b. Semantic Component Rules (Charter)

Enforced in code (a `_vet_constraints` gate), not just documented. Every rule has a runtime check; a
proposal that violates one is dropped + logged, never applied.

**Authority & scope**
1. **Advisory only.** The model proposes; the engine enforces; the user can override. It never sets a
   data value.
2. **Closed scope.** Only a fixed allowlist of constraint kinds is honored: dtype, integer, semantic
   min/max, unit/rounding, categorical value-set, identifier role. Anything else is ignored. No
   free-form code or expressions.

**Data is ground truth (the load-bearing rule)**
3. **A constraint may never contradict observed data.** Proposed bounds must *contain* the observed
   range; "integer" must match observed integrality; a category set must be a superset of observed
   categories. If the model says "Amount ≥ 0" but real rows are negative, the claim is **rejected**
   and logged. The data always wins.
4. **Tighten toward validity only.** Constraints may enforce known-safe limits (e.g. a currency floor
   of 0) but may never invent values the data never exhibited.

**Validation & fallback**
5. **Schema-validated output.** Malformed proposal → one retry → structural fallback.
6. **Per-field confidence.** Below threshold → drop that field, use structural inference. Never
   all-or-nothing.
7. **Offline-safe.** No key / network error / timeout → structural-only path. Generation never blocks.

**Reproducibility & cost**
8. **Recorded input.** Raw proposal + prompt + model id persisted in the manifest; re-runs replay it.
9. **One cached call** per dataset (by schema hash). Never inside a loop.

**Transparency & boundary**
10. **Nothing silent.** Every accepted / rejected / overridden constraint is recorded and surfaced
    (API + UI) with its rationale.
11. **Boundary intact.** Model call outside `engine/`; `engine/` enforces from a plain spec.
    `test_boundary.py` stays green.

Worst case under these rules: a wrong proposal is caught by rule 3/6 or degrades fidelity slightly —
it can never inject fabricated data, contradict the source, or crash the pipeline.

---

## 2. Architecture

```
engine/ingest/profile.py  NEW. Deterministic schema profile (no model): per-column dtype,
                          cardinality, sample values, observed bounds, integrality, naive role
                          guess. This is what the model is *shown*.

contracts/types.py        NEW dataclasses: ColumnSemantics, SemanticProfile, ColumnConstraint —
                          the shared spec the model fills and the engine reads.

engine/constraints.py     NEW. Pure enforcement: clamp to bounds, round integers, snap booleans /
                          categoricals, format units, handle identifiers. Generalizes the current
                          _apply_domain_constraints into a real constraint layer.

regen/semantics.py        NEW. Outside engine/. One model call → fills the SemanticProfile from the
                          schema profile. Runs _vet_constraints (the §1b charter). Cached by schema
                          hash. Offline → returns structural-only profile.
```

Data flow: `ingest → schema profile → (semantics: model or structural) → vetted SemanticProfile →
generate → enforce constraints → audit`.

---

## 3. The semantic contract (what the model returns, L1)

A single structured object, validated on receipt. L1 fields only:

```python
ColumnSemantics:
  name: str
  role: "feature" | "identifier" | "timestamp" | "free_text" | "target"
  dtype: "integer" | "float" | "categorical" | "boolean" | "datetime"
  unit: str | None        # "currency", "ratio[0,1]", "count", "percentage", ...
  min/max: float | None   # semantic bounds (must contain observed range — rule 3)
  notes: str
  confidence: float       # per-column; low → structural fallback (rule 6)
```

The model is shown the deterministic profile and asked to fill this in. Each proposal passes
`_vet_constraints` before it can affect generation.

---

## 4. Milestones (value-ordered, each shippable)

| # | Deliverable | Needs model? | Risk |
|---|-------------|--------------|------|
| **M1** | Deterministic schema profile + `ColumnConstraint` spec + `engine/constraints.py` enforcing **structural** constraints. Refactor today's clamp/round/snap into it. **No model.** | no | low |
| **M2** | Model-advised type / unit / bounds + **identifier handling** (IDs regenerated unique or passed through, not Gaussian noise), gated by `_vet_constraints` (§1b). | yes | low–med |
| **M3** | Advisory wiring: API + UI show the SemanticProfile and let the user edit/confirm; manifest persistence; offline fallback; cost cache; full tests. | — | med |

**M1 stands on its own** even if M2/M3 are never built — it turns today's ad-hoc constraint code into
a tested layer and fixes structural cases immediately. Build M1 first regardless of §7.

---

## 5. Enforcement mechanics (L1)

- **Bounds / type / unit:** clamp to vetted semantic min/max, round integers, snap booleans and
  categoricals to real values, format units (e.g. currency to 2 dp). Generalizes current behavior.
- **Identifiers:** never run through the prior. Regenerate as fresh unique values, or pass through a
  sampled real value if referential integrity matters. (Fixes the order_id / user_id noise problem.)

---

## 6. Reproducibility, offline, cost

- Persist the vetted `SemanticProfile` + prompt + model id in the manifest. Re-runs replay it.
- `REGEN_SEMANTICS=off` / no key → structural-only profile; same code path, no network.
- One call per dataset, cached by `schema_hash`. Cost surfaced in logs.

---

## 7. Open decisions (need owner sign-off — INVARIANTS.md §7)

1. **Provider** — GLM 5.2 (guide's example), Claude, or any OpenAI-compatible endpoint?
2. **Authority** — *advisory* (recommends + explains; user confirms) vs *authoritative* (auto-applies).
   Recommendation: **advisory**.

(Scope is decided: L1 only. M1 needs neither decision; M2 needs both.)

---

## 8. Risks

- First network/LLM dependency in the product path → must stay optional + offline-safe (mitigated §6).
- Model mis-labels a column → bounded blast radius: enforced deterministically, vetted against data
  (rule 3), low-confidence fields fall back (rule 6), and every constraint is shown for review.

---

## 9. First step

Build **M1** now — pure Python, no model, no new dependency. It improves fidelity immediately, fixes
structural cases, and is the foundation M2 plugs into. Hold M2/M3 until the §7 decisions are signed
off. (Likely **post-funding** — M2 adds a paid LLM dependency; see Appendix A.)

---

## Appendix A — M2+ expectations (deferred, likely post-funding)

What turning on the model-backed milestones actually commits us to. Recorded now so it's a
deliberate decision later, not a surprise.

**M2 — model-advised layer (the real commitment):**
- *Requires from owner:* the two §7 decisions (provider, authority) **and** an API key/endpoint.
  This is the first time REGEN talks to a network.
- *Build:* `regen/semantics.py` — assemble the schema profile, one cached model call, parse +
  schema-validate into a `SemanticProfile`, run `_vet_constraints` (the §1b charter), persist the
  proposal to the manifest, offline fallback. Identifier handling lands here (IDs regenerated
  unique / passed through, not Gaussian noise).
- *Signing up for:* a new dependency, added latency on first generate (~seconds, then cached),
  small per-dataset cost, ongoing prompt upkeep, and a determinism caveat (replay proposal from
  manifest). INVARIANTS.md §4 gets updated to document the now-active layer; `test_boundary.py` must
  stay green (call lives outside `engine/`).
- *Extra deliverable:* an **evaluation** on a few datasets proving semantics measurably beats
  structural-only — if it doesn't help, that's a finding. Model is **mocked in CI**.

**M3 — advisory wiring (UX + transparency):**
- API response carries the `SemanticProfile` + accepted/rejected/overridden constraints + rationale.
- UI review panel: per-column inferred meaning, editable before generate; clear applied-vs-rejected.
- `REGEN_SEMANTICS` on/off toggle + key config; off (or no key) is the safe default.
- Mostly front-end + plumbing; no new external risk beyond M2.

**Bottom line:** M1 is free of all this. M2 is a deliberate "yes" to running an LLM in the product
path (cost + key + upkeep), with risk bounded by the charter. M3 makes it transparent and overridable.
