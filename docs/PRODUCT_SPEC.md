# REGEN — System Layout (how the components work together, and why)

**Purpose of this doc:** a build reference that lays out how REGEN's parts connect,
what each one hands to the next, and *why* each exists (what failure it prevents /
what question it answers). It reflects the target version agreed after the 2026-07
review. Component status is tagged **[BUILT]** (exists + tested), **[PARTIAL]**,
**[PLANNED]**. It extends `INVARIANTS.md` (invariants); it does not override it.

---

## 1. What the system is (so the layout has a purpose)

A **scarce-data certified synthetic-surrogate generator**. Given a limited-but-
sufficient real sample, it produces a synthetic dataset **plus a machine-checkable
certificate of how much it preserves and how far it falls short**, so a decision
made on the surrogate is defensible to someone who doesn't trust you. The asset is
the *verifiable certificate + honest diagnosis* — a **known, bounded gap** — not
the rows and not a guaranteed accuracy lift.

## 2. The core mental model: two planes and one spine

Everything below is organized by a single idea. There are **two planes**, and they
are deliberately kept apart:

- **The engine plane — produces every value.** Pure, deterministic, no LLM/network.
  It turns a configuration into grounded numbers and measures them.
- **The assurance/control plane — configures, certifies, and informs. Produces no
  values.** LLM proposals, the vetting gate, explanation, verification, and the
  decision-support surface live here.

They connect through **one spine: the `ScenarioSpec`.** The control plane *writes*
it, the engine plane *reads* it, the manifest *persists* it. That is why the system
can let context (an LLM, a researcher) shape *what* is generated while guaranteeing
it never touches *how* the numbers are made — and why any run collapses to a
deterministic, replayable object.

```
   ┌──────────────────────── ASSURANCE / CONTROL PLANE (no values) ───────────────────────┐
   │  Human intent + real sample                                                            │
   │        │                                                                               │
   │        ▼                                                                               │
   │  Preflight (doctor) ── refuse if out of envelope ───────────────► stop, tell the user  │
   │        │ (in envelope)                                                                 │
   │        ▼                                                                               │
   │  LLM proposer ──draft──►  Vetting gate  ◄──researcher declaration / structural profile │
   │  (metadata only)          (3 sources, fixed rules; a proposal that contradicts data    │
   │                            is dropped + logged)                                        │
   │                                   │                                                    │
   │                                   ▼                                                    │
   │                       ┌───────  ScenarioSpec  ───────┐   ◄── THE SPINE (single source  │
   │                       │  columns · intent · gates ·  │        of truth; persisted in   │
   │                       │  provenance · verdicts       │        the manifest → replay)   │
   └───────────────────────┼──────────────────────────────┼────────────────────────────────┘
                           │  parameterizes (never edits) │
   ┌───────────────────────┼── ENGINE PLANE (produces every value) ─┼──────────────────────┐
   │                        ▼                                                                │
   │  Prior (copula) ─► Scout target ─► Amplifier (TailCorrector) ─► constraints ─► δ-floor     │
   │  grounds in real    where to      structure-aware tail       valid         near-copy    │
   │  marginals+corr     densify       correction                 support        protection  │
   │                        │                                                                │
   │                        ▼   delivered rows                                               │
   │  Auditor: fidelity gate  +  conformance gate      (measured on DELIVERED data)          │
   │                        │  (pass → shippable)                                            │
   │                        ▼                                                                │
   │  Examiner: lift  ·  TSTR harness: surrogate quality  ·  Privacy: measured guarantee     │
   └───────────────────────┼──────────────────────────────────────────────────────────────┘
                           │  report objects
   ┌───────────────────────┼── back up to the ASSURANCE PLANE ───────────────────────────┐
   │                        ▼                                                              │
   │  Explanation (computed) ─► Audit bundle ─► `regen verify` (a skeptic recomputes it)   │
   │                        │                                                              │
   │                        ▼                                                              │
   │  Decision-support: surface tradeoffs + diagnosis + recommend  ──►  HUMAN DECIDES      │
   └───────────────────────────────────────────────────────────────────────────────────┘
```

Two walls make the model work:
- **The LLM/agent never crosses into the engine plane** (never writes a value). It
  only shapes the ScenarioSpec and reads report objects.
- **The engine never makes the value-laden decision** (privacy vs fidelity vs
  utility). It measures; the human, informed by the decision-support surface,
  chooses.

## 3. Component reference — role · consumes · produces · why it's there

### Engine plane (produces values; pure/deterministic)

| Component | Consumes → Produces | **Why it exists** |
|---|---|---|
| **Prior — grounded + mixed-data Gaussian copula** [BUILT] | ScenarioSpec + real sample → a base batch | Grounds every synthetic value in the real marginals **and** cross-column correlations, so the surrogate can preserve structure (what TSTR needs) without copying a real row. |
| **Scout — targeting** [BUILT] | prior + explored memory → a target region | Directs where to densify the rare tail. *Rationale caveat: its incremental value is unproven — treat as optional, not a headline.* |
| **Amplifier — TailCorrector (ARD)** [BUILT] | base batch + target → corrected rare rows | Structure-aware tail correction. Beats SMOTE's linear interpolation on heterogeneous-feature data (stays on-manifold); ties it on simple data. This is the learner that earns its keep. |
| **Constraint layer** [BUILT] | raw synthetic values → in-support values | Folds impossible outputs back onto reality (no negative amount, no fractional count). Never invents values the data never showed. |
| **Privacy floor + verbatim guard** [BUILT] | delivered rare rows → floored rows | Pushes every released rare row ≥ δ from every real rare row and blocks verbatim copies → prevents near-copy re-identification. The resulting gap is the *price of privacy*, not a defect. **NOT differential privacy.** |
| **Auditor — fidelity gate** [BUILT] | delivered batch vs real → pass/fail + stats | Stops a batch whose structure is broken from shipping. Measured on the **delivered** (post-floor) data, so the verdict describes what you actually get. |
| **Auditor — conformance gate** [BUILT] | delivered batch vs ScenarioSpec → pass/fail | The delivered data must obey the *declared* contract (bounds, types, categories, id-uniqueness). Fails the batch like a fidelity failure. |
| **Examiner — lift** [BUILT] | real + synthetic → recall lift (leakage-free) | Measures the *augmentation* benefit honestly. Real only when baseline recall is low; **conditional, not the headline.** |
| **TSTR harness** [BUILT] | train-on-synth, test-on-real → recovered % | The **headline actionable metric**: does the surrogate stand in for real data, and by how much? `measure_tstr` (model panel, ROC-AUC+PR-AUC, multi-seed) + leakage-free `evaluate_surrogate`. Turns "trust me" into "here's the number," paired with the privacy min-distance to catch memorization. |

### Assurance / control plane (configures + certifies + informs; no values)

| Component | Consumes → Produces | **Why it exists** |
|---|---|---|
| **ScenarioSpec** [BUILT] | the whole use case → one typed object | The **spine**: single source of truth for a run; persisted in the manifest so a batch replays bit-for-bit *including its use-case context*, with zero model calls. The unit a user saves/shares/re-runs. |
| **Structural profiler** [BUILT] | real sample → deterministic column profile | Source 1 of the contract: dtype/cardinality/bounds/roles inferred from data, always available, no model needed. |
| **Vetting gate** [BUILT] | structural + researcher + (optional) model proposals → vetted ScenarioSpec + verdicts | Lets *context* parameterize the math under fixed rules (authority: researcher > structural > model; a proposal that contradicts the data is dropped + logged). This is how the system extracts situation without ever violating the invariants. |
| **LLM proposer (`regen/semantics.py`)** [BUILT] | profile + plain-language goal → a *draft* ScenarioSpec (intent + gates + columns, metadata only) | Lowers the expertise barrier. `propose_scenario` / `regen.api.draft_scenario` / CLI `regen propose`: one cached call, all fields validated (invalid ones ignored, never obeyed), **shown to the user to edit** — never auto-committed. Offline → a valid structural draft. |
| **Explanation (`explanation.json`)** [BUILT] | report objects → a computed account | Legibility: every gate's statistic + threshold + verdict, per-column provenance/mechanism, feature informativeness — all *computed*, cited to versioned metric IDs, never narrated by a model. |
| **Audit bundle + `regen verify`** [BUILT] | the run dir → independent recomputation | The moat: a third party who doesn't trust you recomputes every reported statistic from the bundle (integrity-hashed) and it passes or names the discrepancy. Turns claims into checkable facts. |
| **Preflight (`regen doctor`)** [BUILT] | dataset → envelope verdicts | Refuses out-of-envelope shapes *before* a run (time-series, too-few-rare, all-categorical caveats), so failures are named up front, not discovered after. |
| **Decision-support surface** [PLANNED] | report objects + tradeoff sweep → options for the human | Removes toil and surfaces the honest frontier (e.g. δ vs TSTR vs privacy) + plain-language diagnosis, and **recommends with override**. It informs the decision; it never makes the value-laden choice. |

## 4. The trust chain (how a value becomes a certified surrogate)

Read top-to-bottom, this is why the output is defensible — each step is
deterministic and later independently recomputable:

1. **Grounded** — the Prior copula draws it from the real marginals + correlations (not a copy).
2. **Corrected** — the TailCorrector shapes the rare tail on-manifold.
3. **Constrained** — folded onto valid support.
4. **Floored** — pushed ≥ δ from every real rare row (privacy), producing the honest gap.
5. **Gated** — fidelity + conformance checked on the *delivered* data; fails loudly if broken.
6. **Measured** — TSTR (stands-in?), lift (augments?), privacy (min-distance) computed.
7. **Explained** — every number written to `explanation.json`, cited to a versioned metric.
8. **Bundled + verifiable** — hashed into an audit bundle a skeptic recomputes with `regen verify`.

The ScenarioSpec threads through all of it; the manifest persists it; replay reproduces it with zero model calls.

## 5. Principles that keep the two planes honest

1. **Engine produces every value; LLM/agent only decides, describes, orchestrates** existing calculations. Never a model-written cell value.
2. **LLM informs; human decides; engine grounds.** No silent commit of a value-laden tradeoff — tools recommend with override.
3. **A measured gap is a price; an unmeasured gap is a landmine.** Ship consequences measured + recomputable; never present synthetic as real.
4. **Refuse loudly at the edges.** Out-of-envelope or guarantee-can't-hold → say so, never a silent pass.
5. **Reproducible from the spec, zero model calls on replay.**
6. **Honest metrics only** — nothing `regen verify` can't recompute; never a max-over-noise headline (Goodhart).

## 6. Scope & non-goals (stated out loud)

**In scope:** single flat table; continuous + categorical/binary features; rare-event framing; exchangeable rows.
**Out of scope / honest limits:** not differential privacy; not time-series/relational/free-text/images; **scarce ≠ absent** (a minimum viable sample exists — below it, refuse); **correlation, not causation** (a surrogate may validate a pipeline's *engineering*, never serve as evidence for a causal effect).

## 7. Build order (what to add, in what sequence)

1. **TSTR harness** — the metric everything references. **[BUILT]** (`measure_tstr` + `evaluate_surrogate`; `tests/test_tstr.py`)
2. **Intent → ScenarioSpec proposer** — a non-expert drafts a run from a plain-language goal. **[BUILT]** (`propose_scenario`/`draft_scenario`; CLI `regen propose`; `tests/test_scenario_proposal.py`)
3. **Decision-support surface** — frontier + diagnosis + recommend-with-override. [PLANNED]
4. **Certified-surrogate demo** — two-party clean-room (producer emits package; a party who never saw the real data trains + `verify`s). [PLANNED]
5. **Closed-loop repair** — *only if* step 2's single shot underperforms, and only with uncertainty-aware metrics + human-approved final spec + the boundary intact. The prior "agent runtime" was removed for failing this bar; a new one must clear it. [DEFERRED]

## 8. Guardrails every new piece must pass

- [ ] No LLM/agent output becomes a data value; engine produces every number.
- [ ] Agent/LLM code lives outside `engine/`; `test_boundary.py` stays green.
- [ ] Any model decision resolves to a persisted ScenarioSpec; replay = zero calls, bit-identical.
- [ ] Value-laden tradeoffs surfaced with consequences, not auto-selected.
- [ ] No reported number `regen verify` can't recompute; no max-over-noise headline.
- [ ] Out-of-envelope / guarantee-can't-hold → loud refusal.
- [ ] New gate/metric/mechanism adds its `explanation.json` + docs entry in the same change.
