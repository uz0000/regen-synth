# REGEN — design rules and invariants

The constraints this system was built under. §8 lists the invariants, each
enforced by a test.

REGEN is a **deterministic pipeline** that generates **statistically grounded synthetic data** and
**amplifies rare events** so downstream ML models get better at detecting them. It is built on three
research components. Every value comes out of the deterministic engine; there is no agent runtime and
no LLM in the loop today. An optional model-driven layer (narration, reasoning-Scout) is deferred —
see §4 and §7.

---

## 1. The one rule that cannot be broken

> **The deterministic engine produces every number. Nothing an LLM/model emits ever becomes a data value.**

No model ever invents a synthetic data value. If/when a model is added (§4), its output is metadata,
control flow, or narration only. The actual values come out of the Prior, the residual GP, and
the acquisition math. This is what makes REGEN output defensible rather than plausible-looking
fabrication.

Concretely:

- `engine/` is pure Python. It must not import any LLM client, agent framework, or network library.
  A test enforces this (see `tests/test_boundary.py`).
- Any model output (when a model is used at all) is metadata, control flow, and narration. It is never
  written into a synthetic row.
- Every batch is reproducible from its manifest (seed + config + code version). Same manifest → same data.

This is the "fineprint discipline": deterministic core, narration layer on top, full reproducibility.

---

## 2. Methods & prior art

REGEN **composes standard, well-understood techniques** into a verifiable pipeline.
The individual math is not novel and is named as such; the original work is the
composition + the assurance layer (contract, gates, explanation, `regen verify`).
When you describe a method in code comments, name the standard technique — don't
imply it was invented here, and don't paste external paper text.

| Component | Method (standard) | Role in REGEN |
|-----------|-------------------|---------------|
| Prior     | empirical **grounded sampling** + a **mixed-data Gaussian copula** + a class-conditional **Gaussian Naive Bayes** density scorer | Generates the base batch grounded in the real marginals + correlations (never copying a real row). Also scores `P(normal\|x)` for the Amplifier. Strong on the bulk, weak on the tail. Single-table; no relational/FK or temporal structure. |
| Amplifier | **Gaussian-process regression** with an **ARD kernel** (the `TailCorrector`, via GPy) | Corrects the *tail* by modeling the residual (gap between the prior's tail prediction and truth) — smoother and cheaper to learn than regenerating the whole distribution. |
| Scout     | **active-learning acquisition** (information-gain–style targeting score) | Picks which rare region to synthesize next; scores a candidate pool + biases away from explored regions. (Incremental value unproven — optional, not a headline.) |
| Auditor   | **TVD**, **Wasserstein-1**, **Pearson correlation**, coverage radius | Fidelity gate on the delivered batch. |
| Examiner  | **RandomForest** lift (leakage-free) + **TSTR/TRTR** with **ROC-AUC/PR-AUC** | Detection lift (conditional) and surrogate quality (headline). |
| Privacy   | σ-normalized **nearest-neighbour distance** floor (scipy `cKDTree`) + verbatim/**k-anonymity** guard | Near-copy re-identification floor. **NOT differential privacy.** |

Fuller mapping (method ↔ file ↔ why) and how to speak about it:
`docs/COMPONENT_GUIDE.md`. The PDFs in `docs/papers/` are **reference/inspiration
only** — REGEN implements standard techniques informed by them, not verbatim
reimplementations. An earlier attempt wrapped a relational Prior-Fitted Network
(RDB-PFN / TabPFN); it was **removed** (single-table only) — do not reintroduce a
relational/PFN prior without a concrete relational requirement.

---

## 3. Architecture

The system is a deterministic active-learning loop. There is no agent runtime and no separate
"control plane" process — the loop is a single Python function (`regen.api.run_campaign()`).

```
regen.api  (deterministic orchestration — sequences passes, gates batches, reports lift)
    Orchestrator   run_campaign(): runs the active-learning loop, owns pass sequencing
    Scout          targeting: which rare region most improves the detector?
    Examiner       trains/evaluates the downstream detector, measures rare-event lift

Deterministic engine (pure Python — produces all numbers)
    Prior          grounded-sampling base generator + P(normal|x) density scorer
    Amplifier      TailCorrector — corrects the tail over Scout's target region
    Auditor        fidelity gate (marginal calibration, tail dependence, correlation structure)

Entry points
    cli/           `regen` CLI (generate / run / ingest / screen / doctor / verify)
    server/        FastAPI server for frontend integration
    examples/      runnable demo scripts + example ScenarioSpec YAML
```

### API-layer contract & assurance (regen/, above the engine)

The engine implements the math once; a **use case is a validated configuration of
that math, never a fork**. These deterministic layers (all outside `engine/`)
carry the use-case context and the assurances around a batch:

- **ScenarioSpec** (`contracts/scenario.py`) — one typed object carrying the whole
  use case (per-column semantics + intent + gates + provenance). Persisted in the
  manifest; the unit a researcher saves/shares/re-runs. `generate()` /
  `run_campaign()` / `screen()` all accept one (loose params still work).
- **Vetting gate** (`regen/vetting.py`) — merges structural inference (Source 1) +
  researcher declaration (Source 2) + an optional cached model proposal (Source 3,
  `regen/semantics.py`, outside `engine/`) under fixed rules (authority
  researcher > structural > model; a proposal that contradicts the data is dropped
  and logged). Metadata only — never a value.
- **Privacy** (`engine/privacy.py`) — parametric generation + δ-distance floor +
  verbatim guard (Invariant 6; `docs/PRIVACY.md`).
- **Conformance** (`engine/auditor/conformance.py`) — the Auditor also gates the
  delivered batch against the vetted contract.
- **Explainability** (`regen/explain.py`) — every batch ships `explanation.json`
  from computed numbers only (`docs/EXPLAINABILITY.md`).
- **Auditability** (`regen/audit_bundle.py`, `regen/metrics.py`) — `regen verify`
  recomputes every statistic from a self-contained bundle (`docs/METHODS.md`).
- **Estimand preservation** (`regen/estimand.py`) — an optional `EstimandSpec` on
  the ScenarioSpec declares an analysis (`outcome ~ predictors`, family ols|logit)
  whose *estimate* the synthetic data must preserve. REGEN fits θ_real on the real
  reference and θ_synth on the delivered batch and certifies each coefficient is
  recovered (two-sample Wald consistency test at `ci_level`). Distinct from fidelity
  (marginals/correlations) and TSTR (prediction): a batch can pass both while a
  coefficient shifts. θ_real ± SE is a disclosed aggregate, so `regen verify` refits
  θ_synth from the delivered rows and re-certifies (`estimand_delta` metric) — never
  from a cached verdict. Deterministic, no LLM, no new dependency (numpy + scipy):
  a coefficient is a metric, never a value (Invariant 1). Undeclared / unfittable →
  an honest `not_declared` / `uncertifiable` status, never a faked pass.
- **Preflight** (`regen/preflight.py`) — `regen doctor` checks a dataset against
  the supported envelope before generation (`docs/CAPABILITY_MATRIX.md`).

### The active-learning loop (one pass)

```
Scout selects rare-event target
        ↓
Prior generates base batch
        ↓
Amplifier corrects the tail
        ↓
Auditor validates fidelity   ── reject → discard / feed signal back
        ↓ (accept)
Examiner measures detection lift
        ↓
[lift signal] → Scout selects next target   (loop)
```

### Stage responsibilities

- **Orchestrator** (`regen.api.run_campaign`) — sequences the loop, gates batches on the Auditor,
  tracks the best lift across passes, owns output. Computes nothing statistical beyond aggregation.
- **Scout** — runs Scout targeting over a candidate pool, reads the last lift signal and the within-run memory
  of explored regions, emits a target (covariate region / event type / tail percentile). Picks the
  question; the engine answers it.
- **Prior** — grounded-sampling generator. Given a Scout target, generates a base batch by drawing
  real anchor rows and perturbing their continuous features with Gaussian noise scaled to the observed
  spread. Also fits a class-conditional Gaussian scorer `P(normal|x)` consumed by the Amplifier. Strong
  on average-case, weak on the tail (that is the Amplifier's job). Single-table; not a learned or
  relational generative model.
- **Amplifier** — TailCorrector. Corrects and densifies the targeted rare region; does not regenerate
  the whole distribution.
- **Auditor** — hard gate (default). Checks the batch against reference statistics and rejects
  failures. A batch that looks plausible but breaks the real correlation structure is worse than no
  data — this gate exists to stop exactly that.
- **Examiner** — trains/evaluates the downstream detector on accepted data and measures recall/
  precision lift on the tail. That number is Scout's reward signal.
- **Within-run memory** — each pass records its target anchor; Scout biases the next selection away
  from already-explored regions so budget goes to new tail structure. This lives in the engine
  (`engine/scout/targeting.py`), threaded through the loop's `explored_points` accumulator.

---

## 4. Optional model integration (deferred)

The loop runs end-to-end with **no model and no agent runtime** today. A model only earns a place
for one of:

- **Reasoning-Scout** — an LLM that *proposes novel rare scenarios* outside the fixed candidate pool,
  which are then scored by deterministic Scout targeting (it never sets values — §1 still holds).
- **Stylist / narration** — model-grounded narrated content (personas, attack-step text).
- **Result narration** — human-language summaries of a campaign.

If/when any of these is wanted, call a plain hosted model directly (any
OpenAI-compatible endpoint) from a thin module **outside `engine/`**. Do **not** introduce an agent
runtime (the agent runtime or otherwise): REGEN previously carried a agent skill layer and removed it — it was a
no-op wrapper around `regen.api` that added token cost and zero functionality. A single model call per
decision is all that's required.

Rules if a model is added:

- The call site lives outside `engine/` (the boundary test still passes).
- Model output is metadata/proposals/narration only — never a synthetic data value (Invariant 4).
- Every statistical decision that depends on a model proposal must still be reproducible: persist the
  raw proposal text in the manifest so a re-run can replay it deterministically.
- Token cost is a first-class constraint (this is why the agent runtime was dropped). Default to the cheapest
  model that does the job; cache proposals; never call a model inside a tight loop.

---

## 5. Repository layout

```
regen-synth/
  INVARIANTS.md                  # this file
  README.md
  docs/
    papers/                  # the three source PDFs (reference only)
    REGEN.md                 # architecture overview
    REGEN_DOCUMENTATION.md   # full API + stage reference
  engine/                    # DETERMINISTIC. No LLM, no agent, no network imports.
    prior/                   # grounded-sampling generator + P(normal|x) density scorer
    amplifier/               # TailCorrector + correction (Rare Event Amplifier)
    scout/                   # Scout targeting acquisition + explored-region penalty (targeting math only)
    auditor/                 # fidelity statistics + accept/reject
    examiner/                # downstream detector train/eval + lift metric
    ingest/                  # load/clean/split normal vs rare + on-disk persistence
    manifest.py              # batch manifest: seed, schema hash, configs, code version
  contracts/                 # shared dataclasses/enums crossing the engine↔API boundary
  regen/                     # unified API layer (run_campaign, screen, get_results, load_synthetic)
  cli/                       # `regen` CLI entry point
  server/                    # FastAPI server for frontend integration
  examples/                  # sample data + runnable demo
  benchmark/                 # breadth/multipass benchmark runners + results
  tests/
    test_boundary.py         # enforces: no forbidden imports inside engine/
    test_fidelity.py         # Auditor catches a deliberately corrupted batch
    test_reproducibility.py  # same manifest → identical data
    test_api.py              # API + regen/api.py boundary check
    test_memory.py           # Scout within-run explored-region penalty
```

Deferred / not yet built (see §7): a structured run-state store (was "SpacetimeDB"), Neo4j
post-hoc analysis, the Stylist. None are required for the loop to run.

---

## 6. Build order

Each milestone delivers standalone value. M0–M4 are complete and green.

| # | Milestone        | Deliverable | Status |
|---|------------------|-------------|--------|
| M0 | Engine skeleton + boundary test | Grounded-sampling Prior. Generate a reproducible base batch from a fixed schema. `test_boundary.py` passes. | ✅ |
| M1 | Auditor          | Fidelity stats + gate. It must reject a deliberately corrupted batch and accept a clean one. | ✅ |
| M2 | Amplifier        | TailCorrector tail correction on a hardcoded target region. Measurable density increase in the tail with fidelity preserved. | ✅ |
| M3 | Examiner         | Train a simple detector; report tail recall/precision lift of amplified vs base data. Produces a single lift number. | ✅ |
| M4 | Scout (thin)     | Scout targeting picks the next target from a candidate pool using the Examiner signal. One full cycle runs automatically. | ✅ |
| M5 | API + entry points | `regen.api.run_campaign()` unifies the loop; CLI, FastAPI server, and demo wrap it. (The earlier "agent runtime" milestone was removed — the loop is plain Python.) | ✅ |
| M6 | Structured run-state store | (Deferred) Persist batch lineage / target log / fidelity stats to a queryable store for cross-run observability. | ⏳ |
| M7 | (optional)       | (Deferred) Reasoning-Scout or Stylist via a plain model call (§4); Neo4j ingestion for analysis. | ⏳ |
| M8 | Estimand preservation | Declare a regression estimand; certify θ_synth recovers θ_real and recompute it in `regen verify`. Regression-coefficient v1 done (`regen/estimand.py`); power-aware certification + categorical predictors + ATE are v2 (`docs/KNOWN_ISSUES.md`). | ✅ |

---

## 7. Deferred decisions

These are intentionally unresolved. Do not pick a default silently — surface the choice and ask.

- **Stylist in or out.** The semantic-control layer earns a place only if REGEN produces model-grounded
  narrated content (personas, attack-step text). If output is purely tabular/relational, omit it
  entirely. Default assumption until told otherwise: **out**.
- **Scout: thin vs reasoning.** Thin = a wrapper that only runs Scout targeting over a fixed candidate pool
  (fully reproducible, current state). Reasoning = a model that *proposes novel rare scenarios* outside
  the pool, which are then scored by deterministic Scout targeting. Build **thin** first; promote only after the
  loop closes. Even the reasoning version feeds deterministic scoring — it never sets values, and it
  would use a plain model call (§4), not an agent runtime.
- **Auditor: hard gate vs soft penalty.** Default is **hard gate** (reject failing batches). The soft
  alternative feeds a fidelity penalty back into Scout's reward so the system learns which regions are
  hard to fake faithfully. Switch only deliberately.
- **Structured run-state store (M6).** Not built. Today run state lives in tempdirs + a campaign
  summary JSON. If cross-run observability becomes a requirement, pick a store deliberately — do not
  default to one silently.

---

## 8. Invariants (enforced by tests / review)

1. `engine/` imports no LLM client, agent framework, or networking library. Verified by `test_boundary.py`.
2. Every synthetic batch carries a manifest (seed, schema hash, prior config, target, amplifier params,
   Auditor stats, code version) and is bit-reproducible from it. Verified by `test_reproducibility.py`.
3. No batch reaches the Examiner or is persisted without passing the Auditor.
4. Model/LLM output never becomes a synthetic data value — only decisions, metrics, and narration.
5. The active-learning loop is a single deterministic Python function call (`regen.api.run_campaign`),
   runnable and testable with no runtime, no model, and no network.
6. **Privacy (`privacy="floored"`).** When the report says `passed`, every released *rare* row is
   ≥ δ (σ-normalised, continuous features) from every real rare row, and no released row
   verbatim-duplicates a *uniquely-identifying* real record. It is **NOT** differential privacy
   (see `docs/PRIVACY.md`). When the floor cannot apply (no continuous features / no label), the
   report says so (`floor_applied: false` + reason) — never silently. A batch's shippable verdict
   is **fidelity AND conformance AND privacy**.
7. **Contract reproducibility.** Every batch's manifest carries the *vetted `ScenarioSpec`* it was
   generated under; the batch reproduces bit-for-bit from that spec **including its use-case context,
   with zero model calls** (Invariant 2 extended). Every reported statistic is independently
   recomputable from the audit bundle (`regen verify`); no engine statistical routine branches on a
   *use case* — use cases exist only as vetted ScenarioSpec parameters. Model proposals (the optional
   Source 3) are metadata only, vetted by deterministic code before they can affect generation.

---

## 9. Conventions

- Python for the engine, API, CLI, and server.
- Type hints and dataclasses for everything crossing `contracts/`.
- Every engine stage is a pure function of `(input batch, config, seed)` where practical.
- Fail loud: a fidelity failure or a missing manifest is an error, not a warning.
- Comments explain *why* (which paper mechanism, which invariant), not *what*.
