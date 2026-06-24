# REGEN — build guide

This file is the source of truth for building REGEN. Read it before writing code. If a change
contradicts the **Invariants** section, stop and flag it rather than working around it.

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

## 2. Research spine

Three papers, three roles. The PDFs live in `docs/papers/`. Describe their ideas in your own words
in code comments; do not paste paper text.

| Component        | Paper                                                        | Role in REGEN |
|------------------|--------------------------------------------------------------|---------------|
| Prior            | Empirical grounded sampling + class-conditional Gaussian density scorer (no learned generative model) | Generates the base batch by *grounded sampling* — real anchor rows + Gaussian noise scaled to the observed spread (continuous features only). Also fits a class-conditional Gaussian scorer `P(normal\|x)` that the Amplifier uses to weight residual relevance. Intentionally strong on the bulk, weak on the tail. Single-table; no relational/FK or temporal structure. |
| Amplifier        | R-Design — active residual learning (R-EPIG, ResidualGP)     | Corrects the *tail*. Models the residual (gap between the prior's tail predictions and truth) because the residual is smoother and far cheaper to learn than regenerating the full distribution. |
| Stylist (opt.)   | Structured semantic control                                  | Only if generating model-grounded narrated content (personas, attack-step text). Control vectors + drift penalty keep narration on-distribution. **Deferred — see §7.** |

An earlier design proposed wrapping RDB-PFN / TabPFN (a Prior-Fitted Network) for
relational schemas. That path was **removed** — REGEN is single-table, and grounded
sampling plus the Amplifier's residual GP cover the need without it. Do not reintroduce
a relational/PFN prior without a concrete relational requirement; if one appears, wrap
(`https://github.com/MuLabPKU/RDBPFN`) rather than reimplement.

---

## 3. Architecture

The system is a deterministic active-learning loop. There is no agent runtime and no separate
"control plane" process — the loop is a single Python function (`regen.api.run_campaign()`).

```
regen.api  (deterministic orchestration — sequences passes, gates batches, reports lift)
    Orchestrator   run_campaign(): runs the active-learning loop, owns pass sequencing
    Scout          R-EPIG targeting: which rare region most improves the detector?
    Examiner       trains/evaluates the downstream detector, measures rare-event lift

Deterministic engine (pure Python — produces all numbers)
    Prior          grounded-sampling base generator + P(normal|x) density scorer
    Amplifier      ResidualGP correction over Scout's target region
    Auditor        fidelity gate (marginal calibration, tail dependence, correlation structure)

Entry points
    cli/           `regen` CLI (run / ingest / screen / results)
    server/        FastAPI server for frontend integration
    examples/      runnable demo scripts
```

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
- **Scout** — runs R-EPIG over a candidate pool, reads the last lift signal and the within-run memory
  of explored regions, emits a target (covariate region / event type / tail percentile). Picks the
  question; the engine answers it.
- **Prior** — grounded-sampling generator. Given a Scout target, generates a base batch by drawing
  real anchor rows and perturbing their continuous features with Gaussian noise scaled to the observed
  spread. Also fits a class-conditional Gaussian scorer `P(normal|x)` consumed by the Amplifier. Strong
  on average-case, weak on the tail (that is the Amplifier's job). Single-table; not a learned or
  relational generative model.
- **Amplifier** — ResidualGP. Corrects and densifies the targeted rare region; does not regenerate
  the whole distribution.
- **Auditor** — hard gate (default). Checks the batch against reference statistics and rejects
  failures. A batch that looks plausible but breaks the real correlation structure is worse than no
  data — this gate exists to stop exactly that.
- **Examiner** — trains/evaluates the downstream detector on accepted data and measures recall/
  precision lift on the tail. That number is Scout's reward signal.
- **Within-run memory** — each pass records its target anchor; Scout biases the next selection away
  from already-explored regions so budget goes to new tail structure. This lives in the engine
  (`engine/scout/repig.py`), threaded through the loop's `explored_points` accumulator.

---

## 4. Optional model integration (deferred)

The loop runs end-to-end with **no model and no agent runtime** today. A model only earns a place
for one of:

- **Reasoning-Scout** — an LLM that *proposes novel rare scenarios* outside the fixed candidate pool,
  which are then scored by deterministic R-EPIG (it never sets values — §1 still holds).
- **Stylist / narration** — model-grounded narrated content (personas, attack-step text).
- **Result narration** — human-language summaries of a campaign.

If/when any of these is wanted, call a plain hosted model directly (e.g. **GLM 5.2**, Claude, or any
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
    amplifier/               # ResidualGP + correction (Rare Event Amplifier)
    scout/                   # R-EPIG acquisition + explored-region penalty (targeting math only)
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
| M2 | Amplifier        | ResidualGP correction on a hardcoded target region. Measurable density increase in the tail with fidelity preserved. | ✅ |
| M3 | Examiner         | Train a simple detector; report tail recall/precision lift of amplified vs base data. Produces a single lift number. | ✅ |
| M4 | Scout (thin)     | R-EPIG picks the next target from a candidate pool using the Examiner signal. One full cycle runs automatically. | ✅ |
| M5 | API + entry points | `regen.api.run_campaign()` unifies the loop; CLI, FastAPI server, and demo wrap it. (The earlier "agent runtime" milestone was removed — the loop is plain Python.) | ✅ |
| M6 | Structured run-state store | (Deferred) Persist batch lineage / target log / fidelity stats to a queryable store for cross-run observability. | ⏳ |
| M7 | (optional)       | (Deferred) Reasoning-Scout or Stylist via a plain model call (§4); Neo4j ingestion for analysis. | ⏳ |

---

## 7. Deferred decisions

These are intentionally unresolved. Do not pick a default silently — surface the choice and ask.

- **Stylist in or out.** The semantic-control layer earns a place only if REGEN produces model-grounded
  narrated content (personas, attack-step text). If output is purely tabular/relational, omit it
  entirely. Default assumption until told otherwise: **out**.
- **Scout: thin vs reasoning.** Thin = a wrapper that only runs R-EPIG over a fixed candidate pool
  (fully reproducible, current state). Reasoning = a model that *proposes novel rare scenarios* outside
  the pool, which are then scored by deterministic R-EPIG. Build **thin** first; promote only after the
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

---

## 9. Conventions

- Python for the engine, API, CLI, and server.
- Type hints and dataclasses for everything crossing `contracts/`.
- Every engine stage is a pure function of `(input batch, config, seed)` where practical.
- Fail loud: a fidelity failure or a missing manifest is an error, not a warning.
- Comments explain *why* (which paper mechanism, which invariant), not *what*.
