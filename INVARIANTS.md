# REGEN — build guide

This file is the source of truth for building REGEN. Read it before writing code. If a change
contradicts the **Invariants** section, stop and flag it rather than working around it.

REGEN is a staged system that generates **statistically grounded synthetic data** and
**amplifies rare events** so downstream ML models and LLMs get better at detecting them. It is
built on three research components and orchestrated by the agent runtime from Nous Research.

---

## 1. The one rule that cannot be broken

> **The deterministic engine produces every number. The agents only decide, measure, and narrate.**

No LLM and no agent ever invents a synthetic data value. Agents choose *what* to generate, *where*
to spend the generation budget, *whether* to keep a batch, and *how* to describe results. The actual
values come out of the Prior Engine, the residual GP, and the acquisition math. This is what makes
REGEN output defensible rather than plausible-looking fabrication.

Concretely:

- `engine/` is pure Python. It must not import any LLM client, agent framework, or network library.
  A test enforces this (see `tests/test_boundary.py`).
- the agent runtime / LLM output is metadata, control flow, and narration. It is never written into a synthetic row.
- Every batch is reproducible from its manifest (seed + config + code version). Same manifest → same data.

This is the "fineprint discipline": deterministic core, narration layer on top, full reproducibility.

---

## 2. Research spine

Three papers, three roles. The PDFs live in `docs/papers/`. Describe their ideas in your own words
in code comments; do not paste paper text.

| Component        | Paper                                                        | Role in REGEN |
|------------------|--------------------------------------------------------------|---------------|
| Prior Engine     | RDB-PFN — relational in-context learning via synthetic prior | Statistically grounded base data generator. Relational prior + ICL produces structurally consistent rows (correct FK topology, block-diagonal correlation, temporal structure). |
| Amplifier        | R-Design — active residual learning (R-EPIG, ResidualGP)     | Corrects the *tail*. Models the residual (gap between the prior's tail predictions and truth) because the residual is smoother and far cheaper to learn than regenerating the full distribution. |
| Stylist (opt.)   | Structured semantic control                                  | Only if generating LLM-grounded narrated content (personas, attack-step text). Control vectors + drift penalty keep narration on-distribution. **Deferred — see §7.** |

RDB-PFN reference implementation: `https://github.com/MuLabPKU/RDBPFN`. Wrap it; do not reimplement it.

---

## 3. Architecture

Three layers. The boundary between the control plane and the engine is the rule from §1.

```
Control plane (LLM-driven — decides, targets, measures)
    Orchestrator   runs the active-learning loop, spawns stages, schedules
    Scout          R-EPIG targeting: which rare region most improves the detector?
    Examiner       trains/evaluates the downstream detector, measures rare-event lift

Deterministic engine (pure Python — produces all numbers)
    Prior Engine   RDB-PFN generator
    Amplifier      ResidualGP correction over Scout's target region
    Auditor        fidelity gate (marginal calibration, tail dependence, correlation structure)

Substrate & shared services
    SpacetimeDB    live run state, batch lineage, target log (Rust reducers, Python clients)
    Neo4j          post-hoc analysis / threat-graph ingestion
    the agent runtime memory + skills   procedural memory and crystallized generation recipes
```

### The active-learning loop (one pass)

```
Scout selects rare-event target
        ↓
Prior Engine generates base batch
        ↓
Amplifier corrects the tail
        ↓
Auditor validates fidelity   ── reject → discard / feed signal back
        ↓ (accept)
Examiner measures detection lift
        ↓
[lift signal] → Scout selects next target   (loop)
```

### Agent responsibilities

- **Orchestrator** — sequences the loop, spawns one isolated stages per stage job, writes run
  state to SpacetimeDB, owns scheduling. Computes nothing statistical.
- **Scout** — runs R-EPIG over a candidate pool, reads the last lift signal and the memory of
  explored regions, emits a target (covariate region / event type / tail percentile). Picks the
  question; the engine answers it.
- **Prior Engine** — RDB-PFN. Given a schema and a Scout target, generates a structurally consistent
  base batch. Strong on average-case, weak on the tail (that is the Amplifier's job).
- **Amplifier** — ResidualGP. Corrects and densifies the targeted rare region; does not regenerate
  the whole distribution.
- **Auditor** — hard gate (default). Checks the batch against reference statistics and rejects
  failures. A batch that looks plausible but breaks the real correlation structure is worse than no
  data — this gate exists to stop exactly that.
- **Examiner** — trains/evaluates the downstream detector on accepted data and measures recall/
  precision lift on the tail. That number is Scout's reward signal.
- **Memory + skills** — the agent runtime persists what was explored and which fidelity stats held; successful
  recipes crystallize into reusable skill files so the system improves across runs.

---

## 4. the agent runtime Agent integration (Nous Research)

the agent runtime is the orchestration runtime. REGEN's control plane runs *as* the agent runtime; the deterministic
engine runs *outside* it as Python invoked by the agent runtime stages.

> the agent runtime Agent launched Feb 2026 and is moving fast. Verify exact CLI/API/skill-format details
> against `https://agent-runtime-agent.nousresearch.com/docs` before depending on them — treat the mappings
> below as intent, not a frozen contract.

Mapping:

- **Orchestrator → the main the agent runtime loop.** It drives the cycle and schedules campaigns via the agent runtime's
  natural-language scheduling/cron.
- **Engine stages → the agent runtime stages running Python RPC scripts.** Each stage (Prior, Amplifier,
  Auditor, Examiner, Scout's R-EPIG) is a short-lived, isolated stages that shells into the
  corresponding `engine/` script. This is the agent runtime's "contained stages / zero-context-cost pipeline"
  pattern and it maps cleanly onto "one generation job = one isolated worker."
- **Skills → agentskills.io skill files.** The top-level loop is a skill (`agent-runtime/skills/regen-loop/`).
  Domain recipes that prove out (e.g. fraud-tail amplification for transaction RDBs) crystallize into
  their own portable skill files. This is the self-improvement layer — do not hardcode recipes that
  belong here.
- **Memory split.** Structured run state (batch lineage, target log, fidelity stats) lives in
  SpacetimeDB, the system of record. the agent runtime memory holds the higher-level procedural/"what worked"
  layer that informs Scout and skill creation. Do not duplicate structured state into the agent runtime memory.
- **Model provider** — configure via the agent runtime setup (Nous Portal / OpenRouter / own endpoint). The
  engine never calls a model, so the only model consumers are narration, the optional Stylist, and
  (if promoted) a reasoning-Scout.

### Run the agent runtime constrained

This system has no reason to expose a messaging surface. Run the agent runtime headless:

- Disable the messaging gateways (Telegram/Discord/Slack/WhatsApp/etc.) — REGEN uses CLI + scheduler only.
- Use the local or Docker terminal backend.
- Keep the engine as out-of-agent compute. The agent's trust boundary should be as small as possible.

(The the agent runtime gateway as a single trust boundary across messaging platforms is precisely the kind of
attack surface worth scrutinizing in a security context. Keep it closed unless a feature needs it.)

---

## 5. Repository layout

```
regen/
  INVARIANTS.md                  # this file
  docs/
    papers/                  # the three source PDFs (reference only)
  agent-runtime/                    # the agent runtime Agent integration — the control plane
    skills/
      regen-loop/            # the top-level active-learning loop, as a the agent runtime skill
    stages/               # Python RPC scripts the agent runtime stages invoke
      run_prior.py
      run_amplifier.py
      run_auditor.py
      run_examiner.py
      run_scout.py
    config/
      agent-runtime.toml            # model provider, disabled gateways, terminal backend
  engine/                    # DETERMINISTIC. No LLM, no agent, no network imports.
    prior/                   # RDB-PFN wrapper (Prior Engine)
    amplifier/               # ResidualGP + correction (Rare Event Amplifier)
    scout/                   # R-EPIG acquisition (targeting math only)
    auditor/                 # fidelity statistics + accept/reject
    examiner/                # downstream detector train/eval + lift metric
    manifest.py              # batch manifest: seed, schema hash, configs, code version
  stylist/                   # OPTIONAL semantic control — only if narrating (deferred)
  state/                     # SpacetimeDB
    worlds/                  # Rust reducers: run state, batch lineage, target log
    clients/                 # Python SDK clients (the only SpacetimeDB I/O)
  graph/                     # Neo4j post-hoc analysis ingestion
  contracts/                 # shared schemas/types that cross the boundary
  tests/
    test_boundary.py         # enforces: no forbidden imports inside engine/
    test_fidelity.py         # Auditor catches a deliberately corrupted batch
    test_reproducibility.py  # same manifest → identical data
```

---

## 6. Build order

Each milestone delivers standalone value. Do not start a milestone before the previous one is green.

| # | Milestone        | Deliverable |
|---|------------------|-------------|
| M0 | Engine skeleton + boundary test | Wrap RDB-PFN Prior Engine. Generate a reproducible base batch from a fixed schema. `test_boundary.py` passes. |
| M1 | Auditor          | Fidelity stats + gate. It must reject a deliberately corrupted batch and accept a clean one. |
| M2 | Amplifier        | ResidualGP correction on a hardcoded target region. Measurable density increase in the tail with fidelity preserved. |
| M3 | Examiner         | Train a simple detector; report tail recall/precision lift of amplified vs base data. Produces a single lift number. |
| M4 | Scout (thin)     | R-EPIG picks the next target from a candidate pool using the Examiner signal. One full cycle runs automatically. |
| M5 | agent runtime   | Wrap the loop as a the agent runtime skill; engine stages become Python-RPC stages; persistent memory of explored regions; scheduled unattended runs. |
| M6 | SpacetimeDB state| Move batch lineage / target log into reducers; cross-run observability. |
| M7 | (optional)       | Stylist (semantic control) if narrating; Neo4j ingestion for analysis. |

Build the loop in plain Python first (M0–M4), then put the agent runtime on top (M5). Do not couple engine code
to the agent runtime — the engine must run and be testable with no agent runtime present.

---

## 7. Deferred decisions

These are intentionally unresolved. Do not pick a default silently — surface the choice and ask.

- **Stylist in or out.** The semantic-control layer earns a place only if REGEN produces LLM-grounded
  narrated content (personas, attack-step text). If output is purely tabular/relational, omit it
  entirely. Default assumption until told otherwise: **out**.
- **Scout: thin vs reasoning.** Thin = a wrapper that only runs R-EPIG over a fixed candidate pool
  (fully reproducible). Reasoning = an LLM stages that *proposes novel rare scenarios* outside the
  pool, which are then scored by deterministic R-EPIG. Build **thin** first; promote only after the
  loop closes. Even the reasoning version feeds deterministic scoring — it never sets values.
- **Auditor: hard gate vs soft penalty.** Default is **hard gate** (reject failing batches). The soft
  alternative feeds a fidelity penalty back into Scout's reward so the system learns which regions are
  hard to fake faithfully. Switch only deliberately.

---

## 8. Invariants (enforced by tests / review)

1. `engine/` imports no LLM client, agent framework, or networking library. Verified by `test_boundary.py`.
2. Every synthetic batch carries a manifest (seed, schema hash, prior config, target, amplifier params,
   Auditor stats, code version) and is bit-reproducible from it. Verified by `test_reproducibility.py`.
3. No batch reaches the Examiner or is persisted without passing the Auditor.
4. Agent/LLM output never becomes a synthetic data value — only decisions, metrics, and narration.
5. SpacetimeDB is the system of record for structured run state. the agent runtime memory does not duplicate it.
6. the agent runtime runs with messaging gateways disabled unless a feature explicitly requires one.

---

## 9. Conventions

- Python for the engine and orchestration glue; Rust for SpacetimeDB reducers.
- Type hints and dataclasses for everything crossing `contracts/`.
- Every engine stage is a pure function of `(input batch, config, seed)` where practical.
- Fail loud: a fidelity failure or a missing manifest is an error, not a warning.
- Comments explain *why* (which paper mechanism, which invariant), not *what*.
