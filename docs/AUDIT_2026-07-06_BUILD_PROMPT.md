# REGEN — System Audit & Build Prompt (2026-07-06)

**Audience:** a build agent (LLM) completing this system. This document is self-contained: it
states what REGEN is, what was verified working on 2026-07-06, every defect and gap found (with
exact reproduction commands and observed numbers), the ordered build queue, and the documentation
protocol every change must follow. Read `INVARIANTS.md` at the repo root before writing any code —
it is the source of truth for invariants; nothing below overrides it.

**Repo:** `~/Desktop/regen-synth`. Python 3.11 (anaconda). Run everything from the repo root.
**Scope exclusion:** the front-end (server/static) is explicitly out of scope.

**The north-star question this build must answer:** *what is missing for REGEN to be a reliable
synthetic-data generation system, ready to be customized to any (single-table) use case?*

The product frame, set by the owner (2026-07-06): REGEN is a **research tool that supplies
researchers with useful, realistic data on demand**. "Useful" means (a) the data's features are
actually worth observing — the system tells the researcher which features carry signal — and
(b) rare-event amplification is available when the researcher chooses it. The design philosophy is:
**the engine implements the mathematics once, correctly and use-case-agnostically; a use case is a
validated configuration of that mathematics, never a fork of it.** Concretely, the goals are:

1. Synthetic-data generation **fidelity**, with hard checks (Part I fixes what's broken).
2. Reliable, optional **rare-event amplification** with honest utility measurement.
3. A **use-case customization mechanism** — the Scenario Contract (G-B) — that extracts the
   context of the situation being simulated and tailors generation to it, under concrete rules and
   validation policies so it can never violate the math or the invariants. This is REGEN's
   intended differentiator.
4. **Explainability** as a first-class output: every batch ships with a computed account of what
   was generated, how, why it passed or failed, and which features matter — and every reported
   statistic must be **independently verifiable** by a third party who does not trust REGEN
   (the audit mechanism, G-G).
5. **Standing accuracy and performance checks** — a regression harness that catches quality or
   speed drift automatically, not just at audit time.
6. **Privacy on both fronts**: within the generated data (the δ-floor / parametric layer, Part I)
   and within the code and repo itself (secrets, real-data handling, logs — G-F).

This document has two parts. **Part I** (§3) is the defect list from the 2026-07-06 audit — the
system must *work* before it can be generalized, so Part I is a strict prerequisite. **Part II**
(§4) is the capability build-out against the north star. §5 is the build order, §6 the mandatory
documentation protocol, §7 the acceptance criteria.

---

## 1. What REGEN is (30 seconds)

A deterministic pipeline that generates statistically grounded synthetic tabular data and
amplifies rare events (fraud, faults, tail values) so downstream detectors improve. One pass:
**Prior** (base batch) → **Scout** (R-EPIG target) → **Amplifier** (ResidualGP tail correction) →
**Auditor** (fidelity gate: marginals, coverage, correlation structure) → **Examiner** (honest,
leakage-free detection-lift measurement). `regen.api.generate()` is the primary path: it returns a
full dataset (synthetic normal part + amplified rare part at a configurable `rare_ratio`).
`run_campaign()` / `screen()` are the multi-pass diagnostic paths. No LLM anywhere in the engine;
every batch is bit-reproducible from its manifest.

Five invariants (INVARIANTS.md §8) are enforced by tests. The two you will most likely brush against:
**Invariant 2** (manifest → bit-reproducible batch) and **Invariant 3** (no batch is persisted
without passing the Auditor).

---

## 2. Verified-working baseline (observed 2026-07-06, not assumed)

Every claim below was observed by running the command shown. Re-run them before you start to
confirm your environment matches, and re-run them after every change.

| Check | Command | Observed result |
|---|---|---|
| Full test suite | `python -m pytest tests/ -q` | **85 passed** (includes 13 new privacy tests) |
| End-to-end demo | `python examples/run_demo.py` | fidelity PASS (rare 1.00, normal 1.00); privacy PASS (min dist 0.52σ ≥ 0.50σ floor, 0 verbatim copies); detection lift **+0.278**; 3/3 campaign passes accepted |
| Bit-reproducibility with privacy on | `generate(...)` twice, seed 7, hash the parquet | **IDENTICAL** hashes |
| Realistic 30-dim data (label mode) | `generate("benchmark/data/creditcard_subset.csv", label_col="Class", n_rows=400, seed=7)` | passed=True both privacy modes; privacy costs coverage 0.956→0.913 and corr Δ 0.113→0.142, both under gates |
| Engine boundary | part of test suite | no LLM/network imports inside `engine/` |

Milestones M0–M5 (INVARIANTS.md §6) are complete and committed through `c5ebd9c`. The semantic-fidelity
plan (docs/SEMANTIC_FIDELITY_PLAN.md) has M1 + M1.5 done; its M2/M3 are **deliberately deferred**
pending owner decisions — do not build them.

### 2.1 The uncommitted privacy layer (critical context)

The working tree contains a substantially complete, **uncommitted** privacy layer:

- `engine/privacy.py` (new, 377 lines): `enforce_distance_floor` (per-record δ-distance floor —
  every released rare row ≥ δ, σ-normalized, from every real rare row, via projection + respawn,
  clamped in-support), `guard_against_duplicates` (verbatim-attribute guard), `assess_privacy`
  (read-only measurement on delivered data).
- `engine/prior/grounded.py` (+211 lines): `generate_parametric_batch` — Gaussian-copula sampling
  for continuous features + per-class frequency tables for discrete ones. Replaces grounded
  sampling (real anchor + jitter, which emitted near-copies of real individuals) when
  `privacy="floored"`.
- `regen/api.py` (+236/-34): threads `privacy`/`delta` through `generate()` (default
  **`privacy="floored"`, `delta=0.5`**), enforces the floor as the final numeric step (with a
  rounding-margin so integer re-rounding can't re-violate it), reports a separate `privacy` block
  and a top-level shippable verdict `passed = fidelity AND privacy`.
- `contracts/types.py`: `PrivacyReport`; `BatchManifest` gains `privacy` + `delta` fields.
- `tests/test_privacy.py` (new): 13 tests, unit + end-to-end, all green.

This design is sound and tested; it explicitly is **NOT differential privacy** (it prevents
near-copy re-identification; it does not bound membership-inference or aggregate attacks) and the
code says so honestly. The δ-floor is applied only to the **rare** part; the dense normal bulk is
protected by parametric sampling + the verbatim guard (a δ-shell is geometrically infeasible in a
dense bulk — this is deliberate and correct, keep it).

---

## 3. Part I — Audit findings (ranked; each has ID, evidence, and definition of done)

### P0-1 — Commit the privacy layer (work-preservation, do this FIRST)
The entire privacy layer exists only in the working tree. Any reset loses it.
**Do:** run `python -m pytest tests/ -q` (must be 85 green), then commit all modified + untracked
files listed by `git status` as one commit (`feat: privacy — parametric generation + enforced
δ-distance floor + verbatim guard`). The pre-commit hook runs the suite automatically
(`git config core.hooksPath` is already set to `.githooks`). Fix nothing in this commit — it is a
checkpoint of tested work. Everything else below lands as separate follow-up commits.

### P0-2 — Percentile (numeric-tail) rare mode fails the correlation gate under privacy, and privacy is the default
**Evidence (observed):**
```python
from contracts.types import RareEventDef, RareMode
from regen.api import generate
rd = RareEventDef(mode=RareMode.PERCENTILE, percentile=0.05, tail="upper")
s = generate("examples/transactions.csv", label_col="amount", rare_def=rd, n_rows=200, seed=7,
             privacy="floored")   # the default
# → fidelity correlation gate: delta=0.331, passed=False → batch not shippable
# with privacy="none": delta=0.048, passed=True
```
So the **default configuration fails** on percentile-mode rare events for the bundled example.
Root cause is unconfirmed — plausible candidates, in order: (a) the Gaussian copula is fit on a
truncated tail slice (5% of rows), where rank-based correlation estimation is unstable and the
truncation itself induces correlation the copula then distorts; (b) the δ-floor projection/respawn
moves rows in directions that break cross-column structure; (c) the ResidualGP correction
interacts badly with copula-sampled bases.
**Do:** instrument to isolate which stage introduces the correlation error (measure corr-delta on
the raw parametric batch, post-GP, post-floor). Then fix at the source. Acceptable fix shapes:
correlation-preserving floor displacement (project violators along directions that minimize
corr damage), better copula regularization/shrinkage for small-n truncated fits, or — if the case
is genuinely infeasible — a **loud** downgrade: the summary must say the privacy mode was
infeasible for this rare definition and why, and `passed` must be False with a reason field.
Silently loosening the correlation gate is forbidden (it exists to stop exactly this).
**Done when:** the repro above passes both privacy modes on `examples/transactions.csv` (or fails
loudly with a machine-readable reason), a regression test pins it, and label-mode results are
unchanged (demo still: fidelity PASS, privacy PASS, lift +0.278).

### P1-3 — The privacy layer is completely undocumented
`grep -rln "privacy" docs/ README.md server/API_GUIDE.md INVARIANTS.md` → **no matches** (before this
audit file). Code docstrings even say "see docs (Privacy)" — that document does not exist.
**Do:** (a) new `docs/PRIVACY.md`: the threat model (near-copy re-identification from grounded
sampling), the mechanism (parametric copula + frequency tables, δ-floor on the rare part, verbatim
guard on both parts), the exact guarantee, the explicit non-guarantees (**not** differential
privacy — no membership-inference or aggregate bound), why the bulk is not δ-floored, how to read
the `privacy` block of the summary, and how `delta` trades privacy against fidelity (cite the
observed creditcard numbers from §2). (b) Update `INVARIANTS.md` §3 (architecture: privacy stage),
§8 (add the privacy invariant: when `privacy="floored"` and the report says passed, every released
rare row is ≥ δ from every real rare row and no released row verbatim-duplicates a real row).
(c) Update `docs/REGEN_DOCUMENTATION.md` and `README.md` for the new `generate()` params and
summary fields, and note that `passed` is now fidelity AND privacy.

### P1-4 — Server and CLI do not expose privacy
`grep -n "privacy" server/app.py cli/main.py` → no matches. The server's `GenerateRequest`
silently inherits `privacy="floored"`; a user cannot turn it off or tune δ, and the response's
`privacy` block is undocumented in `server/API_GUIDE.md`. The CLI has no generate command surface
for it either.
**Do:** add `privacy: str = "floored"` and `delta: float = 0.5` to `GenerateRequest` (validate
same as `generate()`); pass through. Document the request fields and the response `privacy` block
in `server/API_GUIDE.md`. Give the CLI's relevant command(s) `--privacy {floored,none}` and
`--delta FLOAT`. Front-end rendering of this is out of scope — API/CLI surface only.

### P1-5 — Campaign and screen paths have no privacy story
`run_campaign()` and `screen()` hard-default `privacy="none"` internally (the parameter threads
through `_run_one_pass` but is never exposed). Result: the multi-pass campaign path emits
near-copies of real rows while `generate()` doesn't — an inconsistent product story.
**Do (decide, don't drift):** either thread `privacy`/`delta` through `run_campaign()`/`screen()`
signatures end-to-end (preferred if cheap — the plumbing already exists in `_run_one_pass`), or
explicitly document both as non-private diagnostic paths in their docstrings + docs and have their
summaries carry `privacy: {"mode": "none", ...}` so the regime is always visible. Record the
choice in the build log (§6). This is an **owner-facing decision** — if you cannot decide from the
code, present both options in the build log and pick the reversible one (documentation) while
flagging it.

### P1-6 — All benchmark results predate full-synthesis and privacy; privacy's fidelity/lift cost is unmeasured at scale
`benchmark/RESULTS*.{md,json}` are dated Jun 18–24; the full-synthesis change (`3032c83`) and the
privacy layer both post-date or coincide with them. Nothing measures what `privacy="floored"`
costs in fidelity and detection lift across the 10 benchmark datasets (`benchmark/data/`). The
single observation we have (creditcard_subset: coverage −0.04, corr Δ +0.03, gates still pass)
is one dataset.
**Do:** add a privacy dimension to the benchmark harness: for each dataset run
`generate(privacy="floored")` vs `generate(privacy="none")` at fixed seed, and record per dataset:
fidelity score, coverage, correlation delta, gate pass/fail, tail lift, privacy min-distance,
n_verbatim_duplicates, wall time. Write `benchmark/RESULTS_PRIVACY.md` + `.json` with the run
date and code version (git hash — the manifest already computes it). Mark superseded RESULTS
files with a header line pointing to the newest run rather than deleting them. **Done when:** the
table exists for all 10 datasets and every gate failure it reveals is either fixed or filed as an
explicit finding in the build log with a repro command.

### P2-7 — Detection lift reads 0.0 on very small rare sets, indistinguishable from "no benefit"
Observed: `creditcard_subset.csv` (23 rare rows) → `tail_lift = 0.0` in **both** privacy modes.
With 23 rare rows, the leakage-free lift protocol (fit on a train fold, measure on held-out rare
rows) leaves a handful of test rows; the estimate is degenerate and 0.0 is likely an artifact, not
a measurement. Rare-event amplification is the product's core claim — its headline metric must not
silently degenerate.
**Do:** detect when the held-out rare fold is below a minimum (pick and document a floor, e.g.
<10 test-fold rare rows), and report `lift: {"status": "insufficient_rare_rows", "n_test_rare": k}`
instead of a bare 0.0. Surface it in demo/CLI output. Add a test. Do not "fix" this by weakening
the leakage protections (see git `57a45fc` — the honest protocol is deliberate).

### P2-8 — Privacy scope inconsistencies to reconcile (design tightening, not rewrites)
Three small mismatches between what's enforced, what's measured, and what's implied:
(a) the δ-floor and `assess_privacy`'s distance check run against the **real rare set only** — a
synthetic rare row may in principle sit near a real *normal* row (cross-class near-copy);
(b) `_generate_amp_batch`'s verbatim guard checks against `result.rare_df` only, while
`assess_privacy` counts duplicates against the **full** real set — enforcement is narrower than
measurement, so a cross-class verbatim duplicate would fail the batch with no upstream step that
prevents it; (c) `engine/privacy.enforce_distance_floor`'s docstring says `real_df` is "the real
reference set to stay away from (normal + rare)" but the caller passes rare only.
**Do:** pick one scope and make enforcement, measurement, and docstrings agree. Recommended: keep
the δ-floor rare-vs-rare (the rare set is where isolation is feasible and re-identification risk
concentrates) but run the **verbatim guard against the full real set in both generation paths**,
and fix the docstring. Add a test for the cross-class verbatim case.

### P2-9 — The privacy floor silently skips when it can't apply
In `generate()`, the floor block runs only `if result.label_col and rare_val is not None and
cont_cols:` — a dataset with no continuous features (e.g. all-categorical like Open Payments) or
an unresolved label silently gets **no floor**, while the summary still says `mode: "floored"`.
This violates the repo convention "fail loud" (INVARIANTS.md §9). Note `assess_privacy` will report
`min_distance = inf → passed=True` in the no-continuous case, which is defensible (categoricals
are guarded by frequency sampling + verbatim guard) but must be *stated*, not implied.
**Do:** when the floor is skipped, the summary's `privacy` block must carry an explicit
`floor_applied: false` + reason (`"no_continuous_features"` / `"no_label"`), and the docs (P1-3)
must explain what protection remains in that case. Add a test on an all-categorical dataset.

### P2-10 — Housekeeping
(a) `.env` holds `TABPFN_API_KEY` — dead since the TabPFN backend was removed (`c5ebd9c`); delete
the entry (file is gitignored, low risk, but dead config misleads). (b) `regen-output/` and
`benchmark/regen-output/` contain stale generated parquet from earlier runs — regenerate or
delete; ensure they're gitignored. (c) `docs/KNOWN_ISSUES.md` is fully resolved — either append
new known issues from this build or leave as historical record with a dated header.

---

## 4. Part II — Capability build-out (what is missing for a use-case-ready system)

These are not defects; they are the gaps between "the pipeline works" and "a researcher can point
REGEN at their situation and trust what comes back." Each carries the same discipline as Part I:
tests, observed evidence, documentation in the same commit.

### G-A — The Scenario Contract: one typed object that carries the whole use case

Today the use case is smeared across loose `generate()` parameters (`rare_def`, `rare_ratio`,
`mode`, `privacy`, `delta`, `noise_scale`) plus implicit structural inference. There is no single
artifact that states *what situation is being simulated*. Build one:

```
contracts/scenario.py — ScenarioSpec (dataclass, JSON/YAML-serializable):
  columns:   per-column ColumnSemantics — role (feature|identifier|timestamp|target|free_text),
             dtype, unit, semantic bounds, category value-set, integrality
             (the L1 contract already specified in docs/SEMANTIC_FIDELITY_PLAN.md §3 — reuse it)
  intent:    task ("detector_training" | "data_sharing" | "benchmarking" | "exploration"),
             rare-event definition (label / percentile / imbalance), rare_ratio,
             focus_features (which columns the researcher cares about observing),
             n_rows, seed
  gates:     fidelity thresholds (or "default"), privacy mode + delta, minimum-utility policy
  provenance: for every field — how it was filled (user | structural | model), a confidence,
             and the raw proposal id it came from (see G-B)
```

Rules: `generate()`, `run_campaign()`, and `screen()` accept a `ScenarioSpec` (existing loose
params remain as conveniences that *construct* one — no API break). The **vetted spec is persisted
in the manifest**, so a batch is reproducible from disk including its use-case context
(Invariant 2 extends to the contract). A spec is the unit a researcher saves, shares, and re-runs.
**Done when:** a scenario YAML checked into `examples/` drives the demo end-to-end, the manifest
round-trips it, and a second run from the persisted spec is bit-identical.

### G-B — Situational awareness: the context-extraction mechanism, stated precisely

The owner's ask: the system should *extract the context of the situation being simulated and use
it to tailor the data*, reliably, without ever violating the mathematics. **The mechanism is a
three-source, one-gate contract pipeline.** Nothing about it is speculative — the repo already
contains its skeleton (deterministic schema profile in `engine/ingest/profile.py`, enforcement in
`engine/constraints.py`, and the validation charter in `docs/SEMANTIC_FIDELITY_PLAN.md` §1b). This
workstream generalizes and wires it:

```
        ┌─ Source 1: structural inference (deterministic; always runs; the baseline)
        │    engine/ingest/profile.py — dtype, cardinality, bounds, integrality, id-detection
        │
        ├─ Source 2: researcher declaration (YAML/JSON ScenarioSpec or API fields)
        │    highest authority — the researcher knows the situation
        │
        └─ Source 3: model proposal (OPTIONAL; one cached LLM call OUTSIDE engine/;
             regen/semantics.py) — reads the schema profile, proposes column meanings,
             units, semantic bounds, roles, and suggested rare-event framing
                          │
                          ▼
              THE VETTING GATE (deterministic code — this is what makes it reliable)
                          │
                          ▼
              vetted ScenarioSpec → engine (unchanged math, now parameterized)
                          → manifest (raw proposals + verdicts persisted; replayable)
                          → conformance audit on the delivered batch
```

**The vetting gate's rules (the "concrete rules and validation policies" — enforce every one in
code, with a test each; a proposal that violates one is dropped and logged, never applied):**

1. **Metadata only, never values.** No source — especially not the model — ever emits a synthetic
   data value. Sources emit *constraints and meanings*; the deterministic engine produces every
   number (INVARIANTS.md Invariant 4 — already load-bearing, now extended to the contract).
2. **Closed constraint vocabulary.** Only the fixed allowlist is honored: dtype, integrality,
   semantic min/max, unit/rounding, category value-set, role. No free-form code, expressions, or
   transformations. Anything outside the allowlist is ignored and logged.
3. **Data is ground truth.** A constraint may never contradict observed data: proposed bounds must
   *contain* the observed range; a category set must be a *superset* of observed categories;
   claimed integrality must match observed integrality. Contradiction → rejected + logged.
4. **Tighten toward validity only.** A constraint may enforce known-safe limits (currency ≥ 0);
   it may never invent values the data never exhibited.
5. **Authority order is fixed:** researcher > structural > model. A model proposal can fill a gap
   or tighten within the researcher's declaration; it can never override either.
6. **Per-field confidence with fallback.** Below threshold → that field falls back to structural.
   Never all-or-nothing.
7. **Nothing silent.** Every accepted / rejected / overridden constraint is recorded with its
   rationale and surfaced in the explanation report (G-C). This is also what makes the contract
   *explainable*.
8. **Replayable.** Raw proposals (including the model's raw text, prompt, and model id) persist in
   the manifest; a re-run replays them with **zero** model calls. Offline / no key / model error →
   Sources 1+2 only; generation never blocks and never degrades silently (the spec's provenance
   says which sources ran).
9. **Conformance is audited, not assumed.** The Auditor gains a contract-conformance check: the
   delivered batch must satisfy every vetted constraint (bounds, category sets, dtypes, id
   uniqueness). A conformance failure fails the batch exactly like a fidelity failure.
10. **Cost-bounded.** At most one model call per dataset, cached by schema hash, never in a loop
    (this is why the earlier agent-runtime layer was removed — do not reintroduce one).

**Why this honors the math:** the generators and gates (copula, ResidualGP, R-EPIG, TVD/correlation
statistics, δ-floor) are untouched mathematics; the contract only *parameterizes* them — which
columns are features, what counts as rare, what bounds to clamp to, which gates at which
thresholds. Context can therefore change *what* is simulated but never *how* numbers are made.

**Build staging:** ship Sources 1+2 + the gate + conformance audit first (pure Python, no new
dependency — this alone delivers most of the researcher value). Source 3 lands behind it, in
`regen/semantics.py` outside `engine/` (`test_boundary.py` must stay green), **advisory by
default** (the vetted proposal is shown and applied only with `--accept-contract` / an API flag;
both paths fully logged). The previous deferral of SEMANTIC_FIDELITY_PLAN M2/M3 is **lifted by the
owner as of 2026-07-06** for this shape; the one still-open owner decision is the model provider —
build provider-agnostic against an OpenAI-compatible interface and read the endpoint/key from env.
**Done when:** the same dataset generated under two different scenario specs (e.g. "fraud
detector training, amplify label 1" vs "tail-risk study, amplify top 5% of amount") produces
correspondingly different, gate-passing batches, each manifest replays without a model, and every
gate rule above has a failing-then-passing test.

### G-C — Explainability: every batch explains itself, from computed numbers only

A researcher must be able to answer: *what did I get, why should I trust it, and what should I look
at?* Add an `explain` block to the summary (and persist `explanation.json` next to the manifest)
containing, all **computed from the run** (never narrated by a model):

- **Per-gate account:** each fidelity/privacy/conformance check with its statistic, threshold, and
  verdict (e.g. `correlation: delta=0.048, threshold=0.15, PASS`) — the demo currently prints
  verdicts without the numbers behind them.
- **Per-column provenance:** which mechanism produced the column (copula-sampled / frequency-
  sampled / GP-corrected / identifier-minted / label-attached), which contract constraints were
  applied, and which were rejected and why (from G-B rule 7).
- **Feature informativeness** — the "features worth observing" ask: per-feature relevance for the
  rare class, computed from what already exists (the ResidualGP's ARD inverse-lengthscales,
  class-separation statistics) plus a permutation check on the Examiner's detector. Ranked list in
  the summary so a researcher immediately sees where the signal is.
- **Scout rationale:** the chosen target region and its R-EPIG score versus the alternatives
  considered, plus which regions were skipped as already-explored.
- **Utility with honesty markers:** lift with its train/test protocol stated, n_test_rare, and the
  small-sample status from P2-7 (never a bare 0.0).
- **Privacy account:** the numbers already in the privacy block, plus `floor_applied` and scope
  (from P2-9/P2-8).

If human-language narration is ever added on top, it is a model call outside `engine/` that may
only *cite* these computed statistics (INVARIANTS.md §4 rules apply); the JSON is the ground truth.
**Done when:** `explanation.json` ships with every batch, the demo prints its highlights, docs
gain `docs/EXPLAINABILITY.md` defining every field, and a test asserts the explanation's numbers
equal the report objects they came from.

### G-D — Standing accuracy & performance harness (checks that outlive this build)

P1-6 re-runs the benchmarks once; this makes quality drift impossible to miss thereafter:

- **Metrics registry:** one module enumerating every scored quantity — fidelity (per-column TVD/KS,
  correlation delta, coverage), utility (lift + CI + n_test_rare), privacy (min-distance,
  duplicates), conformance (violation count), performance (wall-time and peak memory per stage) —
  so benchmarks, tests, and explanations all read the same definitions.
- **One-command regression run:** `python benchmark/run_regression.py` — canonical datasets ×
  (privacy on/off) × fixed seeds, compared against committed baseline JSONs with explicit
  tolerances; **exits non-zero on any regression** (fidelity drop, lift drop beyond tolerance,
  gate flip, runtime blow-up past budget). Wire it as an optional pre-push/CI step; the pre-commit
  hook stays tests-only (the regression run is minutes, not seconds).
- **Performance budgets:** measure current per-stage wall-time on the canonical datasets, commit
  the numbers as the budget with headroom, and let the regression run enforce them. (The B1
  leakage fix doubled generation work per pass — see memory of 57a45fc — so campaign latency is a
  known watch-item; measure it rather than assume.)

**Done when:** the regression command exists, a deliberately-degraded run (e.g. noise_scale
cranked) fails it, and `benchmark/BASELINES/` holds dated, provenance-stamped baselines.

### G-E — The generality envelope: preflight instead of surprises

"Applied to multiple use cases" fails at the edges, so define the edges. Build a **preflight
check** (`regen doctor <data>` in the CLI and `preflight()` in the API) that validates a dataset
against the supported envelope *before* generation and reports actionable verdicts:

- minimum rare rows for amplification (GP underdetermination guard exists — surface it here),
  minimum rare rows for a non-degenerate lift estimate (P2-7's floor),
- all-categorical datasets (privacy floor cannot apply — state what protection remains, P2-9),
- high-cardinality columns (top-K TVD path), NaN rates, constant columns, dimensionality vs
  rare-count ratios, dataset size vs memory,
- **out-of-scope shapes named plainly:** time series, relational/multi-table, free text, and
  images are NOT supported — say so in the preflight output and in a `docs/CAPABILITY_MATRIX.md`
  (supported / degraded / unsupported, with the reason and the workaround if any). Do not build
  time-series or relational support in this pass; leave clean seams (the ScenarioSpec's column
  roles already reserve `timestamp`).

**Done when:** preflight exists in CLI + API, each envelope rule has a test with a fixture dataset
that trips it, and the capability matrix is written from observed behavior.

### G-F — Information protection in the code and repo (not just in the generations)

The generated-data privacy layer (Part I) covers the output. These rules cover the system itself;
enforce each with a test or a hook where possible:

1. **No real data values in logs, error messages, exceptions, manifests, or explanations.**
   Manifests and explanations carry statistics, hashes, schemas, and thresholds — never rows.
   Audit current `logger.*` calls and exception strings for leaked values; add a test that runs a
   generation at DEBUG level and greps the captured logs for known sentinel values planted in a
   fixture dataset.
2. **Secrets:** no keys in the repo, ever. `.env` is gitignored (verified) — delete the dead
   `TABPFN_API_KEY` entry (P2-10a). The G-B model endpoint/key comes from env only; add a hook or
   test that fails on high-entropy strings / `sk-`-style patterns in tracked files.
3. **What leaves the machine (G-B Source 3):** the only thing ever sent to a model is the schema
   profile — column names, dtypes, cardinalities, observed bounds, and at most a small, fixed
   number of example values per column with a redaction policy: identifier-role columns send
   **zero** real values; a `REGEN_SEMANTICS_SAMPLES=0` mode sends none at all. Persist exactly
   what was sent in the manifest so the exposure is auditable. Document this in PRIVACY.md.
4. **Real datasets in the repo:** `benchmark/data/` holds public research datasets — keep a
   PROVENANCE.md there recording source and license of each. User-supplied datasets and all
   generated outputs (`regen-output/`, tempdirs) stay gitignored; verify `.gitignore` covers them
   and clean the stale outputs already tracked (P2-10b).
5. **Tests use synthetic fixtures.** No test may embed rows from a real dataset; the bundled
   `examples/transactions.csv` is generated (see `examples/make_sample_data.py`) — keep it that way.

### G-G — Independent auditability: statistics you can check, not just read

In any model-validation or compliance-adjacent setting, a reported statistic that cannot be
independently recomputed is worth nothing. G-C makes the numbers visible; this workstream makes
them **checkable by someone who does not trust REGEN**. The mechanism has five parts:

1. **The audit bundle.** Every generation emits a self-contained directory (also buildable after
   the fact via `regen export-audit <run>`): the delivered parquet, `manifest.json` (extended to
   carry a `manifest_schema_version` plus the SHA-256 of every artifact in the bundle and the
   version of every metric used), `explanation.json`, and `reference_aggregates.json` — the
   aggregate statistics of the real reference data that every gate was computed against
   (per-column histograms/quantiles, correlation matrices, class counts). Aggregates only, under
   the disclosure policy in point 4 — never raw reference rows.
2. **`regen verify <bundle>` — the audit command.** Recomputes every statistic in
   `explanation.json` from the bundle's delivered data + reference aggregates, and reports
   stat-by-stat PASS/FAIL. Integrity first (artifact hashes must match the manifest), then values,
   within an explicit per-metric numeric tolerance (cross-machine floating-point/BLAS differences
   are expected; the tolerance policy lives in METHODS.md, never in someone's head). Any mismatch
   names the statistic, the reported value, and the recomputed value; exit code is non-zero.
   Verification is **pure recomputation from the bundle** — it never reads cached results, so
   there is no circularity: the check would catch a system that lied.
3. **`docs/METHODS.md` — the statistical methods reference.** A formal definition of every metric
   in the G-D registry: the formula, what it assumes, what it detects and what it cannot detect,
   the default threshold and the *rationale* for that threshold (measured provenance, not vibes —
   e.g. the top-K TVD design in docs/KNOWN_ISSUES.md is the standard to meet). Every metric gets
   an ID and a version; `explanation.json` entries cite them. Changing a metric's definition bumps
   its version, and the regression harness refuses to compare results across metric versions
   silently. This is the document a client's model-risk auditor actually reads.
4. **Auditability vs privacy — an explicit disclosure policy.** Reference aggregates reveal
   information about real data, so bound it: histogram/quantile buckets are published only above a
   minimum count (no bucket below k rows), correlation matrices are allowed, no per-row values
   ever, identifier columns summarized as counts only. Document the policy in PRIVACY.md and test
   the bucket floor. If a deployment needs stricter disclosure, the ScenarioSpec `gates` field
   dials it down — and `regen verify` then honestly reports which statistics became uncheckable at
   that disclosure level, rather than pretending to verify them.
5. **Self-verification in the loop.** The regression harness (G-D) runs `regen verify` on every
   bundle it produces, so the audit path cannot rot unnoticed.

**Done when:** a demo-run bundle verifies cleanly on recomputation; a test that tampers with one
value in the delivered parquet makes `verify` fail naming the affected statistics; METHODS.md
covers every registry metric with versioned IDs; the disclosure bucket-floor has a test; and the
regression harness self-verifies its bundles.

---

## 5. Build order

**Stage 1 — make it work (Part I, strict prerequisite for everything else):**

1. **P0-1** commit checkpoint (no code changes).
2. **P0-2** percentile-mode correlation failure (instrument → isolate → fix → regression test).
3. **P2-8 + P2-9** privacy scope + loud-skip (small, touches the same files as P0-2's area; do
   while context is warm).
4. **G-F** code/repo information-protection rules (cheap, and doing it early means every later
   workstream inherits the log/secret/redaction discipline instead of retrofitting it).
5. **P1-5** campaign/screen privacy decision + plumbing.
6. **P1-4** server + CLI exposure.
7. **P1-6** benchmark privacy sweep (validates 2–6 at scale; expect it to surface issues — loop
   back as needed).
8. **P2-7** lift degeneracy reporting.
9. **P2-10** housekeeping.

**Stage 2 — make it customizable and explainable (Part II):**

10. **G-A** ScenarioSpec contract + manifest round-trip (the foundation everything else plugs
    into).
11. **G-B** vetting gate + Sources 1+2 (structural + researcher) + conformance audit. No model yet.
12. **G-C** explainability (`explanation.json` + feature informativeness + demo/CLI surfacing).
13. **G-G** independent auditability (audit bundle + `regen verify` + METHODS.md + disclosure
    policy) — immediately after G-C because it verifies exactly what G-C reports, and before G-D
    so the harness can self-verify from day one.
14. **G-D** standing regression harness + performance budgets (baselines recorded from the
    now-final Stage 1+2 behavior; every run self-verifies via G-G).
15. **G-E** preflight / capability matrix (it reads the ScenarioSpec and the metrics, so it comes
    after 10–14).
16. **G-B Source 3** — the optional model proposal path (`regen/semantics.py`), advisory,
    provider-agnostic, behind env config. Last, because everything it needs (contract, gate,
    conformance, redaction rules) exists by now, and because it is the only step with an external
    dependency.
17. **P1-3 + Part II docs** — documentation written LAST so it describes what was actually built
    and measured, with real numbers from steps 7 and 14.

Each numbered item = at least one commit, tests green before each commit (the hook enforces it).
Do **not** build: the front-end, formal differential privacy, time-series or relational/multi-
table generation, any agent runtime, or the M6 run-state store. The earlier deferral of the
semantic layer (SEMANTIC_FIDELITY_PLAN M2/M3) is lifted **only in the G-B shape** — contract +
vetting gate first, one cached advisory model call outside `engine/` last; nothing beyond that.

---

## 6. Documentation & build-tracking protocol (mandatory, per change)

The owner audits builds against these rules. Every one is checkable; follow them exactly.

1. **Build log.** Create `docs/BUILDLOG.md` (append-only). One entry per work session:
   date, finding IDs addressed, what was observed before (the repro numbers), what changed, what
   was observed after (re-run the same repro), commit hashes, test count. An entry that says
   "fixed X" without a before/after observation is incomplete.
2. **Verify, don't assert.** No claim goes into the build log, docs, or a commit message unless
   you ran a command and saw it. Include the command next to the claim so the owner can re-run
   it. This applies to performance/fidelity numbers especially: they come from actual runs on the
   actual datasets, never from expectation.
3. **Tests first per defect.** For each finding, write the failing test (red), then fix (green).
   The regression test names the finding ID in its docstring (e.g. "P0-2: percentile-mode
   correlation gate under privacy").
4. **Docs move with behavior.** Any change to `generate()`/`run_campaign()`/`screen()` signatures
   or summary shape updates `docs/REGEN_DOCUMENTATION.md` in the same commit; any endpoint change
   updates `server/API_GUIDE.md` in the same commit; anything touching architecture or invariants
   updates `INVARIANTS.md` in the same commit. Stale docs are treated as defects.
5. **Decisions are surfaced, never defaulted silently.** Where a finding says "decide" (P1-5,
   P2-8 scope), record in the build log: the options, the choice, the reason, and what would
   reverse it. This mirrors INVARIANTS.md §7's existing convention.
6. **Benchmark results carry provenance.** Every RESULTS file states the run date, the git hash
   (from the manifest's `code_version`), and the exact command. Superseded results get a header
   pointing to the successor — never silently edited numbers, never deleted history.
7. **Consistency self-review before finishing.** After the last commit, without being asked:
   re-run the full suite, `python examples/run_demo.py`, and the P0-2 repro; confirm every doc
   claim you wrote matches an output you observed this session; confirm no file in `git status`
   is left uncommitted. Record this self-review as the final BUILDLOG entry.
8. **Invariant guardrails.** If any change would violate a INVARIANTS.md §8 invariant, stop and flag
   it in the build log rather than working around it. Specifically re-verify after P0-2, P1-5,
   G-A, and G-B: same manifest → identical parquet hash (Invariant 2), with privacy both on and
   off — and once G-A lands, the manifest replay must reproduce the batch *including* its
   ScenarioSpec with zero model calls.
9. **The math/config boundary is a review criterion.** Any diff that makes an engine statistical
   routine branch on a *use case* (rather than on a parameter of the vetted ScenarioSpec) is
   wrong by construction — push the difference into the contract. One engine, many contracts.
10. **Explainability moves with behavior.** Any new gate, generator mechanism, or metric must add
    its computed entry to `explanation.json` and `docs/EXPLAINABILITY.md` in the same commit —
    an unexplained check is treated as an undocumented check (rule 4 applies).

---

## 7. Acceptance criteria (the whole build is done when all hold)

**Stage 1 — it works:**

- [ ] Full suite green (≥85 tests, plus new regression tests per finding), via the pre-commit hook.
- [ ] `python examples/run_demo.py`: fidelity PASS, privacy PASS, lift reported (+0.27±0.02 on the
      bundled data), campaign 3/3 accepted.
- [ ] The P0-2 percentile repro passes under `privacy="floored"` — or fails loudly with a
      machine-readable reason in the summary.
- [ ] `benchmark/RESULTS_PRIVACY.md` covers all 10 datasets, privacy on vs off, with provenance;
      no unexplained gate failures.
- [ ] Server + CLI accept and validate `privacy`/`delta`; API_GUIDE documents request + response.
- [ ] `docs/PRIVACY.md` exists; INVARIANTS.md, REGEN_DOCUMENTATION.md, README updated; the phrase
      "not differential privacy" appears anywhere the guarantee is described.

**Stage 2 — it is customizable, explainable, and self-checking:**

- [ ] A `ScenarioSpec` YAML in `examples/` drives an end-to-end run; the manifest round-trips it;
      a re-run from the persisted spec is bit-identical with zero model calls (Invariant 2
      extended).
- [ ] Two different scenario specs over the same dataset produce correspondingly different,
      gate-passing batches (the G-B done-when demonstration), and every vetting-gate rule (G-B
      1–10) has its own test.
- [ ] The Auditor enforces contract conformance; a batch violating a vetted constraint is
      rejected, with the violation named in the explanation.
- [ ] Every batch ships `explanation.json` — gate statistics vs thresholds, per-column provenance,
      ranked feature informativeness, Scout rationale, honest-lift status, privacy account — and a
      test proves its numbers equal the report objects they came from.
- [ ] `regen doctor` / `preflight()` exists; each envelope rule has a fixture test;
      `docs/CAPABILITY_MATRIX.md` states supported / degraded / unsupported shapes from observed
      behavior.
- [ ] `python benchmark/run_regression.py` exists with committed, provenance-stamped baselines;
      a deliberately degraded run fails it; performance budgets are recorded and enforced.
- [ ] Every batch exports a self-contained audit bundle; `regen verify <bundle>` recomputes all
      reported statistics from the delivered data + reference aggregates and passes on the demo;
      a tamper test (one value changed in the parquet) fails verification naming the affected
      statistics.
- [ ] `docs/METHODS.md` formally defines every registry metric (formula, assumptions, threshold
      rationale) with versioned IDs that `explanation.json` cites; metric-version changes are
      never silently compared by the regression harness.
- [ ] Reference aggregates respect the minimum-bucket disclosure policy (tested); PRIVACY.md
      documents the auditability-vs-privacy tradeoff and the ScenarioSpec knob that tightens it.
- [ ] G-F holds: sentinel-value log test passes, no secrets in tracked files (hook/test), model
      payload redaction implemented and persisted to the manifest, `benchmark/data/PROVENANCE.md`
      exists, all generated outputs gitignored.

**Always:**

- [ ] `docs/BUILDLOG.md` has a dated, before/after-observed entry per finding/workstream, plus the
      final self-review entry.
- [ ] `git status` clean; every commit passed the test hook.
- [ ] No engine statistical routine branches on a use case — use cases exist only as vetted
      ScenarioSpec parameters (§6 rule 9).
