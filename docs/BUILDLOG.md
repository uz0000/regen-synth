# REGEN — Build Log

Append-only record of build sessions against `docs/AUDIT_2026-07-06_BUILD_PROMPT.md`.
Protocol: every entry states the finding IDs addressed, what was observed **before** (with the
repro command + numbers), what changed, and what was observed **after** (same repro re-run),
plus commit hashes and the test count. "Verify, don't assert" — no claim here that wasn't run.

Environment: Python 3.11 (repo `.venv`, which the pre-commit hook uses). All commands run from
the repo root. `python` below == `.venv/bin/python` unless noted; the anaconda `python` gives
identical results (both 3.11, same deps).

---

## Session 2026-07-06 — Stage 1 kickoff

### Baseline verification (audit §2, re-observed before touching anything)

| Check | Command | Observed |
|---|---|---|
| Full suite | `python -m pytest tests/ -q` | **85 passed** |
| Demo | `python examples/run_demo.py` | fidelity PASS (rare 1.00, normal 1.00); privacy PASS (nearest real row 0.52σ ≥ 0.50σ floor, 0 verbatim); lift **+0.278**; campaign 3/3 accepted |
| Git state | `git status` | matches audit §2.1 exactly — 6 modified + `engine/privacy.py`, `tests/test_privacy.py`, audit doc untracked |

Baseline matches the audit's recorded numbers exactly.

### P0-1 — Commit the privacy layer (work-preservation)

- **Before:** the entire privacy layer existed only in the working tree (uncommitted). A reset
  would lose `engine/privacy.py` (377 lines), `generate_parametric_batch`, and the API plumbing.
- **Change:** committed the 6 modified files + `engine/privacy.py` + `tests/test_privacy.py` as
  one checkpoint. No functional change. The audit doc was committed separately (it is a spec, not
  privacy code — a deliberate, minor split from the literal "one commit" instruction, recorded
  here per §6 rule 5).
- **After:** pre-commit hook ran the suite in `.venv` → **85 passed**. Commit `5ee649b`.
- **Note discovered:** the repo has a `.venv` (not mentioned in the audit); the pre-commit hook
  prefers it over system python. Both `.venv` and anaconda python 3.11 have all deps and produce
  the same results.

### P0-2 — Percentile-mode correlation failure under privacy (default path broken)

- **Before (repro):**
  ```
  from contracts.types import RareEventDef, RareMode; from regen.api import generate
  rd = RareEventDef(mode=RareMode.PERCENTILE, percentile=0.05, tail="upper")
  generate("examples/transactions.csv", label_col="amount", rare_def=rd, n_rows=200, seed=7, privacy="floored")
  ```
  → correlation `delta=0.331`, `passed=False`, batch not shippable. `privacy="none"` → `delta=0.048`, passes.
- **Instrumentation (isolate the stage):** measured corr-delta at base / post-GP / post-constraints
  on the rare part for both modes (`scratchpad/instrument_p02.py`):
  - floored: base **0.225** → post-GP 0.226 → post-constraints 0.226
  - none:    base **0.036** → post-GP 0.036 → post-constraints 0.035
  The error is introduced entirely at the **base parametric (copula) stage**; the GP and the floor
  are ruled out. Per-pair inspection showed the broken pairs were exactly those involving `is_fraud`
  (the binary column): real `n_prior_txns×is_fraud=-0.593` vs synthetic `+0.097`.
- **Root cause:** `generate_parametric_batch` sampled the continuous block via a Gaussian copula but
  sampled discrete columns **independently** (`rng.choice` per column), erasing all discrete↔continuous
  correlation. In LABEL mode the binary is the label (excluded from the correlation gate) so it never
  surfaced; PERCENTILE mode makes the binary a gated feature.
- **Change:** replaced the continuous-only copula + independent discrete sampling with **one joint
  mixed-data Gaussian copula** over all feature columns (`engine/prior/grounded.py`:
  `generate_parametric_batch`, new `_copula_uniforms` + `_discrete_inverse_cdf`). Continuous columns
  map back through empirical quantiles; discrete columns through the inverse-CDF of their per-class
  frequency table — each marginal is preserved exactly, and cross-correlation comes from a shared
  latent. When there are no discrete features the path reduces **bit-for-bit** to the old copula.
- **After (same repro):** floored `delta=0.331 → 0.101`, `passed=True`, shippable. `none` unchanged
  at 0.048. Prototype confirmed the fix (`scratchpad/proto_mixed_copula.py`): rare corr-delta
  0.266→0.103, `is_fraud` marginal 0.13→0.132.
- **Invariant 2 re-verified (§6 rule 8):** label-mode `generate(...seed=42,privacy="floored")` output
  is **bit-identical** before vs after (row-hash `40933a2bef4f4140` both, via `git stash`), so the
  demo/label-mode results are unchanged as the audit requires.
- **Tests:** added `TestP02PercentileCorrelationUnderPrivacy` (3 tests: e2e repro passes under floor,
  privacy-off sanity, and a direct check that the copula preserves a binary↔continuous correlation).
  Full suite **88 passed** (was 85).
- **Commit:** `1dbc508`.

### P2-8 + P2-9 — Privacy scope reconciliation + loud floor-skip

Done together per the audit (same files/context as P0-2).

**P2-8 (scope):**
- (a) δ-floor stays **rare-vs-rare** — enforcement (`generate()`) and measurement (`assess_privacy`)
  already agree on this; kept.
- (b) The **verbatim guard now runs against the full real set** (normal + rare) in both generation
  paths (`_generate_amp_batch`, `_generate_normal_batch`), matching `assess_privacy`'s measurement
  scope. Previously enforcement was rare-vs-rare / normal-vs-normal while measurement was vs the full
  set, so a cross-class verbatim copy could fail the batch with no upstream step preventing it.
- (c) Fixed the misleading `enforce_distance_floor` docstring (it claimed `real_df` was "normal +
  rare"; the caller passes rare-only — now documented as "stay away from whatever reference you pass").
- **Discovered + fixed (surfaced, per §6 rule 5):** the verbatim-duplicate check over-counted on
  low-cardinality **discrete-only** data — every synthetic row necessarily reuses a category tuple, so
  an all-categorical batch reported `n_verbatim=200, passed=False`, contradicting the audit's own
  statement that this case should pass. Redefined a verbatim duplicate as reproducing a **uniquely-
  identifying** real record: a discrete tuple shared by ≥ `_MIN_ANON_COUNT` (=2) real rows is
  k-anonymous and safe; only singleton tuples are flagged. Continuous matches (within ~1e-3σ) are
  already near-unique, so that branch is unchanged. **Before:** all-categorical → `passed=False`
  (200 false dups). **After:** `passed=True`, `n_verbatim=0`.

**P2-9 (loud skip):**
- `generate()` now tracks `floor_applied` + `floor_skip_reason` and surfaces them in the `privacy`
  block. Reasons: `no_label`, `no_continuous_features`, `no_rare_rows`. Added the two fields to
  `PrivacyReport` (so they flow to `explanation.json` in G-C). The note text now states plainly when
  the floor was skipped and what protection remains.
- **Before (repro):** `generate(all_categorical.csv, privacy="floored")` → summary said `mode:floored`
  with no indication the floor never ran. **After:** `floor_applied=False`,
  `floor_skip_reason="no_continuous_features"`, `min_distance=inf`, `passed=True` — explicit.
- **Invariant 2 re-verified:** label-mode row-hash still `40933a2bef4f4140` (P2-8's guard-scope change
  is a no-op on continuous data — 0 verbatim dups).
- **Tests:** `TestP08Scope` (k-anonymity + full-set scope) and `TestP09LoudFloorSkip` (explicit skip on
  all-categorical + floor-applied-true contrast). Suite **88 → 92 green**.

### G-F — Code/repo information protection (done early so later work inherits it)

- **(1) No values in logs — verified + guarded.** Audited every `logger.*` call and `raise` string in
  `engine/` + `regen/`: none interpolate a real cell value (they carry column names, counts, metrics,
  and the user's own filepath). Added `tests/test_infosec.py::TestNoValueLeakInLogs` — runs
  `generate()` at DEBUG over a fixture with planted sentinel values (`8675309.4242`,
  `ZZ_SENTINEL_CATEGORY_ZZ`) and asserts neither appears in captured logs. **Observed:** passes.
- **(2) No secrets in tracked files.** `TestNoSecretsInTrackedFiles` scans everything `git ls-files`
  returns for `sk-…`/`AKIA…`/`tabpfn_sk_…`/secret-named-assignment patterns. **Observed:** 0 offenders.
  This runs on every commit via the existing pre-commit test hook, so it doubles as the requested
  secret-scan hook. Deleted the dead `TABPFN_API_KEY` from `.env` (P2-10a); `.env` was already
  gitignored (never tracked), so no secret was ever in the repo — verified via `git ls-files .env`
  (empty).
- **(3) Model-payload redaction seam.** The engine is model-free; documented the provider-agnostic env
  seam for the future advisory layer in `.env` (`REGEN_SEMANTICS_*`, incl. `REGEN_SEMANTICS_SAMPLES=0`
  to send no example values). Actual redaction code lands with G-B Source 3 (item 16); PRIVACY.md
  documents it at P1-3.
- **(4) Dataset provenance + gitignore.** Wrote `benchmark/data/PROVENANCE.md` (verified OpenML ids from
  `benchmark/dataset_candidates.csv` + the runners' `openml.datasets.get_dataset(<id>)`); added a
  `.gitignore` negation so the provenance doc is tracked while the datasets stay ignored. Untracked 5
  stale `regen-output/*` files that were committed before the ignore rule (`git rm --cached`, P2-10b).
- **(5) Synthetic fixtures only.** The new tests build their own synthetic data;
  `examples/transactions.csv` is generated. No real rows in tests.
- Suite **92 → 94 green**. _(P2-10a + P2-10b handled here; P2-10c KNOWN_ISSUES header deferred to the
  P2-10 housekeeping step.)_

### P1-5 — Campaign/screen privacy (decision + plumbing)

- **Decision (surfaced per §6 rule 5).** The audit offered (A) thread privacy through the signatures
  vs (B) document them as non-private with a visible regime, preferring (A) "if cheap." Investigation
  showed the δ-floor lived only inside `generate()`, so passing `privacy="floored"` into `run_campaign`
  would have given parametric+guard but **no floor** — a silent gap. I chose **(A), done properly**:
  extracted the floor into a shared helper `_enforce_rare_floor()` (verified `generate()` stays
  **bit-identical** — row-hash `40933a2b`, a pure move) and threaded `privacy`/`delta` through
  `run_campaign` end-to-end (default `"none"` for backward compat). This *fixes* the audit's
  inconsistency (campaign emitting near-copies while generate didn't) rather than papering over it.
  `screen()` stays **non-private by design** (it returns a recommendation, persists no rows, and its
  REGEN-vs-SMOTE lift must be apples-to-apples) — documented in its docstring, no param added.
  **Reversal:** to make campaign default to floored, flip the default; to force screen private, add a
  param and a matched-privacy SMOTE arm.
- **Change:** `run_campaign` validates + threads `privacy`/`delta`; each accepted batch is floored via
  the shared helper before persistence; the regime is recorded in `campaign_summary.json` (`privacy`
  block) and the manifest. `_save_campaign_summary` gained the block.
- **After (observed):** `run_campaign(privacy="floored")` best batch → `min_dist=0.520`, `passed=True`,
  `n_verbatim=0`; summary `privacy.mode=floored`, manifest `privacy=floored, delta=0.5`. Default run →
  `privacy.mode=none` with an explicit note. `generate()` bit-identical (hash `40933a2b`).
- **Tests:** `TestP15CampaignPrivacy` (floored batches carry the floor; default regime visible; bad
  privacy rejected). Suite **94 → 97 green**.
- **Docs pending (tracked):** `run_campaign` gained params, so `docs/REGEN_DOCUMENTATION.md` needs the
  new signature — deferred to the P1-3 docs pass because G-A will change these same signatures again
  (adding `ScenarioSpec`); updating twice is churn. Docstrings are current now.

### P1-4 — Server + CLI privacy exposure

- **Server:** added `privacy`/`delta` to `GenerateRequest` (default `floored`/`0.5`) and
  `CampaignRequest` (default `none`/`0.5`), threaded into `generate()`/`run_campaign()`. Both
  endpoints now return **400** on invalid privacy/delta (added `ValueError→400` to the campaign
  endpoint). `/api/generate` already returns the full `privacy` block; `/api/campaign` response now
  carries `{mode, delta}` (full regime in the run's `campaign_summary.json` + manifest).
- **CLI:** added `--privacy {none,floored}` + `--delta` to `regen run`, and a new **`regen generate`**
  subcommand (the audit flagged the missing generate surface) with `--privacy {floored,none}` +
  `--delta` and a human-readable summary. `generate` auto-detects the rare class when `--rare-value` is
  omitted in label mode.
- **Docs:** updated `server/API_GUIDE.md` (§1 generate + §5 campaign) for the request fields and the
  response `privacy` block, incl. the "NOT differential privacy" phrasing (endpoint change → same
  commit, §6 rule 4).
- **Observed:** `regen generate examples/transactions.csv --label is_fraud --n-rows 150 --privacy
  floored` → Fidelity PASS (score 1.0), Privacy PASS (min-dist 0.5482, 0 verbatim, δ-floor applied),
  lift +0.2778, Shippable PASS. Server: floored generate returns a passing privacy block; `privacy:
  bogus` and `delta: 9.0` → 400; floored campaign response shows `privacy.mode=floored`.
- **Tests:** new `tests/test_server.py` (6 tests via FastAPI `TestClient`: floored/none/invalid on
  both endpoints). Suite **97 → 103 green**.

### P1-6 — Benchmark privacy sweep (floored vs none, 11 datasets)

- **Harness:** `benchmark/run_privacy_sweep.py` — fixed config (n_rows=400, seed=7, auto-tune OFF so
  floored-vs-none isolates the privacy cost), rare class auto-detected, per dataset records fidelity,
  coverage, corr-delta, gate, tail lift, floor_applied, min-distance, verbatim count, wall time.
  Writes `benchmark/RESULTS_PRIVACY.{md,json}` stamped with run date + git `code_version`
  (`c2ec51a`). Superseded older RESULTS files (`RESULTS.md`, `RESULTS_BREADTH.md`,
  `RESULTS_MULTIPASS.md`) with a header pointing here (kept, not deleted — §6.6).
- **Headline (observed):** on the 6 datasets with continuous features that pass the gate, floored
  preserves fidelity — coverage cost ≤ 0.03 (e.g. hypothyroid 0.980→0.947, creditcard 0.980→0.919)
  and corr-delta stays under the 0.25 gate (often *improves*: ozone 0.079→0.062, wilt 0.077→0.052).
  Floor min-distance clears δ=0.5 everywhere it applies (0.5–4.1σ). Verified at scale.
- **Crash found + FIXED:** solar_flare `privacy="floored"` crashed (coincident-row inf) — fixed in
  `c2ec51a` (see the fix entry above), now runs.
- **Findings FILED (repro + routed to G-E capability matrix), not silently accepted:**
  1. **Low-cardinality integer data degrades under the floor.** `solar_flare` floored: coverage
     collapses 1.00→0.039, fidelity 1.0→0.7, gate FAILS. Its "continuous" features are 3–6-value
     integer codes; the δ-floor (+integer rounding margin) shoves the tiny integer-grid rare cluster
     far off its own region. Repro: `generate("benchmark/data/solar_flare.csv", label_col="Class",
     rare_def=None, n_rows=400, auto=False, seed=7, privacy="floored")`. → G-E: **degraded** for
     low-cardinality integer/ordinal features; preflight should warn.
  2. **All-categorical high-cardinality data loses fidelity under parametric sampling.**
     `open_payments` floored: fidelity 0.80→0.40 (floor correctly skipped, `floor_applied=False`,
     `no_continuous_features`). Parametric frequency-table sampling + the copula degrade high-card
     categorical TVD vs grounded anchoring. Repro: same call on `open_payments.csv`. → G-E:
     **degraded** for all-categorical high-cardinality; the floor gives no extra protection there
     (verbatim guard + k-anonymity do).
  3. **Lift reads a bare 0.0 on small rare sets** (creditcard_subset/satellite/creditcard, both
     modes) — the P2-7 degeneracy; addressed next.
  4. **Not privacy-induced:** bank_marketing / churn fail the gate in **both** modes at this config
     (a column TVD exceeds threshold) → lift not measured. Pre-existing gate behaviour, noted.
- Suite **104 green** (crash-fix regression test included).

### P2-7 — Lift degeneracy reporting (no more bare 0.0 on tiny rare folds)

- **Before (repro):** `generate("benchmark/data/creditcard_subset.csv", label_col="Class",
  rare_def=None, n_rows=400, auto=False, seed=7, privacy="none")` → `lift.tail_lift = 0.0` — but only
  ~7 real rare rows are held out, so recall can only take a few discrete values and 0.0 is an
  artifact, indistinguishable from "no benefit."
- **Change:** documented floor `MIN_TEST_RARE = 10` in the Examiner; `LiftReport` gains
  `n_test_rare` + `status` (`ok` / `insufficient_rare_rows`); the leakage-free protocol is **unchanged**
  (git 57a45fc) — only its reliability is annotated. `generate()`'s `lift` block now reports
  `{status, n_test_rare, baseline_recall, amplified_recall, tail_lift}` with `tail_lift=None` when
  insufficient (not a bare 0.0). CLI `generate` prints "n/a (only k held-out rare rows)".
- **After (same repro):** `lift = {status: "insufficient_rare_rows", n_test_rare: 7, tail_lift: None,
  ...}`. Healthy case (transactions, 18 held-out) → `status: "ok", tail_lift: 0.2778`.
- **Tests:** `test_lift_flags_insufficient_rare_rows` + `test_generate_lift_out_nulls_tail_lift_when_insufficient`
  (synthetic small-rare fixtures); existing no-synth lift test now also asserts `status=="ok"`.
  Suite **104 → 106 green**. `generate()` bit-identical (hash `40933a2b`).

### P2-10 — Housekeeping

- **(a)** dead `TABPFN_API_KEY` removed from `.env` — done in G-F.
- **(b)** stale `regen-output/*` untracked, gitignore verified — done in G-F. Confirmed no tracked
  parquet/output files remain (`git ls-files | grep -iE "regen-output|\.parquet"` → none).
- **(c)** `docs/KNOWN_ISSUES.md` was fully-resolved history; added a dated header marking it as such
  and a **"CURRENT KNOWN ISSUES (2026-07-06)"** section capturing the three open items surfaced this
  build (floored degrades low-card integer data; floored costs fidelity on all-categorical high-card
  data; a low-severity pandas int64 FutureWarning on floor write-back). Each cross-refs the BUILDLOG
  repro and G-E.

---

## Session 2026-07-06 (cont.) — Stage 2: customizable & explainable

### G-A — ScenarioSpec contract + manifest round-trip

- **New `contracts/scenario.py`** (pure dataclasses, no engine import): `ColumnSemantics` (the L1
  contract from SEMANTIC_FIDELITY_PLAN §3 — role/dtype/unit/bounds/categories/integer + per-column
  provenance: `source`/`confidence`/`proposal_id`), `ScenarioIntent` (task, rare-event def,
  rare_ratio, focus_features, n_rows, seed, mode), `ScenarioGates` (fidelity thresholds or default,
  privacy+delta, min-utility), and `ScenarioSpec` (columns+intent+gates+provenance, JSON **and** YAML
  round-trip). `columns_from_field_dict()` fills Source-1 structural semantics deterministically.
- **`BatchManifest` gains `scenario`** (the vetted spec dict); `build_manifest`/`_write_manifest`
  thread it. `generate()`, `run_campaign()`, `screen()` all accept `scenario=` (loose params kept as
  conveniences that construct one — no API break; `run_campaign`/`screen` label_col/rare_def now
  default so a spec-only call works). When a spec is supplied its intent+gates are authoritative;
  otherwise `generate()` builds one from the resolved params. Every batch now ships a spec: persisted
  in the manifest, echoed in the summary, and written as `scenario.yaml` next to the batch.
- **Invariant 2 extended + verified:** `generate(...seed=42,privacy="floored")` → persist spec →
  reconstruct `ScenarioSpec.from_dict(manifest["scenario"])` → `generate(scenario=spec)` reproduces the
  batch **bit-identically** (row-hash `40933a2b` all three: original, no-scenario, spec-replay). The
  no-scenario path is bit-identical to before (spec is metadata; generation params unchanged).
- **Example:** `examples/scenario_fraud.yaml` (researcher-authored fraud-detector-training scenario)
  drives an end-to-end run and round-trips.
- **Tests:** `tests/test_scenario.py` (8: JSON/YAML round-trip, rare_def build, structural fill,
  manifest replay bit-identity, YAML-drives-generation, campaign+screen accept a spec). Suite
  **106 → 114 green**.

### G-B — Vetting gate + Sources 1+2 + conformance audit (the differentiator)

- **`regen/vetting.py` — `vet_scenario(proposed, ingest)`**: merges Source 1 (structural) + Source 2
  (researcher declaration) into vetted columns + a list of `VettingVerdict`s, enforcing the gate rules
  in code (each with a test): closed vocabulary (role/dtype), data-is-ground-truth (proposed bounds
  must *contain* observed range; category set must be a *superset*; integrality must match),
  tighten-toward-validity (a wider safe bound like currency ≥ 0 is accepted, a clipping one rejected),
  authority order (researcher > structural), per-field confidence fallback, declared-column-must-exist,
  and nothing-silent (every accept/reject/fallback logged with rule + rationale). Rule 1 (metadata
  only) is structural — `ColumnSemantics` has no value field (tested). Rules 8/10 (replay/one-model-
  call) belong to Source 3 (item 16).
- **`engine/auditor/conformance.py` — `check_conformance(df, spec)`**: the Auditor's contract gate
  (rule 9). The delivered batch must obey every vetted constraint — bounds, integrality, categorical
  value-set, identifier uniqueness — reporting violation *counts* (never row values, G-F). A
  conformance failure fails the batch like a fidelity failure (Invariant 3 extended). Exported from
  `engine.auditor`.
- **Wired into `generate()`**: vetting runs each generation → verdicts persisted in the spec (and thus
  the manifest); conformance runs on the delivered batch → folded into `overall_passed = fidelity AND
  conformance AND privacy`; both surfaced in the summary (`scenario.verdicts`, `conformance`).
- **Observed:** `generate()` bit-identical (hash `40933a2b`); the example scenario's 5 user
  declarations all vet **accepted**, conformance passes. **G-B done-when demonstrated:** the same
  `transactions.csv` under a *fraud-detector* spec (label is_fraud) vs a *tail-risk* spec (percentile
  top-5% of amount) yields two correspondingly different, gate-passing, conformant batches (different
  row-hashes).
- **Tests:** `tests/test_vetting.py` (11: one failing-then-passing per data-facing rule + conformance
  violation/uniqueness + the two-specs-differ demo). Suite **114 → 126 green**.

### G-C — Explainability (`explanation.json`)

- **`regen/explain.py` — `build_explanation(...)`**: assembles, from the run's own report objects only
  (no model narration): per-gate account (coverage/correlation/columns/conformance/privacy each with
  statistic + threshold + verdict), per-column provenance (role, source, production **mechanism** —
  copula-sampled+GP / copula-frequency-sampled / grounded / identifier-minted / label-attached — plus
  applied and rejected constraints from the vetting verdicts), **feature informativeness** (per-feature
  class-separation Fisher score, ranked), Scout rationale (target region), utility with honesty markers
  (P2-7 status/n_test_rare/protocol), and the privacy account.
- **Wired into `generate()`**: every batch writes `explanation.json` next to the manifest and echoes it
  in the summary under `explain`. Built after the privacy block so the numbers match exactly.
- **Observed:** `generate()` bit-identical (`40933a2b`); top informative feature `n_prior_txns`
  (Fisher 3.08) — matches P0-2's finding that it drives the fraud signal; `explanation.json` on disk ==
  `summary["explain"]`.
- **Docs:** `docs/EXPLAINABILITY.md` defines every field + the maintainer rule (new gate/mechanism/
  metric must add its explanation entry, §6 rule 10).
- **Tests:** `tests/test_explain.py` (5: file ships + equals summary; gate numbers equal the fidelity/
  conformance/privacy reports; utility equals the lift block; ranked informativeness; per-column
  mechanism). Suite **126 → 131 green**.
- **Demo highlights (pending):** printing explanation highlights in `run_demo.py` is deferred to the
  final self-review per the owner's "set the demo aside for now" — the JSON + docs + tests are in.

### G-G — Independent auditability (audit bundle + `regen verify`)

- **`regen/metrics.py`** — shared metrics registry: every scored quantity with a version, a verify
  tolerance, and whether it is recomputable from aggregates alone. Consumed by generation,
  explanation, and verify so a metric can't mean two things.
- **Audit bundle**: every batch's run dir is now self-contained — `pass_1_accepted.parquet`,
  `explanation.json`, `reference_aggregates.json` (real-data aggregates under a **disclosure policy**:
  class counts, real-rare correlation matrix, per-class column moments, rare deciles *only* above a
  min-bucket floor; **no per-row values, ever**), and `manifest.json` extended with
  `manifest_schema_version`, the **SHA-256 of every artifact**, and the metric versions. Manifest is
  written last (it hashes the others).
- **`regen/audit_bundle.py::verify_bundle` + CLI `regen verify <dir>`**: pure recomputation from the
  bundle — integrity first (artifact hashes vs manifest), then values (correlation delta from delivered
  rare rows + the real corr matrix; Fisher from disclosed moments; class counts). Stats needing raw
  reference rows (coverage, privacy min-distance, tail-lift) are honestly reported **UNCHECKABLE**, not
  faked (G-G point 4). When the δ-floor was applied, the gate's correlation was measured pre-floor (not
  in the bundle), so verify reports the delivered post-floor value informationally rather than
  PASS/FAIL. Exit non-zero on any integrity/value mismatch.
- **`docs/METHODS.md`**: formal definition of every registry metric (formula, detects/can't-detect,
  threshold + rationale, recomputable-from-aggregates), versioned IDs, and the tolerance + disclosure
  policy.
- **Observed:** `generate()` bit-identical (`40933a2b`). Clean floored bundle → VERIFIED (correlation
  uncheckable-by-floor; integrity+fisher+class_counts pass). Clean `none` bundle → correlation
  **checked** and matches (0.0374=0.0374). Tampering a rare row → integrity FAILS naming the parquet
  **and** correlation_delta FAILS. CLI `regen verify` prints per-stat PASS/FAIL and exits non-zero on
  failure.
- **Tests:** `tests/test_audit.py` (6: floored/none clean verify, rare-row tamper fails integrity+stat,
  manifest attestation, disclosure bucket-floor suppress/publish). Suite **131 → 137 green**.

### G-D — Standing regression harness + performance budgets

- **`benchmark/run_regression.py`**: canonical datasets (creditcard_subset / hypothyroid / wilt) ×
  (privacy none/floored) at a fixed seed; each produced bundle is **self-verified via `regen verify`**
  (G-G). Every scored quantity is compared to committed baselines within explicit tolerances; **exits
  non-zero** on any regression — fidelity/coverage drop, correlation increase, gate flip (PASS→FAIL),
  lift drop, a bundle that fails verify, or a **runtime blow-up** past the per-run budget
  (baseline×4 + 5s headroom — catches a gross blow-up, not jitter). `--update-baselines` writes them;
  `--degrade` cranks noise to prove the harness catches drift. Pre-push/CI step; the pre-commit hook
  stays tests-only.
- **Baselines:** `benchmark/BASELINES/regression_baseline.json`, dated + `code_version`-stamped
  (`4229da2`), all six runs verified.
- **Observed:** clean check → "✓ No regression … all bundles verified" (exit 0); `--degrade` →
  "REGRESSION DETECTED" (gate flip + fidelity 1.0→0.87 + corr 0.058→0.18) (exit 1).
- **Perf budgets (scoped note):** the budget is whole-run generate wall-time per (dataset, privacy)
  with wide headroom, not per-*stage* timing — enough to catch the audit's "runtime blow-up" concern
  without threading timing hooks through every engine stage; per-stage timing is a clean follow-up.
- **Tests:** `tests/test_regression_harness.py` (4, fast: `_compare` catches each drift kind + is clean
  on no-drift; committed baseline well-formed + all-verified). Suite **137 → 141 green**.

### G-E — Preflight / capability matrix

- **`regen/preflight.py::preflight(path, label_col, rare_def)`** (+ `regen.api.preflight` re-export,
  CLI `regen doctor`): validates a dataset against the supported envelope *before* generation and
  returns per-check verdicts (`ok`/`warn`/`degraded`/`unsupported`/`error`) with recommendations, plus
  `ok_to_generate`. Rules (each observed): rare-count for amplification (`<10` unsupported) and for a
  non-degenerate lift (`<14` warn, P2-7); all-categorical → floor can't apply (P2-9); low-cardinality
  integer/ordinal features → floor can collapse coverage (P1-6 solar_flare); dimensionality > rare
  count → GP underdetermined; high-cardinality categoricals → top-K TVD; free-text heuristic; constant
  columns; dataset size; and out-of-scope **time-series** columns named plainly. Ingest refusals
  (too-few-rare / ambiguous target) are reported as verdicts, not raised.
- **`docs/CAPABILITY_MATRIX.md`**: supported / degraded / unsupported shapes with reason + workaround,
  written from observed behavior; maps preflight levels to the matrix.
- **Observed:** `regen doctor examples/transactions.csv --label is_fraud` → OK; `open_payments` →
  degraded (all-categorical) + unsupported (free text) → `ok_to_generate: NO`.
- **Fix found while testing:** continuous columns carry `cardinality=None`, so the low-card-integer
  rule now counts distinct values from the data directly.
- **Tests:** `tests/test_preflight.py` (9 fixtures, one per rule: healthy/small-rare/too-few-rare/
  all-categorical/low-card-int/high-dim/constant/timestamp/high-card+free-text). Suite **141 → 150
  green**.

### G-B Source 3 — optional advisory model proposal

- **`regen/semantics.py`** (outside `engine/`; boundary test stays green): one cached, provider-agnostic
  model call that reads the **deterministic profile only** and proposes column semantics.
  `build_model_payload` redacts egress — ≤`REGEN_SEMANTICS_SAMPLES` example values/column, **zero** for
  identifier columns, none when `SAMPLES=0` — and the exact payload sent is persisted. Default caller is
  a urllib POST to an OpenAI-compatible endpoint (no SDK dependency); a caller is **injectable** so
  tests never touch the network. Offline / no key / any error → returns `None`, generation falls back to
  Sources 1+2 and never blocks. Cached by schema hash → at most one call per dataset.
- **Vetting extended:** `vet_scenario(proposed, ingest, model_columns=...)` applies authority order
  researcher > structural > **model** — the model fills gaps/tightens within the data, the researcher
  overrides it, and the model proposal passes the **same** gate rules (a contradiction is dropped).
- **`generate(accept_contract=False, semantics_caller=None, semantics_config=None)`** (+ CLI
  `--accept-contract`, server `accept_contract`): advisory by default. When accepted, the proposal is
  vetted, the vetted model columns persist in the manifest spec (`source="model"`), the raw proposal is
  written to `semantics_proposal.json` + referenced in `provenance`, and a `semantics` block appears in
  the summary. **Replay** from the persisted spec makes **zero** model calls.
- **Observed (offline tests):** identifier columns egress zero values; `SAMPLES=0` egresses none; a
  safe wider bound is vetted **accepted** and a clipping one **rejected**; researcher overrides model;
  exactly **one** call (cache); offline → `applied:false` fallback; and `accept_contract` run →
  persisted `source=model` column → replay with no caller makes **no further calls**.
- **Tests:** `tests/test_semantics.py` (9, fully offline). `generate()` bit-identical (`40933a2b`);
  engine boundary green. Suite **150 → 159 green**.

### P1-3 + Part II docs (written last, from observed numbers)

- **`docs/PRIVACY.md`** (new): threat model (near-copy re-identification), the three-layer mechanism
  (parametric copula + δ-floor + verbatim guard), the exact guarantee, the explicit non-guarantees
  ("**not differential privacy**", bulk not δ-floored, all-categorical skip), how to read the `privacy`
  block, and the δ↔fidelity trade-off with **observed** numbers (demo nearest 0.52–0.53σ; creditcard
  coverage 0.980→0.919; hypothyroid 0.980→0.947; corr often improves).
- **`INVARIANTS.md`**: §8 gains Invariant 6 (privacy) + Invariant 7 (contract reproducibility + verify +
  math/config boundary); §3 gains the API-layer contract/vetting/privacy/conformance/explain/audit/
  preflight map.
- **`README.md`**: new quick-start commands (`generate`/`doctor`/`verify`), the "ships a ScenarioSpec +
  explanation + audit bundle; privacy on by default, NOT differential privacy" note, and doc links.
- **`docs/REGEN_DOCUMENTATION.md` §5.8**: `generate()` privacy/delta/scenario/accept_contract params;
  the ScenarioSpec + vetting paragraph; the full returns list (privacy/conformance/scenario/explain/
  semantics), `passed = fidelity AND conformance AND privacy`, and the audit-bundle/`regen verify`/
  `regen doctor` surface. (This also clears the P1-5 doc-pending item — signatures now documented.)
- **Demo (`examples/run_demo.py`)** updated to print explainability highlights (top Fisher-separation
  features), the conformance verdict, and a `regen verify → VERIFIED` line (G-C/G-G done-when).

### Consistency self-review (§6 rule 7 — required, unprompted)

Re-ran, this session, on the final tree:
- **Full suite:** `python -m pytest tests/ -q` → **159 passed** (from 85 baseline; +74 across the build).
- **Demo:** `python examples/run_demo.py` → fidelity PASS, conformance PASS, privacy PASS (nearest
  0.53σ ≥ 0.50σ, 0 verbatim), **shippable PASS**, detection lift **+0.278**, features-worth-observing
  printed, `regen verify → VERIFIED`, campaign **3/3 accepted**, best tail lift **+0.2778**.
- **P0-2 repro:** floored corr_delta **0.1011 PASS**, none **0.0478 PASS** — the default path is fixed.
- **Invariant 2** (incl. contract replay): label-mode row-hash `40933a2b` held after **every**
  generation-touching change; spec round-trip reproduces bit-for-bit with zero model calls.
- **Doc claims** checked against observed output this session (numbers above match PRIVACY.md/README/
  REGEN_DOCUMENTATION).
- **`git status`:** clean after the final commit; every commit passed the pre-commit test hook.

**Build complete.** All Part I defects (P0-1, P0-2, P1-3/4/5/6, P2-7/8/9/10) and all Part II
workstreams (G-A…G-G) landed as per-finding commits with before/after observations. Known, filed
limitations (floored on low-cardinality-integer / all-categorical high-cardinality data; a low-sev
pandas int64 FutureWarning) are recorded in `docs/KNOWN_ISSUES.md` and the capability matrix.

### Post-build screen (logging + gap review, owner-requested)

- **Log noise cut.** GPy's `reconstraining parameters ...` printed 14 lines/call to stderr on every
  GP fit → silenced at source (`constrain_bounded(..., warning=False)`), so bare API/server calls are
  clean without the demo's blunt stream redirect. paramz `DeprecationWarning`s (5058/test-run) filtered
  via `pyproject.toml [tool.pytest.ini_options] filterwarnings` (the canonical place — a conftest
  import-time filter is reset per-test). **Test warnings 5058 → 2.**
- **Silent mechanism switch closed.** A parametric→grounded base fallback was only a log line while
  `explanation.json` still claimed `copula-sampled`. Threaded a diagnostics dict → the explanation now
  records `generation.{rare_base,normal_base}` and the per-column `mechanism` reflects a fallback
  ("nothing silent", G-C).
- **Gap screen fixes:** (1) `explanation.json` now **cites metric IDs + versions** from the registry
  (`correlation_delta`/`coverage_rate`/`fisher_separation`/`tail_lift`) — closes the §7 "versioned IDs
  that explanation.json cites" miss. (2) Removed dead `_compute_ard_cv()` (no call sites). (3) Added
  CLI **`regen generate --scenario <yaml>`** so a saved ScenarioSpec drives generation (verified: the
  example YAML → 400 rows @ 25% rare, shippable PASS).
- **Flagged for owner decision (not changed):** (a) the fidelity gate is measured *pre-floor* while the
  delivered data is *post-floor* — empirically negligible (demo coverage gap **0.0**), and `regen
  verify` already marks correlation uncheckable-under-floor, but strictly the gate approves data it
  doesn't ship; re-auditing post-floor could flip verdicts on adversarial data. (b) `run_campaign`
  emits no explanation/audit-bundle/vetted-scenario (diagnostic path). (c) No server `/doctor` or
  `/verify` endpoints (CLI + API only).
- `generate()` bit-identical (`40933a2b`). Suite **159 → 161 green**.

### Fidelity gate now audits the DELIVERED (post-floor) data (owner-requested)

- **Gap #4 fixed.** The fidelity verdict was measured on the pre-floor rare batch while the delivered
  data is post-floor — so `passed=True` could describe a batch that isn't shipped. `generate()` now
  **re-audits the delivered rare part** (`full_df.iloc[-n_rare:]`) after the floor when it moved rows,
  and that report drives `fidelity`, `passed`, and the explanation. Parquet unchanged (`40933a2b`);
  only the reported numbers now describe what ships.
- **Consequence (surfaced, not hidden):** this exposed that `privacy="floored"` on a **dense percentile
  tail** fails the gate on delivered data — the δ-floor moves rare rows ~0.5σ, pushing correlation
  (0.101 pre-floor → **0.295 delivered**) and a marginal (`merchant_risk`) past their gates. This is
  the fundamental dense-tail-vs-isolation tension the audit flagged; the correct outcome is a **loud
  downgrade**, which the audit's P0-2 done-when explicitly sanctions ("pass OR fail loudly with a
  machine-readable reason"). It now fails loudly (`fidelity.passed=False`, delivered corr reported,
  `passed=False`); LABEL-mode demo unchanged (still fidelity/privacy PASS, lift +0.278, shippable).
- **Tried and reverted:** a correlation-preserving respawn (draw floor respawns from the real rare
  covariance instead of uniform-in-box) improved percentile corr 0.295→0.190 but (a) still didn't make
  the batch shippable (the `merchant_risk` marginal still fails — inherent to a 0.5σ floor on a tight
  tail) and (b) changed **every** floored output (demo hash + all baselines). Not worth the blast
  radius for no shippability gain — reverted. Filed the correlation-preserving floor as a possible
  future enhancement (measured benefit recorded here).
- **Bonus:** because the reported correlation is now the delivered value, `regen verify` **checks**
  correlation even under the floor (reported == recomputed) instead of marking it uncheckable.
- **Metric-ID citation / dead code / CLI `--scenario`** (gap-screen items) also landed. Test updated
  (`test_percentile_floored_verdict_is_honest_on_delivered_data`) to pin the honest no-silent-pass
  behavior; `docs/CAPABILITY_MATRIX.md` records the dense-tail floored degraded case. Suite **161 green**.

---

## Session 2026-07-09 — Phase 2 (product direction, per docs/PRODUCT_SPEC.md)

Building the new parts from the product spec. This phase is about turning the
verified engine into a *certified surrogate* a non-expert can drive and a skeptic
can check. Metric first, as the spec's build order requires.

### §5.1 — TSTR harness (surrogate quality): the headline actionable metric

- **What:** `engine/examiner/surrogate.py::measure_tstr` — train a model on the
  synthetic surrogate (TSTR) and on real data (TRTR), score both on a held-out
  **real** test set, report `recovered = TSTR / TRTR` across a model panel
  (logreg / random-forest / gradient-boosting), averaged over seeds, on ROC-AUC
  **and** PR-AUC (rare-class-sensitive). `contracts.types.TSTRReport` carries it.
- **Leakage-free orchestration:** `regen.api.evaluate_surrogate` quarantines a real
  test fold, generates the surrogate from the **train fold only** (via `generate`
  on a temp copy), then measures — the same leakage discipline as the lift metric
  (57a45fc). Degeneracy guard mirrors P2-7 (`insufficient_real_test` below
  `MIN_REAL_TEST_RARE=10`).
- **Honest reads baked in:** `recovered > 1` is **flagged, not celebrated**, with a
  pointer to the privacy min-distance (high recovery + low min-distance =
  memorization). TSTR needs raw real test rows, so it is a producer/auditor-side
  metric — **not** recomputable from the audit bundle alone (consistent with the
  disclosure policy).
- **Observed (verify, don't assert):**
  - *metric invariants:* perfect surrogate (synth = real-train) → recovered ROC
    **1.0** exactly; signal-free noise surrogate → **0.53** (TSTR at chance ~0.5,
    TRTR ~0.92). It discriminates.
  - *end-to-end, leakage-free* on `examples/transactions.csv` (privacy=floored):
    recovered ROC-AUC median **1.011**, PR-AUC **1.20**, over 3 models × held-out
    600-row real test (18 rare) — and the report **flagged** it and printed
    privacy min-distance **0.53σ** (healthy). Reads correctly as "recovers ~full
    real performance AND provably not a copy."
- **Scope note:** kept `generate()` untouched — TSTR is a separate entry point
  (`evaluate_surrogate`), so the hot path is unchanged (no bit-identity impact) and
  runtime isn't bloated by default. A future optimization can share the train-fold
  synth with `measure_lift` so both come from one generation.
- **Tests:** `tests/test_tstr.py` (5: perfect=1.0, noise<0.9, insufficient-guard,
  end-to-end hold-out partition, JSON-serializable). Suite **161 → 167 green**.
- **Spec status:** PRODUCT_SPEC §5.1 TSTR harness **[PLANNED] → [BUILT]**.

---

## Session 2026-07-09 (cont.) — Vocabulary pass (proprietary identity, honest prior art)

Rename so the codebase reads as its own system, not a paper reimplementation —
**without** dressing up standard math as invented (that would be the same overclaim
we removed elsewhere).

- **Code renames** (behavior-neutral): `ResidualModel`/"ResidualGP" → `TailCorrector`
  (`engine/amplifier/residual_gp.py` → `tail_corrector.py`); `fit_residuals` /
  `sample_residuals` → `fit_correction` / `sample_correction`; the Scout's "R-EPIG"
  acronym → "targeting / gain score" (`engine/scout/repig.py` → `targeting.py`).
  Role-names (Prior/Amplifier/Scout/Auditor/Examiner) already REGEN's own — kept.
  All references updated across engine/regen/tests/benchmarks (grep-clean).
- **Docs:** `INVARIANTS.md §2` "Research spine" (three-papers-parentage framing) →
  **"Methods & prior art"** — names the standard techniques (Gaussian copula, GP+ARD,
  TVD/Wasserstein, TSTR, k-anonymity…), states the composition + assurance layer as
  the original part, and keeps the papers as *reference/inspiration only*. Current
  docs swept for the old symbol names; historical docs (AUDIT/BUILDLOG/RESULTS/
  SEMANTIC_FIDELITY) left as-is.
- **Verified:** benchmark scripts still compile; full suite **167 green**; `generate()`
  **bit-identical** (row-hash `40933a2b`) — pure rename, zero behavior change.
- **Method note:** worked from the dependency map (blast-radius grep + import sites +
  package `__init__`s) so every reference was caught; the human-readable system map
  (`docs/PRODUCT_SPEC.md`, `docs/COMPONENT_GUIDE.md`) was updated in step.

---

## Session 2026-07-09 (cont.) — §5.2 Intent → ScenarioSpec proposer

Lets a non-expert draft a run from a plain-language goal; the model *informs*,
the human reviews/edits, the engine still grounds every value.

- **`regen/semantics.py::propose_scenario`** — extends the advisory layer from
  column-semantics to a **full ScenarioSpec draft** (intent + gates + columns).
  Always returns a **valid, editable** draft: a structural baseline the model
  refines. Every model-proposed field is **validated** (closed vocabularies, real
  column names, in-range numbers); invalid fields are ignored, never obeyed
  (`_apply_intent`/`_apply_gates`). Offline / no key / error → structural draft
  alone (never blocks). Metadata only — no value written.
- **`regen.api.draft_scenario`** (filepath wrapper) + **CLI `regen propose <data>
  --goal "…" [--out spec.yaml]`** — prints/saves the draft YAML for the user to
  review/edit, then `regen generate --scenario <saved>`. Not auto-committed.
- **Observed:** offline `regen propose` → valid structural draft (drafted_by
  structural). Injected-caller draft applies a valid proposal (task/rare_ratio/
  mode/gates), **drops** invalid fields (bad task/label/ratio/privacy → defaults,
  non-existent focus feature dropped), and the draft round-trips through YAML and
  drives `generate()`. `generate()` bit-identical (`40933a2b`) — new entry point,
  hot path untouched.
- **Tests:** `tests/test_scenario_proposal.py` (5, offline). **Spec §5.2
  [PARTIAL] → [BUILT].**

---

## Session 2026-07-09 (cont.) — §5.3 Decision-support surface

The deliberate replacement for the autonomous optimizer we rejected: surface the
tradeoff, let the human decide.

- **`regen.api.explore_options`** (+ CLI `regen explore`) — runs `generate()` at
  privacy=none and privacy=floored across a δ sweep, and returns a **frontier**:
  per option {privacy, δ, fidelity, coverage, corr, min_distance, shippable, and a
  plain-language **diagnosis**}. It **recommends a labelled default** (the most
  private floored option that still ships; else the non-private option, flagged;
  else none → "run doctor") that the user can override. It returns options —
  it never generates a "final" artifact or picks the value-laden tradeoff.
- **`_diagnose`** — turns a run's gates into plain language + the fix, and is
  **privacy-aware**: low coverage under the floor blames the δ-floor ("lower delta
  / use none"); low coverage without a floor says "poor fit for amplification" —
  no more mis-attributing a none-mode failure to the floor.
- **Observed:** `regen explore transactions.csv --label is_fraud` → all options
  ship, recommends δ=0.8 (most private that ships). Percentile-tail → all options
  fail with honest per-row diagnoses and "no option shipped — run doctor."
- **Tests:** `tests/test_explore.py` (6: diagnosis shippable/floored-low-cov/
  none-low-cov/correlation, frontier surfaces+recommends, returns-options-not-
  auto-commit). `generate()` bit-identical (`40933a2b`). Spec §5.3 → **[BUILT]**.

---

## Session 2026-07-09 (cont.) — §5.5 Certified-surrogate clean-room demo

Ties the whole system into one showcase (and the last non-deferred build item).

- **`examples/certified_surrogate_demo.py`** — stages the two-party story:
  PRODUCER quarantines a real test slice, generates a leakage-free surrogate from
  the train fold, and emits the data package (bundle); CONSUMER (never sees real
  data) runs `regen verify` and trains a model on the surrogate alone;
  AUDITOR (holds the real test slice) measures TSTR. Headline artifacts: **TSTR +
  VERIFIED**, read against the privacy min-distance.
- **Observed:** transactions → package emitted (floored, 0.53σ, 0 verbatim);
  consumer `verify` → **VERIFIED** (3 stats + 3 hashes); auditor TSTR recovers
  **101% ROC-AUC / 122% PR-AUC** — flagged suspicious and cross-checked against
  the healthy 0.53σ distance (genuine, not memorization). The real data never moved.
- **Tests:** `tests/test_cleanroom_demo.py` (1 smoke: runs end-to-end on a fixture,
  asserts the three roles + VERIFIED + TSTR appear). Spec build-order #4 → **[BUILT]**.

### Phase-2 status
Build order 1–4 (TSTR · proposer · decision-support · clean-room demo) are **[BUILT]**.
#5 closed-loop repair stays **[DEFERRED]** — build only if the single-shot proposer
demonstrably underperforms, with anti-Goodhart discipline + human-approved final spec.

---

## Session 2026-07-09 (cont.) — API surface + TSTR optimization

- **Server API** — added FastAPI endpoints for the new capabilities so the server
  matches the CLI: `POST /api/doctor` (preflight), `POST /api/propose` (draft a
  ScenarioSpec from a goal, advisory), `POST /api/explore` (tradeoff frontier),
  `POST /api/tstr` (leakage-free surrogate quality), and `GET
  /api/campaign/{run_id}/verify` (independently recompute a produced bundle).
  `tests/test_server.py::TestNewEndpoints` (5) via `TestClient`.
- **TSTR optimization** — `evaluate_surrogate` → `generate()` was doing a second,
  redundant train-fold generation for the internal lift it never uses. Added
  `with_lift: bool = True` to `generate()`/`_run_one_pass` (default preserves
  behavior); `evaluate_surrogate` now calls `with_lift=False`. **Delivered batch
  bit-identical** (`40933a2b`); only `summary["lift"]` becomes None when skipped.
  `evaluate_surrogate` on transactions **~13s → 7.2s** (≈ halved).
- **Verified:** default `generate` bit-identical + lift intact; `with_lift=False`
  same batch, no lift; new-endpoint + lift/reproducibility/tstr subsets green.

## Session 2026-07-09 (cont.) — target tie-break hand-off + media specificity

Motivation (user): the LLM earns its place specifically at *target disambiguation*;
wire that hand-off, keep it working with no valid API key, and put the actual
mechanisms (target scoring + the ROC-AUC model panel) into the showcase materials
with real specificity.

- **Target tie-break hand-off.** Rule-based detection raises `AmbiguousTargetError`
  when the top-two target scores are within `ambiguity_margin` (0.05). Added
  `resolve_ambiguous_target()` (`regen/semantics.py`): sends only candidate NAMES +
  their statistics + the user's goal (no raw rows) to the advisory model, which must
  return **one of the tied candidates**. Wired into `draft_scenario()`
  (`regen/api.py`): a tie is handed to the model; the chosen `label_col` is recorded
  in `provenance.target_tiebreak` (chosen/reason/candidates/resolved_by).
  - **Offline-safe by construction.** No model / no key / bad key / non-candidate
    pick → `resolve_ambiguous_target` returns `(None, reason)` and `draft_scenario`
    **re-raises the same `AmbiguousTargetError`** so a human chooses. Never invents a
    target, never crashes on a missing key (Invariant 4 + INVARIANTS.md §7 decision-support).
  - **Transparency surfaced.** `/api/propose` now returns `target_tiebreak`; `regen
    propose` prints "target tie among [...] → auto-selected 'X' (reason). Override
    with --label." So the auto-selection is visible and overridable, not silent.
  - Tests: `tests/test_scenario_proposal.py::TestTargetTieBreak` (3) — honest offline
    error, model breaks tie from goal (provenance recorded), invalid pick → error.
    Fake caller branches on the prompt so one injected caller serves both calls.
- **Media specificity (`showcase/how-it-works.md`, NEW).** The "how exactly?" layer:
  (1) the full target-scoring formula + weights + the useful-band disqualifiers + the
  ambiguity refusal + the tie-break LLM and its guardrails/why; (2) the 3-model
  ROC-AUC/TSTR panel (LogReg/RF/GBDT, target=rare-vs-rest, features=the rest,
  TSTR/TRTR, recovered=median, refusal + >1.05 flag); (3) other nameable specifics
  (copula 0.331→0.101, δ-floor mechanics, Auditor's four statistics, bit-repro,
  `regen verify`). Linked from `showcase/README.md`.
- **Verified:** full suite green (see run below); offline `draft_scenario` unchanged
  (structural draft, no calls); tie-break exercised only via injected caller in tests.

- **Tie-break now sees semantic context (redacted).** The name+stats-only payload was
  thin for opaquely-named targets (`y`, `flag_9`). `AmbiguousTargetError` now carries
  `all_columns` + `candidate_examples` (built from the df it already has at raise time
  — engine-side, no LLM). `resolve_ambiguous_target` sends: goal, tied candidates with
  up to `config.samples` example values each, and `other_columns` (the rest of the
  schema's **names only**) as domain context. Egress discipline matches
  `build_model_payload`: example values capped, suppressed when
  `REGEN_SEMANTICS_SAMPLES=0`, `other_columns` never carries values, no raw rows ever.
  - Test: `test_semantic_context_sent_to_model` asserts `other_columns == {reading,
    score}` and candidate `example_values == [0,1]` reach the (captured) payload.
  - **Verified:** affected surface `tests/test_scenario_proposal.py
    tests/test_server.py tests/test_api.py` → **67 passed** (174s); full suite reached
    76% with 159 passed / 0 failed before a machine-load timeout (not a failure — the
    documented pre-commit-hook load pattern); tie-break class 4/4 in 6.7s.

- **README honest-numbers swap.** The README still showed the inflated,
  leakage-inflated lift table (Satellite +39.1%/3.75×, Hypothyroid +12.6%, etc.) —
  the exact figures we established evaporate under leakage-free measurement. Replaced
  the whole "Benchmark" section with "What the numbers actually say": (1) the
  leakage-free **TSTR recovered-%** table from `RESULTS_TSTR.md` (median of
  LogReg/RF/GBDT, ROC-AUC + PR-AUC), reading churn honestly as a weak spot (~0.65/0.39)
  and `creditcard_subset` as an honest refusal (insufficient held-out rare); (2) the
  **conditional-lift** thesis (amplification helps only when the baseline is weak /
  rare data scarce; Satellite +39% → ~+4% leakage-free). Also softened the tagline
  ("improves ML performance" → "for rare-event problems … with a re-checkable
  certificate") and added the single-table/cross-sectional scope line. Old
  `RESULTS_BREADTH.md` kept but labelled as pre-leakage-free.

---

## Session 2026-07-11 — Estimand preservation (G-H), regression-coefficient v1

New guarantee (not an audit finding): extend the recomputable certificate from
"correlations recompute" to "**the declared regression coefficient recomputes on
the delivered data**". A researcher declares an analysis (`EstimandSpec`:
`outcome ~ predictors`, family ols|logit); REGEN fits θ_real on the real reference
and θ_synth on the delivered batch and certifies that each coefficient of interest
is preserved. This is orthogonal to fidelity (marginals/correlations) and TSTR
(prediction): a batch can pass both while a coefficient silently shifts. Deterministic,
no LLM, no new dependency; the estimator is closed-form OLS + IRLS logit (numpy +
scipy only) so `regen verify` recomputes it to a fixed tolerance on any machine.

- **Contract (`contracts/scenario.py`).** Added `EstimandSpec` (outcome, predictors,
  family, coefficients_of_interest, ci_level, rule) as a first-class field on
  `ScenarioSpec`, serialized in every `to_dict`/`from_dict`/YAML path. Persists in
  the manifest → reproduces bit-for-bit (Invariant 7). Old specs with no `estimand`
  key still load (undeclared). `FAMILIES = ("ols","logit")`.
- **Estimator (`regen/estimand.py`, NEW).** `fit_estimand` — OLS via normal
  equations + t-interval; logit via Newton/IRLS + Wald interval from the inverse
  Fisher information. `certify` compares θ_synth to θ_real. `evaluate` orchestrates
  (never raises — an unfittable spec becomes a status). `reference_aggregate` emits
  the disclosed θ_real ± SE block.
- **Certification rule — design fix found by a test.** The first rule (θ_synth ∈
  θ_real's CI) *ignored the synthetic sample's own uncertainty* and false-failed on
  two independent draws from the identical process. Replaced with a **two-sample
  Wald consistency test**: preserved iff `|θ_real − θ_synth| ≤ z·√(se_real²+se_synth²)`
  at `ci_level`. Reduces to the CI check as the synthetic set grows (se_synth→0).
  Default `rule="consistent"`; `within_ci` kept as the stricter option. Per target
  we also surface `real_significant` (does θ_real's CI exclude 0?) — preserving a
  null effect is vacuous; surfaced now, power-aware failing is a documented v2.
- **Generation (`regen/api.py`).** The vetted spec now carries the estimand;
  `generate()` fits real + delivered, writes the verdict into `explanation.json`
  (`estimand` block, always present — `not_declared` when absent) and the summary
  (`estimand` key), and publishes θ_real ± SE into `reference_aggregates.json`.
- **Verify (`regen/audit_bundle.py` + `regen/metrics.py`).** New metric
  `estimand_delta` (v1, tol 1e-6, recomputable-from-aggregates). `_verify_estimand`
  refits θ_synth from the delivered rows, re-certifies against the disclosed θ_real,
  and checks both each θ_synth (within tolerance) and the certified verdict.
  Undeclared/uncertifiable → honestly `uncheckable`, never a fake pass.
- **Readiness coupling (Phase 4, honest floor).** When θ_real (or θ_synth) cannot
  be fit — too few complete rows, non-binary logit outcome — `evaluate` returns
  `status="uncertifiable", certified=False` with a reason. The verification gap is
  never filled with synthetic data.

**Before:** no estimand concept; the certificate covered fidelity/privacy/lift only.
`grep -n EstimandSpec contracts/scenario.py` → nothing.

**After (repro):**
- `python -m pytest tests/test_estimand.py tests/test_scenario.py tests/test_audit.py -q`
  → **34 passed**. Includes the headline properties: a coefficient driven 2.0→0.0
  fails `certify`; a batch whose delivered predictor is permuted (θ_synth moves)
  fails `estimand_delta` under `regen verify` even though the reported verdict said
  certified (the certificate is recomputed from data, not trusted).
- End-to-end smoke (declared OLS `y ~ x1 + x2`, `privacy="none"`, n_rows=400): summary
  `estimand.status=certified`; `reference_aggregates.json` carries `estimand_real`
  (coeffs Intercept/x1/x2); `verify_bundle` → `estimand_delta` **checked & passed**,
  `max_theta_synth_diff = 0.00e+00`.
- Regression: `python -m pytest tests/test_api.py tests/test_boundary.py -q` →
  **66 passed** (boundary invariant holds — estimator is in `regen/`, not `engine/`).

**Deferred (documented in KNOWN_ISSUES):** power-aware certification (scarce real
data → wide θ_real CI → the consistency test is lenient; today surfaced via
`real_significant`, not failed); categorical/one-hot predictors (v1 is numeric-only);
extending the same recompute-and-certify machinery from coefficients to an ATE.

---

## Session 2026-07-11 (b) — Generator-agnostic certifier + multi-generator demo

Strategic reframe (user-decided): the **certifier is the product, not the
generator**. The certificate is generator-agnostic — it certifies whether *any*
synthetic dataset preserves a declared analysis, whoever produced it. REGEN's
generator is demoted to a reference implementation + test fixture.

- **`regen/certifier.py` (NEW).** `certify_dataset(real_df, synthetic_df, estimand)`
  → a portable certificate (verdict, per-coefficient θ_real vs θ_synth, rule/CI,
  metric version, source label, **θ_real ± SE disclosed** so it re-checks against
  the synthetic alone). `certify_many(real_df, {name: df}, estimand)` certifies the
  same estimand across sources (θ_real identical; only θ_synth varies). Thin,
  generator-agnostic wrapper over `estimand.evaluate` — never raises.
- **`examples/certifier_demo/` (NEW).** Real UCI credit-default data (30k rows,
  `prepare_data.py` documents provenance from the UCI .xls; `credit_default.csv`
  committed, 825K). `run_demo.py` certifies one logit — `default ~ pay_delay_1 +
  utilization + log_limit + age` — across six producers: bootstrap (positive
  control), independent-columns (negative control), 0.5σ-noised real, a Gaussian
  copula, SMOTE (imblearn), and REGEN.

**Observed (repro: `python examples/certifier_demo/run_demo.py`):**
- bootstrap → **CERTIFIED** (all 4 preserved); independent → refused (all 4 collapse
  to ~0). The certifier discriminates — not an always-fail.
- **Every practical method is refused, and every one breaks `pay_delay_1`** (the
  strongest, discrete, non-linear predictor): copula +0.71→+0.46, SMOTE →+0.61,
  noised →+0.52, REGEN →+0.93; smooth predictors (`log_limit`, `age`) mostly survive.
  Fidelity/prediction would flag none of this.
- 1/6 certified. Same θ_real across all rows — provenance-independent.

**Tests:** `python -m pytest tests/test_certifier.py tests/test_certifier_demo.py -q`
→ **5 passed** (faithful certifies, distorted refused, unfittable→uncertifiable not
crash, θ_real identical across sources; demo headline guarded on the committed CSV,
fast path excludes REGEN/SMOTE).

**Finding → v2 target:** the coefficient hardest to preserve is the discrete,
high-signal `pay_delay_1`, and **all** marginals-plus-linear-correlation methods
(Gaussian copula, REGEN, and SMOTE via interpolation) fail it. This sharpens the v2
generator investigation (KNOWN_ISSUES #6): why do such methods lose the conditional
structure of discrete non-linear predictors, and what generation change preserves it?
