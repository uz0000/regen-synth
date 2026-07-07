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
