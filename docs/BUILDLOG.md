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
