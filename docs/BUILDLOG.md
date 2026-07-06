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
- **Commit:** _(below)_
