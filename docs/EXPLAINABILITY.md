# Explainability — `explanation.json` (G-C)

Every `generate()` batch ships an `explanation.json` next to its `manifest.json`
(and the same object is returned in the summary under `explain`). It answers, for
a researcher: **what did I get, why should I trust it, and what should I look at?**

**Every value is computed from the run** — from the fidelity / privacy /
conformance / lift report objects, the vetted `ScenarioSpec`, the Scout target,
and class-separation statistics. Nothing here is narrated by a model. If
human-language narration is ever layered on top, it is a model call *outside*
`engine/` that may only **cite** these numbers (INVARIANTS.md §4); this JSON is the
ground truth. A test (`tests/test_explain.py`) asserts the numbers equal the
report objects they came from.

## Fields

### `gates`
Per-gate account — each check with its statistic, threshold, and verdict.
- `fidelity.coverage`: `{value, threshold, passed}` — fraction of real rare rows covered.
- `fidelity.correlation`: `{value, threshold, passed}` — mean abs correlation-matrix
  delta (`value` is `null` when there are too few numeric columns to estimate).
- `fidelity.columns`: `{n_passed, n_total}` per-column marginal checks.
- `normal_fidelity`: the same for the synthetic normal half (coverage is n/a there).
- `conformance`: `{passed, n_checked, violations[]}` — the vetted-contract gate
  (G-B rule 9). Each violation is `{column, violation, n_rows}` (counts only, no values).
- `privacy`: the privacy block (below), or `null` when `privacy="none"`.

### `feature_informativeness`
The "features worth observing" answer.
- `method`: the statistic used — class-separation Fisher score
  `(μ_rare − μ_normal)² / (σ²_rare + σ²_normal)`, computed from the real data.
- `ranked`: `[{feature, fisher_score, rank}]`, highest separation first. This is
  where the rare-vs-normal signal concentrates.

### `column_provenance`
Per column: `role`, `source` (who set its semantics — structural / user / model),
`mechanism` (how its values were produced — `copula-sampled + GP tail correction`,
`copula-frequency-sampled`, `grounded-sampled`, `identifier-minted`,
`label-attached`), `constraints_applied` (vetted attributes that took effect), and
`constraints_rejected` (proposals the vetting gate dropped, with the rule and
rationale — G-B rule 7).

### `scout`
The chosen target region (feature band / percentile) and any scalar R-EPIG
metadata the Scout recorded.

### `utility`
Detection lift with honesty markers (P2-7):
`{status, n_test_rare, baseline_recall, amplified_recall, tail_lift, protocol}`.
`tail_lift` is `null` when `status = insufficient_rare_rows` (the held-out rare
fold was too small to trust); `status = not_measured` when the batch failed the
fidelity gate (lift is only measured on a passing batch). `protocol` states the
leakage-free train/test discipline.

### `generation`
`{rare_base, normal_base}` — which base generator actually ran for each part
(`"parametric"` or `"grounded_fallback"`). Under `privacy="floored"` the parametric
copula is used; if it can't fit (a degenerate class) generation falls back to
grounded sampling, and that switch is recorded here **and** reflected in each
column's `mechanism` — a mechanism change never hides in a log.

### `privacy`
The privacy account: `{mode, delta, floor_applied, floor_skip_reason, min_distance,
distance_p50, n_verbatim_duplicates, passed, note}`. `floor_applied=false` (with a
`floor_skip_reason`) says plainly when no δ-shell was carved and what protection
remains. NOT differential privacy — see `docs/PRIVACY.md`.

## Rule for maintainers

Any new gate, generator mechanism, or metric **must** add its computed entry here
and to `build_explanation()` in the same change (BUILDLOG protocol §6 rule 10) —
an unexplained check is treated as an undocumented one.
