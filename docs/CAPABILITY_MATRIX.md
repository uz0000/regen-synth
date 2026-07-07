# Capability matrix (G-E)

What REGEN supports, degrades on, and does not support — written from observed
behavior (the degraded rows came out of the P1-6 privacy sweep). `regen doctor
<data>` / `regen.api.preflight(...)` check a dataset against this envelope
*before* generation and report actionable verdicts, so the edges fail loudly
instead of producing a surprising batch.

## Data shapes

| Shape | Status | Notes / workaround |
|---|---|---|
| Single flat table (CSV/JSON/Parquet), tabular | **Supported** | The design target. |
| Continuous + binary features, ≥ ~14 rare rows | **Supported** | Full fidelity + floored privacy + honest lift. |
| Categorical features (low/medium cardinality) | **Supported** | Frequency-sampled; verbatim guard + k-anonymity. |
| High-cardinality categoricals (> 50) | **Degraded** | Top-K TVD path; per-category fidelity is approximate. |
| All-categorical (no continuous features) | **Degraded** | δ-distance floor **cannot apply** (P2-9); privacy = parametric sampling + verbatim guard + k-anonymity. Floored fidelity can drop on high-card categoricals (P1-6 `open_payments`). Use `privacy="none"` if fidelity matters more. |
| Low-cardinality **integer/ordinal** "continuous" features | **Degraded** | The δ-floor can collapse coverage (P1-6 `solar_flare`). Use `privacy="none"` or declare them categorical. |
| Very few rare rows (< 14) | **Degraded** | Amplifies, but the held-out lift reports `insufficient_rare_rows` (P2-7). |
| Very few rare rows (< 10) | **Unsupported** | Below the amplification minimum; ingest refuses. |
| High dimensionality vs rare count (D > n_rare) | **Degraded** | Residual GP underdetermined; set `max_features` 6–10. |
| Time series (temporal order matters) | **Unsupported** | Rows treated as exchangeable; temporal structure is **not** modeled or preserved. A `timestamp`-role column is reserved in the ScenarioSpec but not yet used. |
| Relational / multi-table (foreign keys) | **Unsupported** | Single-table only; no FK/topology modeling. |
| Free text | **Unsupported** | Long, near-unique string columns are not modeled — drop or encode them. |
| Images / audio / other non-tabular | **Unsupported** | Out of scope. |

## Privacy envelope (summary — see docs/PRIVACY.md, docs/METHODS.md)

- The δ-distance floor is a **rare-vs-rare** guarantee on **continuous** features.
  It is skipped (loudly, `floor_applied=false`) when there are no continuous
  features; the verbatim guard + k-anonymity still hold.
- It is **not** differential privacy — it prevents near-copy re-identification, not
  membership-inference or aggregate attacks.

## How preflight levels map here

`ok` → Supported · `warn`/`degraded` → Degraded (runs, with caveats) ·
`unsupported`/`error` → Unsupported (preflight sets `ok_to_generate=false`).
