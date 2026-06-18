# regen
### synthetic data generation platform · v0.4 specification

---

## What REGEN Is Trying to Do

Machine learning models fail on the cases that matter most — fraud that looks slightly different from past fraud, a rare disease presentation that never appeared in training data, a security event that sits just outside known attack patterns. This happens because training data reflects history, and history underrepresents tail events by definition. You cannot fix this by collecting more data: rare events are rare. You fix it by generating synthetic data that is statistically grounded in the real distribution but deliberately amplified at the tail.

REGEN's goal is to produce synthetic tabular datasets where rare, high-stakes events are represented accurately enough that a model trained on them will recognize those events in production. The output must be statistically faithful — not invented, but derived from the real distribution with principled deviation. And the output must be explainable — a domain expert needs to understand what was generated and why before they will trust it enough to train on it.

---

## Core Design Principle

**Statistical models do the generation. The LLM does the translation.**

The LLM never invents data. It receives structured, statistically grounded output from the statistical pipeline and renders it into language a human can evaluate. Every claim in the narrative must be traceable to a number in the statistical output. This constraint is what makes the system trustworthy rather than just fluent.

---

## Differentiation

|  | Gretel / Mostly AI | REGEN |
|---|---|---|
| Returns synthetic records | ✓ | ✓ |
| Models tail / rare event distribution | — | ✓ |
| GP uncertainty per generated row | — | ✓ |
| Plain-language narrative grounded in stats | — | ✓ |
| Domain expert can verify output before training | — | ✓ |

---

## Component Specification

---

### Component 01 — Data Ingestion and Schema Mapping
`entry point`

**What it does**

Takes a real dataset and produces everything downstream needs: a typed field dictionary, a normal events baseline, and an isolated rare events set. Identifies relational structure if present and encodes it as a schema graph. The rare event definition is configurable — by explicit label, class imbalance ratio, or tail percentile on a continuous target variable.

**Input**
Raw tabular file (CSV / dataframe / JSON) + metadata: label column, rare event definition

**Output**
Typed field dictionary · normal events dataframe · rare events dataframe · schema graph

**Success condition**
Two clean labeled dataframes with consistent schema, no missing values, and a field dictionary the user can inspect and correct.

---

### Component 02 — Prior Fitting — Normal Baseline
`statistical`

**What it does**

Learns the statistical fingerprint of normal activity. A prior-fitted network (PFN-style transformer) is trained on the normal events dataframe and produces calibrated probability estimates with minimal data. If relational structure is present, a graph neural network propagates latent states across table relationships so each row's score reflects its connected context, not just its own values.

**Input**
Normal events dataframe · schema graph

**Output**
Fitted prior model · `predicted_normal` probability scores on full dataset

**Success condition**
The model produces visibly different probability scores on held-out normal events versus known rare events.

---

### Component 03 — Residual Learning — Tail Deviation
`statistical`

**What it does**

Learns how rare events deviate from the normal prior. Residuals are computed as the difference between actual rare event feature values and what the prior model predicted as normal. A Gaussian Process with ARD kernel is fit on these residuals — learning which features drive deviation. R-EPIG acquisition scores candidate samples by information gain on the tail, prioritizing the most informative rare event candidates for amplification.

**Input**
Rare events dataframe · `predicted_normal` scores from Component 02

**Output**
Fitted GP with posterior mean and variance · R-EPIG scores per candidate · per-feature relevance weights

**Success condition**
GP produces higher uncertainty on unseen event types and lower uncertainty on patterns consistent with known rare events. R-EPIG scores discriminate meaningfully between informative and redundant candidates.

---

### Component 04 — Synthetic Scenario Sampler
`generative`

**What it does**

Combines prior and residual models to generate synthetic records grounded in the real distribution but amplified at the tail. Samples from the GP posterior to produce a residual vector, adds it back to the normal baseline, and repeats N times. For relational data, schema graph constraints ensure generated rows respect table relationships. A tone conditioning layer enforces distributional coherence across the batch — R-EPIG scores modulate this: high-stakes rare candidates get stricter enforcement.

**Input**
Fitted GP · fitted prior · schema graph · R-EPIG scores · target batch size

**Output**
Synthetic records batch · GP posterior mean and variance per row · R-EPIG score per row

**Success condition**
Synthetic records resemble real rare event telemetry, with feature distributions matching empirical statistics from training data.

---

### Component 05 — Fidelity Validator
`validation`

**What it does**

Compares the generated synthetic batch against the real data distribution before returning output to the user. Computes TVD per column, Wasserstein distance per continuous column, and rare event coverage rate. The coverage rate is the primary metric — a batch can pass TVD on bulk columns and still fail if the tail is not adequately covered. That failure is surfaced explicitly. Failing batches are returned with a warning or fed back to the sampler with tighter constraints.

**Input**
Real data distribution (from Component 01) · synthetic batch · configurable thresholds

**Output**
Fidelity report: TVD per column · Wasserstein per continuous column · rare event coverage rate · per-metric and overall pass/fail

**Success condition**
Validator reliably distinguishes high-fidelity from low-fidelity batches, validated against a small set of manually labeled examples.

---

### Component 06 — Narrative Generator
`llm layer`

**What it does**

Takes the synthetic batch and its statistical metadata — GP posteriors, R-EPIG scores, fidelity report — and produces a plain-language explanation of what was generated and why it is statistically meaningful. The LLM receives structured numerical context and must reference specific features and deviation magnitudes. Generic narrative that cannot be traced back to input statistics is a failure mode. The LLM translates; the statistical pipeline generates.

**Input**
Synthetic batch · GP posterior mean/variance per row · R-EPIG scores · fidelity report · domain context string

**Output**
Batch summary (1 paragraph) · per-row annotations for high-scoring rare event candidates · training value statement

**Success condition**
A domain expert can trace every narrative claim to a specific row and a specific statistical value in the input.

---

## Build Order

Components must be built in sequence because each depends on the previous one's output.

- **Component 01 first** — Without clean ingested data nothing downstream is grounded.
- **Components 02 and 03 next, in order** — Residuals depend on the prior. Component 03 cannot run before 02 has fitted.
- **Component 04 after 02 and 03** — Sampling requires both the fitted prior and the fitted GP.
- **Component 05 before 06** — The fidelity report is part of the narrative context. Validate before narrating.
- **Component 06 last** — Narration only makes sense once statistical output is complete and validated.

---

## Full Pipeline

```
[Component 01: Ingestion]
  → normal_df · rare_df · schema_graph · field_dict

[Component 02: Prior Fitting]
  ← normal_df · schema_graph
  → fitted_prior · predicted_normal scores

[Component 03: Residual Learning]
  ← rare_df · predicted_normal scores
  → fitted_GP · R-EPIG scores · feature relevance weights

[Component 04: Sampler]
  ← fitted_GP · fitted_prior · schema_graph · R-EPIG scores
  → synthetic_batch · GP posteriors per row · R-EPIG scores per row

[Component 05: Fidelity Validator]
  ← real distribution · synthetic_batch
  → fidelity_report

[Component 06: Narrative Generator]
  ← synthetic_batch · GP posteriors · R-EPIG scores · fidelity_report · domain_context
  → batch summary · per-row annotations · training value statement
```

---

## What Makes This Different

Every other synthetic data tool returns a table. REGEN returns a table, a statistical explanation of each row's deviation from normal, and a plain-language statement of what a model trained on this data will learn. The LLM layer is not the product — the statistical pipeline is the product. The LLM is what makes it legible. That combination — statistical rigor plus interpretability — is what earns trust from the ML engineers and domain experts who have to stake their model's performance on the output.
