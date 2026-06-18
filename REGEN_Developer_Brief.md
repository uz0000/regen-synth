# REGEN — Developer Brief

Companion to the REGEN Product Specification. Read the spec sheet first for component definitions, inputs, outputs, and success conditions. This brief answers the questions the spec sheet leaves open: libraries, file layout, prompt structure, default values, and hard constraints.

> **Starting from scratch.** This is a new repository. There is no prior codebase to reference or preserve. Do not carry forward any existing implementation. Build each component clean.

---

## Architectural Constraint — Do Not Violate

**The statistical pipeline generates. The LLM translates.**

The Narrative Generator (Component 06) receives structured numerical output — GP posterior means, variances, R-EPIG scores, fidelity metrics — as explicit fields in its prompt. It does not receive a vague instruction to "generate a description." Every claim in the narrative must be traceable to a number in the prompt context. If the LLM cannot cite a specific value, the output is invalid.

---

## Repository Structure

```
regen/
├── ingestion.py          # Component 01 — Data Ingestion and Schema Mapping
├── prior.py              # Component 02 — Prior Fitting (TabPFN wrapper)
├── residual.py           # Component 03 — Residual GP + R-EPIG acquisition
├── sampler.py            # Component 04 — Synthetic Scenario Sampler
├── validator.py          # Component 05 — Fidelity Validator
├── narrator.py           # Component 06 — Narrative Generator (LLM layer)
├── pipeline.py           # Orchestrator — calls components 01→06 in sequence
├── config.py             # Single source of truth for all configurable values
└── types.py              # Shared dataclasses: SchemaGraph, RareEventDef, etc.

tests/
├── test_ingestion.py
├── test_prior.py
├── test_residual.py
├── test_sampler.py
├── test_validator.py
└── test_narrator.py

requirements.txt
README.md
```

---

## Library Decisions

| Component | Library | Why |
|---|---|---|
| 01  Ingestion | `pandas`, `numpy` | Standard tabular parsing. No ML libraries needed at this stage. |
| 02  Prior Fitting | `tabpfn` | Pretrained PFN transformer. Fits on small data with no tuning. Call `.fit()` on `normal_df`, `.predict_proba()` for scores. |
| 02  Relational structure | `torch_geometric` | Graph construction and message passing for multi-table schemas. Only used when `schema_graph` is non-empty. |
| 03  Gaussian Process | `GPy` | Multi-output GP with ARD kernel. Straightforward API for `.fit()` and `.predict()`. Cholesky updates for rolling buffer. |
| 03  R-EPIG acquisition | custom (`numpy`) | Implement R-EPIG scoring in `residual.py` directly. No external library. ~50 lines of numpy. |
| 04  Sampler | `numpy`, `scipy` | GP posterior sampling via `scipy.stats.multivariate_normal`. No additional ML library needed. |
| 05  Fidelity Validator | `scipy.stats`, `numpy` | TVD and Wasserstein via `scipy.stats.wasserstein_distance`. Coverage rate is custom numpy logic. |
| 06  Narrative Generator | `anthropic` | Claude API. Model: `claude-sonnet-4-20250514`. Structured prompt → structured JSON response. |
| Config / types | `dataclasses` (stdlib) | No third-party config library. Pure Python dataclasses in `config.py` and `types.py`. |
| Testing | `pytest` | One test file per component. Tests run against small synthetic fixtures, not real data. |

---

## requirements.txt

```
tabpfn
torch
torch-geometric
GPy
numpy
pandas
scipy
anthropic
pytest
```

---

## Config Defaults (`config.py`)

All configurable values live in a single dataclass in `config.py`. Nothing is hardcoded in component files. Components import `RegenConfig` and read from it.

```python
from dataclasses import dataclass

@dataclass
class RegenConfig:

    # Component 01 — Ingestion
    rare_percentile: float = 0.05       # Bottom 5% of target column = rare
    min_rare_rows: int = 10             # Raise error if fewer rare rows found

    # Component 02 — Prior Fitting
    tabpfn_device: str = 'cpu'          # 'cuda' if GPU available
    gnn_layers: int = 3                 # Message-passing rounds for relational data
    latent_dim: int = 64                # Per-row latent vector size

    # Component 03 — Residual GP
    gp_kernel: str = 'ARD'             # ARD = automatic relevance determination
    gp_max_obs: int = 300              # Rolling buffer cap (Cholesky stays stable)
    gp_noise_variance: float = 0.1     # Observation noise sigma^2
    repig_num_candidates: int = 100    # Candidate pool size for acquisition scoring

    # Component 04 — Sampler
    default_batch_size: int = 500      # Synthetic rows per generation call
    tone_lambda_drift: float = 0.5     # Drift penalty weight. Higher = more rigid.
    tone_lambda_struct: float = 1.0    # Structural consistency weight

    # Component 05 — Fidelity Validator
    tvd_threshold: float = 0.15        # Max acceptable TVD per column
    wasserstein_threshold: float = 0.20 # Max acceptable Wasserstein per continuous col
    coverage_threshold: float = 0.80   # Min rare event coverage rate (PRIMARY METRIC)

    # Component 06 — Narrative Generator
    llm_model: str = 'claude-sonnet-4-20250514'
    llm_max_tokens: int = 1500
    llm_top_n_rows: int = 5            # Annotate top-N rows by R-EPIG score
```

---

## Component Interfaces

Each component exposes one primary function. Signatures are fixed. Internal implementation is the developer's choice within these contracts.

### `ingestion.py`
```python
def ingest(
    filepath: str,
    label_col: str,
    rare_def: RareEventDef,   # label | percentile | imbalance_ratio
    config: RegenConfig
) -> IngestResult:
    # Returns: normal_df, rare_df, schema_graph, field_dict
    # Raises ValueError if min_rare_rows not met
```

### `prior.py`
```python
def fit_prior(
    normal_df: pd.DataFrame,
    schema_graph: SchemaGraph,
    config: RegenConfig
) -> PriorModel:
    # Wraps TabPFN. Runs GNN pass if schema_graph is non-empty.
    # Returns fitted model with .score(df) -> pd.Series[float]
```

### `residual.py`
```python
def fit_residuals(
    rare_df: pd.DataFrame,
    prior_model: PriorModel,
    config: RegenConfig
) -> ResidualModel:
    # Computes residuals, fits GP, exposes .posterior(X) and .repig(candidates)
```

### `sampler.py`
```python
def sample(
    prior_model: PriorModel,
    residual_model: ResidualModel,
    schema_graph: SchemaGraph,
    config: RegenConfig,
    n: int = None                 # defaults to config.default_batch_size
) -> SampleResult:
    # Returns: records (DataFrame), gp_mean, gp_var, repig_scores per row
```

### `validator.py`
```python
def validate(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    field_dict: FieldDict,
    config: RegenConfig
) -> FidelityReport:
    # Returns per-column TVD, Wasserstein, rare coverage rate, pass/fail
    # coverage_threshold is the primary gate — checked first
```

### `narrator.py`
```python
def narrate(
    sample_result: SampleResult,
    fidelity_report: FidelityReport,
    domain_context: str,
    config: RegenConfig
) -> Narrative:
    # Builds structured prompt, calls Claude API, parses JSON response
    # Returns: summary (str), row_annotations (list), training_value (str)
```

### `pipeline.py`
```python
def run(
    filepath: str,
    label_col: str,
    rare_def: RareEventDef,
    domain_context: str,
    config: RegenConfig = None
) -> PipelineResult:
    # Calls 01 -> 02 -> 03 -> 04 -> 05 -> 06 in sequence
    # Each step's output is passed directly to the next
    # Returns all intermediate results plus the final Narrative
```

---

## Narrative Generator — Prompt Structure (`narrator.py`)

The prompt is built programmatically from structured fields. The LLM must respond in JSON only — no prose preamble, no markdown fences. Parse the response with `json.loads()`. If parsing fails, retry once then raise.

```
SYSTEM:
You are a synthetic data analyst. You receive structured statistical output
from a data generation pipeline and translate it into plain-language explanations.
You never invent claims. Every statement must reference a specific field
from the data provided. Respond only in valid JSON matching the schema below.

USER:
{
  "domain_context": "<user-provided string>",
  "batch_summary": {
    "n_rows": <int>,
    "n_rare_candidates": <int>,
    "fidelity_passed": <bool>,
    "rare_coverage_rate": <float>,
    "tvd_worst_column": "<col_name>",
    "tvd_worst_value": <float>
  },
  "top_rows": [
    {
      "row_index": <int>,
      "repig_score": <float>,
      "gp_mean_deviation": <float>,
      "gp_variance": <float>,
      "top_deviating_features": [
        {"feature": "<name>", "deviation": <float>},
        ...
      ]
    },
    ... (top N rows by repig_score)
  ]
}

REQUIRED RESPONSE SCHEMA:
{
  "batch_summary": "<1 paragraph. Must cite n_rows, rare_coverage_rate, domain>",
  "row_annotations": [
    {"row_index": <int>, "annotation": "<cite repig_score and top features>"},
    ...
  ],
  "training_value": "<What a model trained on this batch will learn. Cite coverage rate.>"
}
```

---

## Ingestion Edge Cases

Component 01 is the entry point. These edge cases must be handled explicitly — do not silently proceed with bad input.

| Situation | Required handling |
|---|---|
| No label column provided | Attempt to infer: if a column named `label`, `target`, or `y` exists, use it. Otherwise raise `ValueError` with a clear message listing available columns. |
| Rare event definition not provided | Default to percentile mode: bottom 5% of target column (`config.rare_percentile`). Log a warning so the user knows the default was applied. |
| Fewer than `min_rare_rows` rare events found | Raise `ValueError`. Do not proceed. Message must state how many were found and what the minimum is. The GP cannot fit on too few observations. |
| Missing values in dataset | Drop rows with missing values in the label column. For feature columns, impute with column median (continuous) or mode (categorical). Log imputation counts per column. |
| No relational structure (single flat table) | `schema_graph` is returned as an empty `SchemaGraph`. Components 02 and 04 check for this and skip GNN passes silently. |
| All rows flagged as rare | Raise `ValueError`. This indicates a misconfigured `rare_def`, not a valid dataset. |

---

## Test Plan

One test file per component. Tests use small synthetic fixtures — never real data. All tests must pass with `pytest` in under 30 seconds total.

| File | What to verify |
|---|---|
| `test_ingestion.py` | Happy path with a 200-row synthetic CSV. Verify `normal_df` and `rare_df` are non-empty, schema is consistent, no missing values. Test all six edge cases with `pytest.raises` assertions. |
| `test_prior.py` | Fit on 100-row normal fixture. Verify `.score()` returns values in `[0,1]`. Verify rare event rows score lower on average than normal rows. |
| `test_residual.py` | Fit GP on 20 synthetic residual vectors. Verify posterior returns mean and variance arrays of correct shape. Verify R-EPIG scores discriminate between a clearly-anomalous and a clearly-normal candidate. |
| `test_sampler.py` | Generate a batch of 50 rows. Verify output DataFrame has correct column schema. Verify `gp_mean`, `gp_var`, `repig_scores` are present and have length 50. |
| `test_validator.py` | Compare two identical DataFrames — expect TVD = 0, coverage = 1.0, pass = True. Compare `real_df` against random noise — expect failures. Verify `coverage_rate` is the first gate checked. |
| `test_narrator.py` | Mock the Anthropic API call. Verify prompt contains all required fields (`n_rows`, `repig_score`, `gp_mean_deviation`, `domain_context`). Verify JSON response is parsed correctly into a `Narrative` object. Verify that a response without row-level citations raises a `ValidationError`. |
