# REGEN API — Frontend Integration Guide

The REGEN API server exposes HTTP endpoints that a web frontend can call
to upload data, screen it, run amplification campaigns, and download
synthetic data.

## Quick Start

```bash
# Install deps
pip install fastapi uvicorn

# Start the server
uvicorn server.app:app --port 8000 --reload
```

The server runs at `http://localhost:8000`. Interactive docs (Swagger UI)
are available at `http://localhost:8000/docs`.

**Web UI:** open `http://localhost:8000/` in a browser for the built-in
single-page frontend. It's a three-step flow — (1) data, (2) how many rows +
what for, (3) generate — with REGEN auto-tuning all technical parameters and
showing both quality numbers (distribution fidelity + detection lift). No build
step; it's one self-contained `server/static/index.html`.

## Endpoints

### 0. Web UI + demo data
```
GET /                    → the single-page frontend (index.html)
GET /api/demo            → generate + ingest the built-in fraud demo dataset
```

### 1. Generate (simple, auto-configured) — the primary path
```
POST /api/generate       → auto-tune + generate a synthetic dataset
```
Body: `{ label_col?, rare_def?: {mode, label_value}, n_rows, mode, seed, auto }`
where `mode` is `"faithful"` | `"balanced"` (default) | `"boost"`.

`label_col` and `rare_def` are **optional** — omit `label_col` (or send `""`)
and omit/`null` `rare_def` to let REGEN auto-detect the target column and rare
class structurally (the most imbalanced low-cardinality column + its minority
class). Send them to override. `rare_def.label_value` accepts any scalar — ints
(`1`), or strings (`"fraud"`); `null` means "auto-pick the minority class."

The user supplies only what only they know (their data, how many rows, what
it's for); REGEN picks the target and the auto-tuner picks `noise_scale`.
Returns: `detection` (what was auto-selected — label column, rare value, ratio,
and runner-up columns; `null` if both were supplied), `fidelity` (per-column
distribution match **plus** `fidelity.correlation` = `{delta, passed}`, the
cross-column correlation-structure gate; `delta` is `null` when there are too
few numeric columns to estimate), `lift` (held-out detection lift, or null),
`config_used` (the chosen noise/mode), `candidates` (the auto-tune trail — one
entry per noise candidate, for the fidelity-vs-lift frontier chart), and `seed`
+ `manifest_path` (the batch ships with a manifest that regenerates it
bit-for-bit). The batch is saved in campaign layout, so
`/api/campaign/{run_id}/download`, `/preview`, and `/manifest` retrieve it with
the returned `run_id`. If the target is ambiguous (two equally plausible
columns) the endpoint returns **400** with the candidate list — resend with an
explicit `label_col`.

**Modes:** `faithful` maximizes distributional fidelity (model-agnostic — a
faithful copy for any model); `balanced` maximizes detection-lift subject to
the fidelity gate; `boost` is the same with a looser gate (more lift, more
distortion). Detection lift is measured by the internal Examiner, a Random
Forest used as a generic tabular-classifier proxy — and it is **leakage-free**:
the real rare rows are split into train/test first, the synthetic the amplified
detector trains on is generated from the train fold only, and both detectors are
scored on the held-out real rows. The Auditor gate now also checks cross-column
correlation structure, not just per-column marginals.

### 2. Health Check
```
GET /api/health
```
Returns `{"status": "ok", "version": "0.1.0"}`. Use this to check if the
server is running.

### 3. Upload & Profile Data
```
POST /api/ingest
Content-Type: multipart/form-data

file:        <CSV file>
label_col:   "Class"           # column name marking rare events
rare_mode:   "label"           # "label" | "percentile" | "imbalance_ratio"
label_value: 0                 # which value = rare (for label mode)
percentile:  (optional)        # for percentile mode
imbalance_ratio: (optional)    # for imbalance_ratio mode
```

Returns:
```json
{
  "filepath": "/tmp/.../upload_abc123.csv",
  "n_rows": 3163,
  "n_normal": 3012,
  "n_rare": 151,
  "rare_ratio": 0.0477,
  "n_features": 25,
  "label_col": "Class",
  "columns": [
    {"name": "age", "type": "continuous", "nullable": false, "min": 0.0, "max": 92.0},
    {"name": "on_thyroxine", "type": "binary", "nullable": false},
    ...
  ]
}
```

### 4. Screen Data (Win Predictor)
```
POST /api/screen
Content-Type: application/json

{
  "label_col": "Class",
  "rare_def": {"mode": "label", "label_value": 0},
  "seed": 42,
  "quick_campaign": false
}
```

Returns:
```json
{
  "recommended_method": "REGEN",
  "heterogeneity_score": 0.45,
  "confidence": 0.67,
  "predicted_lift_band": "+5% to +25%",
  "rationale": "ARD inverse-lengthscale CV = 0.45 — features vary in informativeness...",
  "n_rare": 151,
  "n_features": 25
}
```

### 5. Run Campaign
```
POST /api/campaign
Content-Type: application/json

{
  "label_col": "Class",
  "rare_def": {"mode": "label", "label_value": 0},
  "seed": 42,
  "n_rows": 200,
  "max_passes": 5,
  "coverage_threshold": 0.50,
  "noise_scale": 0.10,
  "gp_noise": 0.1,
  "max_features": 0
}
```

Returns:
```json
{
  "best_lift": 0.1522,
  "passes": [
    {"pass_num": 1, "status": "accepted", "tail_lift": 0.0435, ...},
    {"pass_num": 2, "status": "accepted", "tail_lift": 0.1522, ...},
    ...
  ],
  "n_accepted": 5,
  "n_rejected": 0,
  "n_normal": 3012,
  "n_rare": 151,
  "n_features": 25,
  "n_rows_per_pass": 200,
  "run_id": "abc123def456",
  "best_batch_path": "/tmp/.../pass_5_accepted.parquet"
}
```

### 6. Get Campaign Results
```
GET /api/campaign/{run_id}
```
Retrieves a previously saved campaign result (same shape as the POST response).

### 7. Preview Synthetic Data
```
GET /api/campaign/{run_id}/preview?n=10
```
Returns the first N rows as JSON records:
```json
{
  "run_id": "abc123",
  "total_rows": 200,
  "preview_rows": 10,
  "columns": ["age", "sex", ...],
  "data": [{"age": 71.8, "sex": 2.1, ...}, ...]
}
```

### 8. Download Synthetic Data
```
GET /api/campaign/{run_id}/download?format=csv
GET /api/campaign/{run_id}/download?format=parquet
```
Returns a file download (CSV or Parquet).

### 9. Download Manifest
```
GET /api/campaign/{run_id}/manifest
```
Returns the batch manifest JSON (seed, schema hash, prior/amplifier configs,
Scout target, code version) — everything needed to regenerate the batch
bit-for-bit. **404** if the run has no manifest.

## Frontend Flow

The built-in UI (`GET /`) is a three-step session:

```
1. Data:  "Use demo dataset" → GET /api/demo   (or upload → POST /api/ingest)
2. Pick:  rows + intent (faithful/balanced/boost); label auto-detected, editable
3. Go:    "Generate"         → POST /api/generate
         show fidelity + lift + frontier, then:
         "Download CSV"      → GET  /api/campaign/{run_id}/download?format=csv
```

The full campaign / screen endpoints (sections 4–5) remain available for
power users and the API, but the simple path is `/api/generate`.

## Key Types for Frontend Objects

### CampaignResult
```typescript
interface CampaignResult {
  best_lift: number;          // e.g. 0.1522 = +15.22% recall
  passes: PassDetail[];
  n_accepted: number;
  n_rejected: number;
  n_normal: number;
  n_rare: number;
  n_features: number;
  n_rows_per_pass: number;
  run_id: string;
  best_batch_path: string | null;
}
```

### PassDetail
```typescript
interface PassDetail {
  pass_num: number;
  status: "accepted" | "rejected";
  tail_lift: number;           // recall improvement
  baseline_recall: number;
  amplified_recall: number;
  baseline_precision: number;
  amplified_precision: number;
  coverage: number;
}
```

### ScreenResult
```typescript
interface ScreenResult {
  recommended_method: "REGEN" | "SMOTE";
  heterogeneity_score: number;
  confidence: number;          // 0-1
  predicted_lift_band: string;
  rationale: string;
  n_rare: number;
  n_features: number;
}
```

### IngestProfile
```typescript
interface IngestProfile {
  filepath: string;
  n_rows: number;
  n_normal: number;
  n_rare: number;
  rare_ratio: number;
  n_features: number;
  label_col: string;
  columns: ColumnInfo[];
}

interface ColumnInfo {
  name: string;
  type: "continuous" | "categorical" | "binary" | "identifier";
  nullable: boolean;
  cardinality?: number;  // categorical only
  min?: number;          // continuous only
  max?: number;          // continuous only
}
```

### GenerateResult  (POST /api/generate — the primary path)
```typescript
interface GenerateResult {
  run_id: string;
  n_rows: number;              // rows generated
  label_col: string;
  n_normal: number; n_rare: number; n_features: number;
  detection: {
    label_col: string; rare_value: any; n_rare: number; minority_ratio: number;
    auto_label: boolean; auto_rare: boolean; alternatives: object[];
  } | null;                    // what was auto-selected; null if both supplied
  fidelity: {
    score: number;             // 0-1, fraction of columns matching
    passed: boolean;           // cleared the Auditor gate
    coverage: number;
    correlation: { delta: number | null; passed: boolean };  // cross-column joint structure
    columns: { col: string; passed: boolean; metric: "wasserstein"|"tvd"; value: number }[];
  };
  lift: { tail_lift: number; baseline_recall: number; amplified_recall: number } | null; // held-out
  config_used: { mode: string; noise_scale: number; coverage_threshold: number; auto: boolean };
  candidates: { noise: number; fidelity: number; lift: number | null; passed: boolean }[]; // frontier
  seed: number;                // reproduces the batch with config_used
  manifest_path: string;       // also downloadable via /api/campaign/{run_id}/manifest
}
```

## Notes

- The `/api/campaign` endpoint is synchronous and may take 10-60 seconds
  depending on dataset size and max_passes. For a production frontend,
  consider running it as a background task with polling.
- CORS is enabled for all origins (`*`). Tighten this in production.
- Uploaded files are stored in the system temp directory. They persist
  across server restarts but are cleared by the OS periodically.
- The `noise_scale` parameter defaults to 0.10 (tuned via benchmark sweep).
  Lower values produce tighter distribution matches; higher values explore
  more of the feature space.
