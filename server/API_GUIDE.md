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
single-page frontend (upload or demo data → screen → run campaign → view
per-pass recall chart + download synthetic rows). No build step; it's one
self-contained `server/static/index.html`.

## Endpoints

### 0. Web UI
```
GET /                    → the single-page frontend (index.html)
GET /api/demo            → generate + ingest the built-in fraud demo dataset
```

### 1. Health Check
```
GET /api/health
```
Returns `{"status": "ok", "version": "0.1.0"}`. Use this to check if the
server is running.

### 2. Upload & Profile Data
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

### 3. Screen Data (Win Predictor)
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

### 4. Run Campaign
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

### 5. Get Campaign Results
```
GET /api/campaign/{run_id}
```
Retrieves a previously saved campaign result (same shape as the POST response).

### 6. Preview Synthetic Data
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

### 7. Download Synthetic Data
```
GET /api/campaign/{run_id}/download?format=csv
GET /api/campaign/{run_id}/download?format=parquet
```
Returns a file download (CSV or Parquet).

## Frontend Flow

A typical user session maps to these API calls:

```
1. User uploads CSV          → POST /api/ingest
2. Show dataset profile      → (display response)
3. User clicks "Screen"      → POST /api/screen
4. Show REGEN vs SMOTE rec   → (display response)
5. User clicks "Run Campaign"→ POST /api/campaign
6. Show pass-by-pass results → (display response, poll if needed)
7. User clicks "Download"    → GET /api/campaign/{run_id}/download
8. User clicks "Preview"     → GET /api/campaign/{run_id}/preview
```

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
