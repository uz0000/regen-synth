"""
REGEN API Server — FastAPI wrapper around regen.api.

Exposes the core REGEN functionality as HTTP endpoints so a web frontend
(React, Vue, vanilla JS, etc.) can:
  1. Upload data and get a dataset profile
  2. Screen data to predict REGEN vs SMOTE
  3. Run a full amplification campaign
  4. Download the synthetic data

Run:
    uvicorn server.app:app --reload --port 8000

All endpoints accept/return JSON. File uploads use multipart/form-data.
"""

import io
import os
import json
import uuid
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="REGEN API",
    description="Rare-Event Generation & Noise amplification — synthetic data for rare events",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for uploaded datasets and campaign results
DATA_DIR = Path(tempfile.gettempdir()) / "regen_uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CAMPAIGN_DIR = Path(tempfile.gettempdir()) / "regen_campaigns"
CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)


# ── Request / response models ─────────────────────────────────────────────────

class RareEventDefModel(BaseModel):
    mode: str = "label"  # "label" | "percentile" | "imbalance_ratio"
    label_value: Optional[int] = None
    percentile: Optional[float] = None
    imbalance_ratio: Optional[float] = None


class ScreenRequest(BaseModel):
    label_col: str
    rare_def: RareEventDefModel
    seed: int = 42
    quick_campaign: bool = False


class CampaignRequest(BaseModel):
    label_col: str
    rare_def: RareEventDefModel
    seed: int = 42
    n_rows: int = 200
    max_passes: int = 5
    coverage_threshold: float = 0.50
    noise_scale: float = 0.25
    gp_noise: float = 0.1
    max_features: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rare_def_from_model(model: RareEventDefModel):
    from contracts.types import RareEventDef, RareMode
    mode_map = {
        "label": RareMode.LABEL,
        "percentile": RareMode.PERCENTILE,
        "imbalance_ratio": RareMode.IMBALANCE,
    }
    return RareEventDef(
        mode=mode_map[model.mode],
        label_value=model.label_value,
        percentile=model.percentile,
        imbalance_ratio=model.imbalance_ratio,
    )


def _save_upload(file: UploadFile) -> str:
    """Save uploaded file to disk, return path."""
    ext = Path(file.filename).suffix if file.filename else ".csv"
    file_id = uuid.uuid4().hex[:12]
    path = DATA_DIR / f"upload_{file_id}{ext}"
    content = file.file.read()
    path.write_bytes(content)
    return str(path)


def _profile(result, filepath: str) -> dict:
    """Build the dataset-profile JSON returned by /api/ingest and /api/demo."""
    columns = []
    for name, meta in result.field_dict.items():
        col_info = {
            "name": name,
            "type": meta.field_type.value,
            "nullable": meta.nullable,
        }
        if meta.field_type.value == "categorical" and meta.cardinality is not None:
            col_info["cardinality"] = meta.cardinality
        if meta.field_type.value == "continuous":
            if meta.min_val is not None:
                col_info["min"] = meta.min_val
            if meta.max_val is not None:
                col_info["max"] = meta.max_val
        columns.append(col_info)

    n_total = len(result.normal_df) + len(result.rare_df)
    return {
        "filepath": filepath,
        "n_rows": n_total,
        "n_normal": len(result.normal_df),
        "n_rare": len(result.rare_df),
        "rare_ratio": round(len(result.rare_df) / n_total, 4) if n_total else 0.0,
        "n_features": len(result.field_dict) - (1 if result.label_col else 0),
        "label_col": result.label_col,
        "columns": columns,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the single-page web UI."""
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/ingest")
async def ingest_data(
    file: UploadFile = File(...),
    label_col: str = Form(""),
    rare_mode: str = Form("label"),
    label_value: Optional[int] = Form(None),
    percentile: Optional[float] = Form(None),
    imbalance_ratio: Optional[float] = Form(None),
):
    """Upload a dataset and get a profile: row count, columns, rare/normal split."""
    from regen.api import ingest
    from contracts.types import RareEventDef, RareMode

    filepath = _save_upload(file)

    mode_map = {"label": RareMode.LABEL, "percentile": RareMode.PERCENTILE, "imbalance_ratio": RareMode.IMBALANCE}
    rare_def = RareEventDef(
        mode=mode_map.get(rare_mode, RareMode.LABEL),
        label_value=label_value,
        percentile=percentile,
        imbalance_ratio=imbalance_ratio,
    )

    try:
        result = ingest(filepath, label_col, rare_def)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingest failed: {str(e)}")

    return _profile(result, filepath)


@app.get("/api/demo")
async def demo_dataset(
    label_col: str = "is_fraud",
    rare_mode: str = "label",
    label_value: Optional[int] = 1,
    percentile: Optional[float] = None,
    imbalance_ratio: Optional[float] = None,
):
    """Generate the built-in sample fraud dataset, ingest it, return a profile.

    Lets the UI work with zero setup — no manual upload required.
    """
    from regen.api import ingest
    from contracts.types import RareEventDef, RareMode
    from examples.make_sample_data import make_dataset

    file_id = uuid.uuid4().hex[:12]
    filepath = str(DATA_DIR / f"upload_{file_id}.csv")
    df = make_dataset(n=2000, fraud_rate=0.03)
    df.to_csv(filepath, index=False)

    mode_map = {"label": RareMode.LABEL, "percentile": RareMode.PERCENTILE,
                "imbalance_ratio": RareMode.IMBALANCE}
    rare_def = RareEventDef(
        mode=mode_map.get(rare_mode, RareMode.LABEL),
        label_value=label_value,
        percentile=percentile,
        imbalance_ratio=imbalance_ratio,
    )

    try:
        result = ingest(filepath, label_col, rare_def)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingest failed: {str(e)}")

    return _profile(result, filepath)


@app.post("/api/screen")
async def screen_data(req: ScreenRequest):
    """Predict whether REGEN or SMOTE will win on this data."""
    from regen.api import screen

    # Find the uploaded file
    uploads = sorted(DATA_DIR.glob("upload_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not uploads:
        raise HTTPException(status_code=404, detail="No uploaded dataset found. Upload first via /api/ingest.")

    filepath = str(uploads[0])
    rare_def = _rare_def_from_model(req.rare_def)

    try:
        result = screen(filepath, req.label_col, rare_def,
                        seed=req.seed, quick_campaign=req.quick_campaign)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Screen failed: {str(e)}")

    return result.to_dict()


@app.post("/api/campaign")
async def run_campaign_endpoint(req: CampaignRequest):
    """Run a full multi-pass REGEN amplification campaign."""
    from regen.api import run_campaign

    uploads = sorted(DATA_DIR.glob("upload_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not uploads:
        raise HTTPException(status_code=404, detail="No uploaded dataset found. Upload first via /api/ingest.")

    filepath = str(uploads[0])
    rare_def = _rare_def_from_model(req.rare_def)
    run_id = uuid.uuid4().hex[:12]
    out_dir = str(CAMPAIGN_DIR / run_id)

    try:
        result = run_campaign(
            filepath,
            label_col=req.label_col,
            rare_def=rare_def,
            seed=req.seed,
            n_rows=req.n_rows,
            max_passes=req.max_passes,
            out_dir=out_dir,
            coverage_threshold=req.coverage_threshold,
            noise_scale=req.noise_scale,
            gp_noise=req.gp_noise,
            max_features=req.max_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Campaign failed: {str(e)}")

    response = result.to_dict()
    response["run_id"] = run_id
    return response


@app.get("/api/campaign/{run_id}")
async def get_campaign(run_id: str):
    """Retrieve a previously saved campaign result."""
    from regen.api import get_results

    run_dir = str(CAMPAIGN_DIR / run_id)
    try:
        result = get_results(run_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Campaign {run_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response = result.to_dict()
    response["run_id"] = run_id
    return response


@app.get("/api/campaign/{run_id}/download")
async def download_synthetic(run_id: str, format: str = "csv"):
    """Download the synthetic data from a completed campaign."""
    from regen.api import load_synthetic

    run_dir = str(CAMPAIGN_DIR / run_id)
    try:
        df = load_synthetic(run_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No synthetic data for campaign {run_id}.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if format == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=regen_synthetic_{run_id}.parquet"},
        )
    else:
        # CSV (default)
        csv_str = df.to_csv(index=False)
        return StreamingResponse(
            io.StringIO(csv_str),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=regen_synthetic_{run_id}.csv"},
        )


@app.get("/api/campaign/{run_id}/preview")
async def preview_synthetic(run_id: str, n: int = 10):
    """Preview the first N rows of synthetic data as JSON."""
    from regen.api import load_synthetic

    run_dir = str(CAMPAIGN_DIR / run_id)
    try:
        df = load_synthetic(run_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No synthetic data for campaign {run_id}.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Return first n rows as JSON records
    preview = df.head(n)
    return {
        "run_id": run_id,
        "total_rows": len(df),
        "preview_rows": n,
        "columns": list(df.columns),
        "data": json.loads(preview.to_json(orient="records")),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
