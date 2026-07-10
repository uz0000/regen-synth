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
from typing import Any, Optional

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
    # Any scalar: rare classes are encoded as ints (0/1), strings ("fraud"), etc.
    # None in label mode → auto-detect the minority class.
    label_value: Optional[Any] = None
    percentile: Optional[float] = None
    imbalance_ratio: Optional[float] = None
    tail: str = "lower"   # percentile mode: "lower" (bottom pct) | "upper" (top pct)


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
    # Campaign is a diagnostic path; privacy defaults to "none" (matches
    # run_campaign). Set "floored" for released pass batches. See API_GUIDE.
    privacy: str = "none"         # "none" | "floored"
    delta: float = 0.5            # δ-distance floor in σ-units (privacy="floored")


class GenerateRequest(BaseModel):
    """Minimal user-facing request: data + how much + intent.

    All technical knobs (noise, target column, gate strictness) are auto-decided
    by regen.api.generate()'s auto-tuner; the user only owns the irreducible
    inputs (their data, how many rows, what they want it for).
    """
    label_col: str = ""           # "" → auto-detect the target column
    rare_def: Optional[RareEventDefModel] = None  # None → auto-detect the rare class
    n_rows: int = 300
    mode: str = "balanced"        # "faithful" | "balanced" | "boost"
    seed: int = 42
    auto: bool = True
    rare_ratio: Optional[float] = None  # None → auto (amplify minority to >=25%)
    # Privacy of the delivered dataset. "floored" (default) = parametric
    # generation + verbatim guard + δ-distance floor on the rare part; "none" =
    # legacy grounded sampling. NOT differential privacy. The response's
    # "privacy" block reports exactly what held. See API_GUIDE.
    privacy: str = "floored"      # "floored" | "none"
    delta: float = 0.5            # δ-distance floor in σ-units (privacy="floored")
    # Apply the OPTIONAL advisory model semantic proposal (Source 3), vetted by
    # the deterministic gate. Off by default; needs REGEN_SEMANTICS_* env.
    accept_contract: bool = False


class DoctorRequest(BaseModel):
    """Preflight: is the dataset in the supported envelope?"""
    label_col: str = ""
    rare_def: Optional[RareEventDefModel] = None


class ProposeRequest(BaseModel):
    """Draft a ScenarioSpec from a plain-language goal (advisory, editable)."""
    goal: str = ""
    label_col: str = ""
    rare_def: Optional[RareEventDefModel] = None
    n_rows: int = 300
    seed: int = 42


class ExploreRequest(BaseModel):
    """The privacy↔fidelity tradeoff frontier for the human to choose from."""
    label_col: str = ""
    rare_def: Optional[RareEventDefModel] = None
    deltas: list[float] = [0.3, 0.5, 0.8]
    n_rows: int = 300
    seed: int = 42


class TSTRRequest(BaseModel):
    """Surrogate quality: how much real performance does a model trained only on
    the surrogate recover (leakage-free)?"""
    label_col: str = ""
    rare_def: Optional[RareEventDefModel] = None
    privacy: str = "none"
    seed: int = 42


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _coerce_scalar(value):
    """Form fields arrive as strings; recover int/float so '1' matches int 1.

    Empty/None → None (auto-detect). Non-numeric strings (e.g. 'fraud') pass through.
    """
    if value is None or value == "":
        return None
    try:
        f = float(value)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return value


def _rare_def_from_model(model: Optional[RareEventDefModel]):
    from contracts.types import RareEventDef, RareMode
    if model is None:
        return None  # → regen.api.generate() auto-detects the rare class
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
        tail=model.tail,
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
        # What auto-detection chose (label column + rare value), or None if the
        # caller supplied both. Lets the UI show "auto-selected X" and offer override.
        "detection": result.detection.as_dict() if result.detection else None,
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
    label_value: Optional[str] = Form(None),
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
        label_value=_coerce_scalar(label_value),  # "" / None → auto-detect minority class
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
    label_col: str = "",
    rare_mode: str = "label",
    label_value: Optional[Any] = None,
    percentile: Optional[float] = None,
    imbalance_ratio: Optional[float] = None,
):
    """Generate the built-in sample fraud dataset, ingest it, return a profile.

    Lets the UI work with zero setup — no manual upload required. By default it
    exercises the same auto-detection path as a real upload (label_col="",
    rare value auto) so the demo shows the system picking the target itself; the
    detected column/value come back in the profile's `detection` field. Pass
    explicit params to override.
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
        label_value=label_value,   # None → auto-detect the minority class
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


def _latest_upload() -> str:
    uploads = sorted(DATA_DIR.glob("upload_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not uploads:
        raise HTTPException(status_code=404,
                            detail="No uploaded dataset found. Upload first via /api/ingest.")
    return str(uploads[0])


@app.post("/api/doctor")
async def doctor_data(req: DoctorRequest):
    """Preflight the latest upload against the supported envelope (before generating)."""
    from regen.api import preflight
    try:
        return preflight(_latest_upload(), label_col=req.label_col,
                         rare_def=_rare_def_from_model(req.rare_def))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Doctor failed: {str(e)}")


@app.post("/api/propose")
async def propose_scenario_endpoint(req: ProposeRequest):
    """Draft a ScenarioSpec from a plain-language goal (advisory; user reviews/edits)."""
    from regen.api import draft_scenario
    try:
        draft, proposal = draft_scenario(
            _latest_upload(), label_col=req.label_col,
            rare_def=_rare_def_from_model(req.rare_def),
            goal=req.goal, n_rows=req.n_rows, seed=req.seed)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Propose failed: {str(e)}")
    return {"scenario": draft.to_dict(), "yaml": draft.to_yaml(),
            "drafted_by": draft.provenance.get("drafted_by"),
            "model_used": proposal is not None,
            # surfaced so the UI can show "auto-selected <col> because <reason>" and
            # offer an override when the LLM broke a target tie (decision-support).
            "target_tiebreak": draft.provenance.get("target_tiebreak")}


@app.post("/api/explore")
async def explore_endpoint(req: ExploreRequest):
    """The privacy↔fidelity tradeoff frontier — options + recommend-with-override."""
    from regen.api import explore_options
    try:
        return explore_options(_latest_upload(), label_col=req.label_col,
                               rare_def=_rare_def_from_model(req.rare_def),
                               deltas=tuple(req.deltas), n_rows=req.n_rows, seed=req.seed)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explore failed: {str(e)}")


@app.post("/api/tstr")
async def tstr_endpoint(req: TSTRRequest):
    """Surrogate quality (TSTR): leakage-free train-on-synthetic / test-on-real."""
    from regen.api import evaluate_surrogate
    if req.privacy not in ("none", "floored"):
        raise HTTPException(status_code=400, detail="privacy must be 'none' or 'floored'")
    try:
        return evaluate_surrogate(_latest_upload(), label_col=req.label_col,
                                  rare_def=_rare_def_from_model(req.rare_def),
                                  privacy=req.privacy, seed=req.seed)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"TSTR failed: {str(e)}")


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
            privacy=req.privacy,
            delta=req.delta,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Campaign failed: {str(e)}")

    response = result.to_dict()
    response["run_id"] = run_id
    # Surface the privacy regime in the response (full detail is in the run's
    # campaign_summary.json + manifest).
    response["privacy"] = {"mode": req.privacy,
                           "delta": req.delta if req.privacy == "floored" else 0.0}
    return response


@app.post("/api/generate")
async def generate_data(req: GenerateRequest):
    """The simple primary path: generate a synthetic dataset, auto-configured.

    The user supplies data (already uploaded) + how many rows + an intent
    (faithful copy / balanced / boost). The auto-tuner picks noise and target;
    the response includes both quality numbers (fidelity + lift) and the
    candidate trail (for the fidelity-vs-lift frontier chart). The batch is
    saved in campaign layout so /download and /preview work with the run_id.
    """
    from regen.api import generate

    uploads = sorted(DATA_DIR.glob("upload_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not uploads:
        raise HTTPException(status_code=404, detail="No uploaded dataset found. Upload first via /api/ingest or /api/demo.")

    filepath = str(uploads[0])
    rare_def = _rare_def_from_model(req.rare_def)
    run_id = uuid.uuid4().hex[:12]
    out_dir = str(CAMPAIGN_DIR / run_id)

    try:
        summary = generate(
            filepath,
            label_col=req.label_col,
            rare_def=rare_def,
            n_rows=req.n_rows,
            mode=req.mode,
            seed=req.seed,
            auto=req.auto,
            rare_ratio=req.rare_ratio,
            privacy=req.privacy,
            delta=req.delta,
            accept_contract=req.accept_contract,
            out_dir=out_dir,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generate failed: {str(e)}")

    summary["run_id"] = run_id
    return summary


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


@app.get("/api/campaign/{run_id}/manifest")
async def download_manifest(run_id: str):
    """Download the batch manifest (seed + configs + schema hash + code version).

    The manifest is what makes a batch reproducible from disk (Invariant 2).
    """
    manifest_path = CAMPAIGN_DIR / run_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"No manifest for campaign {run_id}.")
    return StreamingResponse(
        io.StringIO(manifest_path.read_text()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=regen_manifest_{run_id}.json"},
    )


@app.get("/api/campaign/{run_id}/verify")
async def verify_bundle_endpoint(run_id: str):
    """Independently recompute a produced batch's reported statistics from its
    bundle (integrity + values), so a third party can check the certificate."""
    from regen.audit_bundle import verify_bundle
    run_dir = CAMPAIGN_DIR / run_id
    if not (run_dir / "manifest.json").exists():
        raise HTTPException(status_code=404, detail=f"No audit bundle for {run_id}.")
    try:
        return verify_bundle(str(run_dir))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verify failed: {str(e)}")


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
