"""Compare GaussianPrior vs PFN backend on credit card fraud benchmark."""
import json, os, time, sys
from pathlib import Path

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in open(env_path):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)

import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior, generate_base_batch
from engine.amplifier import AmplifierConfig, fit_correction, sample_correction
from engine.auditor import AuditorConfig, audit
from engine.examiner import ExaminerConfig, measure_lift
from engine.scout import ScoutConfig, select_target
import pandas as pd

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")

REPO_ROOT = Path(__file__).parent.parent
data_path = str(REPO_ROOT / "benchmark" / "creditcard.csv")

print("=" * 62)
print("  REGEN — Backend Comparison")
print("=" * 62)

result = do_ingest(data_path, "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
print(f"  Data: {len(result.normal_df)} normal, {len(result.rare_df)} rare\n")

for backend in ["gaussian", "pfn"]:
    print(f"  --- Backend: {backend} ---")

    rng = np.random.default_rng(42)
    t0 = time.time()
    prior = fit_prior(result, PriorConfig(backend=backend), rng)
    print(f"  Prior fit: {time.time()-t0:.1f}s", flush=True)

    amp = AmplifierConfig(max_features=6)
    residual = fit_correction(result, prior, amp)

    base = generate_base_batch(prior, 300, {}, rng)
    _, _, X_res = sample_correction(residual, base.values.astype(np.float64), rng)
    amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
    amp_df["Class"] = 0

    report = audit(result, amp_df, AuditorConfig(coverage_threshold=0.50))
    print(f"  Audit: passed={report.overall_passed}  coverage={report.coverage_rate:.3f}", flush=True)

    lift = measure_lift(result, amp_df, ExaminerConfig())
    print(f"  Lift:  {lift.tail_lift:+.4f}  (recall {lift.baseline_recall:.3f} → {lift.amplified_recall:.3f})", flush=True)
    print()
