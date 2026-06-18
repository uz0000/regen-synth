"""
Quick diagnostic — time each stage of the REGEN pipeline.

Usage: python benchmark/diagnose.py
"""
import logging, time, sys
logging.basicConfig(level=logging.WARNING, format="%(message)s")

import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior, generate_base_batch
from engine.amplifier import AmplifierConfig, fit_residuals, sample_residuals
from engine.auditor import AuditorConfig, audit
from engine.scout import ScoutConfig, select_target
import pandas as pd

result = do_ingest("benchmark/creditcard.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
print(f"Normal: {len(result.normal_df)}, Rare: {len(result.rare_df)}, Features: {len(result.field_dict)}")

rng = np.random.default_rng(42)

t0 = time.time()
prior = fit_prior(result, PriorConfig(device="cpu"), rng)
print(f"  Prior fit:          {time.time()-t0:.1f}s")

t0 = time.time()
amp = AmplifierConfig(max_features=6)
residual = fit_residuals(result, prior, amp)
print(f"  Residual fit (6D): {time.time()-t0:.1f}s")

t0 = time.time()
base = generate_base_batch(prior, 300, {}, rng)
print(f"  Base batch:         {time.time()-t0:.1f}s")

t0 = time.time()
_, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng)
print(f"  Amplify:            {time.time()-t0:.1f}s")

t0 = time.time()
amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
amp_df["Class"] = 0
report = audit(result, amp_df, AuditorConfig(coverage_threshold=0.50))
print(f"  Audit:              {time.time()-t0:.1f}s  passed={report.overall_passed}  coverage={report.coverage_rate:.3f}")

t0 = time.time()
target = select_target(residual, prior._feature_cols, rng, ScoutConfig(), explored_points=[])
print(f"  Scout:              {time.time()-t0:.1f}s")
print(f"  Target:             {target}")
print(f"\nTotal: {time.time()-t0:.1f}s (just the loop)")
