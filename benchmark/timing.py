"""Single pass timing for speed diagnosis."""
import time, sys, logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior, generate_base_batch
from engine.amplifier import AmplifierConfig, fit_residuals, sample_residuals
from engine.auditor import AuditorConfig, audit
from engine.examiner import ExaminerConfig, measure_lift
import pandas as pd

result = do_ingest("benchmark/creditcard.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
print(f"Ingest: {time.time():.0f}", flush=True)

rng = np.random.default_rng(42)

t0 = time.time()
prior = fit_prior(result, PriorConfig(), rng)
print(f"  Prior fit: {time.time()-t0:.1f}s  [{time.time():.0f}]", flush=True)

t0 = time.time()
amp = AmplifierConfig(max_features=6)
residual = fit_residuals(result, prior, amp)
print(f"  Residual: {time.time()-t0:.1f}s  [{time.time():.0f}]", flush=True)

t0 = time.time()
base = generate_base_batch(prior, 300, {}, rng)
print(f"  Base gen: {time.time()-t0:.1f}s  [{time.time():.0f}]", flush=True)

t0 = time.time()
_, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng)
print(f"  Amplify:  {time.time()-t0:.1f}s  [{time.time():.0f}]", flush=True)

amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
amp_df["Class"] = 0

t0 = time.time()
report = audit(result, amp_df, AuditorConfig(coverage_threshold=0.50))
print(f"  Audit:    {time.time()-t0:.1f}s  [{time.time():.0f}]", flush=True)

t0 = time.time()
lift = measure_lift(result, amp_df, ExaminerConfig())
print(f"  Examine:  {time.time()-t0:.1f}s  [{time.time():.0f}]", flush=True)
print(f"  Lift: {lift.tail_lift:.4f}", flush=True)
print(f"Total: {time.time()-t0:.0f}s", flush=True)
