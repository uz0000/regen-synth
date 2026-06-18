"""Exercise PFN backend end-to-end and capture backend used in output."""
import os, sys, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior, generate_base_batch
from engine.amplifier import AmplifierConfig, fit_residuals, sample_residuals
from engine.auditor import AuditorConfig, audit
from engine.examiner import ExaminerConfig, measure_lift
import pandas as pd

result = do_ingest("benchmark/creditcard.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
rng = np.random.default_rng(42)

print("=" * 60)
print("PFN BACKEND VERIFICATION")
print("=" * 60)

prior = fit_prior(result, PriorConfig(backend="pfn", max_train_rows=500), rng)
print(f"\nBackend used: {prior._backend_used}")
assert prior._backend_used == "pfn", f"Expected 'pfn', got '{prior._backend_used}'"
print("  ✅ PFN backend confirmed in PriorModel")

amp = AmplifierConfig(max_features=6)
residual = fit_residuals(result, prior, amp)
print(f"GP optimized: {residual._gp_optimized}")
print(f"ARD lengthscales present: {hasattr(residual._gp.kern, 'lengthscale')}")

base = generate_base_batch(prior, 300, {}, rng)
_, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng)
amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
amp_df["Class"] = 0

report = audit(result, amp_df, AuditorConfig(coverage_threshold=0.50))
print(f"\nAudit: passed={report.overall_passed}, coverage={report.coverage_rate:.3f}")

lift = measure_lift(result, amp_df, ExaminerConfig())
print(f"\nLift: {lift.tail_lift:+.4f} (recall {lift.baseline_recall:.3f} → {lift.amplified_recall:.3f})")

print("\n✅ PFN backend end-to-end pass")
