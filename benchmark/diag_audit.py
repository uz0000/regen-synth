"""Diagnose why Hypothyroid fails the Auditor."""
import logging, sys
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")
import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior, generate_base_batch
from engine.amplifier import AmplifierConfig, fit_residuals, sample_residuals
from engine.auditor import AuditorConfig, audit
import pandas as pd

result = do_ingest("benchmark/data/hypothyroid.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=0))
rng = np.random.default_rng(42)
prior = fit_prior(result, PriorConfig(), rng)
amp = AmplifierConfig(max_features=0)
residual = fit_residuals(result, prior, amp)
base = generate_base_batch(prior, 200, {}, rng)
rng2 = np.random.default_rng(42)
_, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng2)
amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
amp_df["Class"] = 0

report = audit(result, amp_df, AuditorConfig(coverage_threshold=0.50))
print(f"Overall: {report.overall_passed}")
print(f"Coverage: {report.coverage_rate:.4f} passed={report.coverage_passed}")
print(f"Numeric cols: {sum(1 for r in report.column_results if r.wasserstein is not None)}")
failed = [r for r in report.column_results if not r.passed]
print(f"Failed columns ({len(failed)}/{len(report.column_results)}):")
for r in failed:
    t = f"tvd={r.tvd:.4f}" if r.tvd is not None else ""
    w = f"wasserstein={r.wasserstein:.4f}" if r.wasserstein is not None else ""
    print(f"  {r.col}: {t} {w}")
