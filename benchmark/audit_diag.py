"""Check which columns fail the Auditor on the credit card dataset."""
import logging, sys
logging.basicConfig(level=logging.WARNING, format="%(message)s")
import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior, generate_base_batch
from engine.amplifier import AmplifierConfig, fit_residuals, sample_residuals
from engine.auditor import AuditorConfig, audit
import pandas as pd

result = do_ingest("benchmark/creditcard.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
rng = np.random.default_rng(42)
prior = fit_prior(result, PriorConfig(device="cpu"), rng)
amp = AmplifierConfig(max_features=6)
residual = fit_residuals(result, prior, amp)
base = generate_base_batch(prior, 300, {}, rng)
_, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng)
amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
amp_df["Class"] = 0

# Audit with a lenient column threshold to see failing columns
report = audit(result, amp_df, AuditorConfig(coverage_threshold=0.50, wasserstein_threshold=0.50))

failed = [r for r in report.column_results if not r.passed]
print(f"Coverage: {report.coverage_rate:.3f} passed={report.coverage_passed}")
print(f"Failed columns ({len(failed)}/{len(report.column_results)}):")
for r in failed:
    t = f"tvd={r.tvd:.4f}" if r.tvd is not None else ""
    w = f"wasserstein={r.wasserstein:.4f}" if r.wasserstein is not None else ""
    print(f"  {r.col}: {t} {w}")
