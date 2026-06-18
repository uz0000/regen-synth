"""Test amplifier timing only (after prior fit)."""
import time, sys
import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior
from engine.amplifier import AmplifierConfig, fit_residuals, sample_residuals

result = do_ingest("benchmark/creditcard.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
rng = np.random.default_rng(42)
prior = fit_prior(result, PriorConfig(device="cpu"), rng)
print(f"Prior done", flush=True)

t0 = time.time()
amp = AmplifierConfig(max_features=6)
residual = fit_residuals(result, prior, amp)
print(f"Residual fit (6D): {time.time()-t0:.1f}s", flush=True)

print(f"GP dims: {len(residual._gp_feature_idx)}/{residual._n_total_features}", flush=True)
print("OK", flush=True)
