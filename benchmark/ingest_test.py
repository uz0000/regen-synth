"""Test ingest + prior timing."""
import time, sys
import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode

t0 = time.time()
result = do_ingest("benchmark/creditcard.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
print(f"ingest: {time.time()-t0:.1f}s ({len(result.normal_df)} normal, {len(result.rare_df)} rare)", flush=True)

t0 = time.time()
from engine.prior import PriorConfig, fit_prior
rng = np.random.default_rng(42)
prior = fit_prior(result, PriorConfig(device="cpu"), rng)
print(f"prior:  {time.time()-t0:.1f}s", flush=True)

print("OK", flush=True)
