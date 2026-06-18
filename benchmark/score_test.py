"""Check what model type the Prior loaded and time its score()."""
import time, sys
import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior

result = do_ingest("benchmark/creditcard.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
rng = np.random.default_rng(42)
prior = fit_prior(result, PriorConfig(device="cpu"), rng)
print(f"Prior model type: {type(prior._model).__name__}", flush=True)

t0 = time.time()
scores = prior.score(result.rare_df)
print(f"score(rare_df, n={len(result.rare_df)}): {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
scores2 = prior.score(result.normal_df.head(100))
print(f"score(normal_df, n=100): {time.time()-t0:.1f}s", flush=True)

print("OK", flush=True)
