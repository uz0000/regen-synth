"""Benchmark GP fitting: before (no opt) vs after (ARD+optimize)."""
import time, logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior
from engine.amplifier import AmplifierConfig, fit_residuals

result = do_ingest("benchmark/creditcard.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
rng = np.random.default_rng(42)
prior = fit_prior(result, PriorConfig(), rng)

# Test 1: max_features=0 (all 30 dims) with ARD+optimize
print("--- 30 dims (all features) ---")
t0 = time.time()
res = fit_residuals(result, prior, AmplifierConfig(max_features=0, gp_optimize_iters=500))
t1 = time.time()
print(f"  Fitted: {t1-t0:.1f}s, optimized={res._gp_optimized}, top-5 relevance={np.argsort(res._feature_relevance)[::-1][:5].tolist()}", flush=True)

# Test 2: max_features=6 with ARD+optimize  
print("--- 6 dims (max_features=6) ---")
t0 = time.time()
res = fit_residuals(result, prior, AmplifierConfig(max_features=6, gp_optimize_iters=500))
t1 = time.time()
print(f"  Fitted: {t1-t0:.1f}s, optimized={res._gp_optimized}, top-5 relevance={np.argsort(res._feature_relevance)[::-1][:5].tolist()}", flush=True)

# Test 3: max_features=10 with ARD+optimize
print("--- 10 dims (max_features=10) ---")
t0 = time.time()
res = fit_residuals(result, prior, AmplifierConfig(max_features=10, gp_optimize_iters=500))
t1 = time.time()
print(f"  Fitted: {t1-t0:.1f}s, optimized={res._gp_optimized}, top-5 relevance={np.argsort(res._feature_relevance)[::-1][:5].tolist()}", flush=True)
