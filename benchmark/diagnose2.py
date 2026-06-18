"""Ultra-fine granularity timing."""
import logging, time, sys
logging.basicConfig(level=logging.WARNING, format="%(message)s")

import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode

t0 = time.time()
result = do_ingest("benchmark/creditcard.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
t1 = time.time()
print(f"ingest:            {t1-t0:.1f}s")

from engine.prior import PriorConfig, fit_prior
rng = np.random.default_rng(42)
prior = fit_prior(result, PriorConfig(device="cpu"), rng)
t2 = time.time()
print(f"prior_fit:         {t2-t1:.1f}s")

# now test just the GP construction
import GPy
from engine.amplifier.residual_gp import AmplifierConfig, _encode_features

feature_cols = prior._feature_cols
X_rare = _encode_features(result.rare_df[feature_cols]).astype(np.float64)
prior_scores = prior.score(result.rare_df).values
residuals = 1.0 - prior_scores

t3 = time.time()
print(f"encode_rare:       {t3-t2:.1f}s")

# test GP construction only
kernel = GPy.kern.RBF(input_dim=6, variance=1.0, lengthscale=1.0, ARD=False)
gp = GPy.models.GPRegression(X_rare[:300, :6], residuals[:300].reshape(-1, 1), kernel=kernel, noise_var=0.1)
t4 = time.time()
print(f"gp_construct:      {t4-t3:.1f}s")

# test GP predict (no optimization)
mean, var = gp.predict(X_rare[:5, :6])
t5 = time.time()
print(f"gp_predict(5):     {t5-t4:.4f}s")

# test GP predict for full scout batch
mean, var = gp.predict(X_rare[:300, :6])
t6 = time.time()
print(f"gp_predict(300):   {t6-t5:.4f}s")

# test with optimization
gp.optimize(messages=False)
t7 = time.time()
print(f"gp_optimize:       {t7-t6:.1f}s")

print(f"\nTotal: {t7-t0:.1f}s")
