"""Ultra-fine amplifier timing."""
import time, sys
import numpy as np
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior, PriorModel
from engine.prior.rdbpfn import _encode_features

result = do_ingest("benchmark/creditcard.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
rng = np.random.default_rng(42)
prior = fit_prior(result, PriorConfig(device="cpu"), rng)
print("Prior done", flush=True)

# Manual fit_residuals breakdown
rare_df = result.rare_df
feature_cols = prior._feature_cols

t0 = time.time()
X_rare = _encode_features(rare_df[feature_cols]).astype(np.float64)
print(f"  encode: {time.time()-t0:.3f}s", flush=True)

t0 = time.time()
prior_scores = prior.score(rare_df).values
print(f"  score:  {time.time()-t0:.3f}s", flush=True)

t0 = time.time()
residuals = 1.0 - prior_scores
print(f"  res:    {time.time()-t0:.3f}s", flush=True)

# Feature selection
variances = X_rare.var(axis=0)
gp_feature_idx = np.argsort(variances)[::-1][:6]
X_gp = X_rare[:, gp_feature_idx].copy()
print(f"  select: {time.time()-t0:.3f}s ({X_gp.shape})", flush=True)

# Cap at gp_max_obs=300
if len(X_gp) > 300:
    X_gp = X_gp[-300:]
    residuals = residuals[-300:]
print(f"  buffer: {time.time()-t0:.3f}s", flush=True)

# GP fit
import GPy
t0 = time.time()
kernel = GPy.kern.RBF(input_dim=6, variance=1.0, lengthscale=1.0, ARD=False)
gp = GPy.models.GPRegression(X_gp, residuals.reshape(-1, 1), kernel=kernel, noise_var=0.1)
print(f"  gp_construct: {time.time()-t0:.3f}s", flush=True)

# That GP predict (no opt)
t0 = time.time()
mean, var = gp.predict(X_gp[:5])
print(f"  gp_predict: {time.time()-t0:.3f}s", flush=True)

print("OK", flush=True)
