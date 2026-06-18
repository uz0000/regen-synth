"""Minimal test: does GPy import and predict work?"""
import time, sys
t0 = time.time()
import GPy
import numpy as np
print(f"import GPy: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
X = np.random.randn(100, 6)
y = np.random.randn(100)
kernel = GPy.kern.RBF(input_dim=6, variance=1.0, lengthscale=1.0, ARD=False)
gp = GPy.models.GPRegression(X, y.reshape(-1, 1), kernel=kernel, noise_var=0.1)
print(f"GP construct: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
mean, var = gp.predict(X[:10])
print(f"GP predict: {time.time()-t0:.1f}s", flush=True)
print("Done", flush=True)
