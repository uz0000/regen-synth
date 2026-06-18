"""Verify the PFN backend loads and runs end-to-end."""
import os, sys, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior, generate_base_batch
from engine.prior.rdbpfn import _load_pfn_backend
import numpy as np

# Quick test: does the PFN backend actually load?
result = do_ingest("benchmark/creditcard.csv", "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))

print("Testing PFN backend directly...", flush=True)
rng = np.random.default_rng(42)

# Call _load_pfn_backend with small data to test
X_small = np.random.randn(100, 5).astype(np.float32)
y_small = np.array([1]*80 + [0]*20, dtype=np.int64)
config = PriorConfig(backend="pfn")

try:
    scorer = _load_pfn_backend(config, X_small, y_small)
    print(f"  Backend loaded: {type(scorer).__name__}", flush=True)
    
    # Test scoring
    probs = scorer.predict_proba(X_small[:5])
    print(f"  Score shape: {probs.shape}, OK", flush=True)
    
    # Now run full prior with pfn backend
    print("\nRunning fit_prior with backend='pfn'...", flush=True)
    prior = fit_prior(result, PriorConfig(backend="pfn"), rng)
    print(f"  Prior scorer type: {type(prior._scorer).__name__}", flush=True)
    print(f"  PFN backend confirmed operational", flush=True)
except Exception as e:
    print(f"  ERROR: {e}", flush=True)
    # Check what TabPFN reports
    try:
        from tabpfn import TabPFNClassifier
        print(f"  TabPFNClassifier imported OK", flush=True)
    except ImportError as ie:
        print(f"  TabPFN import failed: {ie}", flush=True)
    except Exception as te:
        print(f"  TabPFN init failed: {te}", flush=True)
