"""
Estimand-preserving generation (v2): synthetic data that a declared estimand
survives — certified on the same real credit data where the copula/SMOTE/REGEN
baselines are refused (docs/KNOWN_ISSUES #6). Covers logit + OLS, determinism,
and the honesty that it generates novel rows (no verbatim real copies).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.scenario import EstimandSpec
from regen.certifier import certify_dataset
from regen.estimand_preserving import generate_estimand_preserving

CSV = Path(__file__).resolve().parent.parent / "examples" / "certifier_demo" / "credit_default.csv"


@pytest.mark.skipif(not CSV.exists(), reason="demo dataset not present")
def test_certifies_where_baselines_fail_logit():
    real = pd.read_csv(CSV)
    spec = EstimandSpec(outcome="default",
                        predictors=["pay_delay_1", "utilization", "log_limit", "age"],
                        family="logit")
    synth = generate_estimand_preserving(real, spec, n_rows=6000, seed=7)
    cert = certify_dataset(real, synth, spec)
    # The headline: it certifies the full analysis (copula/SMOTE/REGEN do not).
    assert cert["certified"] is True
    assert all(t["preserved"] for t in cert["targets"])
    # Honesty: novel rows, not verbatim copies of the real predictors.
    assert not synth[spec.predictors].round(6).apply(tuple, axis=1).isin(
        real[spec.predictors].round(6).apply(tuple, axis=1)).all()


def test_ols_continuous_outcome():
    rng = np.random.default_rng(1)
    n = 4000
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    y = 1.5 + 2.0 * x1 - 3.0 * x2 + rng.normal(0, 1.0, n)
    real = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    spec = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="ols")
    synth = generate_estimand_preserving(real, spec, n_rows=4000, seed=3)
    assert certify_dataset(real, synth, spec)["certified"] is True


def test_deterministic():
    rng = np.random.default_rng(2)
    real = pd.DataFrame({"y": (rng.uniform(size=2000) < 0.3).astype(int),
                         "x1": rng.normal(0, 1, 2000), "x2": rng.normal(0, 1, 2000)})
    spec = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="logit")
    a = generate_estimand_preserving(real, spec, n_rows=1500, seed=9)
    b = generate_estimand_preserving(real, spec, n_rows=1500, seed=9)
    pd.testing.assert_frame_equal(a, b)


def test_guards():
    real = pd.DataFrame({"y": [0, 1, 0, 1], "x1": [1.0, 2, 3, 4]})
    with pytest.raises(ValueError):
        generate_estimand_preserving(real, EstimandSpec(), n_rows=10)
    with pytest.raises(ValueError):
        generate_estimand_preserving(real, EstimandSpec(outcome="y", predictors=["nope"]), n_rows=10)
