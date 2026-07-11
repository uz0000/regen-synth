"""
Guards the certifier-demo headline on the committed real dataset: a faithful
source (bootstrap of real) certifies; a structure-destroying source (independent
columns) is refused on every coefficient. Fast — excludes the REGEN/SMOTE paths.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.scenario import EstimandSpec
from regen.certifier import certify_many

CSV = Path(__file__).resolve().parent.parent / "examples" / "certifier_demo" / "credit_default.csv"
PREDICTORS = ["pay_delay_1", "utilization", "log_limit", "age"]
COLS = ["default"] + PREDICTORS


@pytest.mark.skipif(not CSV.exists(), reason="demo dataset not present")
def test_positive_control_certifies_negative_control_refused():
    real = pd.read_csv(CSV)
    rng = np.random.default_rng(7)
    bootstrap = real[COLS].sample(6000, replace=True, random_state=7).reset_index(drop=True)
    independent = pd.DataFrame({c: rng.choice(real[c].to_numpy(), size=6000) for c in COLS})

    spec = EstimandSpec(outcome="default", predictors=PREDICTORS, family="logit")
    certs = certify_many(real, {"bootstrap": bootstrap, "independent": independent}, spec)

    # Faithful source certifies; θ_real is a property of the real data.
    assert certs["bootstrap"]["certified"] is True
    assert {"pay_delay_1", "utilization"} <= set(
        certs["bootstrap"]["theta_real_disclosed"]["coefficients"])
    # Destroying the joint structure collapses every coefficient → refused.
    assert certs["independent"]["certified"] is False
    assert all(not t["preserved"] for t in certs["independent"]["targets"])
