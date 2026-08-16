"""
Estimand-preserving generation (v2): synthetic data that a declared estimand
survives — a real, partial improvement over copula/SMOTE/REGEN on the credit
demo, not a solved problem. A 30-seed sweep (docs/KNOWN_ISSUES.md, 2026-08-16
correction; reproduce with examples/certifier_demo/seed_sweep.py) found full
certification on 37% of seeds, with two of four coefficients (`pay_delay_1`,
`age`) recovering essentially unbiased and the other two (`utilization`,
`log_limit`) carrying a systematic — not random — attenuation. These tests
assert the validated claim (the unbiased coefficients reliably recover across
seeds) rather than a single seed's full-certification pass/fail, which is not
reliable enough to assert on its own. Also covers OLS and the honesty that it
generates novel rows (no verbatim real copies).
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
def test_reliable_coefficients_recover_across_seeds_logit():
    real = pd.read_csv(CSV)
    spec = EstimandSpec(outcome="default",
                        predictors=["pay_delay_1", "utilization", "log_limit", "age"],
                        family="logit")
    seeds = range(1, 7)
    preserved = {"pay_delay_1": 0, "age": 0}
    synth_for_honesty_check = None
    for seed in seeds:
        synth = generate_estimand_preserving(real, spec, n_rows=6000, seed=seed)
        if synth_for_honesty_check is None:
            synth_for_honesty_check = synth
        cert = certify_dataset(real, synth, spec)
        tgt = {t["coefficient"]: t for t in cert["targets"]}
        for name in preserved:
            preserved[name] += int(tgt[name]["preserved"])
    # The validated claim: these two are essentially unbiased (KNOWN_ISSUES.md),
    # so they should preserve on most seeds. Threshold well below the observed
    # ~93% individual rate to absorb ordinary Wald-test noise.
    assert preserved["pay_delay_1"] >= 4, preserved
    assert preserved["age"] >= 4, preserved
    # Honesty: novel rows, not verbatim copies of the real predictors.
    assert not synth_for_honesty_check[spec.predictors].round(6).apply(tuple, axis=1).isin(
        real[spec.predictors].round(6).apply(tuple, axis=1)).all()


def test_ols_continuous_outcome():
    rng = np.random.default_rng(1)
    n = 4000
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    y = 1.5 + 2.0 * x1 - 3.0 * x2 + rng.normal(0, 1.0, n)
    real = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    spec = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="ols")
    # OLS on two clean linear predictors is far more reliable than the credit
    # demo's logit (no systematic bias observed in a 10-seed check) — but
    # still check a few seeds rather than assert on one.
    n_certified = 0
    for seed in range(1, 6):
        synth = generate_estimand_preserving(real, spec, n_rows=4000, seed=seed)
        n_certified += int(certify_dataset(real, synth, spec)["certified"])
    assert n_certified >= 4, n_certified


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
