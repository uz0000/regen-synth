"""
Estimand preservation — the estimator recovers known coefficients, and certify()
catches a distorted one. This is the differentiator's core in miniature: a fit on
data with a KNOWN relationship must land on the true coefficient (within CI), and
a synthetic fit whose coefficient has moved out of the real CI must fail to certify.

Deterministic: fixed-seed numpy RNG, no network, pure recomputation.
"""

import numpy as np
import pandas as pd
import pytest

from contracts.scenario import EstimandSpec
from regen.estimand import fit_estimand, certify, EstimandError


def _linear(n, b1, b2, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1.5 + b1 * x1 + b2 * x2 + rng.normal(0, noise, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


class TestOLS:
    def test_recovers_known_coefficients(self):
        df = _linear(4000, b1=2.0, b2=-3.0, noise=1.0, seed=1)
        spec = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="ols")
        fit = fit_estimand(df, spec)
        assert fit["family"] == "ols" and fit["n"] == 4000
        c = fit["coefficients"]
        # Point estimates near truth (recovery), and each CI is a proper interval
        # bracketing its own estimate. (Single-95%-CI coverage of the *true* value
        # is ~95% by construction — asserting it on one seed would be flaky.)
        assert abs(c["x1"]["coef"] - 2.0) < 0.1
        assert abs(c["x2"]["coef"] - (-3.0)) < 0.1
        for name in ("x1", "x2"):
            assert c[name]["ci_low"] < c[name]["coef"] < c[name]["ci_high"]

    def test_narrower_ci_with_more_data(self):
        spec = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="ols")
        small = fit_estimand(_linear(200, 2.0, -3.0, seed=2), spec)["coefficients"]["x1"]
        big = fit_estimand(_linear(8000, 2.0, -3.0, seed=2), spec)["coefficients"]["x1"]
        width = lambda d: d["ci_high"] - d["ci_low"]
        assert width(big) < width(small)


class TestLogit:
    def test_recovers_sign_and_magnitude(self):
        rng = np.random.default_rng(3)
        n = 6000
        x1 = rng.normal(0, 1, n)
        eta = 0.5 + 1.2 * x1
        p = 1.0 / (1.0 + np.exp(-eta))
        y = (rng.uniform(size=n) < p).astype(float)
        df = pd.DataFrame({"y": y, "x1": x1})
        spec = EstimandSpec(outcome="y", predictors=["x1"], family="logit")
        fit = fit_estimand(df, spec)
        assert fit["family"] == "logit"
        c = fit["coefficients"]["x1"]
        assert c["ci_low"] <= 1.2 <= c["ci_high"]
        assert abs(c["coef"] - 1.2) < 0.15

    def test_rejects_non_binary_outcome(self):
        df = _linear(100, 2.0, -3.0)
        with pytest.raises(EstimandError):
            fit_estimand(df, EstimandSpec(outcome="y", predictors=["x1"], family="logit"))


class TestCertify:
    def test_preserved_coefficient_certifies(self):
        spec = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="ols")
        real = fit_estimand(_linear(4000, 2.0, -3.0, seed=10), spec)
        # A "synthetic" set drawn from the SAME process → coefficients preserved.
        synth = fit_estimand(_linear(4000, 2.0, -3.0, seed=11), spec)
        verdict = certify(real, synth, spec)
        assert verdict["certified"] is True
        assert all(t["preserved"] for t in verdict["targets"])

    def test_distorted_coefficient_fails(self):
        spec = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="ols")
        real = fit_estimand(_linear(4000, 2.0, -3.0, seed=12), spec)
        # x1's effect has been driven far outside the real CI (2.0 -> 0.0):
        # fidelity/TSTR could still pass, but the estimand is NOT preserved.
        synth = fit_estimand(_linear(4000, 0.0, -3.0, seed=13), spec)
        verdict = certify(real, synth, spec)
        assert verdict["certified"] is False
        bad = next(t for t in verdict["targets"] if t["coefficient"] == "x1")
        assert bad["preserved"] is False

    def test_targets_respect_coefficients_of_interest(self):
        # Only x2 is of interest; a distorted x1 must not fail certification.
        spec = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="ols",
                            coefficients_of_interest=["x2"])
        real = fit_estimand(_linear(4000, 2.0, -3.0, seed=14), spec)
        synth = fit_estimand(_linear(4000, 0.0, -3.0, seed=15), spec)
        verdict = certify(real, synth, spec)
        assert [t["coefficient"] for t in verdict["targets"]] == ["x2"]
        assert verdict["certified"] is True


class TestGuards:
    def test_undeclared_raises(self):
        with pytest.raises(EstimandError):
            fit_estimand(_linear(100, 1.0, 1.0), EstimandSpec())

    def test_missing_column_raises(self):
        spec = EstimandSpec(outcome="y", predictors=["nope"], family="ols")
        with pytest.raises(EstimandError):
            fit_estimand(_linear(100, 1.0, 1.0), spec)

    def test_too_few_rows_raises(self):
        spec = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="ols")
        with pytest.raises(EstimandError):
            fit_estimand(_linear(2, 1.0, 1.0), spec)
