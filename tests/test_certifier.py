"""
Generator-agnostic certifier: it certifies a faithful synthetic set, refuses a
distorted one, and does so identically regardless of provenance. The θ_real in the
certificate is the same across sources; only θ_synth (and the verdict) changes.
"""

import numpy as np
import pandas as pd

from contracts.scenario import EstimandSpec
from regen.certifier import certify_dataset, certify_many


def _linear(n, b1, b2, seed):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    y = 1.0 + b1 * x1 + b2 * x2 + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


SPEC = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="ols")


class TestCertifyDataset:
    def test_faithful_source_certifies(self):
        real = _linear(4000, 2.0, -3.0, seed=1)
        synth = _linear(4000, 2.0, -3.0, seed=2)  # same process
        cert = certify_dataset(real, synth, SPEC, source="faithful")
        assert cert["certified"] is True
        assert cert["source"] == "faithful"
        assert cert["metric"] == "estimand_delta"
        # Portable: the certificate carries θ_real ± SE for re-checking.
        assert "theta_real_disclosed" in cert
        assert {"x1", "x2"} <= set(cert["theta_real_disclosed"]["coefficients"])

    def test_distorted_source_is_refused(self):
        real = _linear(4000, 2.0, -3.0, seed=3)
        synth = _linear(4000, 0.0, -3.0, seed=4)  # x1 effect destroyed
        cert = certify_dataset(real, synth, SPEC, source="distorted")
        assert cert["certified"] is False
        bad = next(t for t in cert["targets"] if t["coefficient"] == "x1")
        assert bad["preserved"] is False

    def test_unfittable_is_uncertifiable_not_crash(self):
        real = _linear(100, 1.0, 1.0, seed=5)
        cert = certify_dataset(real, real, EstimandSpec(outcome="y",
                               predictors=["nope"], family="ols"), source="bad")
        assert cert["status"] == "uncertifiable" and cert["certified"] is False


class TestCertifyMany:
    def test_same_theta_real_discriminates_across_sources(self):
        real = _linear(4000, 2.0, -3.0, seed=10)
        sources = {
            "faithful": _linear(4000, 2.0, -3.0, seed=11),
            "distorted": _linear(4000, 0.0, -3.0, seed=12),
        }
        certs = certify_many(real, sources, SPEC)
        assert certs["faithful"]["certified"] is True
        assert certs["distorted"]["certified"] is False
        # θ_real is a property of the real data — identical across sources.
        tr = lambda c: c["theta_real_disclosed"]["coefficients"]["x1"]["coef"]
        assert tr(certs["faithful"]) == tr(certs["distorted"])
