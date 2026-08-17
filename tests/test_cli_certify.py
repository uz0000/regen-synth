"""
`regen certify` — the certifier's command-line surface.

The exit code is the contract: 0 certified, 1 a declared coefficient shifted,
2 the check could not run at all. A pipeline has to be able to tell "the data
failed" from "the check never happened", so those are separate codes.
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def _linear(n, b1, b2, seed):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1.0 + b1 * x1 + b2 * x2 + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _run(*args):
    """Invoke the CLI the way a user would, in a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "cli.main", "certify", *map(str, args)],
        capture_output=True, text=True,
    )


@pytest.fixture
def real_csv(tmp_path):
    p = tmp_path / "real.csv"
    _linear(4000, 2.0, -3.0, seed=1).to_csv(p, index=False)
    return p


class TestExitCodes:
    def test_faithful_source_certifies_and_exits_zero(self, real_csv, tmp_path):
        synth = tmp_path / "synth.csv"
        _linear(4000, 2.0, -3.0, seed=2).to_csv(synth, index=False)   # same process
        r = _run(real_csv, synth, "--outcome", "y", "--predictors", "x1,x2")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "CERTIFIED" in r.stdout

    def test_distorted_source_is_refused_and_exits_one(self, real_csv, tmp_path):
        synth = tmp_path / "synth.csv"
        _linear(4000, 0.5, -3.0, seed=2).to_csv(synth, index=False)   # x1 effect broken
        r = _run(real_csv, synth, "--outcome", "y", "--predictors", "x1,x2")
        assert r.returncode == 1, r.stdout + r.stderr
        assert "REFUSED" in r.stdout
        assert "x1" in r.stdout

    def test_missing_file_exits_two_not_one(self, real_csv, tmp_path):
        r = _run(real_csv, tmp_path / "absent.csv", "--outcome", "y", "--predictors", "x1")
        assert r.returncode == 2, r.stdout + r.stderr

    def test_unknown_column_exits_two_not_one(self, real_csv, tmp_path):
        synth = tmp_path / "synth.csv"
        _linear(500, 2.0, -3.0, seed=2).to_csv(synth, index=False)
        r = _run(real_csv, synth, "--outcome", "nope", "--predictors", "x1")
        assert r.returncode == 2, r.stdout + r.stderr
        assert "nope" in (r.stdout + r.stderr)


class TestOutput:
    def test_json_is_machine_readable(self, real_csv, tmp_path):
        import json
        synth = tmp_path / "synth.csv"
        _linear(4000, 2.0, -3.0, seed=2).to_csv(synth, index=False)
        r = _run(real_csv, synth, "--outcome", "y", "--predictors", "x1,x2", "--json")
        cert = json.loads(r.stdout)
        assert cert["certified"] is True
        assert {t["coefficient"] for t in cert["targets"]} == {"x1", "x2"}
        # the certificate carries θ_real so a third party can re-check it
        assert "theta_real_disclosed" in cert

    def test_coefficients_flag_narrows_what_is_certified(self, real_csv, tmp_path):
        synth = tmp_path / "synth.csv"
        _linear(4000, 0.5, -3.0, seed=2).to_csv(synth, index=False)   # only x1 broken
        broken = _run(real_csv, synth, "--outcome", "y", "--predictors", "x1,x2")
        assert broken.returncode == 1
        # certifying x2 alone passes, because x2 was never distorted
        narrowed = _run(real_csv, synth, "--outcome", "y",
                        "--predictors", "x1,x2", "--coefficients", "x2")
        assert narrowed.returncode == 0, narrowed.stdout + narrowed.stderr
