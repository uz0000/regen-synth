"""
Tests for engine.privacy — the enforced δ-distance floor + verbatim guard.

Two layers are covered:
  1. Unit tests on the pure functions (enforce_distance_floor,
     guard_against_duplicates, assess_privacy) against a small hand-built
     field_dict, so the geometric guarantee and determinism are pinned down
     directly.
  2. End-to-end tests through regen.api.generate(privacy="floored"), asserting
     the guarantee holds on the *delivered* data, that fidelity and privacy
     verdicts are reported separately, and that the privacy regime is recorded
     in the manifest (Invariant 2 — reproducible from disk).

The privacy module is pure Python (numpy/scipy/pandas); the engine-boundary
test (tests/test_boundary.py) covers the no-LLM/no-network invariant — these
tests focus on the guarantee's correctness.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.types import (
    FieldDict, FieldMeta, FieldType, PrivacyReport, RareEventDef, RareMode,
)
from engine.privacy import (
    enforce_distance_floor, guard_against_duplicates, assess_privacy,
)

SAMPLE_CSV = str(Path(__file__).parent.parent / "examples" / "transactions.csv")
LABEL_COL = "is_fraud"
RARE_DEF = RareEventDef(mode=RareMode.LABEL, label_value=1)


# Five continuous columns + one binary. The δ-floor is a *sparse-set* guarantee:
# in low dimensions a 0.5σ shell has no empty space to move into once a handful
# of real points are present (this is exactly why the floor is applied to the
# rare set, never the dense bulk — see engine.privacy). Five continuous features
# mirror a realistic rare-event schema where the floor is feasible.
CONT = [f"c{i}" for i in range(5)]


def _toy_field_dict():
    fd = {c: FieldMeta(name=c, field_type=FieldType.CONTINUOUS) for c in CONT}
    fd["flag"] = FieldMeta(name="flag", field_type=FieldType.BINARY)
    return fd


def _toy_real(n=40, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(0, 1, n) for c in CONT}
    data["flag"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


# ── enforce_distance_floor ──────────────────────────────────────────────────

class TestDistanceFloor:
    def test_floor_holds_on_near_copies(self):
        """Synthetic rows that start ON real rows are pushed to ≥ delta."""
        real = _toy_real()
        fd = _toy_field_dict()
        # Worst case: synthetic rows start as exact copies of real rows (a sparse
        # subset, as the rare set is sparse relative to its feature space).
        synth = real.iloc[:8].copy()
        delta = 0.5
        out, report = enforce_distance_floor(
            synth, real, fd, label_col="", delta=delta,
            rng=np.random.default_rng(1),
        )
        assert isinstance(report, PrivacyReport)
        assert report.passed
        assert report.min_distance >= delta - 1e-9
        # The flag (binary, non-continuous) is not part of the metric — untouched.
        assert (out["flag"].to_numpy() == synth["flag"].to_numpy()).all()

    def test_floor_is_deterministic(self):
        real = _toy_real()
        fd = _toy_field_dict()
        synth = real.iloc[:8].copy()
        a, _ = enforce_distance_floor(synth, real, fd, "", 0.5, np.random.default_rng(7))
        b, _ = enforce_distance_floor(synth, real, fd, "", 0.5, np.random.default_rng(7))
        pd.testing.assert_frame_equal(a, b)

    def test_empty_or_no_continuous_passes_trivially(self):
        fd = {"flag": FieldMeta(name="flag", field_type=FieldType.BINARY)}
        synth = pd.DataFrame({"flag": [0, 1, 1]})
        real = pd.DataFrame({"flag": [0, 1, 0]})
        out, report = enforce_distance_floor(synth, real, fd, "", 0.5,
                                             np.random.default_rng(0))
        assert report.passed
        assert report.min_distance == float("inf")
        pd.testing.assert_frame_equal(out, synth)


# ── guard_against_duplicates ──────────────────────────────────────────────────

class TestVerbatimGuard:
    def test_exact_duplicate_is_nudged(self):
        real = _toy_real(n=50)
        fd = _toy_field_dict()
        # One synthetic row is a verbatim copy of a real row; the rest are far away.
        dup = real.iloc[[3]].copy()
        far = pd.DataFrame({c: [100.0, 101.0] for c in CONT})
        far["flag"] = [0, 1]
        synth = pd.concat([dup, far], ignore_index=True)
        out, n = guard_against_duplicates(synth, real, fd, "",
                                          np.random.default_rng(2))
        assert n == 1
        # The duplicate row's continuous values moved (no longer verbatim).
        moved = out.iloc[0][CONT].to_numpy()
        orig = synth.iloc[0][CONT].to_numpy()
        assert not np.allclose(moved, orig)

    def test_no_duplicates_is_noop(self):
        real = _toy_real(n=50)
        fd = _toy_field_dict()
        synth = pd.DataFrame({c: [100.0] for c in CONT})
        synth["flag"] = [1]
        out, n = guard_against_duplicates(synth, real, fd, "",
                                          np.random.default_rng(0))
        assert n == 0
        pd.testing.assert_frame_equal(out, synth)


# ── assess_privacy ────────────────────────────────────────────────────────────

class TestAssess:
    def test_passes_when_floored_and_no_dups(self):
        real = _toy_real()
        fd = _toy_field_dict()
        synth = real.iloc[:8].copy()
        floored, _ = enforce_distance_floor(synth, real, fd, "", 0.5,
                                            np.random.default_rng(3))
        report = assess_privacy(floored, real, floored, real, fd, "", 0.5)
        assert report.passed
        assert report.min_distance >= 0.5 - 1e-9
        assert report.n_respawned == 0  # no verbatim duplicates

    def test_fails_on_near_copy(self):
        real = _toy_real()
        fd = _toy_field_dict()
        # Un-floored exact copies → min distance 0, and verbatim duplicates.
        synth = real.iloc[:8].copy()
        report = assess_privacy(synth, real, synth, real, fd, "", 0.5)
        assert not report.passed
        assert report.min_distance < 0.5


# ── End-to-end through generate() ─────────────────────────────────────────────

class TestGeneratePrivacy:
    def _run(self, privacy, delta=0.5):
        from regen.api import generate
        with tempfile.TemporaryDirectory() as out:
            s = generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
                         n_rows=300, auto=False, privacy=privacy, delta=delta,
                         out_dir=out)
            manifest = json.loads(
                (Path(s["output_dir"]) / "manifest.json").read_text()
            )
        return s, manifest

    def test_floored_guarantee_holds_on_delivered_data(self):
        s, _ = self._run("floored", delta=0.5)
        pv = s["privacy"]
        assert pv is not None
        assert pv["passed"]
        # Delivered (post-constraint, post-round) distance clears the floor.
        assert pv["min_distance"] >= 0.5 - 1e-3
        assert pv["n_verbatim_duplicates"] == 0

    def test_fidelity_and_privacy_verdicts_are_separate(self):
        """The fidelity block reports the Auditor gate only; the top-level
        `passed` is fidelity AND privacy. A privacy result never silently
        flips the fidelity verdict."""
        s, _ = self._run("floored")
        assert s["fidelity"]["passed"] is True          # Auditor gate (Invariant 3)
        assert s["passed"] == (s["fidelity"]["passed"] and s["privacy"]["passed"])

    def test_privacy_regime_recorded_in_manifest(self):
        s, manifest = self._run("floored", delta=0.5)
        m = manifest["manifest"] if "manifest" in manifest else manifest
        assert m["privacy"] == "floored"
        assert m["delta"] == 0.5

    def test_privacy_none_skips_floor_and_records_regime(self):
        s, manifest = self._run("none")
        assert s["privacy"] is None
        m = manifest["manifest"] if "manifest" in manifest else manifest
        assert m["privacy"] == "none"

    def test_invalid_privacy_and_delta_rejected(self):
        from regen.api import generate
        with pytest.raises(ValueError):
            generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
                     n_rows=50, privacy="bogus")
        with pytest.raises(ValueError):
            generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
                     n_rows=50, delta=5.0)
