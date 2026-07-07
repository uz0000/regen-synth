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

    def test_coincident_rows_at_nonleading_positions_do_not_crash(self):
        """P1-6 regression: many synthetic rows landing exactly on real rows —
        common on integer-coded / low-cardinality continuous columns — must not
        produce inf via delta/nd. The coincident-row nudge previously indexed the
        first n_zero rows of the violating subset instead of the actual zero
        rows, so a coincident row past that prefix stayed at distance 0 and the
        next KD-tree query crashed on a non-finite value (solar_flare)."""
        fd = {c: FieldMeta(name=c, field_type=FieldType.CONTINUOUS, is_integer=True)
              for c in CONT}
        rng = np.random.default_rng(0)
        # Low-cardinality integer grid → exact coincidences are common.
        real = pd.DataFrame({c: rng.integers(0, 4, 60) for c in CONT})
        # Synth: first row is far away (non-violating), the rest are exact copies
        # of real rows — so the zero-distance rows are NOT the leading entries of
        # the violating subset, which is exactly what tripped the old bug.
        far = pd.DataFrame({c: [99] for c in CONT})
        copies = real.iloc[5:20].copy()
        synth = pd.concat([far, copies], ignore_index=True)
        out, report = enforce_distance_floor(
            synth, real, fd, label_col="", delta=0.5, rng=np.random.default_rng(1),
        )
        assert np.isfinite(out[CONT].to_numpy()).all()   # no inf/nan escaped
        assert np.isfinite(report.min_distance)

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


# ── P0-2: percentile-mode correlation gate under privacy ──────────────────────

class TestP02PercentileCorrelationUnderPrivacy:
    """P0-2: percentile (numeric-tail) rare mode + privacy="floored".

    Root cause of the original failure: the parametric generator sampled discrete
    columns independently of the continuous copula, erasing discrete↔continuous
    correlation. In LABEL mode the binary is the label (excluded from the gate);
    in PERCENTILE mode (amount as target, is_fraud a gated binary feature) it
    surfaced. The fix draws all features from ONE joint (mixed-data) Gaussian
    copula — see test_mixed_copula_preserves_discrete_continuous_correlation.

    The audit's done-when allows the repro to "pass OR fail loudly with a
    machine-readable reason." The base copula fix is real (correlation on the
    generated base is preserved), but the fidelity gate is measured on the
    DELIVERED (post-floor) data, and on this dense percentile tail the δ-floor
    perturbs the marginals/correlation enough to exceed the gate — so the batch is
    reported not-shippable, loudly, never a silent pass. (privacy="none", no floor,
    still passes.) See docs/CAPABILITY_MATRIX.md.
    """

    RD = RareEventDef(mode=RareMode.PERCENTILE, percentile=0.05, tail="upper")

    def test_percentile_floored_verdict_is_honest_on_delivered_data(self):
        from regen.api import generate
        with tempfile.TemporaryDirectory() as out:
            s = generate(SAMPLE_CSV, label_col="amount", rare_def=self.RD,
                         n_rows=200, seed=7, privacy="floored", out_dir=out)
        # No silent pass: the shippable verdict is exactly the conjunction of the
        # delivered-data gates (fidelity AND conformance AND privacy). If the floor
        # breaks delivered fidelity, `passed` is False and the reason is visible.
        priv_ok = s["privacy"] is None or s["privacy"]["passed"]
        assert s["passed"] == bool(
            s["fidelity"]["passed"] and s["conformance"]["passed"] and priv_ok)
        # The reported correlation is the DELIVERED (post-floor) value, and it is
        # not silently stamped shippable when it exceeds the gate.
        if not s["fidelity"]["correlation"]["passed"]:
            assert not s["passed"]

    def test_privacy_off_still_passes(self):
        """Sanity: the non-private path was never broken and stays green."""
        from regen.api import generate
        with tempfile.TemporaryDirectory() as out:
            s = generate(SAMPLE_CSV, label_col="amount", rare_def=self.RD,
                         n_rows=200, seed=7, privacy="none", out_dir=out)
        assert s["fidelity"]["correlation"]["passed"]

    def test_mixed_copula_preserves_discrete_continuous_correlation(self):
        """Direct check on the generator: a discrete feature strongly correlated
        with a continuous one keeps that correlation under the joint copula,
        where independent discrete sampling would drop it toward zero."""
        import numpy as np
        from engine.prior import fit_prior, PriorConfig, generate_parametric_batch
        from regen.api import ingest as _ingest

        result = _ingest(SAMPLE_CSV, "amount", self.RD)
        rng = np.random.default_rng(0)
        prior = fit_prior(result, PriorConfig(), rng)
        batch = generate_parametric_batch(prior, 400, np.random.default_rng(1),
                                          which_class="rare")
        # In the rare tail is_fraud is strongly (negatively) correlated with
        # n_prior_txns in the real data; the synthetic batch should keep the sign
        # and a non-trivial magnitude rather than collapsing to ~0.
        real_r = result.rare_df[["n_prior_txns", "is_fraud"]].corr().iloc[0, 1]
        synth_r = batch[["n_prior_txns", "is_fraud"]].corr().iloc[0, 1]
        assert abs(real_r) > 0.2                       # precondition on the fixture
        assert np.sign(synth_r) == np.sign(real_r)
        assert abs(synth_r) > 0.15                     # not erased to independence


# ── P2-8: privacy scope reconciliation (floor / guard / measurement) ──────────

class TestP08Scope:
    """P2-8: verbatim-attribute detection must (a) be k-anonymity aware for
    discrete-only data — reusing a category tuple shared by many real rows is not
    a leak — and (b) treat matching a real *normal* row the same as a real rare
    row (the guard now runs against the full real set in generation)."""

    def _cat_fd(self):
        return {
            "a": FieldMeta(name="a", field_type=FieldType.CATEGORICAL,
                           categories=["x", "y", "z"]),
            "b": FieldMeta(name="b", field_type=FieldType.CATEGORICAL,
                           categories=["p", "q"]),
        }

    def test_kanonymous_discrete_tuple_is_not_a_duplicate(self):
        from engine.privacy import _count_duplicates
        # 'x','p' appears many times → k-anonymous; 'z','q' appears once → unique.
        real = pd.DataFrame({
            "a": ["x", "x", "x", "x", "z"],
            "b": ["p", "p", "p", "p", "q"],
        })
        fd = self._cat_fd()
        shared = pd.DataFrame({"a": ["x"], "b": ["p"]})   # reuses a shared tuple
        unique = pd.DataFrame({"a": ["z"], "b": ["q"]})   # reproduces the singleton
        assert _count_duplicates(shared, real, fd, "") == 0
        assert _count_duplicates(unique, real, fd, "") == 1

    def test_guard_scope_is_the_full_real_set(self):
        """A synthetic row verbatim-matching a UNIQUE real row is caught whether
        that real row is 'normal' or 'rare' — the guard sees the full set."""
        from engine.privacy import _count_duplicates
        real_full = pd.DataFrame({           # every tuple unique → all identifying
            "a": ["x", "y", "z"],
            "b": ["p", "q", "p"],
        })
        fd = self._cat_fd()
        synth = pd.DataFrame({"a": ["y"], "b": ["q"]})    # matches the middle row
        assert _count_duplicates(synth, real_full, fd, "") == 1


# ── P2-9: the floor must never skip silently ──────────────────────────────────

class TestP09LoudFloorSkip:
    """P2-9: when the δ-floor can't apply (no continuous features / no label) the
    privacy block must say so — floor_applied=False + a reason — never imply a
    δ-shell was carved while mode still reads 'floored'."""

    def _all_categorical_csv(self, tmp):
        rng = np.random.default_rng(0)
        n = 600
        df = pd.DataFrame({
            "region": rng.choice(["north", "south", "east", "west"], size=n,
                                 p=[0.4, 0.3, 0.2, 0.1]),
            "plan": rng.choice(["basic", "plus", "pro"], size=n),
            "device": rng.choice(["ios", "android"], size=n),
            "churn": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
        })
        p = str(Path(tmp) / "cat.csv")
        df.to_csv(p, index=False)
        return p

    def test_floor_skip_is_explicit_on_all_categorical(self):
        from regen.api import generate
        with tempfile.TemporaryDirectory() as tmp:
            path = self._all_categorical_csv(tmp)
            s = generate(path, label_col="churn", n_rows=200, seed=3,
                         privacy="floored", auto=False, out_dir=tmp)
        pv = s["privacy"]
        assert pv["floor_applied"] is False
        assert pv["floor_skip_reason"] == "no_continuous_features"
        # k-anonymous categorical reuse is not a leak → still passes, as documented.
        assert pv["min_distance"] == float("inf")
        assert pv["n_verbatim_duplicates"] == 0
        assert pv["passed"] is True

    def test_floor_applied_true_when_continuous_present(self):
        """Contrast: the bundled continuous dataset carves a real δ-shell."""
        from regen.api import generate
        with tempfile.TemporaryDirectory() as tmp:
            s = generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
                         n_rows=200, seed=1, privacy="floored", auto=False,
                         out_dir=tmp)
        assert s["privacy"]["floor_applied"] is True
        assert s["privacy"]["floor_skip_reason"] is None


# ── P1-5: campaign privacy plumbing + visible regime ──────────────────────────

class TestP15CampaignPrivacy:
    """P1-5: run_campaign threads privacy end-to-end (same floor helper as
    generate) and always records the regime; screen is non-private by design."""

    def test_floored_campaign_batches_carry_the_floor(self):
        import pandas as pd
        from regen.api import run_campaign, ingest
        from engine.privacy import assess_privacy
        with tempfile.TemporaryDirectory() as out:
            cr = run_campaign(SAMPLE_CSV, LABEL_COL, RARE_DEF, seed=42, n_rows=150,
                              max_passes=2, out_dir=out, privacy="floored", delta=0.5)
            summ = json.loads((Path(out) / "campaign_summary.json").read_text())
            man = json.loads((Path(out) / "manifest.json").read_text())
            res = ingest(SAMPLE_CSV, LABEL_COL, RARE_DEF)
            df = pd.read_parquet(cr.best_batch_path)
        assert summ["privacy"]["mode"] == "floored"
        assert man["privacy"] == "floored" and man["delta"] == 0.5
        rep = assess_privacy(df, res.rare_df, df,
                             pd.concat([res.normal_df, res.rare_df], ignore_index=True),
                             res.field_dict, LABEL_COL, 0.5)
        assert rep.passed
        assert rep.min_distance >= 0.5 - 1e-3

    def test_default_campaign_regime_is_none_and_visible(self):
        from regen.api import run_campaign
        with tempfile.TemporaryDirectory() as out:
            run_campaign(SAMPLE_CSV, LABEL_COL, RARE_DEF, seed=42, n_rows=120,
                         max_passes=1, out_dir=out)
            summ = json.loads((Path(out) / "campaign_summary.json").read_text())
        assert summ["privacy"]["mode"] == "none"

    def test_campaign_rejects_bad_privacy(self):
        from regen.api import run_campaign
        with pytest.raises(ValueError):
            run_campaign(SAMPLE_CSV, LABEL_COL, RARE_DEF, n_rows=50, max_passes=1,
                         privacy="bogus")
