"""
Decision-support surface tests (PRODUCT_SPEC §5.3): a transparent tradeoff
frontier + plain-language diagnosis + recommend-with-override — it surfaces
options, it does NOT pick for the user.
"""

from pathlib import Path

import pytest

SAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "transactions.csv")


def _summary(passed, coverage=1.0, corr=(0.1, True), cols=(), conf=True, privacy=None):
    return {
        "passed": passed,
        "fidelity": {
            "passed": passed if not cols else all(c[1] for c in cols),
            "coverage": coverage,
            "correlation": {"delta": corr[0], "passed": corr[1]},
            "columns": [{"col": c[0], "passed": c[1]} for c in cols],
        },
        "conformance": {"passed": conf},
        "privacy": privacy,
    }


class TestDiagnosis:
    def test_shippable(self):
        from regen.api import _diagnose
        assert "shippable" in _diagnose(_summary(True))

    def test_low_coverage_floored_blames_the_floor(self):
        from regen.api import _diagnose
        s = _summary(False, coverage=0.2,
                     privacy={"passed": True, "floor_applied": True, "min_distance": 1.0})
        msg = _diagnose(s)
        assert "δ-floor" in msg and msg.startswith("not shippable")

    def test_low_coverage_none_does_not_blame_the_floor(self):
        from regen.api import _diagnose
        s = _summary(False, coverage=0.2, privacy=None)   # no floor in the none case
        msg = _diagnose(s)
        assert "δ-floor" not in msg
        assert "poor fit" in msg

    def test_correlation_failure_named(self):
        from regen.api import _diagnose
        s = _summary(False, corr=(0.4, False))
        assert "correlation" in _diagnose(s)


class TestFrontier:
    def test_frontier_surfaces_options_and_recommends(self):
        from regen.api import explore_options
        rep = explore_options(SAMPLE, label_col="is_fraud", deltas=(0.5,),
                              n_rows=150, seed=1)
        opts = rep["options"]
        assert len(opts) == 2                                  # none + floored@0.5
        assert opts[0]["privacy"] == "none" and opts[1]["privacy"] == "floored"
        assert all("diagnosis" in o for o in opts)
        # transactions ships under the floor → recommend the floored option
        assert rep["recommended"] == 1
        assert opts[1]["shippable"]
        assert "override" in rep["note"].lower()

    def test_returns_options_does_not_auto_commit(self):
        """It returns a report for the human — it does not generate a 'final'
        artifact or pick silently."""
        from regen.api import explore_options
        rep = explore_options(SAMPLE, label_col="is_fraud", deltas=(0.5,),
                              n_rows=120, seed=2)
        assert set(rep.keys()) == {"options", "recommended", "note"}
        # 'recommended' is an index into options (a labelled default), or None
        assert rep["recommended"] is None or 0 <= rep["recommended"] < len(rep["options"])
