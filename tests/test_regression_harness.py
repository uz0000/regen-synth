"""
Regression harness tests (G-D) — fast checks of the comparison logic and the
committed baseline (the full harness run is a minutes-long pre-push step, not a
unit test).
"""

import json
from pathlib import Path

from benchmark.run_regression import _compare, CANONICAL

BASELINE = Path(__file__).resolve().parent.parent / "benchmark" / "BASELINES" / "regression_baseline.json"


def _row(**kw):
    base = {"fidelity_score": 1.0, "coverage": 1.0, "corr_delta": 0.05,
            "gate_passed": True, "tail_lift": 0.2, "verified": True, "wall_time_s": 5.0}
    base.update(kw)
    return base


def test_no_drift_is_clean():
    base = {"k": _row()}
    assert _compare(dict(base), base) == []


def test_each_drift_kind_is_caught():
    base = {"k": _row()}
    assert _compare({"k": _row(gate_passed=False)}, base)          # gate flip
    assert _compare({"k": _row(fidelity_score=0.7)}, base)         # fidelity drop
    assert _compare({"k": _row(coverage=0.5)}, base)               # coverage drop
    assert _compare({"k": _row(corr_delta=0.4)}, base)             # correlation worse
    assert _compare({"k": _row(tail_lift=0.0)}, base)              # lift drop
    assert _compare({"k": _row(verified=False)}, base)            # bundle failed verify
    assert _compare({"k": _row(wall_time_s=999.0)}, base)          # runtime blow-up


def test_missing_baseline_is_flagged():
    assert _compare({"newkey": _row()}, {})


def test_committed_baseline_is_wellformed():
    payload = json.loads(BASELINE.read_text())
    assert "code_version" in payload and "run_date" in payload
    results = payload["results"]
    # every canonical dataset × privacy has a baseline row that verified
    for fname, _ in CANONICAL:
        for privacy in ("none", "floored"):
            key = f"{fname}__{privacy}"
            assert key in results, f"missing baseline {key}"
            assert results[key]["verified"] is True
