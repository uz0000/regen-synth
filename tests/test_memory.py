"""
M5 — persistent memory of explored regions.

Two things must hold:
  1. The store round-trips across processes: regions written in one run are
     visible to the next (this is what makes scheduled unattended runs improve
     over time instead of repeating themselves).
  2. Scout's R-EPIG selection is biased away from already-explored anchors when
     that memory is supplied — without it, selection is unchanged.
"""

import numpy as np
import pytest

from agent-runtime.memory import ExploredRegion, ExploredRegionMemory


# ── Memory persistence ────────────────────────────────────────────────────────

def _region(anchor, lift=0.5, accepted=True, idx=0):
    return ExploredRegion(
        feature_idx=idx,
        feature_name="amount",
        percentile_low=0.9,
        percentile_high=1.0,
        anchor_point=list(anchor),
        accepted=accepted,
        coverage_rate=0.8,
        tail_lift=lift,
        pass_index=0,
    )


def test_memory_round_trips_across_processes(tmp_path):
    path = str(tmp_path / "explored.json")

    mem = ExploredRegionMemory.load(path)
    assert mem.regions == []
    mem.record(_region([1.0, 2.0, 3.0], lift=0.4))
    mem.record(_region([5.0, 6.0, 7.0], lift=0.7))
    mem.save()

    # Fresh load simulates a separate process / a later scheduled run
    reloaded = ExploredRegionMemory.load(path)
    assert len(reloaded.regions) == 2
    assert reloaded.best().tail_lift == 0.7
    assert reloaded.anchor_points() == [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]]


def test_record_dedups_on_anchor(tmp_path):
    mem = ExploredRegionMemory(path=tmp_path / "m.json")
    mem.record(_region([1.0, 1.0, 1.0], lift=0.2))
    mem.record(_region([1.0, 1.0, 1.0], lift=0.9))  # same anchor, newer outcome
    assert len(mem.regions) == 1
    assert mem.regions[0].tail_lift == 0.9


def test_summary_counts_accepted(tmp_path):
    mem = ExploredRegionMemory(path=tmp_path / "m.json")
    mem.record(_region([1.0, 0.0], lift=0.5, accepted=True))
    mem.record(_region([2.0, 0.0], lift=0.0, accepted=False))
    s = mem.summary()
    assert s["n_explored"] == 2
    assert s["n_accepted"] == 1
    assert s["best_lift"] == 0.5


# ── Scout diversity penalty ───────────────────────────────────────────────────

def test_explored_penalty_downweights_near_anchors():
    """A candidate sitting on an explored anchor must lose score; a far one keeps it."""
    from engine.scout.repig import ScoutConfig, _apply_explored_penalty

    candidates = np.array([
        [0.0, 0.0],   # sits exactly on the explored anchor below
        [10.0, 10.0], # far away
    ])
    scores = np.array([1.0, 1.0])
    explored = [[0.0, 0.0]]
    std = np.array([1.0, 1.0])
    config = ScoutConfig(explored_penalty=0.7)

    out = _apply_explored_penalty(scores, candidates, explored, std, config)

    # Near candidate suppressed by ~0.7 → ~0.3; far candidate ~unchanged
    assert out[0] < 0.4
    assert out[1] > 0.95
    assert out[0] < out[1]


def test_explored_penalty_noop_without_memory():
    """With no explored points, scores pass through unchanged."""
    from engine.scout.repig import ScoutConfig, _apply_explored_penalty

    scores = np.array([1.0, 2.0, 3.0])
    candidates = np.random.default_rng(0).standard_normal((3, 4))
    out = _apply_explored_penalty(scores, candidates, None, np.ones(4), ScoutConfig())
    np.testing.assert_array_equal(scores, out)


def test_explored_penalty_handles_shape_mismatch():
    """A schema change (different feature count) must not crash selection."""
    from engine.scout.repig import ScoutConfig, _apply_explored_penalty

    scores = np.array([1.0, 2.0])
    candidates = np.zeros((2, 3))
    explored = [[0.0, 0.0]]  # wrong dimensionality
    out = _apply_explored_penalty(scores, candidates, explored, np.ones(3), ScoutConfig())
    np.testing.assert_array_equal(scores, out)
