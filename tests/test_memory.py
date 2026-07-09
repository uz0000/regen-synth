"""
Scout within-run memory — the explored-region diversity penalty.

The active-learning loop tracks every target Scout has already amplified this
campaign and biases the next selection away from those anchors. Three things
must hold:

  1. A candidate sitting on an already-explored anchor loses score.
  2. A candidate far from every explored anchor is left ~unchanged.
  3. The penalty is a no-op when no memory is supplied, and never crashes on
     a schema/shape mismatch (e.g. a re-run after the feature set changed).

This is the within-run memory that lives in the engine (engine.scout.targeting),
threaded through regen.api.run_campaign()'s `explored_points` accumulator.
"""

import numpy as np
import pytest


# ── Scout diversity penalty ───────────────────────────────────────────────────

def test_explored_penalty_downweights_near_anchors():
    """A candidate sitting on an explored anchor must lose score; a far one keeps it."""
    from engine.scout.targeting import ScoutConfig, _apply_explored_penalty

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
    from engine.scout.targeting import ScoutConfig, _apply_explored_penalty

    scores = np.array([1.0, 2.0, 3.0])
    candidates = np.random.default_rng(0).standard_normal((3, 4))
    out = _apply_explored_penalty(scores, candidates, None, np.ones(4), ScoutConfig())
    np.testing.assert_array_equal(scores, out)


def test_explored_penalty_handles_shape_mismatch():
    """A schema change (different feature count) must not crash selection."""
    from engine.scout.targeting import ScoutConfig, _apply_explored_penalty

    scores = np.array([1.0, 2.0])
    candidates = np.zeros((2, 3))
    explored = [[0.0, 0.0]]  # wrong dimensionality
    out = _apply_explored_penalty(scores, candidates, explored, np.ones(3), ScoutConfig())
    np.testing.assert_array_equal(scores, out)
