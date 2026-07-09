"""
TSTR / surrogate-quality tests (PRODUCT_SPEC §5.1).

Covers the metric's core invariants (a perfect surrogate recovers ~1.0; a noise
surrogate recovers far less; too-few real test rare → degenerate status) and the
end-to-end leakage-free path (`evaluate_surrogate`). Synthetic fixtures only.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from engine.examiner import measure_tstr, MIN_REAL_TEST_RARE
from contracts.types import RareEventDef, RareMode


def _fixture_csv(tmp_path, n_norm=600, n_rare=150, seed=0):
    rng = np.random.default_rng(seed)
    norm = pd.DataFrame({"a": rng.normal(0, 1, n_norm), "b": rng.normal(0, 1, n_norm), "y": 0})
    rare = pd.DataFrame({"a": rng.normal(3, 1, n_rare), "b": rng.normal(3, 1, n_rare), "y": 1})
    df = pd.concat([norm, rare], ignore_index=True)
    p = str(tmp_path / "f.csv")
    df.to_csv(p, index=False)
    return p


def _ingest(path):
    from regen.api import ingest
    return ingest(path, "y", RareEventDef(mode=RareMode.LABEL, label_value=1))


def _split(res, test_size=0.3, seed=1):
    real = pd.concat([res.normal_df, res.rare_df], ignore_index=True)
    strat = (real["y"] == 1).astype(int)
    return train_test_split(real, test_size=test_size, random_state=seed, stratify=strat)


class TestMetricInvariants:
    def test_perfect_surrogate_recovers_one(self, tmp_path):
        res = _ingest(_fixture_csv(tmp_path))
        tr, te = _split(res)
        # synth == real train → identical models → TSTR == TRTR → recovered == 1.0
        rep = measure_tstr(tr, tr, te, "y", res.field_dict, rare_value=1, seeds=(1,))
        assert rep.status == "ok"
        assert rep.recovered_roc_auc_median == 1.0
        assert all(m["recovered_roc_auc"] == 1.0 for m in rep.per_model)

    def test_noise_surrogate_recovers_far_less(self, tmp_path):
        res = _ingest(_fixture_csv(tmp_path))
        tr, te = _split(res)
        noise = tr.copy()
        rng = np.random.default_rng(0)
        for c in ("a", "b"):
            noise[c] = rng.permutation(noise[c].values)   # destroy the label↔feature signal
        rep = measure_tstr(noise, tr, te, "y", res.field_dict, rare_value=1, seeds=(1,))
        assert rep.status == "ok"
        # a signal-free surrogate can't match real; recovery is well below perfect
        assert rep.recovered_roc_auc_median < 0.9

    def test_insufficient_real_test_is_flagged(self, tmp_path):
        # 12 rare total → ~4 in a 30% test fold → below MIN_REAL_TEST_RARE
        res = _ingest(_fixture_csv(tmp_path, n_rare=12))
        tr, te = _split(res)
        rep = measure_tstr(tr, tr, te, "y", res.field_dict, rare_value=1, seeds=(1,))
        assert rep.status == "insufficient_real_test"
        assert rep.n_real_test_rare < MIN_REAL_TEST_RARE
        assert rep.recovered_roc_auc_median is None


class TestEndToEndLeakageFree:
    def test_evaluate_surrogate_runs_and_holds_out_test(self, tmp_path):
        from regen.api import evaluate_surrogate
        path = _fixture_csv(tmp_path, n_norm=700, n_rare=200)
        r = evaluate_surrogate(path, label_col="y", privacy="none", auto=False,
                               seed=1, tstr_seeds=(1,))
        t = r["tstr"]
        assert t["status"] == "ok"
        assert t["recovered_roc_auc_median"] is not None
        # the split is a genuine hold-out: train + test partition the real data
        assert r["n_train_real"] + r["n_test_real"] == 900
        # on a cleanly separable fixture a faithful surrogate recovers most of it
        assert t["recovered_roc_auc_median"] > 0.5

    def test_report_serializes(self, tmp_path):
        import json
        res = _ingest(_fixture_csv(tmp_path))
        tr, te = _split(res)
        rep = measure_tstr(tr, tr, te, "y", res.field_dict, rare_value=1, seeds=(1,))
        json.dumps(rep.to_dict())   # must be JSON-clean for the bundle/summary
