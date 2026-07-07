"""
Preflight / capability-envelope tests (G-E). Each envelope rule has a synthetic
fixture that trips it. All fixtures are generated — no real dataset rows (G-F).
"""

import numpy as np
import pandas as pd
import pytest

from regen.api import preflight
from contracts.types import RareEventDef, RareMode


def _csv(tmp_path, df, name="f.csv"):
    p = str(tmp_path / name)
    df.to_csv(p, index=False)
    return p


def _levels(rep):
    return {c["check"]: c["level"] for c in rep["checks"]}


def _healthy(n_norm=300, n_rare=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.concat([
        pd.DataFrame({"a": rng.normal(0, 1, n_norm), "b": rng.normal(0, 1, n_norm), "y": 0}),
        pd.DataFrame({"a": rng.normal(3, 1, n_rare), "b": rng.normal(3, 1, n_rare), "y": 1}),
    ], ignore_index=True)


def test_healthy_dataset_ok(tmp_path):
    rep = preflight(_csv(tmp_path, _healthy()), label_col="y")
    assert rep["ok_to_generate"]
    assert _levels(rep)["rare_count"] == "ok"


def test_small_rare_warns(tmp_path):
    rep = preflight(_csv(tmp_path, _healthy(n_rare=12)), label_col="y")
    assert _levels(rep)["rare_count"] == "warn"        # <14, ≥10 → insufficient lift
    assert rep["ok_to_generate"]                       # still generates


def test_too_few_rare_unsupported(tmp_path):
    rep = preflight(_csv(tmp_path, _healthy(n_rare=6)), label_col="y")
    assert not rep["ok_to_generate"]
    assert _levels(rep)["rare_count"] == "unsupported"


def test_all_categorical_degraded(tmp_path):
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "region": rng.choice(["n", "s", "e", "w"], size=n),
        "plan": rng.choice(["a", "b", "c"], size=n),
        "y": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
    })
    rep = preflight(_csv(tmp_path, df), label_col="y")
    assert _levels(rep)["privacy_floor"] == "degraded"


def test_low_cardinality_integer_degraded(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.concat([
        pd.DataFrame({"code": rng.integers(0, 6, 300), "x": rng.normal(0, 1, 300), "y": 0}),
        pd.DataFrame({"code": rng.integers(0, 6, 40), "x": rng.normal(3, 1, 40), "y": 1}),
    ], ignore_index=True)
    rep = preflight(_csv(tmp_path, df), label_col="y")
    assert _levels(rep).get("low_cardinality_integer") == "degraded"


def test_high_dimensionality_warns(tmp_path):
    rng = np.random.default_rng(0)
    cols = {f"f{i}": rng.normal(0, 1, 340) for i in range(40)}   # 40 features
    cols["y"] = np.r_[np.zeros(320), np.ones(20)]                # 20 rare < 40 feats
    rep = preflight(_csv(tmp_path, pd.DataFrame(cols)), label_col="y")
    assert _levels(rep).get("dimensionality") == "warn"


def test_constant_column_warns(tmp_path):
    df = _healthy()
    df["const"] = 7.0
    rep = preflight(_csv(tmp_path, df), label_col="y")
    assert _levels(rep).get("constant_column") == "warn"


def test_timestamp_column_unsupported(tmp_path):
    df = _healthy()
    df["event_timestamp"] = np.arange(len(df))
    rep = preflight(_csv(tmp_path, df), label_col="y")
    assert _levels(rep).get("time_series") == "unsupported"
    assert not rep["ok_to_generate"]


def test_high_cardinality_and_free_text(tmp_path):
    rng = np.random.default_rng(0)
    n_norm, n_rare = 300, 40
    long_text = [f"a rather long free-text note number {i} with detail" for i in range(n_norm + n_rare)]
    df = pd.concat([
        pd.DataFrame({"a": rng.normal(0, 1, n_norm), "y": 0}),
        pd.DataFrame({"a": rng.normal(3, 1, n_rare), "y": 1}),
    ], ignore_index=True)
    df["note"] = long_text
    rep = preflight(_csv(tmp_path, df), label_col="y")
    lv = _levels(rep)
    assert lv.get("high_cardinality") == "warn"
    assert lv.get("free_text") == "unsupported"
