"""
ScenarioSpec contract tests (G-A): serialization round-trip, structural fill,
and — the load-bearing property — a batch is reproducible from its persisted
spec, bit-for-bit (Invariant 2 extended to the use-case contract).
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from contracts.scenario import (
    ScenarioSpec, ScenarioIntent, ScenarioGates, ColumnSemantics, EstimandSpec,
    columns_from_field_dict,
)
from contracts.types import RareEventDef, RareMode

SAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "transactions.csv")
SCENARIO_YAML = str(Path(__file__).resolve().parent.parent / "examples" / "scenario_fraud.yaml")


def _hash(df):
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


class TestSerialization:
    def test_json_round_trip(self):
        spec = ScenarioSpec(
            columns={"a": ColumnSemantics(name="a", role="feature", dtype="float",
                                          min=0.0, max=1.0, source="user")},
            intent=ScenarioIntent(label_col="y", rare_mode="label", rare_value=1,
                                  n_rows=200, seed=7),
            gates=ScenarioGates(privacy="floored", delta=0.5),
            notes="t",
        )
        back = ScenarioSpec.from_json(spec.to_json())
        assert back.to_dict() == spec.to_dict()

    def test_yaml_round_trip(self):
        spec = ScenarioSpec(intent=ScenarioIntent(label_col="y", n_rows=50))
        back = ScenarioSpec.from_yaml(spec.to_yaml())
        assert back.to_dict() == spec.to_dict()

    def test_intent_builds_rare_def(self):
        spec = ScenarioSpec(intent=ScenarioIntent(rare_mode="percentile",
                                                  percentile=0.05, tail="upper"))
        rd = spec.rare_def()
        assert rd.mode == RareMode.PERCENTILE and rd.percentile == 0.05 and rd.tail == "upper"


class TestEstimandSpec:
    def test_undeclared_by_default(self):
        # A fresh spec carries an empty estimand → the certificate omits it.
        spec = ScenarioSpec()
        assert spec.estimand.declared() is False
        assert spec.estimand.targets() == []

    def test_declared_and_targets(self):
        e = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="ols")
        assert e.declared() is True
        # No explicit coefficients_of_interest → certify every predictor.
        assert e.targets() == ["x1", "x2"]
        e2 = EstimandSpec(outcome="y", predictors=["x1", "x2"],
                          coefficients_of_interest=["x2"])
        assert e2.targets() == ["x2"]

    def test_outcome_without_predictors_is_undeclared(self):
        assert EstimandSpec(outcome="y").declared() is False

    def test_round_trip(self):
        e = EstimandSpec(outcome="y", predictors=["x1", "x2"], family="logit",
                         coefficients_of_interest=["x1"], ci_level=0.9,
                         rule="within_ci")
        assert EstimandSpec.from_dict(e.to_dict()) == e

    def test_scenario_round_trip_with_estimand(self):
        spec = ScenarioSpec(
            intent=ScenarioIntent(label_col="y"),
            estimand=EstimandSpec(outcome="y", predictors=["a", "b"], family="ols"),
        )
        back = ScenarioSpec.from_json(spec.to_json())
        assert back.to_dict() == spec.to_dict()
        assert back.estimand.declared() is True

    def test_backward_compat_missing_estimand_key(self):
        # A spec persisted before estimands existed has no "estimand" key; it must
        # still load, as an undeclared estimand (never crash on old manifests).
        legacy = ScenarioSpec(intent=ScenarioIntent(label_col="y")).to_dict()
        legacy.pop("estimand")
        spec = ScenarioSpec.from_dict(legacy)
        assert spec.estimand.declared() is False


class TestStructuralFill:
    def test_columns_from_field_dict(self):
        from regen.api import ingest
        res = ingest(SAMPLE, "is_fraud", RareEventDef(mode=RareMode.LABEL, label_value=1))
        cols = columns_from_field_dict(res.field_dict, res.label_col)
        assert cols["is_fraud"].role == "target"
        assert all(c.source == "structural" for c in cols.values())
        # bounds are populated for continuous columns from the profile
        assert cols["amount"].min is not None


class TestManifestRoundTrip:
    def test_generate_persists_spec_and_replays_bit_identical(self):
        from regen.api import generate
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            s = generate(SAMPLE, label_col="is_fraud", n_rows=200, seed=42,
                         privacy="floored", out_dir=d1)
            df1 = pd.read_parquet(s["best_batch_path"])
            man = json.loads((Path(d1) / "manifest.json").read_text())
            assert man["scenario"] is not None
            assert s["scenario"]["intent"]["label_col"] == "is_fraud"

            # Reconstruct the spec from the manifest and re-run from it alone.
            spec = ScenarioSpec.from_dict(man["scenario"])
            s2 = generate(SAMPLE, scenario=spec, out_dir=d2)
            df2 = pd.read_parquet(s2["best_batch_path"])
        assert _hash(df1) == _hash(df2), "replay from persisted spec not bit-identical"

    def test_scenario_yaml_drives_generation(self):
        """The example scenario YAML drives an end-to-end run and round-trips."""
        from regen.api import generate
        spec = ScenarioSpec.load_yaml(SCENARIO_YAML)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            s = generate(SAMPLE, scenario=spec, out_dir=d1)
            assert s["label_col"] == "is_fraud"
            assert s["rare_ratio"] == 0.25
            df1 = pd.read_parquet(s["best_batch_path"])
            # A scenario.yaml is saved next to the batch (the shareable unit).
            assert (Path(d1) / "scenario.yaml").exists()
            # Re-run from the persisted (manifest) spec → identical.
            man = json.loads((Path(d1) / "manifest.json").read_text())
            s2 = generate(SAMPLE, scenario=ScenarioSpec.from_dict(man["scenario"]), out_dir=d2)
            df2 = pd.read_parquet(s2["best_batch_path"])
        assert _hash(df1) == _hash(df2)


class TestOtherEntryPointsAcceptScenario:
    def test_campaign_accepts_scenario(self):
        from regen.api import run_campaign
        spec = ScenarioSpec(intent=ScenarioIntent(label_col="is_fraud", rare_mode="label",
                                                  rare_value=1, n_rows=120, seed=1),
                            gates=ScenarioGates(privacy="none"))
        with tempfile.TemporaryDirectory() as out:
            cr = run_campaign(SAMPLE, scenario=spec, max_passes=1, out_dir=out)
        assert cr.n_rare > 0

    def test_screen_accepts_scenario(self):
        from regen.api import screen
        spec = ScenarioSpec(intent=ScenarioIntent(label_col="is_fraud", rare_mode="label",
                                                  rare_value=1, seed=1))
        res = screen(SAMPLE, scenario=spec, quick_campaign=False)
        assert res.recommended_method in ("REGEN", "SMOTE")
