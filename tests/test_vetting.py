"""
Vetting gate + conformance audit tests (G-B).

Each of the gate's data-facing rules gets a failing-then-passing check, plus the
G-B done-when demonstration: two different scenario specs over the same dataset
produce correspondingly different, gate-passing batches.
"""

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from contracts.scenario import (
    ScenarioSpec, ScenarioIntent, ScenarioGates, ColumnSemantics,
)
from contracts.types import RareEventDef, RareMode
from regen.vetting import vet_scenario, CONFIDENCE_FLOOR
from engine.auditor import check_conformance

SAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "transactions.csv")


@pytest.fixture(scope="module")
def ingest_fraud():
    from regen.api import ingest
    return ingest(SAMPLE, "is_fraud", RareEventDef(mode=RareMode.LABEL, label_value=1))


def _spec_with_col(col: ColumnSemantics) -> ScenarioSpec:
    return ScenarioSpec(columns={col.name: col})


def _verdict(verdicts, field):
    return next((v for v in verdicts if v.field == field), None)


class TestGateRules:
    def test_bounds_must_contain_observed(self, ingest_fraud):
        obs_min = ingest_fraud.field_dict["amount"].min_val
        # PASS: a wider (safe) lower bound is accepted.
        col = ColumnSemantics(name="amount", source="user", min=obs_min - 100.0)
        cols, v = vet_scenario(_spec_with_col(col), ingest_fraud)
        assert _verdict(v, "amount.min").decision == "accepted"
        assert cols["amount"].min == obs_min - 100.0
        # FAIL: a lower bound above the observed min would clip real data → rejected.
        col = ColumnSemantics(name="amount", source="user", min=obs_min + 100.0)
        cols, v = vet_scenario(_spec_with_col(col), ingest_fraud)
        assert _verdict(v, "amount.min").decision == "rejected"
        assert cols["amount"].min != obs_min + 100.0     # kept structural

    def test_max_must_contain_observed(self, ingest_fraud):
        obs_max = ingest_fraud.field_dict["amount"].max_val
        col = ColumnSemantics(name="amount", source="user", max=1.0)  # below observed
        _, v = vet_scenario(_spec_with_col(col), ingest_fraud)
        assert _verdict(v, "amount.max").decision == "rejected"
        col = ColumnSemantics(name="amount", source="user", max=obs_max + 1000.0)
        _, v = vet_scenario(_spec_with_col(col), ingest_fraud)
        assert _verdict(v, "amount.max").decision == "accepted"

    def test_integrality_must_match_observed(self, ingest_fraud):
        # amount is not integral → claiming integer is rejected.
        col = ColumnSemantics(name="amount", source="user", integer=True)
        _, v = vet_scenario(_spec_with_col(col), ingest_fraud)
        assert _verdict(v, "amount.integer").decision == "rejected"

    def test_closed_vocabulary(self, ingest_fraud):
        col = ColumnSemantics(name="amount", source="user", role="wormhole")
        _, v = vet_scenario(_spec_with_col(col), ingest_fraud)
        assert _verdict(v, "amount.role").decision == "rejected"
        assert _verdict(v, "amount.role").rule == "closed_vocabulary"

    def test_authority_order_valid_role_applies(self, ingest_fraud):
        col = ColumnSemantics(name="amount", source="user", role="identifier")
        cols, v = vet_scenario(_spec_with_col(col), ingest_fraud)
        assert _verdict(v, "amount.role").decision == "accepted"
        assert cols["amount"].role == "identifier"

    def test_confidence_fallback(self, ingest_fraud):
        col = ColumnSemantics(name="amount", source="model", role="identifier",
                              confidence=CONFIDENCE_FLOOR - 0.1)
        cols, v = vet_scenario(_spec_with_col(col), ingest_fraud)
        assert _verdict(v, "amount").decision == "fallback"
        assert cols["amount"].role == "feature"          # structural, not the proposal

    def test_declared_column_not_in_data_rejected(self, ingest_fraud):
        col = ColumnSemantics(name="ghost_col", source="user")
        _, v = vet_scenario(_spec_with_col(col), ingest_fraud)
        assert _verdict(v, "ghost_col").decision == "rejected"

    def test_metadata_only_no_value_field(self):
        """Rule 1: a source can only emit metadata — ColumnSemantics has no field
        that carries a data value."""
        import dataclasses
        names = {f.name for f in dataclasses.fields(ColumnSemantics)}
        assert "value" not in names and "values" not in names


class TestConformance:
    def test_violation_fails_the_batch(self, ingest_fraud):
        # A batch with a value above the vetted max is non-conformant.
        spec = ScenarioSpec(columns={
            "amount": ColumnSemantics(name="amount", role="feature", max=100.0),
        })
        good = pd.DataFrame({"amount": [10.0, 50.0, 99.0]})
        bad = pd.DataFrame({"amount": [10.0, 50.0, 250.0]})
        assert check_conformance(good, spec).passed
        rep = check_conformance(bad, spec)
        assert not rep.passed
        assert rep.violations[0]["column"] == "amount"
        assert rep.violations[0]["n_rows"] == 1

    def test_identifier_uniqueness(self):
        spec = ScenarioSpec(columns={
            "id": ColumnSemantics(name="id", role="identifier"),
        })
        assert check_conformance(pd.DataFrame({"id": [1, 2, 3]}), spec).passed
        assert not check_conformance(pd.DataFrame({"id": [1, 2, 2]}), spec).passed


class TestTwoScenariosDiffer:
    """G-B done-when: the same dataset under two different scenario specs produces
    correspondingly different, gate-passing batches."""

    def _hash(self, df):
        return hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:16]

    def test_fraud_vs_tailrisk(self):
        from regen.api import generate
        fraud = ScenarioSpec(
            intent=ScenarioIntent(task="detector_training", label_col="is_fraud",
                                  rare_mode="label", rare_value=1, n_rows=300,
                                  seed=7, rare_ratio=0.25),
            gates=ScenarioGates(privacy="none"),
        )
        tail = ScenarioSpec(
            intent=ScenarioIntent(task="benchmarking", label_col="amount",
                                  rare_mode="percentile", percentile=0.05, tail="upper",
                                  n_rows=300, seed=7, rare_ratio=0.25),
            gates=ScenarioGates(privacy="none"),
        )
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            s1 = generate(SAMPLE, scenario=fraud, out_dir=d1)
            s2 = generate(SAMPLE, scenario=tail, out_dir=d2)
            df1 = pd.read_parquet(s1["best_batch_path"])
            df2 = pd.read_parquet(s2["best_batch_path"])
        assert s1["fidelity"]["passed"] and s2["fidelity"]["passed"]
        assert s1["conformance"]["passed"] and s2["conformance"]["passed"]
        assert s1["label_col"] == "is_fraud" and s2["label_col"] == "amount"
        assert self._hash(df1) != self._hash(df2)   # correspondingly different batches
