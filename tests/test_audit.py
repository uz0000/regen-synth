"""
Independent auditability tests (G-G): a bundle verifies cleanly, tampering is
caught (integrity + the affected statistic), the disclosure bucket-floor holds,
and the manifest carries artifact hashes + metric versions.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regen.audit_bundle import verify_bundle, build_reference_aggregates, DEFAULT_MIN_BUCKET
from regen.metrics import metric_versions

SAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "transactions.csv")


def _gen(privacy, out):
    from regen.api import generate
    return generate(SAMPLE, label_col="is_fraud", n_rows=250, seed=42,
                    privacy=privacy, out_dir=out)


class TestCleanVerify:
    def test_floored_bundle_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            _gen("floored", d)
            rep = verify_bundle(d)
        assert rep["passed"]
        assert all(a["passed"] for a in rep["integrity"])
        # correlation is uncheckable under the floor; fisher + class_counts pass
        checked = {s["metric"]: s["passed"] for s in rep["stats"] if s["status"] == "checked"}
        assert checked.get("fisher_separation") and checked.get("class_counts")

    def test_none_bundle_checks_correlation(self):
        with tempfile.TemporaryDirectory() as d:
            _gen("none", d)
            rep = verify_bundle(d)
        assert rep["passed"]
        corr = [s for s in rep["stats"] if s["metric"] == "correlation_delta"][0]
        assert corr["status"] == "checked" and corr["passed"]


class TestTamperDetection:
    def test_tampering_a_rare_row_fails_verify(self):
        with tempfile.TemporaryDirectory() as d:
            s = _gen("none", d)
            n_rare = s["n_synthetic_rare"]
            df = pd.read_parquet(Path(d) / "pass_1_accepted.parquet")
            # Tamper a RARE row (they are the last n_rare rows) so both the hash
            # and the recomputed rare-correlation change.
            df.iloc[len(df) - 1, df.columns.get_loc("amount")] += 1e5
            df.to_parquet(Path(d) / "pass_1_accepted.parquet", index=False)
            rep = verify_bundle(d)
        assert not rep["passed"]
        # Integrity names the affected artifact...
        assert any(not a["passed"] and a["artifact"].endswith(".parquet")
                   for a in rep["integrity"])
        # ...and the value recomputation names the affected statistic.
        corr = [x for x in rep["stats"] if x["metric"] == "correlation_delta"][0]
        assert corr["status"] == "checked" and not corr["passed"]


def _gen_with_estimand(out, seed=7):
    """Generate a batch whose ScenarioSpec declares an OLS estimand y ~ x1 + x2."""
    from contracts.scenario import (ScenarioSpec, ScenarioIntent, ScenarioGates,
                                     EstimandSpec)
    from regen.api import generate
    rng = np.random.default_rng(0)
    n = 1500
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    y = 1.5 + 2.0 * x1 - 3.0 * x2 + rng.normal(0, 1.0, n)
    lab = (0.8 * x1 + rng.normal(0, 1, n) > 1.28).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y, "is_rare": lab})
    with tempfile.TemporaryDirectory() as d:
        csv = Path(d) / "data.csv"; df.to_csv(csv, index=False)
        spec = ScenarioSpec(
            intent=ScenarioIntent(label_col="is_rare", rare_mode="label",
                                  rare_value=1, n_rows=400, seed=seed),
            gates=ScenarioGates(privacy="none"),
            estimand=EstimandSpec(outcome="y", predictors=["x1", "x2"], family="ols"),
        )
        return generate(str(csv), scenario=spec, out_dir=out)


class TestEstimandVerify:
    def test_declared_estimand_certifies_and_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            s = _gen_with_estimand(d)
            # Summary + explanation carry the verdict; aggregates carry θ_real ± SE.
            assert s["estimand"]["declared"] is True
            assert s["estimand"]["status"] in ("certified", "not_preserved")
            agg = json.loads((Path(d) / "reference_aggregates.json").read_text())
            assert "estimand_real" in agg
            assert {"x1", "x2"} <= set(agg["estimand_real"]["coefficients"])
            rep = verify_bundle(d)
        est = [x for x in rep["stats"] if x["metric"] == "estimand_delta"][0]
        assert est["status"] == "checked" and est["passed"]

    def test_undeclared_estimand_is_uncheckable(self):
        with tempfile.TemporaryDirectory() as d:
            _gen("none", d)  # no scenario → no estimand declared
            rep = verify_bundle(d)
        est = [x for x in rep["stats"] if x["metric"] == "estimand_delta"][0]
        assert est["status"] == "uncheckable"

    def test_tampering_breaks_the_recomputed_coefficient(self):
        # The load-bearing property: the certified verdict is RECOMPUTED from the
        # delivered rows. Break a predictor's relationship and θ_synth must move —
        # so estimand_delta fails even though the reported verdict said certified.
        with tempfile.TemporaryDirectory() as d:
            _gen_with_estimand(d)
            path = Path(d) / "pass_1_accepted.parquet"
            df = pd.read_parquet(path)
            rng = np.random.default_rng(1)
            df["x1"] = rng.permutation(df["x1"].to_numpy())  # kills x1↔y association
            df.to_parquet(path, index=False)
            rep = verify_bundle(d)
        assert not rep["passed"]
        est = [x for x in rep["stats"] if x["metric"] == "estimand_delta"][0]
        assert est["status"] == "checked" and not est["passed"]


class TestManifestAttestation:
    def test_manifest_has_hashes_and_metric_versions(self):
        with tempfile.TemporaryDirectory() as d:
            _gen("floored", d)
            man = json.loads((Path(d) / "manifest.json").read_text())
        assert set(man["artifact_sha256"]) >= {
            "pass_1_accepted.parquet", "explanation.json", "reference_aggregates.json"}
        assert man["metric_versions"] == metric_versions()
        assert man["manifest_schema_version"] >= 1


class TestDisclosurePolicy:
    def _ingest(self, n_rare=15):
        from regen.api import ingest
        from contracts.types import RareEventDef, RareMode
        rng = np.random.default_rng(0)
        norm = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300), "y": 0})
        rare = pd.DataFrame({"a": rng.normal(3, 1, n_rare), "b": rng.normal(3, 1, n_rare), "y": 1})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.csv"
            pd.concat([norm, rare], ignore_index=True).to_csv(p, index=False)
            return ingest(str(p), "y", RareEventDef(mode=RareMode.LABEL, label_value=1))

    def test_quantiles_suppressed_below_bucket_floor(self):
        res = self._ingest(n_rare=15)   # rare count 15 < the stricter min_bucket 30
        agg = build_reference_aggregates(res, n_normal=100, n_rare=20, min_bucket=30)
        assert agg["quantiles_rare"] is None
        assert "quantiles_suppressed" in agg["disclosure"]
        # aggregates still expose correlation + moments (allowed), never rows
        assert "column_moments" in agg
        assert "correlation_rare" in agg

    def test_quantiles_published_above_floor(self):
        res = self._ingest(n_rare=15)   # rare count 15 >= min_bucket 5
        agg = build_reference_aggregates(res, n_normal=100, n_rare=20, min_bucket=5)
        assert agg["quantiles_rare"] is not None
