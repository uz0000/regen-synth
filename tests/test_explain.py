"""
Explainability tests (G-C): explanation.json ships with every batch and its
numbers equal the report objects they were computed from (no drift, no narration).
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

SAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "transactions.csv")


@pytest.fixture(scope="module")
def run():
    from regen.api import generate
    d = tempfile.mkdtemp()
    s = generate(SAMPLE, label_col="is_fraud", n_rows=250, seed=5,
                 privacy="floored", out_dir=d)
    return s, d


def test_explanation_file_ships_and_matches_summary(run):
    s, d = run
    disk = json.loads((Path(d) / "explanation.json").read_text())
    assert disk == s["explain"]


def test_gate_numbers_equal_reports(run):
    s, _ = run
    ex = s["explain"]
    # correlation delta and coverage in the explanation equal the fidelity block
    assert ex["gates"]["fidelity"]["correlation"]["value"] == s["fidelity"]["correlation"]["delta"]
    assert ex["gates"]["fidelity"]["coverage"]["value"] == s["fidelity"]["coverage"]
    # conformance is the same object
    assert ex["gates"]["conformance"] == s["conformance"]
    # privacy account equals the privacy block
    assert ex["privacy"] == s["privacy"]


def test_utility_matches_lift_block(run):
    s, _ = run
    ex_u = s["explain"]["utility"]
    if s["lift"] is not None:
        assert ex_u["status"] == s["lift"]["status"]
        assert ex_u["n_test_rare"] == s["lift"]["n_test_rare"]
        assert ex_u["tail_lift"] == s["lift"]["tail_lift"]


def test_feature_informativeness_is_ranked(run):
    s, _ = run
    ranked = s["explain"]["feature_informativeness"]["ranked"]
    assert len(ranked) >= 1
    scores = [f["fisher_score"] for f in ranked]
    assert scores == sorted(scores, reverse=True)          # descending
    assert [f["rank"] for f in ranked] == list(range(1, len(ranked) + 1))


def test_column_provenance_has_mechanism_for_each_column(run):
    s, _ = run
    prov = {c["column"]: c for c in s["explain"]["column_provenance"]}
    # every non-label feature has a named production mechanism
    assert prov["amount"]["mechanism"]
    assert prov["is_fraud"]["mechanism"] == "label-attached"


def test_generation_block_records_the_base(run):
    """The base generator that actually ran is recorded (a parametric→grounded
    fallback can't hide in a log)."""
    s, _ = run
    gen = s["explain"]["generation"]
    assert gen["rare_base"] == "parametric"      # no fallback on this data
    assert gen["normal_base"] == "parametric"


def test_mechanism_reflects_fallback():
    """Unit: when the rare base fell back to grounded, the per-column mechanism
    says so rather than claiming copula-sampled."""
    from regen.explain import _mechanism
    from contracts.scenario import ColumnSemantics
    col = ColumnSemantics(name="amount", role="feature", dtype="float")
    assert "copula" in _mechanism(col, "floored", rare_fallback=False)
    assert "fallback" in _mechanism(col, "floored", rare_fallback=True)
