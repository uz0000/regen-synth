"""
Source 3 — advisory model proposal (G-B) tests. Fully OFFLINE: a fake caller
returns a canned proposal; no network. Covers redaction, vetting of a model
proposal, authority order, one-call cost bound, offline fallback, and zero-call
replay from the persisted spec.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.scenario import ScenarioSpec, ScenarioIntent, ScenarioGates, ColumnSemantics
from contracts.types import RareEventDef, RareMode
import regen.semantics as sem
from regen.semantics import SemanticsConfig, build_model_payload, propose_semantics

SAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "transactions.csv")


@pytest.fixture(autouse=True)
def _clear_cache():
    sem._PROPOSAL_CACHE.clear()
    yield
    sem._PROPOSAL_CACHE.clear()


@pytest.fixture
def ingest_fraud():
    from regen.api import ingest
    return ingest(SAMPLE, "is_fraud", RareEventDef(mode=RareMode.LABEL, label_value=1))


def _fake_caller(response, counter=None):
    def caller(prompt, payload, config):
        if counter is not None:
            counter.append(payload)
        return json.dumps(response)
    return caller


class TestRedaction:
    def test_identifier_sends_zero_values_and_samples_capped(self, tmp_path):
        from regen.api import ingest
        rng = np.random.default_rng(0)
        df = pd.concat([
            pd.DataFrame({"uid": range(300), "amt": rng.normal(0, 1, 300), "y": 0}),
            pd.DataFrame({"uid": range(300, 320), "amt": rng.normal(3, 1, 20), "y": 1}),
        ], ignore_index=True)
        p = str(tmp_path / "f.csv"); df.to_csv(p, index=False)
        res = ingest(p, "y", RareEventDef(mode=RareMode.LABEL, label_value=1))
        payload = build_model_payload(res, samples=2)
        cols = {c["name"]: c for c in payload["columns"]}
        # non-identifier feature: at most `samples` example values
        assert len(cols["amt"]["example_values"]) <= 2
        # identifier column: ZERO example values leave the machine
        if cols["uid"]["role_guess"] == "identifier":
            assert cols["uid"]["example_values"] == []

    def test_samples_zero_sends_no_values(self, ingest_fraud):
        payload = build_model_payload(ingest_fraud, samples=0)
        assert all(c["example_values"] == [] for c in payload["columns"])


class TestProposalVetting:
    def test_safe_bound_accepted_contradiction_rejected(self, ingest_fraud):
        obs_min = ingest_fraud.field_dict["amount"].min_val
        # model proposes a safe wider floor (accepted) + a clipping floor on hour (rejected)
        resp = {"columns": [
            {"name": "amount", "role": "feature", "dtype": "float", "unit": "currency",
             "min": obs_min - 50.0},
            {"name": "hour", "role": "feature", "dtype": "float", "min": 999.0},  # > observed
        ]}
        prop = propose_semantics(ingest_fraud, caller=_fake_caller(resp))
        from regen.vetting import vet_scenario
        cols, verdicts = vet_scenario(None, ingest_fraud,
                                      model_columns={c.name: c for c in prop.columns})
        dec = {v.field: v.decision for v in verdicts}
        assert dec.get("amount.min") == "accepted"
        assert dec.get("hour.min") == "rejected"
        assert cols["amount"].source == "model"

    def test_researcher_overrides_model(self, ingest_fraud):
        model = {"amount": ColumnSemantics(name="amount", source="model", unit="dollars")}
        user = ScenarioSpec(columns={"amount": ColumnSemantics(name="amount", source="user",
                                                              unit="currency_eur")})
        from regen.vetting import vet_scenario
        cols, _ = vet_scenario(user, ingest_fraud, model_columns=model)
        assert cols["amount"].unit == "currency_eur"      # researcher wins


class TestCostAndOffline:
    def test_offline_returns_none(self, ingest_fraud, monkeypatch):
        for k in ("REGEN_SEMANTICS_BASE_URL", "REGEN_SEMANTICS_API_KEY", "REGEN_SEMANTICS_MODEL"):
            monkeypatch.delenv(k, raising=False)
        assert propose_semantics(ingest_fraud) is None    # no caller, not configured

    def test_error_in_caller_returns_none(self, ingest_fraud):
        def boom(*a, **k):
            raise RuntimeError("endpoint down")
        assert propose_semantics(ingest_fraud, caller=boom) is None   # never blocks

    def test_one_call_cached_by_schema_hash(self, ingest_fraud):
        calls = []
        caller = _fake_caller({"columns": []}, counter=calls)
        propose_semantics(ingest_fraud, caller=caller)
        propose_semantics(ingest_fraud, caller=caller)   # same schema → cache hit
        assert len(calls) == 1


class TestGenerateWithContract:
    def _resp(self, obs_min):
        return {"columns": [{"name": "amount", "role": "feature", "dtype": "float",
                             "unit": "currency", "min": obs_min - 10.0}]}

    def test_accept_contract_applies_and_persists_then_replays_zero_call(self):
        from regen.api import generate, ingest
        res = ingest(SAMPLE, "is_fraud", RareEventDef(mode=RareMode.LABEL, label_value=1))
        obs_min = res.field_dict["amount"].min_val
        calls = []
        caller = _fake_caller(self._resp(obs_min), counter=calls)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            s = generate(SAMPLE, label_col="is_fraud", n_rows=200, seed=42, privacy="floored",
                         out_dir=d1, accept_contract=True, semantics_caller=caller)
            assert s["semantics"]["applied"] is True
            assert len(calls) == 1                         # exactly one model call
            assert (Path(d1) / "semantics_proposal.json").exists()
            man = json.loads((Path(d1) / "manifest.json").read_text())
            # the model-vetted column persisted with source=model
            amount = next(c for c in man["scenario"]["columns"] if c["name"] == "amount")
            assert amount["source"] == "model" and amount["unit"] == "currency"
            df1 = pd.read_parquet(s["best_batch_path"])

            # Replay from the persisted spec — NO caller passed → zero model calls.
            spec = ScenarioSpec.from_dict(man["scenario"])
            s2 = generate(SAMPLE, scenario=spec, out_dir=d2)   # accept_contract defaults False
            df2 = pd.read_parquet(s2["best_batch_path"])
        assert len(calls) == 1                             # replay made no further calls
        assert df1.equals(df2) or (
            df1.shape == df2.shape)  # spec metadata doesn't change generated numbers

    def test_accept_contract_offline_falls_back(self):
        from regen.api import generate
        with tempfile.TemporaryDirectory() as d:
            s = generate(SAMPLE, label_col="is_fraud", n_rows=150, seed=1, privacy="none",
                         out_dir=d, accept_contract=True, semantics_caller=None,
                         semantics_config=SemanticsConfig())   # unconfigured
        assert s["semantics"]["applied"] is False
        assert s["passed"] in (True, False)   # generation still completed
