"""
Intent → ScenarioSpec proposer tests (PRODUCT_SPEC §5.2). Fully offline: a fake
caller returns a canned scenario JSON; no network. Verifies the draft is always
valid, the model *informs* under validation (invalid fields ignored, never
obeyed), and the draft drives generation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from contracts.scenario import ScenarioSpec

SAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "transactions.csv")


def _fake(resp):
    def caller(prompt, payload, config):
        return json.dumps(resp)
    return caller


class TestOfflineStructuralDraft:
    def test_structural_draft_when_no_model(self):
        from regen.api import draft_scenario
        draft, proposal = draft_scenario(SAMPLE, label_col="is_fraud", goal="x", n_rows=250)
        assert proposal is None                                  # offline
        assert isinstance(draft, ScenarioSpec)
        assert draft.intent.label_col == "is_fraud"
        assert draft.intent.n_rows == 250
        assert draft.provenance["drafted_by"] == "structural"
        assert draft.columns["amount"].source == "structural"
        ScenarioSpec.from_yaml(draft.to_yaml())                  # valid + round-trips


class TestModelInforms:
    def test_valid_proposal_applied(self):
        from regen.api import draft_scenario
        resp = {
            "intent": {"task": "data_sharing", "rare_mode": "label", "rare_value": 1,
                       "rare_ratio": 0.3,
                       "focus_features": ["amount", "merchant_risk", "ghost"],
                       "mode": "boost"},
            "gates": {"privacy": "none", "delta": 0.7},
            "columns": [{"name": "amount", "role": "feature", "dtype": "float",
                         "unit": "currency", "min": 0.0}],
        }
        draft, proposal = draft_scenario(SAMPLE, label_col="is_fraud", caller=_fake(resp))
        assert proposal is not None
        assert draft.intent.task == "data_sharing"
        assert draft.intent.rare_ratio == 0.3
        assert draft.intent.mode == "boost"
        assert draft.intent.focus_features == ["amount", "merchant_risk"]   # 'ghost' dropped
        assert draft.gates.privacy == "none" and draft.gates.delta == 0.7
        assert draft.columns["amount"].source == "model"
        assert draft.columns["amount"].unit == "currency"
        assert draft.provenance["drafted_by"] == "model+structural"

    def test_invalid_proposal_is_ignored_not_obeyed(self):
        from regen.api import draft_scenario
        resp = {"intent": {"task": "wormhole", "label_col": "ghost_col",
                           "rare_ratio": 5, "mode": "turbo"},
                "gates": {"privacy": "bogus", "delta": 9}}
        draft, _ = draft_scenario(SAMPLE, label_col="is_fraud", caller=_fake(resp))
        assert draft.intent.task == "detector_training"   # invalid task → default
        assert draft.intent.label_col == "is_fraud"        # non-existent column → structural
        assert draft.intent.rare_ratio is None             # out-of-range → default (auto)
        assert draft.intent.mode == "balanced"             # invalid mode → default
        assert draft.gates.privacy == "floored"            # invalid → default
        assert draft.gates.delta == 0.5                    # out-of-range → default

    def test_model_error_falls_back(self):
        from regen.api import draft_scenario
        def boom(p, pl, c):
            raise RuntimeError("endpoint down")
        draft, proposal = draft_scenario(SAMPLE, label_col="is_fraud", caller=boom)
        assert proposal is None
        assert draft.provenance["drafted_by"] == "structural"


class TestDraftDrivesGeneration:
    def test_draft_round_trips_and_generates(self):
        from regen.api import draft_scenario, generate
        resp = {"intent": {"task": "detector_training", "rare_mode": "label",
                           "rare_value": 1, "rare_ratio": 0.25, "mode": "balanced"},
                "gates": {"privacy": "floored", "delta": 0.5}, "columns": []}
        draft, _ = draft_scenario(SAMPLE, label_col="is_fraud", caller=_fake(resp))
        with tempfile.TemporaryDirectory() as out:
            reloaded = ScenarioSpec.from_yaml(draft.to_yaml())   # user saves/edits the YAML
            s = generate(SAMPLE, scenario=reloaded, out_dir=out)
        assert s["label_col"] == "is_fraud"
        assert s["rare_ratio"] == 0.25
        assert s["passed"] in (True, False)                      # ran end-to-end
