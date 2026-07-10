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


class TestTargetTieBreak:
    """When rule-based scoring ties on the target (AmbiguousTargetError), the goal is
    handed to the advisory model to break the tie; offline it stays an honest error."""

    def _ambiguous_csv(self, path):
        import numpy as np
        import pandas as pd
        rng = np.random.RandomState(0)
        n = 120
        # Two binary columns with IDENTICAL imbalance (20/120) and no name bonus →
        # they score equal → AmbiguousTargetError. Continuous features are excluded
        # (high cardinality), so these two are the only candidates.
        flag_a = np.array([1] * 20 + [0] * 100)
        flag_b = np.array([1] * 20 + [0] * 100)
        rng.shuffle(flag_b)
        pd.DataFrame({
            "reading": rng.normal(0, 1, n),
            "score": rng.normal(5, 2, n),
            "churned": flag_a,          # not in _LABEL_CANDIDATES
            "defaulted": flag_b,        # not in _LABEL_CANDIDATES
        }).to_csv(path, index=False)

    def _tiebreak_caller(self, chosen):
        def caller(prompt, payload, config):
            if "selecting the rare-event TARGET" in prompt:      # the tie-break call
                return json.dumps({"label_col": chosen, "reason": "goal names churn"})
            return json.dumps({"intent": {}, "gates": {}, "columns": []})  # scenario call
        return caller

    def test_offline_tie_is_honest_error(self):
        from regen.api import draft_scenario
        from engine.ingest.loader import AmbiguousTargetError
        with tempfile.TemporaryDirectory() as d:
            csv = str(Path(d) / "amb.csv")
            self._ambiguous_csv(csv)
            with pytest.raises(AmbiguousTargetError):             # no model → human chooses
                draft_scenario(csv, goal="predict churn")

    def test_model_breaks_tie_from_goal(self):
        from regen.api import draft_scenario
        with tempfile.TemporaryDirectory() as d:
            csv = str(Path(d) / "amb.csv")
            self._ambiguous_csv(csv)
            draft, _ = draft_scenario(csv, goal="predict churn",
                                      caller=self._tiebreak_caller("churned"))
        assert draft.intent.label_col == "churned"
        tb = draft.provenance["target_tiebreak"]
        assert tb["chosen"] == "churned" and tb["resolved_by"] == "model"
        assert set(tb["candidates"]) == {"churned", "defaulted"}

    def test_model_invalid_pick_falls_back_to_error(self):
        from regen.api import draft_scenario
        from engine.ingest.loader import AmbiguousTargetError
        with tempfile.TemporaryDirectory() as d:
            csv = str(Path(d) / "amb.csv")
            self._ambiguous_csv(csv)
            with pytest.raises(AmbiguousTargetError):             # non-candidate → declined
                draft_scenario(csv, goal="x", caller=self._tiebreak_caller("nonexistent"))

    def test_semantic_context_sent_to_model(self):
        """The tie-break payload carries the other column names (domain context) and
        example values for the tied candidates — names+values only, never raw rows."""
        from regen.api import draft_scenario
        seen = {}

        def capturing(prompt, payload, config):
            if "selecting the rare-event TARGET" in prompt:
                seen["payload"] = payload
                return json.dumps({"label_col": "churned", "reason": "goal"})
            return json.dumps({"intent": {}, "gates": {}, "columns": []})

        with tempfile.TemporaryDirectory() as d:
            csv = str(Path(d) / "amb.csv")
            self._ambiguous_csv(csv)
            draft_scenario(csv, goal="predict churn", caller=capturing)

        p = seen["payload"]
        assert set(p["other_columns"]) == {"reading", "score"}     # non-candidate names as context
        cand = {c["name"]: c for c in p["candidates"]}
        assert set(cand) == {"churned", "defaulted"}
        assert sorted(cand["churned"]["example_values"]) == [0, 1]  # candidate values sent


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
