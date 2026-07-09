"""
Server API tests (P1-4): the FastAPI surface exposes privacy on /api/generate
and /api/campaign, validates it, and reports the regime in the response.

Uses the generated examples/transactions.csv fixture (not a real dataset).
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from server.app import app  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "transactions.csv"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _upload(client):
    with open(SAMPLE, "rb") as f:
        r = client.post(
            "/api/ingest",
            files={"file": ("transactions.csv", f, "text/csv")},
            data={"label_col": "is_fraud", "rare_mode": "label"},
        )
    assert r.status_code == 200, r.text


class TestGenerateEndpointPrivacy:
    def test_floored_returns_privacy_block(self, client):
        _upload(client)
        r = client.post("/api/generate", json={
            "label_col": "is_fraud", "n_rows": 120, "auto": False,
            "seed": 1, "privacy": "floored", "delta": 0.5,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        pv = body["privacy"]
        assert pv is not None
        assert pv["mode"] == "floored"
        assert "floor_applied" in pv and "passed" in pv
        # top-level verdict is fidelity AND privacy
        assert body["passed"] == (body["fidelity"]["passed"] and pv["passed"])

    def test_privacy_none_has_null_block(self, client):
        _upload(client)
        r = client.post("/api/generate", json={
            "label_col": "is_fraud", "n_rows": 120, "auto": False,
            "seed": 1, "privacy": "none",
        })
        assert r.status_code == 200, r.text
        assert r.json()["privacy"] is None

    def test_invalid_privacy_returns_400(self, client):
        _upload(client)
        r = client.post("/api/generate", json={
            "label_col": "is_fraud", "n_rows": 60, "auto": False, "privacy": "bogus",
        })
        assert r.status_code == 400

    def test_invalid_delta_returns_400(self, client):
        _upload(client)
        r = client.post("/api/generate", json={
            "label_col": "is_fraud", "n_rows": 60, "auto": False,
            "privacy": "floored", "delta": 9.0,
        })
        assert r.status_code == 400


class TestCampaignEndpointPrivacy:
    def test_floored_regime_visible(self, client):
        _upload(client)
        r = client.post("/api/campaign", json={
            "label_col": "is_fraud", "rare_def": {"mode": "label", "label_value": 1},
            "seed": 42, "n_rows": 120, "max_passes": 1, "privacy": "floored", "delta": 0.5,
        })
        assert r.status_code == 200, r.text
        assert r.json()["privacy"]["mode"] == "floored"

    def test_invalid_privacy_returns_400(self, client):
        _upload(client)
        r = client.post("/api/campaign", json={
            "label_col": "is_fraud", "rare_def": {"mode": "label", "label_value": 1},
            "n_rows": 60, "max_passes": 1, "privacy": "bogus",
        })
        assert r.status_code == 400


class TestNewEndpoints:
    def test_doctor(self, client):
        _upload(client)
        r = client.post("/api/doctor", json={"label_col": "is_fraud",
                        "rare_def": {"mode": "label", "label_value": 1}})
        assert r.status_code == 200, r.text
        assert "ok_to_generate" in r.json() and "checks" in r.json()

    def test_propose(self, client):
        _upload(client)
        r = client.post("/api/propose", json={"label_col": "is_fraud", "goal": "fraud detector",
                        "rare_def": {"mode": "label", "label_value": 1}, "n_rows": 200})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["scenario"]["intent"]["label_col"] == "is_fraud"
        assert "yaml" in body and body["drafted_by"] == "structural"   # offline in tests

    def test_explore(self, client):
        _upload(client)
        r = client.post("/api/explore", json={"label_col": "is_fraud",
                        "rare_def": {"mode": "label", "label_value": 1},
                        "deltas": [0.5], "n_rows": 150})
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["options"]) == 2 and "recommended" in body

    def test_verify_a_produced_bundle(self, client):
        _upload(client)
        g = client.post("/api/generate", json={"label_col": "is_fraud", "n_rows": 150,
                        "auto": False, "seed": 1, "privacy": "floored"})
        assert g.status_code == 200, g.text
        run_id = g.json()["run_id"]
        v = client.get(f"/api/campaign/{run_id}/verify")
        assert v.status_code == 200, v.text
        assert v.json()["passed"] is True

    def test_tstr(self, client):
        _upload(client)
        r = client.post("/api/tstr", json={"label_col": "is_fraud",
                        "rare_def": {"mode": "label", "label_value": 1}, "privacy": "none"})
        assert r.status_code == 200, r.text
        assert "tstr" in r.json() and "recovered_roc_auc_median" in r.json()["tstr"]
