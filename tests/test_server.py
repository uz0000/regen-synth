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
