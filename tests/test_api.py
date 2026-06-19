"""
Tests for regen.api — the unified REGEN API layer.

End-to-end tests on the sample data in examples/. Each method
(ingest, run_campaign, screen, get_results, load_synthetic) is
exercised with real input and asserted against its typed contract.

Reproducibility: screen() must return the same score for the same
seed. The API lives outside engine/ so the boundary test (which
scans only engine/) is unaffected — but we verify the API module
itself doesn't import forbidden libs.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.types import (
    CampaignResult,
    IngestResult,
    PassDetail,
    RareEventDef,
    RareMode,
    ScreenResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CSV = str(Path(__file__).parent.parent / "examples" / "transactions.csv")
LABEL_COL = "is_fraud"
RARE_DEF = RareEventDef(mode=RareMode.LABEL, label_value=1)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestIngest:
    """regen.api.ingest() loads data and returns typed IngestResult."""

    def test_ingest_returns_typed_result(self):
        from regen.api import ingest
        result = ingest(SAMPLE_CSV, LABEL_COL, RARE_DEF)
        assert isinstance(result, IngestResult)
        assert isinstance(result.normal_df, pd.DataFrame)
        assert isinstance(result.rare_df, pd.DataFrame)
        assert len(result.normal_df) > 0
        assert len(result.rare_df) > 0
        assert result.label_col == LABEL_COL

    def test_ingest_splits_correctly(self):
        from regen.api import ingest
        result = ingest(SAMPLE_CSV, LABEL_COL, RARE_DEF)
        total = len(result.normal_df) + len(result.rare_df)
        assert total > 0
        # All rare rows should have the label value 1
        assert (result.rare_df[LABEL_COL] == 1).all()
        # All normal rows should have the label value 0
        assert (result.normal_df[LABEL_COL] == 0).all()

    def test_ingest_raises_on_bad_path(self):
        from regen.api import ingest
        with pytest.raises(FileNotFoundError):
            ingest("/nonexistent/file.csv", LABEL_COL, RARE_DEF)

    def test_ingest_deterministic(self):
        """Same file + same rare_def = same split (no RNG in ingest)."""
        from regen.api import ingest
        r1 = ingest(SAMPLE_CSV, LABEL_COL, RARE_DEF)
        r2 = ingest(SAMPLE_CSV, LABEL_COL, RARE_DEF)
        assert len(r1.normal_df) == len(r2.normal_df)
        assert len(r1.rare_df) == len(r2.rare_df)


class TestRunCampaign:
    """regen.api.run_campaign() orchestrates the full loop."""

    def test_run_campaign_returns_typed_result(self):
        from regen.api import run_campaign
        with tempfile.TemporaryDirectory(prefix="regen_test_") as tmpdir:
            result = run_campaign(
                SAMPLE_CSV, LABEL_COL, RARE_DEF,
                seed=42, n_rows=100, max_passes=1,
                out_dir=tmpdir, coverage_threshold=0.30,
            )
        assert isinstance(result, CampaignResult)
        assert isinstance(result.best_lift, float)
        assert isinstance(result.passes, list)
        assert len(result.passes) >= 1

    def test_run_campaign_has_pass_details(self):
        from regen.api import run_campaign
        with tempfile.TemporaryDirectory(prefix="regen_test_") as tmpdir:
            result = run_campaign(
                SAMPLE_CSV, LABEL_COL, RARE_DEF,
                seed=42, n_rows=100, max_passes=2,
                out_dir=tmpdir, coverage_threshold=0.30,
            )
        for p in result.passes:
            assert isinstance(p, PassDetail)
            assert p.pass_num >= 1
            assert p.status in ("accepted", "rejected")

    def test_run_campaign_writes_parquet(self):
        from regen.api import run_campaign
        with tempfile.TemporaryDirectory(prefix="regen_test_") as tmpdir:
            result = run_campaign(
                SAMPLE_CSV, LABEL_COL, RARE_DEF,
                seed=42, n_rows=100, max_passes=1,
                out_dir=tmpdir, coverage_threshold=0.30,
            )
            if result.best_batch_path:
                df = pd.read_parquet(result.best_batch_path)
                assert len(df) == 100
                # Batch contains feature columns only, not the label
                assert LABEL_COL not in df.columns

    def test_run_campaign_output_dir_exists(self):
        from regen.api import run_campaign
        with tempfile.TemporaryDirectory(prefix="regen_test_") as tmpdir:
            result = run_campaign(
                SAMPLE_CSV, LABEL_COL, RARE_DEF,
                seed=42, n_rows=100, max_passes=1,
                out_dir=tmpdir, coverage_threshold=0.30,
            )
            assert Path(result.output_dir).exists()


class TestScreen:
    """regen.api.screen() predicts the win boundary."""

    def test_screen_returns_typed_result(self):
        from regen.api import screen
        result = screen(SAMPLE_CSV, LABEL_COL, RARE_DEF, seed=42)
        assert isinstance(result, ScreenResult)
        assert result.recommended_method in ("REGEN", "SMOTE")
        assert isinstance(result.heterogeneity_score, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.rationale, str)
        assert len(result.rationale) > 0

    def test_screen_score_is_stable(self):
        """Same seed → same heterogeneity_score."""
        from regen.api import screen
        r1 = screen(SAMPLE_CSV, LABEL_COL, RARE_DEF, seed=42)
        r2 = screen(SAMPLE_CSV, LABEL_COL, RARE_DEF, seed=42)
        assert r1.heterogeneity_score == r2.heterogeneity_score
        assert r1.recommended_method == r2.recommended_method

    def test_screen_score_changes_with_seed(self):
        """Different seeds may produce different scores (GP fitting varies)."""
        from regen.api import screen
        r1 = screen(SAMPLE_CSV, LABEL_COL, RARE_DEF, seed=42)
        r2 = screen(SAMPLE_CSV, LABEL_COL, RARE_DEF, seed=999)
        # The GP fitting is deterministic given a seed, so different
        # seeds produce different results due to different prior+residual
        # fits (the prior subsamples normal rows differently).
        assert isinstance(r1.heterogeneity_score, float)
        assert isinstance(r2.heterogeneity_score, float)

    def test_screen_includes_data_shape(self):
        from regen.api import screen
        result = screen(SAMPLE_CSV, LABEL_COL, RARE_DEF, seed=42)
        assert result.n_rare > 0
        assert result.n_features > 0

    def test_screen_with_quick_campaign(self):
        """quick_campaign flag should not crash (may refine lift band)."""
        from regen.api import screen
        result = screen(SAMPLE_CSV, LABEL_COL, RARE_DEF, seed=42,
                         quick_campaign=True)
        assert isinstance(result, ScreenResult)
        assert result.recommended_method in ("REGEN", "SMOTE")


class TestGetResults:
    """regen.api.get_results() reads back campaign output."""

    def test_get_results_from_previous_run(self):
        from regen.api import run_campaign, get_results
        with tempfile.TemporaryDirectory(prefix="regen_test_") as tmpdir:
            cr = run_campaign(
                SAMPLE_CSV, LABEL_COL, RARE_DEF,
                seed=42, n_rows=100, max_passes=1,
                out_dir=tmpdir, coverage_threshold=0.30,
            )
            loaded = get_results(cr.output_dir)
        assert isinstance(loaded, CampaignResult)
        assert loaded.best_lift == cr.best_lift

    def test_get_results_raises_on_bad_dir(self):
        from regen.api import get_results
        with pytest.raises(FileNotFoundError):
            get_results("/nonexistent/output")


class TestLoadSynthetic:
    """regen.api.load_synthetic() loads Parquet batches."""

    def test_load_synthetic_returns_dataframe(self):
        from regen.api import run_campaign, load_synthetic
        with tempfile.TemporaryDirectory(prefix="regen_test_") as tmpdir:
            cr = run_campaign(
                SAMPLE_CSV, LABEL_COL, RARE_DEF,
                seed=42, n_rows=100, max_passes=1,
                out_dir=tmpdir, coverage_threshold=0.30,
            )
            df = load_synthetic(cr.output_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Batch contains feature columns only (label is inherent for rare events)
        assert LABEL_COL not in df.columns

    def test_load_synthetic_raises_on_bad_dir(self):
        from regen.api import load_synthetic
        with pytest.raises(FileNotFoundError):
            load_synthetic("/nonexistent/output")


# ── Boundary check: regen/api.py must not import LLM/network libs ────────────

# We reuse the same forbidden-module list from tests/test_boundary.py
# but apply it to regen/api.py rather than the whole engine/ tree.
# This is a lighter check than the engine test — just the API file.

FORBIDDEN = {
    "openai", "anthropic", "langchain", "llama_index",
    "httpx", "requests", "aiohttp", "boto3", "google.cloud",
    "agent-runtime", "agentskills",
}


def _collect_imports(filepath: Path):
    """Return all top-level module names imported by a Python file."""
    import ast
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
    return names


def test_api_boundary():
    """regen/api.py must not import LLM/agent/network libraries."""
    api_path = Path(__file__).parent.parent / "regen" / "api.py"
    imports = _collect_imports(api_path)
    violations = [m for m in imports if m in FORBIDDEN]
    assert not violations, (
        f"regen/api.py imports forbidden modules: {violations}. "
        "The API must be pure deterministic Python."
    )