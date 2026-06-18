"""
Invariant 2 — the Auditor must reject a deliberately corrupted batch and
accept a clean one.

We synthesize a small reference dataset (normal + rare events) and then:
  - Pass a clean synthetic batch → expect overall_passed = True
  - Pass a corrupted batch (scrambled values, broken coverage) → expect False
"""

import numpy as np
import pandas as pd
import pytest

from contracts.types import FieldDict, FieldMeta, FieldType, IngestResult, SchemaGraph
from engine.auditor import AuditorConfig, audit


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ingest(rng: np.random.Generator) -> IngestResult:
    """Create a small IngestResult with a clear rare region."""
    n_normal = 200
    n_rare   = 20

    normal = pd.DataFrame({
        "feat_a": rng.normal(0.0, 1.0, n_normal),
        "feat_b": rng.normal(0.0, 1.0, n_normal),
    })
    rare = pd.DataFrame({
        "feat_a": rng.normal(5.0, 0.3, n_rare),   # rare events are far in feat_a
        "feat_b": rng.normal(0.0, 1.0, n_rare),
    })

    field_dict: FieldDict = {
        "feat_a": FieldMeta(name="feat_a", field_type=FieldType.CONTINUOUS),
        "feat_b": FieldMeta(name="feat_b", field_type=FieldType.CONTINUOUS),
    }

    return IngestResult(
        normal_df=normal,
        rare_df=rare,
        schema_graph=SchemaGraph(),
        field_dict=field_dict,
        label_col="",
    )


def _clean_synthetic(ingest: IngestResult, rng: np.random.Generator) -> pd.DataFrame:
    """A clean rare-event amplification batch.

    REGEN's output is concentrated in the rare region, so a clean batch
    matches the *rare* distribution's marginals (feat_a ≈ 5, feat_b ≈ 0)
    and covers the real rare events. This is what the Auditor compares against.
    """
    n = 200
    return pd.DataFrame({
        "feat_a": rng.normal(5.0, 0.3, n),   # matches rare marginal
        "feat_b": rng.normal(0.0, 1.0, n),   # matches rare marginal
    })


def _corrupted_synthetic(ingest: IngestResult, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic batch that completely misses the rare region."""
    n = 200
    return pd.DataFrame({
        "feat_a": rng.normal(-10.0, 0.1, n),  # far from both normal and rare
        "feat_b": rng.normal(-10.0, 0.1, n),
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_auditor_accepts_clean_batch():
    rng    = np.random.default_rng(42)
    ingest = _make_ingest(rng)
    synth  = _clean_synthetic(ingest, rng)
    config = AuditorConfig(coverage_threshold=0.80)

    report = audit(ingest, synth, config)

    assert report.overall_passed, (
        f"Expected clean batch to pass. "
        f"coverage_rate={report.coverage_rate:.3f}, "
        f"failed_cols={[r.col for r in report.column_results if not r.passed]}"
    )


def test_auditor_rejects_corrupted_batch():
    rng    = np.random.default_rng(42)
    ingest = _make_ingest(rng)
    synth  = _corrupted_synthetic(ingest, rng)
    config = AuditorConfig(coverage_threshold=0.80)

    report = audit(ingest, synth, config)

    assert not report.overall_passed, (
        "Expected corrupted batch to be rejected by Auditor. "
        "If this passes, the fidelity gate is not working correctly."
    )
    assert not report.coverage_passed, (
        f"Expected coverage failure. Got coverage_rate={report.coverage_rate:.3f}"
    )


def test_auditor_report_has_manifest_when_provided():
    from contracts.types import BatchManifest, SchemaGraph
    rng    = np.random.default_rng(7)
    ingest = _make_ingest(rng)
    synth  = _clean_synthetic(ingest, rng)
    config = AuditorConfig()

    manifest = BatchManifest(
        seed=7,
        schema_hash="abc123",
        prior_config={},
        target_region={},
        amplifier_params={},
        code_version="test",
        n_rows=len(synth),
    )
    report = audit(ingest, synth, config, manifest=manifest)
    assert report.manifest is manifest
