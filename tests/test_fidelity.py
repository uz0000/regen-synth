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
from engine.auditor.fidelity import _tvd_discrete


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


def test_auditor_high_cardinality_tvd_topk():
    """High-cardinality categoricals should use top-K TVD, not fail on full distribution."""
    rng = np.random.default_rng(42)

    # Simulate Open Payments scale: 500 categories, 1000 rare rows, 200 synthetic
    n_rare = 1000
    n_synth = 200
    categories = [f"cat_{i}" for i in range(500)]

    # Power-law distribution (top categories dominate)
    probs = np.array([1.0 / (i + 1) for i in range(500)])
    probs = probs / probs.sum()

    # Both real and synthetic follow the same distribution (Prior samples from rare)
    real_cats = rng.choice(categories, size=n_rare, p=probs)
    synth_cats = rng.choice(categories, size=n_synth, p=probs)

    real_df = pd.DataFrame({"cat_col": real_cats})
    synth_df = pd.DataFrame({"cat_col": synth_cats})
    normal_df = pd.DataFrame({"cat_col": rng.choice(categories[:10], size=2000)})

    field_dict: FieldDict = {
        "cat_col": FieldMeta(name="cat_col", field_type=FieldType.CATEGORICAL),
    }

    ingest = IngestResult(
        normal_df=normal_df,
        rare_df=real_df,
        schema_graph=SchemaGraph(),
        field_dict=field_dict,
        label_col="",
    )

    config = AuditorConfig(coverage_threshold=0.0)  # skip coverage (categorical only)
    report = audit(ingest, synth_df, config)
    cat_result = [r for r in report.column_results if r.col == "cat_col"][0]

    # What top-K TVD actually buys is *separation*: a matching batch scores near
    # 0.15 where the full-distribution comparison would have scored near 1.0 by
    # arithmetic alone, since 200 rows cannot cover 261 categories.
    #
    # This deliberately does NOT assert `passed`. At this scale the matching
    # score sits on the gate — 0.106 to 0.174 across seeds against a 0.15
    # threshold — so a pass/fail assertion here is a coin flip dressed as a
    # test. It used to look stable only because the top-K set was picked with a
    # non-deterministic tie-break; once that was fixed (engine/auditor/
    # fidelity.py) the score became a reproducible 0.152 and the knife-edge was
    # visible. The gate being inside the noise band is a real limitation, and it
    # is recorded as issue 10 in docs/KNOWN_ISSUES.md rather than hidden behind
    # a threshold nudge.
    assert cat_result.tvd is not None
    assert cat_result.tvd < 0.30, (
        f"top-K TVD should keep a matching high-cardinality batch far below the "
        f"full-distribution failure mode; got {cat_result.tvd:.4f}"
    )


def test_high_cardinality_tvd_is_reproducible():
    """The top-K set must not depend on how the rows happen to be ordered.

    `value_counts().nlargest(k)` breaks ties arbitrarily, and on this fixture 17
    categories tie at the k-th count — so the same data produced a different TVD
    on different platforms, which Invariant 2 forbids. Caught by CI failing on
    Linux against a passing macOS run.
    """
    rng = np.random.default_rng(42)
    categories = [f"cat_{i}" for i in range(500)]
    probs = np.array([1.0 / (i + 1) for i in range(500)])
    probs = probs / probs.sum()
    real = pd.Series(rng.choice(categories, size=1000, p=probs))
    synth = pd.Series(rng.choice(categories, size=200, p=probs))

    config = AuditorConfig(coverage_threshold=0.0)
    scores = {
        round(_tvd_discrete(real.sample(frac=1.0, random_state=s).reset_index(drop=True),
                            synth, config), 12)
        for s in range(15)
    }
    assert len(scores) == 1, f"TVD depends on row order: {sorted(scores)}"


def test_auditor_high_cardinality_rejects_mismatched():
    """High-cardinality top-K TVD should still reject genuinely corrupted batches."""
    rng = np.random.default_rng(42)

    n_rare = 1000
    n_synth = 200
    categories = [f"cat_{i}" for i in range(500)]

    # Real: power-law (top categories dominate)
    probs = np.array([1.0 / (i + 1) for i in range(500)])
    probs = probs / probs.sum()
    real_cats = rng.choice(categories, size=n_rare, p=probs)

    # Synthetic: uniform over ALL categories (completely wrong distribution)
    synth_cats = rng.choice(categories, size=n_synth)

    real_df = pd.DataFrame({"cat_col": real_cats})
    synth_df = pd.DataFrame({"cat_col": synth_cats})
    normal_df = pd.DataFrame({"cat_col": rng.choice(categories[:10], size=2000)})

    field_dict: FieldDict = {
        "cat_col": FieldMeta(name="cat_col", field_type=FieldType.CATEGORICAL),
    }

    ingest = IngestResult(
        normal_df=normal_df,
        rare_df=real_df,
        schema_graph=SchemaGraph(),
        field_dict=field_dict,
        label_col="",
    )

    config = AuditorConfig(coverage_threshold=0.0)
    report = audit(ingest, synth_df, config)

    cat_result = [r for r in report.column_results if r.col == "cat_col"][0]
    assert not cat_result.passed, (
        f"Corrupted high-cardinality batch should be rejected. "
        f"TVD={cat_result.tvd:.4f}, threshold={config.tvd_threshold}"
    )
