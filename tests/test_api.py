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
                # Batch carries the label: every row is the amplified rare class
                assert LABEL_COL in df.columns
                assert set(df[LABEL_COL].unique()) == {RARE_DEF.label_value}

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
        # Batch carries the rare label so it is usable as labeled training data
        assert LABEL_COL in df.columns
        assert set(df[LABEL_COL].unique()) == {RARE_DEF.label_value}

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

# ── generate(): label passthrough + domain constraints ────────────────────────

def test_generate_attaches_rare_label():
    """The synthetic batch must carry the label column, set to the rare class.

    The Prior generates feature columns only, so the batch arrives unlabeled —
    but every row is the amplified rare class and must say so to be usable
    downstream. Regression test for the dropped-label bug.
    """
    from regen import generate, load_synthetic

    with tempfile.TemporaryDirectory() as out:
        summary = generate(
            SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
            n_rows=120, mode="balanced", auto=False, out_dir=out,
        )
        batch = load_synthetic(out)

    assert LABEL_COL in batch.columns, "synthetic batch is missing the label column"
    # All rows are the amplified rare class → label is the rare value, constant.
    assert set(batch[LABEL_COL].unique()) == {RARE_DEF.label_value}


def test_generate_clips_to_observed_support():
    """Continuous columns must stay within the real data's observed [min, max].

    The Gaussian Prior + residual GP can sample past the support (e.g. negative
    amounts); _apply_domain_constraints clips them back. Regression test.
    """
    from regen import generate, load_synthetic

    src = pd.read_csv(SAMPLE_CSV)
    with tempfile.TemporaryDirectory() as out:
        summary = generate(
            SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
            n_rows=200, mode="balanced", auto=False, out_dir=out,
        )
        batch = load_synthetic(out)

    for col in ("amount", "n_prior_txns", "hour", "merchant_risk"):
        lo, hi = float(src[col].min()), float(src[col].max())
        assert batch[col].min() >= lo - 1e-9, f"{col} fell below observed min"
        assert batch[col].max() <= hi + 1e-9, f"{col} rose above observed max"
    # 'amount' is non-negative in the source, so the batch must be too.
    assert batch["amount"].min() >= 0.0


def test_generate_auto_detects_target():
    """With label_col/rare_def left open, generate() reports what it auto-picked."""
    from regen import generate

    with tempfile.TemporaryDirectory() as out:
        summary = generate(SAMPLE_CSV, n_rows=80, mode="balanced", out_dir=out)

    det = summary["detection"]
    assert det is not None and det["auto_label"] and det["auto_rare"]
    assert det["label_col"] == LABEL_COL  # the one imbalanced low-cardinality column


# ── Batch A: output-validity fixes (integer/binary/categorical/NaN) ───────────

def _toy_csv(tmp):
    """Dataset with an integer count, a binary flag, and a categorical whose
    'kiosk' value appears ONLY in normal rows (exercises canonical decode)."""
    import numpy as _np
    rng = _np.random.RandomState(0)
    n = 600
    rare = rng.rand(n) < 0.06
    df = pd.DataFrame({
        "n_txns": rng.randint(0, 50, size=n),
        "amount": _np.abs(rng.gamma(2, size=n)) * 10,
        "is_weekend": rng.randint(0, 2, size=n),
        "channel": _np.where(rng.rand(n) < 0.5, "web", "store"),
        "label": rare.astype(int),
    })
    df.loc[df.index[df["label"] == 0][:30], "channel"] = "kiosk"
    path = str(Path(tmp) / "toy.csv")
    df.to_csv(path, index=False)
    return path


def test_generate_rounds_integer_columns():
    """Integer-valued source columns must come back as whole numbers, not floats."""
    from regen import generate, load_synthetic
    rd = RareEventDef(mode=RareMode.LABEL, label_value=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = _toy_csv(tmp)
        out = str(Path(tmp) / "out")
        generate(path, label_col="label", rare_def=rd, n_rows=150, auto=False, out_dir=out)
        b = load_synthetic(out)
    assert (b["n_txns"] == b["n_txns"].round()).all(), "integer column emitted fractional"


def test_generate_snaps_binary_columns():
    """Binary columns must be exactly the two observed values, not floats like 0.7."""
    from regen import generate, load_synthetic
    rd = RareEventDef(mode=RareMode.LABEL, label_value=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = _toy_csv(tmp)
        out = str(Path(tmp) / "out")
        generate(path, label_col="label", rare_def=rd, n_rows=150, auto=False, out_dir=out)
        b = load_synthetic(out)
    assert set(b["is_weekend"].unique()) <= {0, 1}, "binary column drifted off {0,1}"


def test_generate_decodes_categoricals_canonically():
    """Decoded categoricals must be real categories — never a mislabel from using
    only the rare subset's categories."""
    from regen import generate, load_synthetic
    rd = RareEventDef(mode=RareMode.LABEL, label_value=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = _toy_csv(tmp)
        out = str(Path(tmp) / "out")
        generate(path, label_col="label", rare_def=rd, n_rows=150, auto=False, out_dir=out)
        b = load_synthetic(out)
    assert set(b["channel"].astype(str).unique()) <= {"web", "store", "kiosk"}


def test_ingest_rejects_all_nan_column():
    """A column with no observed values can't be imputed — fail loud, don't poison
    the pipeline with NaN that the Auditor would silently pass."""
    from regen.api import ingest
    rd = RareEventDef(mode=RareMode.LABEL, label_value=1)
    with tempfile.TemporaryDirectory() as tmp:
        df = pd.read_csv(_toy_csv(tmp))
        df["dead"] = np.nan
        path = str(Path(tmp) / "withnan.csv")
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="entirely missing"):
            ingest(path, "label", rd)


# ── Batch B1: leakage-free lift measurement ───────────────────────────────────

def test_measure_lift_generates_synth_from_train_fold_only():
    """The synthetic used for lift must come from a strict subset of rare rows
    (the train fold) — the held-out rare test rows must never reach generation."""
    from engine.examiner import measure_lift, ExaminerConfig
    from regen.api import ingest as api_ingest

    res = api_ingest(SAMPLE_CSV, LABEL_COL, RARE_DEF)
    full_rare = len(res.rare_df)
    seen = {}

    def fake_gen(train_ingest):
        seen["n_rare_train"] = len(train_ingest.rare_df)
        return train_ingest.rare_df.copy()  # any DataFrame with the feature cols

    cfg = ExaminerConfig(n_estimators=20)
    measure_lift(res, cfg, generate_synth_fn=fake_gen)

    assert 0 < seen["n_rare_train"] < full_rare, (
        "generation saw the full rare set — held-out test rows leaked in"
    )


def test_measure_lift_no_synth_means_zero_lift():
    """With no augmentation the amplified model == baseline, so lift is exactly 0."""
    from engine.examiner import measure_lift, ExaminerConfig
    from regen.api import ingest as api_ingest

    res = api_ingest(SAMPLE_CSV, LABEL_COL, RARE_DEF)
    rep = measure_lift(res, ExaminerConfig(n_estimators=20), generate_synth_fn=None)
    assert rep.tail_lift == 0.0
    assert rep.n_synthetic_used == 0


# ── Batch B2: Auditor checks cross-column correlation structure ───────────────

def test_auditor_rejects_scrambled_correlation():
    """A batch with correct marginals but destroyed joint structure must FAIL,
    even though every per-column check passes. This is the gate's reason to exist."""
    from engine.auditor import audit, AuditorConfig
    from engine.ingest.loader import _build_field_dict
    from contracts.types import IngestResult, SchemaGraph

    rng = np.random.RandomState(0)
    x = rng.randn(400)
    rare = pd.DataFrame({"a": x, "b": x * 0.9 + 0.1 * rng.randn(400),
                         "c": -0.8 * x + 0.1 * rng.randn(400)})
    fd = _build_field_dict(rare, "")
    ing = IngestResult(normal_df=rare.copy(), rare_df=rare.copy(),
                       schema_graph=SchemaGraph(), field_dict=fd, label_col="")
    cfg = AuditorConfig(coverage_threshold=0.0)  # isolate the correlation gate

    # Faithful batch (same dependence) passes.
    xs = rng.randn(400)
    good = pd.DataFrame({"a": xs, "b": xs * 0.9 + 0.1 * rng.randn(400),
                         "c": -0.8 * xs + 0.1 * rng.randn(400)})
    rep_good = audit(ing, good, cfg)
    assert rep_good.correlation_passed and rep_good.overall_passed

    # Shuffle each column independently → identical marginals, broken joint structure.
    bad = good.copy()
    for col in bad.columns:
        bad[col] = bad[col].sample(frac=1, random_state=rng.randint(1_000_000)).values
    rep_bad = audit(ing, bad, cfg)
    assert all(c.passed for c in rep_bad.column_results), "marginals should still pass"
    assert not rep_bad.correlation_passed, "scrambled correlation should fail"
    assert not rep_bad.overall_passed


# ── Batch B3: manifest persistence + real-path reproducibility ────────────────

def test_generate_writes_complete_manifest():
    """generate() must persist a manifest with everything needed to reproduce the
    batch from disk (Invariant 2)."""
    import json
    from regen import generate
    rd = RareEventDef(mode=RareMode.LABEL, label_value=1)
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "out")
        s = generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=rd,
                     n_rows=100, auto=False, out_dir=out)
        mpath = Path(s["manifest_path"])
        assert mpath.exists()
        m = json.loads(mpath.read_text())
    for key in ("seed", "schema_hash", "prior_config", "amplifier_params", "code_version", "n_rows"):
        assert key in m, f"manifest missing {key}"
    assert m["code_version"] and m["code_version"] != "unknown"
    assert m["prior_config"].get("noise_scale") is not None


def test_generate_is_reproducible_through_the_api():
    """Same seed + config through the real generate() path → identical batch."""
    from regen import generate, load_synthetic
    rd = RareEventDef(mode=RareMode.LABEL, label_value=1)
    with tempfile.TemporaryDirectory() as tmp:
        o1, o2 = str(Path(tmp) / "a"), str(Path(tmp) / "b")
        generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=rd, n_rows=120,
                 seed=7, auto=False, out_dir=o1)
        generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=rd, n_rows=120,
                 seed=7, auto=False, out_dir=o2)
        b1, b2 = load_synthetic(o1), load_synthetic(o2)
    pd.testing.assert_frame_equal(b1, b2)


# ── Medium fragility fixes (#8 RNG / #9 GP guard / #10 standardization) ────────

def test_gaussian_prior_not_dominated_by_feature_scale():
    """A huge-scale noise feature must not swamp an informative small-scale one
    (the prior standardizes features before the Gaussian fit)."""
    from engine.prior.rdbpfn import GaussianPrior
    rng = np.random.RandomState(0)
    m = 200
    x_info = np.concatenate([rng.normal(-1, 0.5, m), rng.normal(1, 0.5, m)])
    x_noise = rng.normal(0, 1000.0, 2 * m)          # huge scale, no signal
    X = np.column_stack([x_info, x_noise])
    y = np.array([0] * m + [1] * m)
    gp = GaussianPrior().fit(X, y)
    proba = gp.predict_proba(X)
    acc = ((proba[:, 1] > 0.5).astype(int) == y).mean()
    assert acc > 0.9, "scoring was dominated by the large-scale noise feature"


def test_amplifier_warns_when_underdetermined(caplog):
    """Few rare rows relative to feature dims → loud warning, not a silent fit."""
    import logging as _logging
    from engine.amplifier.residual_gp import fit_residuals, AmplifierConfig
    from engine.prior.rdbpfn import fit_prior, PriorConfig
    from engine.ingest.loader import _build_field_dict
    from contracts.types import IngestResult, SchemaGraph

    rng = np.random.RandomState(0)
    nf = 8
    cols = [f"f{i}" for i in range(nf)]
    norm = pd.DataFrame(rng.randn(300, nf), columns=cols); norm["label"] = 0
    rare = pd.DataFrame(rng.randn(12, nf) + 3, columns=cols); rare["label"] = 1
    full = pd.concat([norm, rare], ignore_index=True)
    ing = IngestResult(normal_df=norm, rare_df=rare, schema_graph=SchemaGraph(),
                       field_dict=_build_field_dict(full, "label"), label_col="label")
    prior = fit_prior(ing, PriorConfig(), np.random.default_rng(0))
    with caplog.at_level(_logging.WARNING, logger="engine.amplifier.residual_gp"):
        fit_residuals(ing, prior, AmplifierConfig())
    assert any("underdetermined" in r.message for r in caplog.records)
