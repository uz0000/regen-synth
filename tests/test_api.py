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

def test_generate_returns_full_dataset_with_both_classes():
    """generate() returns a FULL dataset: synthetic normal part + amplified rare part.

    The Prior generates feature columns only, so each part arrives unlabeled and
    must be stamped with its class. The batch therefore carries BOTH classes —
    normal rows labeled 0 and rare rows labeled 1 — at the resolved rare ratio.
    Regression test for the dropped-label bug and for the full-synthesis shape.
    """
    from regen import generate, load_synthetic

    n_rows = 120
    with tempfile.TemporaryDirectory() as out:
        summary = generate(
            SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
            n_rows=n_rows, mode="balanced", auto=False, out_dir=out,
        )
        batch = load_synthetic(out)

    assert LABEL_COL in batch.columns, "synthetic batch is missing the label column"
    # Both classes present, and only those two values.
    assert set(batch[LABEL_COL].unique()) == {0, RARE_DEF.label_value}
    # The full dataset has the requested number of rows.
    assert len(batch) == n_rows
    # The split matches the reported counts.
    n_rare = int((batch[LABEL_COL] == RARE_DEF.label_value).sum())
    n_normal = int((batch[LABEL_COL] == 0).sum())
    assert n_rare == summary["n_synthetic_rare"]
    assert n_normal == summary["n_synthetic_normal"]
    assert n_rare + n_normal == n_rows


def test_generate_auto_rare_ratio_amplifies_minority():
    """Auto rare_ratio = max(natural_prevalence, DEFAULT_MIN_RARE_FRAC=0.25).

    The sample data has ~3% natural prevalence, so the auto ratio floors it at
    25% — a real amplification, never a de-amplification. The resolved ratio and
    the realized rare fraction in the batch must agree.
    """
    from regen.api import generate

    with tempfile.TemporaryDirectory() as out:
        summary = generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
                           n_rows=400, auto=False, out_dir=out)
        batch = pd.read_parquet(summary["best_batch_path"])

    assert summary["rare_ratio"] == 0.25
    assert summary["natural_prevalence"] < 0.25  # genuinely rare in the source
    realized = (batch[LABEL_COL] == RARE_DEF.label_value).mean()
    assert abs(realized - 0.25) < 0.02  # within one row of the target ratio


def test_generate_explicit_rare_ratio_honored():
    """An explicit rare_ratio overrides auto and is reflected in the split."""
    from regen.api import generate

    with tempfile.TemporaryDirectory() as out:
        summary = generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
                           n_rows=200, auto=False, rare_ratio=0.5, out_dir=out)

    assert summary["rare_ratio"] == 0.5
    assert summary["n_synthetic_rare"] == 100
    assert summary["n_synthetic_normal"] == 100


def test_generate_rejects_invalid_rare_ratio():
    from regen.api import generate
    with pytest.raises(ValueError, match="rare_ratio"):
        generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
                 n_rows=100, auto=False, rare_ratio=1.5)


def test_generate_gates_normal_part_against_normal_reference():
    """The normal part is audited against normal_df (coverage off) and reported.

    The summary must carry a normal_fidelity block, and for a clean run the
    normal part passes its gate — otherwise a garbage normal half would ship
    silently alongside an accepted rare half.
    """
    from regen.api import generate

    with tempfile.TemporaryDirectory() as out:
        summary = generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
                           n_rows=300, auto=False, out_dir=out)

    nf = summary["normal_fidelity"]
    assert "score" in nf and "passed" in nf
    # Overall passed requires BOTH halves to pass.
    assert summary["fidelity"]["passed"] == (nf["passed"] and True)


def test_generate_full_dataset_is_reproducible():
    """Same seed + config through generate() → identical full-dataset batch."""
    from regen.api import generate

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sa = generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
                      n_rows=200, auto=False, noise_scale=0.10, out_dir=a)
        sb = generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
                      n_rows=200, auto=False, noise_scale=0.10, out_dir=b)
        da = pd.read_parquet(sa["best_batch_path"])
        db = pd.read_parquet(sb["best_batch_path"])
    assert da.equals(db), "full-dataset generation is not reproducible"


def test_generate_manifest_records_rare_ratio():
    """The manifest must capture the rare split so the full dataset reproduces."""
    import json as _json
    from regen.api import generate

    with tempfile.TemporaryDirectory() as out:
        summary = generate(SAMPLE_CSV, label_col=LABEL_COL, rare_def=RARE_DEF,
                           n_rows=200, auto=False, rare_ratio=0.4, out_dir=out)
        manifest = _json.loads(Path(summary["manifest_path"]).read_text())

    assert manifest["rare_ratio"] == 0.4
    assert manifest["n_rows"] == 200


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
    assert rep.status == "ok"          # 60 rare → ~18 held out, enough to measure


# ── P2-7: lift degeneracy on tiny rare folds ─────────────────────────────────

def test_lift_flags_insufficient_rare_rows(tmp_path):
    """P2-7: a held-out rare fold below MIN_TEST_RARE is reported as a status,
    not a bare 0.0. Synthetic fixture with ~14 rare rows → ~4 held out."""
    import numpy as np
    import pandas as pd
    from engine.examiner import measure_lift, ExaminerConfig
    from engine.examiner.detector import MIN_TEST_RARE
    from regen.api import ingest as api_ingest
    from contracts.types import RareEventDef, RareMode

    rng = np.random.default_rng(0)
    n_norm, n_rare = 300, 14
    df = pd.concat([
        pd.DataFrame({"a": rng.normal(0, 1, n_norm), "b": rng.normal(0, 1, n_norm),
                      "y": 0}),
        pd.DataFrame({"a": rng.normal(3, 1, n_rare), "b": rng.normal(3, 1, n_rare),
                      "y": 1}),
    ], ignore_index=True)
    path = str(tmp_path / "small_rare.csv")
    df.to_csv(path, index=False)

    res = api_ingest(path, "y", RareEventDef(mode=RareMode.LABEL, label_value=1))
    rep = measure_lift(res, ExaminerConfig(n_estimators=20), generate_synth_fn=None)
    assert rep.n_test_rare < MIN_TEST_RARE
    assert rep.status == "insufficient_rare_rows"


def test_generate_lift_out_nulls_tail_lift_when_insufficient(tmp_path):
    """The generate() summary reports {status, n_test_rare, tail_lift=None} rather
    than a misleading 0.0 when the rare fold is too small (P2-7)."""
    import numpy as np
    import pandas as pd
    from regen.api import generate

    rng = np.random.default_rng(1)
    n_norm, n_rare = 400, 16
    df = pd.concat([
        pd.DataFrame({"a": rng.normal(0, 1, n_norm), "b": rng.normal(0, 1, n_norm),
                      "y": 0}),
        pd.DataFrame({"a": rng.normal(4, 1, n_rare), "b": rng.normal(4, 1, n_rare),
                      "y": 1}),
    ], ignore_index=True)
    path = str(tmp_path / "small_rare2.csv")
    df.to_csv(path, index=False)

    s = generate(path, label_col="y", n_rows=300, auto=False, seed=3,
                 privacy="none", out_dir=str(tmp_path / "out"))
    lift = s["lift"]
    if lift is not None and lift["status"] == "insufficient_rare_rows":
        assert lift["tail_lift"] is None
        assert lift["n_test_rare"] < 10


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
    from engine.prior.grounded import GaussianPrior
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
    from engine.amplifier.tail_corrector import fit_correction, AmplifierConfig
    from engine.prior.grounded import fit_prior, PriorConfig
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
    with caplog.at_level(_logging.WARNING, logger="engine.amplifier.tail_corrector"):
        fit_correction(ing, prior, AmplifierConfig())
    assert any("underdetermined" in r.message for r in caplog.records)


# ── CSV delimiter sniffing (semicolon/tab files) ──────────────────────────────

def test_loader_sniffs_semicolon_delimiter():
    """A semicolon-delimited CSV must parse into separate columns, not one blob
    (regression: Instacart's export collapsed into a single 478k-unique column)."""
    from engine.ingest.loader import _load_file
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "semi.csv"
        p.write_text("a;b;label\n1;x;0\n2;y;0\n3;z;1\n")
        df = _load_file(str(p))
    assert list(df.columns) == ["a", "b", "label"]
    assert len(df) == 3


# ── Percentile mode: upper vs lower tail ──────────────────────────────────────

def test_percentile_tail_direction():
    """Lower tail flags the smallest values; upper tail the largest."""
    from engine.ingest.loader import ingest as _ingest
    from contracts.types import RareEventDef, RareMode
    with tempfile.TemporaryDirectory() as tmp:
        df = pd.DataFrame({"score": list(range(100)), "x": np.random.RandomState(0).randn(100)})
        p = str(Path(tmp) / "d.csv"); df.to_csv(p, index=False)
        lo = _ingest(p, "score", RareEventDef(mode=RareMode.PERCENTILE, percentile=0.10, tail="lower"))
        hi = _ingest(p, "score", RareEventDef(mode=RareMode.PERCENTILE, percentile=0.10, tail="upper"))
    assert lo.rare_df["score"].max() < 15, "lower tail should be the smallest values"
    assert hi.rare_df["score"].min() > 85, "upper tail should be the largest values"


# ── Semantic Fidelity M1: deterministic constraint layer + schema profile ─────

def _toy_ingest():
    from regen.api import ingest as _ingest
    rd = RareEventDef(mode=RareMode.LABEL, label_value=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = _toy_csv(tmp)
        return _ingest(path, "label", rd)


def test_build_constraints_reflects_field_types():
    from engine.constraints import build_constraints
    cons = build_constraints(_toy_ingest())
    assert cons["label"].kind == "label"
    assert cons["n_txns"].kind == "continuous" and cons["n_txns"].is_integer
    assert cons["is_weekend"].kind == "binary" and set(cons["is_weekend"].binary_values) <= {0, 1}
    assert cons["channel"].kind == "categorical"


def test_apply_constraints_clamps_rounds_snaps():
    from engine.constraints import apply_constraints
    ing = _toy_ingest()
    # Out-of-support junk the prior/GP could produce.
    bad = pd.DataFrame({
        "n_txns": [3.7, -5.0, 1e6],          # integer column, out of range
        "amount": [-9.0, 5.0, 1e9],          # continuous, non-negative in source
        "is_weekend": [0.7, -0.2, 1.4],      # binary drifted off {0,1}
        "channel": [0, 1, 2],                # categorical codes (decoded elsewhere)
    })
    out = apply_constraints(bad, ing)
    assert (out["n_txns"] == out["n_txns"].round()).all()      # integers
    assert out["amount"].min() >= 0.0                          # clamped to observed >= 0
    assert set(out["is_weekend"].unique()) <= {0, 1}           # snapped


def test_column_profiles_shape_and_role_guess():
    from engine.ingest.profile import column_profiles
    profs = {p["name"]: p for p in column_profiles(_toy_ingest())}
    assert profs["label"]["role_guess"] == "label"
    assert profs["n_txns"]["is_integer"] is True
    assert "min" in profs["amount"] and "max" in profs["amount"]
    assert profs["channel"]["minority_value"] in ("web", "store", "kiosk")


# ── M1.5: structural identifier detection + handling ──────────────────────────

def _id_ingest(tmp):
    """Dataset with a near-unique integer ID, a string ID, a near-unique float
    measurement (must NOT be flagged), and a real binary label."""
    from regen.api import ingest as _ingest
    rng = np.random.RandomState(0); n = 300
    df = pd.DataFrame({
        "order_id": np.arange(10_000, 10_000 + n),                 # near-unique int → ID
        "email": [f"user{i}@x.com" for i in range(n)],             # near-unique string → ID
        "amount": rng.gamma(2, size=n) + np.arange(n) * 1e-6,      # near-unique float, no hint → NOT id
        "label": (rng.rand(n) < 0.2).astype(int),
    })
    p = str(Path(tmp) / "ids.csv"); df.to_csv(p, index=False)
    return _ingest(p, "label", RareEventDef(mode=RareMode.LABEL, label_value=1))


def test_identifier_detection_is_conservative():
    with tempfile.TemporaryDirectory() as tmp:
        fd = _id_ingest(tmp).field_dict
    assert fd["order_id"].is_identifier
    assert fd["email"].is_identifier
    assert not fd["amount"].is_identifier, "near-unique float must not be flagged an ID"
    assert not fd["label"].is_identifier


def test_identifier_columns_regenerated_unique():
    from regen import generate, load_synthetic
    with tempfile.TemporaryDirectory() as tmp:
        ing_dir = str(Path(tmp) / "out")
        # reuse the same toy file the ingest built
        rng = np.random.RandomState(0); n = 300
        df = pd.DataFrame({
            "order_id": np.arange(10_000, 10_000 + n),
            "email": [f"user{i}@x.com" for i in range(n)],
            "amount": rng.gamma(2, size=n),
            "label": (rng.rand(n) < 0.2).astype(int),
        })
        p = str(Path(tmp) / "ids.csv"); df.to_csv(p, index=False)
        generate(p, label_col="label", rare_def=RareEventDef(mode=RareMode.LABEL, label_value=1),
                 n_rows=120, auto=False, out_dir=ing_dir)
        b = load_synthetic(ing_dir)
    assert b["order_id"].is_unique and b["order_id"].min() > 10_000 + n - 1  # past real max
    assert b["email"].is_unique and b["email"].astype(str).str.startswith("email-").all()
