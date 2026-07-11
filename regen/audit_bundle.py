"""
Independent auditability (G-G) — the audit bundle + `verify_bundle`.

Every generation emits a self-contained bundle (the run directory):
  pass_1_accepted.parquet   the delivered data
  manifest.json             seed/config/scenario + SHA-256 of every artifact + metric versions
  explanation.json          the reported statistics (G-C)
  reference_aggregates.json aggregate stats of the REAL reference every gate was
                            computed against — under a disclosure policy (no per-row
                            values; histogram/quantile buckets only above a minimum count)

`verify_bundle` recomputes the reported statistics from the delivered data +
reference aggregates and reports stat-by-stat PASS / FAIL / UNCHECKABLE. Integrity
first (artifact hashes must match the manifest), then values within each metric's
tolerance. It is **pure recomputation from the bundle** — it never reads a cached
result — so it would catch a system that lied. Statistics that need the raw
reference rows (coverage, privacy min-distance) are honestly reported as
UNCHECKABLE at aggregate disclosure, not faked.

Pure Python (numpy/pandas). Lives outside engine/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from contracts.types import FieldType, IngestResult
from regen.metrics import METRICS, metric_versions, tolerance

# Disclosure: histogram/quantile buckets are published only for a class with at
# least this many real rows (no bucket reveals a near-unique individual). The
# ScenarioSpec gates can dial this up; verify then reports which stats became
# uncheckable at the stricter level.
DEFAULT_MIN_BUCKET = 10

BATCH_NAME = "pass_1_accepted.parquet"
AGG_NAME = "reference_aggregates.json"
EXPLAIN_NAME = "explanation.json"
MANIFEST_NAME = "manifest.json"


# ── Bundle emission ───────────────────────────────────────────────────────────

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_reference_aggregates(
    result: IngestResult, n_normal: int, n_rare: int,
    min_bucket: int = DEFAULT_MIN_BUCKET,
    estimand_real: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate statistics of the REAL reference, under the disclosure policy.

    Publishes: class counts, the real-rare correlation matrix over numeric
    feature columns, per-class per-column encoded moments (mean/var/count — enough
    to recompute the Fisher separation), and rare-class deciles per numeric column
    (only when the rare class has ≥ min_bucket rows). When an estimand is declared,
    also the θ_real ± SE coefficient aggregate (from ``estimand.reference_aggregate``)
    so ``regen verify`` can re-certify without raw rows. Never any per-row value.
    """
    from engine.prior.grounded import _encode_features

    fd = result.field_dict
    label_col = result.label_col
    feats = [c for c in result.normal_df.columns if c != label_col]
    numeric_cols = [
        c for c in feats
        if c in fd and fd[c].field_type in (FieldType.CONTINUOUS, FieldType.BINARY)
        and not getattr(fd[c], "is_identifier", False)
    ]

    n_norm_real, n_rare_real = len(result.normal_df), len(result.rare_df)

    agg: Dict[str, Any] = {
        "disclosure": {
            "min_bucket_count": int(min_bucket),
            "note": ("Aggregates only — no per-row values. Quantiles/histograms "
                     "published only for a class with >= min_bucket_count rows."),
        },
        "class_counts": {"real_normal": n_norm_real, "real_rare": n_rare_real,
                         "label_col": label_col},
        "synthetic_split": {"n_normal": int(n_normal), "n_rare": int(n_rare)},
        "numeric_columns": numeric_cols,
    }

    # Real-rare correlation matrix over numeric feature columns (allowed).
    if len(numeric_cols) >= 2 and n_rare_real >= 3:
        corr = result.rare_df[numeric_cols].corr().to_numpy()
        agg["correlation_rare"] = {
            "columns": numeric_cols,
            "matrix": [[None if not np.isfinite(x) else round(float(x), 8)
                        for x in row] for row in corr],
        }

    # Per-class encoded column moments (for Fisher separation recomputation).
    Xn = _encode_features(result.normal_df[feats], fd).astype(np.float64)
    Xr = _encode_features(result.rare_df[feats], fd).astype(np.float64)
    moments: Dict[str, Any] = {}
    for i, c in enumerate(feats):
        moments[c] = {
            "normal": {"mean": float(Xn[:, i].mean()), "var": float(Xn[:, i].var()),
                       "count": int(n_norm_real)},
            "rare": {"mean": float(Xr[:, i].mean()), "var": float(Xr[:, i].var()),
                     "count": int(n_rare_real)},
        }
    agg["column_moments"] = moments

    # Rare-class deciles per numeric column — only above the disclosure floor.
    if n_rare_real >= min_bucket:
        q = {}
        for c in numeric_cols:
            vals = result.rare_df[c].dropna().to_numpy(dtype=float)
            if vals.size:
                q[c] = [round(float(x), 8) for x in np.percentile(vals, np.arange(0, 101, 10))]
        agg["quantiles_rare"] = q
    else:
        agg["quantiles_rare"] = None
        agg["disclosure"]["quantiles_suppressed"] = (
            f"rare class has {n_rare_real} < {min_bucket} rows")

    # Declared-estimand coefficient aggregate (θ_real ± SE) — a disclosable
    # aggregate, same policy as the correlation matrix above.
    if estimand_real is not None:
        agg["estimand_real"] = estimand_real

    return agg


# ── Verification ──────────────────────────────────────────────────────────────

def verify_bundle(bundle_dir: str | Path) -> Dict[str, Any]:
    """Recompute every reported statistic from the bundle and check it.

    Returns {passed: bool, integrity: [...], stats: [...]}. `passed` is False if
    any artifact hash mismatches or any checkable statistic disagrees beyond its
    tolerance. UNCHECKABLE stats (need raw reference rows) never fail the run;
    they are reported so the disclosure limit is explicit.
    """
    bundle = Path(bundle_dir)
    manifest = json.loads((bundle / MANIFEST_NAME).read_text())
    explanation = json.loads((bundle / EXPLAIN_NAME).read_text())
    agg = json.loads((bundle / AGG_NAME).read_text())
    delivered = pd.read_parquet(bundle / BATCH_NAME)

    report: Dict[str, Any] = {"integrity": [], "stats": [], "notes": []}

    # 1. Integrity — recomputed artifact hashes must match the manifest.
    recorded = manifest.get("artifact_sha256") or {}
    artifacts = {BATCH_NAME: bundle / BATCH_NAME, EXPLAIN_NAME: bundle / EXPLAIN_NAME,
                 AGG_NAME: bundle / AGG_NAME}
    integrity_ok = True
    for name, path in artifacts.items():
        actual = sha256_file(path)
        exp = recorded.get(name)
        ok = (exp is not None and actual == exp)
        integrity_ok = integrity_ok and ok
        report["integrity"].append({
            "artifact": name, "passed": bool(ok),
            "recorded": exp, "recomputed": actual,
        })

    # 2. Metric-version guard — never compare across metric-definition changes.
    recorded_versions = manifest.get("metric_versions") or {}
    if recorded_versions and recorded_versions != metric_versions():
        report["notes"].append(
            f"metric versions differ (bundle={recorded_versions}, "
            f"current={metric_versions()}) — recomputation uses current definitions")

    # 3. Value recomputation from delivered data + reference aggregates.
    label_col = agg.get("class_counts", {}).get("label_col", "")
    n_rare = agg.get("synthetic_split", {}).get("n_rare", 0)
    # Delivered rare rows are the last n_rare (generate concatenates normal then rare).
    rare_synth = delivered.iloc[len(delivered) - n_rare:] if n_rare else delivered.iloc[:0]

    _verify_correlation(report, explanation, agg, rare_synth)
    _verify_fisher(report, explanation, agg)
    _verify_class_counts(report, agg, delivered, label_col)
    _verify_estimand(report, explanation, agg, manifest, delivered)
    _mark_uncheckable(report, explanation)

    stats_ok = all(s["passed"] for s in report["stats"] if s["status"] == "checked")
    report["passed"] = bool(integrity_ok and stats_ok)
    return report


def _stat(report, metric, status, passed=None, reported=None, recomputed=None, note=""):
    report["stats"].append({
        "metric": metric, "version": METRICS.get(metric, {}).get("version"),
        "status": status, "passed": passed, "reported": reported,
        "recomputed": recomputed, "note": note,
    })


def _verify_correlation(report, explanation, agg, rare_synth):
    reported = explanation.get("gates", {}).get("fidelity", {}).get("correlation", {}).get("value")
    real = agg.get("correlation_rare")
    if reported is None or real is None:
        _stat(report, "correlation_delta", "uncheckable",
              note="no correlation reported / no real correlation matrix in aggregates")
        return
    cols = [c for c in real["columns"] if c in rare_synth.columns]
    if len(cols) < 2 or len(rare_synth) < 3:
        _stat(report, "correlation_delta", "uncheckable",
              note="too few delivered rare rows/columns to recompute")
        return
    synth_corr = rare_synth[cols].corr().to_numpy()
    idx = [real["columns"].index(c) for c in cols]
    real_corr = np.array([[real["matrix"][i][j] for j in idx] for i in idx], dtype=float)
    iu = np.triu_indices_from(real_corr, k=1)
    diffs = np.abs(real_corr[iu] - synth_corr[iu])
    diffs = diffs[np.isfinite(diffs)]
    recomputed = round(float(diffs.mean()), 4) if diffs.size else None
    # The reported correlation is now measured on the DELIVERED (post-floor) rare
    # rows — the same rows in the bundle — so it is recomputable and checked even
    # under the floor. (Before the gate re-audited delivered data, this had to be
    # marked uncheckable when a floor was applied.)
    ok = recomputed is not None and abs(recomputed - reported) <= max(tolerance("correlation_delta"), 5e-4)
    _stat(report, "correlation_delta", "checked", bool(ok), reported, recomputed)


def _verify_fisher(report, explanation, agg):
    reported = {r["feature"]: r["fisher_score"]
                for r in explanation.get("feature_informativeness", {}).get("ranked", [])}
    moments = agg.get("column_moments") or {}
    if not reported or not moments:
        _stat(report, "fisher_separation", "uncheckable", note="no moments/scores")
        return
    worst = 0.0
    for feat, rep in reported.items():
        m = moments.get(feat)
        if not m:
            continue
        var = m["normal"]["var"] + m["rare"]["var"] + 1e-8
        recomputed = (m["rare"]["mean"] - m["normal"]["mean"]) ** 2 / var
        worst = max(worst, abs(round(recomputed, 6) - rep))
    ok = worst <= max(tolerance("fisher_separation"), 1e-4)
    _stat(report, "fisher_separation", "checked", bool(ok),
          reported="ranked scores", recomputed=f"max abs diff {worst:.2e}")


def _verify_class_counts(report, agg, delivered, label_col):
    split = agg.get("synthetic_split", {})
    n_rare = split.get("n_rare")
    if n_rare is None:
        _stat(report, "class_counts", "uncheckable")
        return
    delivered_rare = len(delivered) - split.get("n_normal", 0)
    ok = (delivered_rare == n_rare)
    _stat(report, "class_counts", "checked", bool(ok),
          reported=n_rare, recomputed=delivered_rare)


def _verify_estimand(report, explanation, agg, manifest, delivered):
    """Recompute θ_synth from the DELIVERED rows and re-run certification.

    θ_real ± SE is disclosed in ``agg['estimand_real']``; the spec comes from the
    manifest. We refit θ_synth on the delivered batch, re-certify against the
    disclosed θ_real, and check both (a) each θ_synth matches what was reported
    (within ``estimand_delta`` tolerance) and (b) the recomputed certified verdict
    matches the reported one. Undeclared / uncertifiable estimands are honestly
    marked uncheckable, never faked as a pass.
    """
    block = explanation.get("estimand") or {}
    real = agg.get("estimand_real")
    if not block.get("declared") or block.get("status") in (None, "not_declared",
                                                            "uncertifiable") or real is None:
        _stat(report, "estimand_delta", "uncheckable",
              note="no declared/certifiable estimand, or θ_real not in aggregates")
        return

    from contracts.scenario import EstimandSpec
    from regen.estimand import fit_estimand, certify, EstimandError

    spec_dict = (manifest.get("scenario") or {}).get("estimand") or {}
    spec = EstimandSpec.from_dict(spec_dict) if spec_dict else EstimandSpec(
        outcome=real.get("outcome", ""), predictors=real.get("predictors", []),
        family=real.get("family", "ols"), rule=real.get("rule", "consistent"),
        ci_level=real.get("ci_level", 0.95),
    )
    try:
        synth_fit = fit_estimand(delivered, spec)
    except EstimandError as e:
        _stat(report, "estimand_delta", "uncheckable",
              note=f"could not refit θ_synth on delivered data: {e}")
        return

    real_fit = {"coefficients": real.get("coefficients", {}),
                "n": real.get("n"), "dof": real.get("dof")}
    recomputed = certify(real_fit, synth_fit, spec)

    tol = max(tolerance("estimand_delta"), 1e-6)
    reported_targets = {t["coefficient"]: t for t in block.get("targets", [])}
    worst = 0.0
    for t in recomputed["targets"]:
        rep = reported_targets.get(t["coefficient"])
        if rep is None or rep.get("theta_synth") is None or t.get("theta_synth") is None:
            continue
        worst = max(worst, abs(float(rep["theta_synth"]) - float(t["theta_synth"])))
    coefs_ok = worst <= tol
    verdict_ok = bool(recomputed["certified"]) == bool(block.get("certified"))
    _stat(report, "estimand_delta", "checked", bool(coefs_ok and verdict_ok),
          reported={"certified": block.get("certified")},
          recomputed={"certified": recomputed["certified"],
                      "max_theta_synth_diff": f"{worst:.2e}"},
          note=("θ_synth refit from delivered rows; certified verdict re-derived "
                "against disclosed θ_real ± SE"))


def _mark_uncheckable(report, explanation):
    # These need the raw reference rows / full protocol — honestly out of scope
    # at aggregate disclosure (G-G point 4).
    if explanation.get("gates", {}).get("fidelity", {}).get("coverage") is not None:
        _stat(report, "coverage_rate", "uncheckable",
              note="needs raw real rare rows (not disclosed as aggregates)")
    if explanation.get("privacy"):
        _stat(report, "privacy_min_distance", "uncheckable",
              note="needs raw real rare rows to recompute nearest-neighbour distance")
    util = explanation.get("utility", {})
    if util.get("tail_lift") is not None:
        _stat(report, "tail_lift", "uncheckable",
              note="needs the full held-out detector protocol, not in the bundle")
