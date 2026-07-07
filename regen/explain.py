"""
Explainability (G-C) — every batch explains itself, from computed numbers only.

Answers, for a researcher: *what did I get, why should I trust it, and what
should I look at?* Assembled purely from the run's own report objects (fidelity /
privacy / conformance / lift), the vetted ScenarioSpec, the Scout target, and
class-separation statistics — **never narrated by a model**. If human-language
narration is ever layered on top it may only cite these numbers (INVARIANTS.md §4);
this JSON is the ground truth. A test asserts the numbers here equal the reports
they came from.

Pure Python; lives outside engine/. Reads only already-computed objects, so it
adds no statistical decision of its own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from regen.metrics import METRICS


def _cite(metric_id: str) -> Dict[str, Any]:
    """A metric-registry citation (id + version) so an explanation number is
    traceable to its formal definition in docs/METHODS.md (G-G)."""
    return {"metric_id": metric_id, "metric_version": METRICS.get(metric_id, {}).get("version")}


def feature_informativeness(result) -> List[Dict[str, Any]]:
    """Per-feature relevance for the rare class, ranked. Uses the class-separation
    (Fisher) statistic — (μ_rare − μ_normal)² / (σ²_rare + σ²_normal) — computed
    from the real data, so it is verifiable and needs no model. Higher = the
    feature separates rare from normal more, i.e. is more worth observing.
    """
    from engine.prior.grounded import _encode_features

    label_col = result.label_col
    feats = [c for c in result.normal_df.columns if c != label_col]
    if not feats:
        return []
    fd = result.field_dict
    Xn = _encode_features(result.normal_df[feats], fd).astype(np.float64)
    Xr = _encode_features(result.rare_df[feats], fd).astype(np.float64)
    out = []
    for i, name in enumerate(feats):
        n_col, r_col = Xn[:, i], Xr[:, i]
        var = n_col.var() + r_col.var() + 1e-8
        score = float((r_col.mean() - n_col.mean()) ** 2 / var)
        out.append({"feature": name, "fisher_score": round(score, 6)})
    out.sort(key=lambda d: d["fisher_score"], reverse=True)
    for rank, d in enumerate(out, 1):
        d["rank"] = rank
    return out


def _mechanism(col, privacy: str, rare_fallback: bool = False) -> str:
    """How the column's values were produced. Reflects a parametric→grounded
    fallback when it actually happened (never hidden in a log)."""
    if col.role == "identifier":
        return "identifier-minted"
    if col.role == "target":
        return "label-attached"
    if privacy == "floored" and not rare_fallback:
        if col.dtype in ("categorical", "boolean"):
            return "copula-frequency-sampled"
        return "copula-sampled + GP tail correction (rare)"
    if privacy == "floored" and rare_fallback:
        return "grounded-sampled (parametric fallback) + GP tail correction (rare)"
    return "grounded-sampled + GP tail correction (rare)"


def _column_provenance(vetted_spec, privacy: str, rare_fallback: bool = False) -> List[Dict[str, Any]]:
    verdicts = vetted_spec.verdicts
    prov = []
    for name, col in vetted_spec.columns.items():
        applied = [v.field for v in verdicts
                   if v.field.startswith(name + ".") and v.decision == "accepted"]
        rejected = [{"field": v.field, "rule": v.rule, "rationale": v.rationale}
                    for v in verdicts
                    if v.field.startswith(name + ".") and v.decision == "rejected"]
        prov.append({
            "column": name,
            "role": col.role,
            "source": col.source,
            "mechanism": _mechanism(col, privacy, rare_fallback),
            "constraints_applied": applied,
            "constraints_rejected": rejected,
        })
    return prov


def build_explanation(
    *,
    result,
    vetted_spec,
    rare_report,
    normal_report,
    conformance,
    privacy_out: Optional[Dict[str, Any]],
    lift,
    target_region: Dict[str, Any],
    aud_cfg,
    coverage_threshold: float,
    generation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the explanation dict for a generate() run (all computed numbers)."""
    generation = generation or {}
    rare_fallback = generation.get("rare_base") == "grounded_fallback"
    # -- per-gate account (statistic, threshold, verdict) --------------------
    def _cols_summary(rep):
        return {"n_passed": sum(1 for c in rep.column_results if c.passed),
                "n_total": len(rep.column_results)}

    gates: Dict[str, Any] = {
        "fidelity": {
            "coverage": {"value": round(rare_report.coverage_rate, 4),
                         "threshold": coverage_threshold,
                         "passed": bool(rare_report.coverage_passed), **_cite("coverage_rate")},
            "correlation": {"value": (round(rare_report.correlation_delta, 4)
                                      if rare_report.correlation_delta is not None else None),
                            "threshold": aud_cfg.correlation_threshold,
                            "passed": bool(rare_report.correlation_passed),
                            **_cite("correlation_delta")},
            "columns": _cols_summary(rare_report),
            "passed": bool(rare_report.overall_passed),
        },
        "normal_fidelity": {
            "correlation": {"value": (round(normal_report.correlation_delta, 4)
                                      if normal_report.correlation_delta is not None else None),
                            "threshold": aud_cfg.correlation_threshold,
                            "passed": bool(normal_report.correlation_passed)},
            "columns": _cols_summary(normal_report),
            "passed": bool(normal_report.overall_passed),
        },
        "conformance": conformance.to_dict(),
        "privacy": privacy_out,   # already threshold-annotated, or None
    }

    # -- utility with honesty markers (P2-7) ---------------------------------
    if lift is not None:
        utility = {
            "status": lift.status,
            "n_test_rare": lift.n_test_rare,
            "baseline_recall": round(lift.baseline_recall, 4),
            "amplified_recall": round(lift.amplified_recall, 4),
            "tail_lift": (round(lift.tail_lift, 4) if lift.status == "ok" else None),
            "protocol": ("Leakage-free: real rare rows split train/test first; "
                         "synthetic generated from the train fold only; both "
                         "detectors scored on held-out real rare rows."),
            **_cite("tail_lift"),
        }
    else:
        utility = {"status": "not_measured",
                   "reason": "batch did not pass the fidelity gate",
                   "protocol": "lift is only measured on a gate-passing batch (Invariant 3)"}

    # -- scout rationale ------------------------------------------------------
    scout = {k: v for k, v in (target_region or {}).items()
             if isinstance(v, (int, float, str, bool, type(None)))}

    return {
        "gates": gates,
        "feature_informativeness": {
            "method": "class-separation Fisher score (μ_rare−μ_normal)²/(σ²_rare+σ²_normal)",
            **_cite("fisher_separation"),
            "ranked": feature_informativeness(result),
        },
        "column_provenance": _column_provenance(
            vetted_spec, vetted_spec.gates.privacy, rare_fallback),
        "scout": scout,
        "utility": utility,
        "privacy": privacy_out,
        # Which base generator actually ran per part — records a
        # parametric→grounded fallback so a mechanism switch is never silent.
        "generation": {"rare_base": generation.get("rare_base"),
                       "normal_base": generation.get("normal_base")},
    }
