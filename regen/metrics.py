"""
Metrics registry (G-D/G-G) — one place that names every scored quantity, its
version, and where it is formally defined (docs/METHODS.md).

Benchmarks, the explanation, and the independent verifier all read the same
definitions from here, so a metric can't mean one thing in generation and
another in verification. Changing a metric's definition bumps its version; the
verifier and the regression harness refuse to compare across versions silently.

Pure data. No imports beyond the standard library.
"""

from __future__ import annotations

from typing import Dict

# id -> {version, per-metric numeric tolerance for cross-machine recomputation,
#        one-line summary, whether it is recomputable from AGGREGATES alone (vs
#        needing the raw reference rows — see the disclosure policy in G-G)}.
METRICS: Dict[str, Dict] = {
    "correlation_delta": {
        "version": 1, "tol": 1e-6,
        "summary": "mean abs difference between real and synthetic correlation matrices",
        "recomputable_from_aggregates": True,   # needs only the real corr matrix
    },
    "coverage_rate": {
        "version": 1, "tol": 1e-6,
        "summary": "fraction of real rare rows within sqrt(D) of a synthetic row",
        "recomputable_from_aggregates": False,  # needs raw real rare rows
    },
    "fisher_separation": {
        "version": 1, "tol": 1e-4,
        "summary": "per-feature (mu_rare-mu_normal)^2/(var_rare+var_normal)",
        "recomputable_from_aggregates": True,   # needs only per-class column moments
    },
    "privacy_min_distance": {
        "version": 1, "tol": 1e-4,
        "summary": "min sigma-normalized distance from a synthetic rare row to any real rare row",
        "recomputable_from_aggregates": False,  # needs raw real rare rows
    },
    "class_counts": {
        "version": 1, "tol": 0,
        "summary": "delivered per-class row counts",
        "recomputable_from_aggregates": True,
    },
    "tail_lift": {
        "version": 1, "tol": 1e-4,
        "summary": "amplified_recall - baseline_recall on held-out real rare rows",
        "recomputable_from_aggregates": False,  # needs the full detector protocol
    },
    "estimand_delta": {
        "version": 1, "tol": 1e-6,
        "summary": ("|theta_synth - theta_real| per declared regression coefficient; "
                    "preserved iff synth and real estimates are indistinguishable "
                    "beyond combined SE (two-sample Wald at ci_level)"),
        # theta_synth refits from the delivered rows; theta_real +/- SE is a
        # disclosed aggregate (a coefficient is an aggregate — no per-row values).
        "recomputable_from_aggregates": True,
    },
}


def metric_versions() -> Dict[str, int]:
    """{metric_id: version} — embedded in the manifest so results are never
    silently compared across metric-definition changes."""
    return {mid: m["version"] for mid, m in METRICS.items()}


def tolerance(metric_id: str) -> float:
    return METRICS.get(metric_id, {}).get("tol", 1e-6)
