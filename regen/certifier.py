"""
Generator-agnostic estimand certification — the certifier as a standalone tool.

Given a real reference dataset, **any** synthetic dataset (however it was
produced — REGEN, SMOTE, a GAN, a hand-rolled sampler), and a declared analysis
(``EstimandSpec``), decide whether the synthetic data PRESERVES the estimate: fit
the analysis on both, compare each coefficient, and emit a portable, recomputable
certificate. The synthetic data's provenance is irrelevant — only whether the
conclusion survives.

This is the core the product is built around: it does not generate anything. The
certificate carries θ_real ± SE (a disclosed aggregate), so a third party can
recompute θ_synth from the synthetic data alone and re-check the verdict without
the real rows — the "certificate you attach to synthetic data you share" model.

Deterministic, no LLM, numpy + scipy only (via regen.estimand).
"""

from __future__ import annotations

from typing import Any, Dict

from contracts.scenario import EstimandSpec
from regen.estimand import evaluate, reference_aggregate
from regen.metrics import METRICS


def certify_dataset(real_df, synthetic_df, estimand: EstimandSpec,
                    source: str = "") -> Dict[str, Any]:
    """Certify whether ``synthetic_df`` preserves ``estimand`` measured on ``real_df``.

    Returns a portable certificate: the verdict (``certified`` / ``status``),
    per-coefficient θ_real vs θ_synth with the consistency test, the rule + CI
    level, the metric version, the source label, and the disclosed θ_real ± SE
    (so the certificate is re-checkable against the synthetic data alone). Never
    raises — an unfittable spec becomes an honest ``uncertifiable`` status.
    """
    assessment, real_fit = evaluate(real_df, synthetic_df, estimand)
    cert = dict(assessment)
    cert["source"] = source
    cert["metric"] = "estimand_delta"
    cert["metric_version"] = METRICS["estimand_delta"]["version"]
    if real_fit is not None:
        cert["theta_real_disclosed"] = reference_aggregate(real_fit, estimand)
    return cert


def certify_many(real_df, synthetics: Dict[str, Any],
                 estimand: EstimandSpec) -> Dict[str, Dict[str, Any]]:
    """Certify the *same* estimand across many synthetic sources → {name: certificate}.

    θ_real is identical across all of them (it comes from ``real_df``); what varies
    is θ_synth per source. This is the generator-agnostic comparison: it is not
    about who made the data, only whether each preserves the conclusion.
    """
    return {name: certify_dataset(real_df, df, estimand, source=name)
            for name, df in synthetics.items()}
