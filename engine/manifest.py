"""
Batch manifest helpers.

Every synthetic batch must carry a manifest so it can be reproduced exactly
from (seed + config + code_version). This module builds manifests and seeds
the numpy RNG from them.

Invariant verified by tests/test_reproducibility.py:
  seed_from_manifest(m); generate() == seed_from_manifest(m); generate()
"""

import importlib.metadata
import subprocess
from typing import Any, Dict

import numpy as np

from contracts.types import BatchManifest, SchemaGraph


# ── Code version ──────────────────────────────────────────────────────────────

def _code_version() -> str:
    """Return the current git commit hash, or package version if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    try:
        return importlib.metadata.version("regen")
    except Exception:
        return "unknown"


# ── Manifest construction ─────────────────────────────────────────────────────

def build_manifest(
    seed: int,
    schema: SchemaGraph,
    prior_config: Dict[str, Any],
    target_region: Dict[str, Any],
    amplifier_params: Dict[str, Any],
    n_rows: int,
    rare_ratio: float = 0.0,
    privacy: str = "none",
    delta: float = 0.0,
    scenario: Any = None,
    artifact_sha256: Any = None,
    metric_versions: Any = None,
) -> BatchManifest:
    return BatchManifest(
        seed=seed,
        schema_hash=schema.schema_hash(),
        prior_config=prior_config,
        target_region=target_region,
        amplifier_params=amplifier_params,
        code_version=_code_version(),
        n_rows=n_rows,
        rare_ratio=rare_ratio,
        privacy=privacy,
        delta=delta,
        scenario=scenario,
        artifact_sha256=artifact_sha256,
        metric_versions=metric_versions,
    )


# ── RNG seeding ───────────────────────────────────────────────────────────────

def seed_rng(manifest: BatchManifest) -> np.random.Generator:
    """
    Return a seeded numpy Generator for this manifest.
    All stochastic engine operations take a Generator argument so results
    are fully determined by the manifest seed.
    """
    return np.random.default_rng(manifest.seed)
