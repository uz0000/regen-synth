"""
Prior Engine — RDB-PFN base batch generator.

Wraps the RDB-PFN reference implementation (MuLabPKU/RDBPFN).
Given a schema and a Scout target region, produces a structurally consistent
base batch that is statistically grounded in the real distribution.

RDB-PFN uses relational in-context learning via a synthetic prior:
  - Block-diagonal correlation structure (correct per-table covariance)
  - FK topology preserved across tables
  - Temporal ordering maintained where present

The base batch covers the average case well; the Amplifier corrects the tail.
"""

from .rdbpfn import PriorConfig, PriorModel, fit_prior, generate_base_batch

__all__ = ["PriorConfig", "PriorModel", "fit_prior", "generate_base_batch"]
