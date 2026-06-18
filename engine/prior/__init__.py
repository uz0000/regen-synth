"""
Prior Engine — statistical base data generator.

Given a schema and a Scout target region, produces a base batch that is
statistically grounded in the real distribution. The default backend is a
Gaussian Naive Bayes scorer (fast, air-gapped). An optional PFN backend
(TabPFN / RDB-PFN) provides relational in-context learning for linked
multi-table schemas.

The base batch covers the average case well; the Amplifier corrects the tail.
"""

from .rdbpfn import PriorConfig, PriorModel, fit_prior, generate_base_batch

__all__ = ["PriorConfig", "PriorModel", "fit_prior", "generate_base_batch"]
