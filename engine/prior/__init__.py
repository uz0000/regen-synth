"""
Prior — grounded-sampling base generator + normal-density scorer.

Produces a base batch that is statistically grounded in the real distribution,
by empirical grounded sampling (real anchor row + noise scaled to the observed
spread), not a learned generative model. A class-conditional Gaussian density
scorer (P(normal|x)) is fit alongside and consumed by the Amplifier to weight
residual relevance.

The base batch covers the average case well; the Amplifier corrects the tail.
"""

from .grounded import (
    PriorConfig, PriorModel, fit_prior,
    generate_base_batch, generate_normal_batch, generate_parametric_batch,
)

__all__ = [
    "PriorConfig", "PriorModel", "fit_prior",
    "generate_base_batch", "generate_normal_batch", "generate_parametric_batch",
]
