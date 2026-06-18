"""
Amplifier — ResidualGP tail correction.

Corrects the prior's weakness in the tail. The prior is strong on the
average case but systematically underestimates rare-event density. The
ResidualGP learns the *residual* (gap between prior predictions and
the truth on observed rare events) because residuals are smoother than
raw outcomes — they converge with far fewer observations (R-Design, Lemma 1).
"""

from .residual_gp import AmplifierConfig, ResidualModel, fit_residuals, sample_residuals

__all__ = ["AmplifierConfig", "ResidualModel", "fit_residuals", "sample_residuals"]
