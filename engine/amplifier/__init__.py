"""
Amplifier — TailCorrector tail correction.

Corrects the prior's weakness in the tail. The prior is strong on the
average case but systematically underestimates rare-event density. The
TailCorrector learns the *residual* (gap between prior predictions and
the truth on observed rare events) because residuals are smoother than
raw outcomes — they converge with far fewer observations (R-Design, Lemma 1).
"""

from .tail_corrector import AmplifierConfig, TailCorrector, fit_correction, sample_correction

__all__ = ["AmplifierConfig", "TailCorrector", "fit_correction", "sample_correction"]
