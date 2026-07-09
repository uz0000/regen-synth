"""
Examiner — downstream detector training and lift measurement.

Trains a simple classifier on real data (baseline) and on real + synthetic
amplified data (augmented), then measures recall/precision on the rare-event
tail. The difference is the *tail lift* — Scout's reward signal.

The Examiner never sees a batch that has not passed the Auditor gate.
"""

from .detector import ExaminerConfig, measure_lift
from .surrogate import measure_tstr, MIN_REAL_TEST_RARE, PANEL

__all__ = ["ExaminerConfig", "measure_lift", "measure_tstr",
           "MIN_REAL_TEST_RARE", "PANEL"]
