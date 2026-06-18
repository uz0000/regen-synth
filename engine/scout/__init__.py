"""
Scout — R-EPIG acquisition (targeting math only).

The Scout selects which rare-event region to amplify next by scoring a
candidate pool with R-EPIG (Residual Expected Predictive Information Gain).
This file contains only the deterministic scoring math.

The Scout *picks the question*. The Amplifier answers it.

Thin vs reasoning: we implement thin Scout first (fixed candidate pool,
fully reproducible). A reasoning Scout (LLM proposes novel scenarios) is
deferred until the loop closes end-to-end (see INVARIANTS.md §7).
"""

from .repig import ScoutConfig, score_candidates, select_target

__all__ = ["ScoutConfig", "score_candidates", "select_target"]
