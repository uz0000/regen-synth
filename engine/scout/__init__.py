"""
Scout — Scout targeting acquisition (targeting math only).

The Scout selects which rare-event region to amplify next by scoring a
candidate pool with Scout targeting (Scout targeting (gain score)).
This file contains only the deterministic scoring math.

The Scout *picks the question*. The Amplifier answers it.

Thin vs reasoning: we implement thin Scout first (fixed candidate pool,
fully reproducible). A reasoning Scout (LLM proposes novel scenarios) is
deferred until the loop closes end-to-end (see INVARIANTS.md §7).
"""

from .targeting import ScoutConfig, score_candidates, select_target

__all__ = ["ScoutConfig", "score_candidates", "select_target"]
