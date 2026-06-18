"""
Auditor — fidelity gate.

A batch that looks plausible but breaks the real correlation structure is
*worse than no data* — it silently corrupts downstream detectors. This gate
exists to stop exactly that.

Default mode: hard gate. A batch that fails any fidelity check is rejected
outright; it never reaches the Examiner or any persistent store.

Soft-penalty mode (alternative): feed fidelity failures back as a penalty
into Scout's reward so the system learns which regions are hard to fake
faithfully. Switch to soft only deliberately (see INVARIANTS.md §7).
"""

from .fidelity import AuditorConfig, audit

__all__ = ["AuditorConfig", "audit"]
