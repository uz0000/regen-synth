"""
REGEN Deterministic Engine

Pure Python. No LLM client, no agent framework, no network library.
Enforced by tests/test_boundary.py.

Entry points:
  engine.prior    — grounded-sampling base generator + Gaussian density scorer
  engine.amplifier — TailCorrector tail correction
  engine.scout    — Scout targeting acquisition (targeting math)
  engine.auditor  — fidelity gate (hard reject on failure)
  engine.examiner — downstream detector + lift measurement
  engine.manifest — BatchManifest construction helpers
"""
