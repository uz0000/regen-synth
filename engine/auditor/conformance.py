"""
Contract conformance check (G-B rule 9) — the Auditor's gate on the *vetted
ScenarioSpec*.

The fidelity Auditor asks "does the batch look like the real data?"; this asks
"does the batch obey the vetted contract?" — every delivered row must satisfy the
vetted semantic constraints: continuous values within [min, max], integrality,
categorical values inside the declared value-set, and identifier uniqueness. A
conformance failure fails the batch exactly like a fidelity failure (Invariant 3
extends to the contract) — a batch that violates its own declared meaning is not
shippable.

Pure Python (pandas/numpy). Imports only contracts + pandas — no LLM/network, so
the engine boundary holds. Enforcement lives here (in the engine) even though the
constraints were *decided* by the vetting gate above the engine: producing every
number and checking it is the engine's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from contracts.scenario import ScenarioSpec


@dataclass
class ConformanceReport:
    passed: bool = True
    violations: List[Dict[str, Any]] = field(default_factory=list)
    n_checked: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "n_checked": self.n_checked,
                "violations": self.violations}


def check_conformance(df: pd.DataFrame, spec: ScenarioSpec,
                      label_col: str = "") -> ConformanceReport:
    """Verify a delivered batch satisfies every vetted column constraint.

    Reports the first-order violation per column (count of offending rows), never
    a row value (G-F: no real/generated values in reports — only counts). Columns
    not present in the batch are skipped. The target/label column is exempt from
    the value-set check (it is set constant to the rare value, by design).
    """
    report = ConformanceReport()
    for name, col in spec.columns.items():
        if name not in df.columns:
            continue
        report.n_checked += 1
        s = df[name]

        # Identifier uniqueness — a minted key column must have no duplicates.
        if col.role == "identifier":
            n_dup = int(s.duplicated().sum())
            if n_dup:
                _add(report, name, "identifier_not_unique", n_dup)
            continue

        if col.role == "target" or name == label_col:
            continue  # label is metadata, set constant to the rare value

        is_numeric = pd.api.types.is_numeric_dtype(s)

        # Semantic bounds.
        if is_numeric and col.min is not None:
            n = int((s < col.min).sum())
            if n:
                _add(report, name, f"below_min({col.min})", n)
        if is_numeric and col.max is not None:
            n = int((s > col.max).sum())
            if n:
                _add(report, name, f"above_max({col.max})", n)

        # Integrality.
        if is_numeric and col.integer:
            vals = s.dropna().to_numpy(dtype=float)
            n = int(np.count_nonzero(vals != np.round(vals)))
            if n:
                _add(report, name, "non_integer", n)

        # Categorical value-set (superset of observed, so any delivered value must
        # be inside it). Only checked when a value-set was vetted in.
        if col.categories is not None and not is_numeric:
            allowed = set(map(str, col.categories))
            n = int((~s.dropna().astype(str).isin(allowed)).sum())
            if n:
                _add(report, name, "value_outside_declared_set", n)

    report.passed = len(report.violations) == 0
    return report


def _add(report: ConformanceReport, col: str, kind: str, n_rows: int) -> None:
    report.violations.append({"column": col, "violation": kind, "n_rows": int(n_rows)})
