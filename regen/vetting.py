"""
The vetting gate (G-B) — deterministic code that merges the three context
sources into ONE vetted ScenarioSpec the engine can trust, under concrete rules.

Sources (authority order, rule 5): researcher declaration > structural inference
> model proposal. This module wires Sources 1 (structural) + 2 (researcher). The
optional model proposal (Source 3) lands in ``regen/semantics.py`` later and is
vetted by exactly the same gate.

What makes it reliable is that every proposed constraint must survive these
rules before it can affect generation; a violator is dropped, the field falls
back to the structural baseline, and the decision is logged (rule 7 — nothing
silent). The generators/gates (copula, GP, TVD/correlation, δ-floor) are never
touched — the contract only *parameterizes* them.

Pure Python. Lives outside engine/ (it is policy/orchestration, not the
statistical math); the delivered-batch conformance CHECK is in the Auditor
(engine/auditor/conformance.py), because enforcement-by-audit is the engine's job.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

from contracts.scenario import (
    ColumnSemantics, ScenarioSpec, VettingVerdict, columns_from_field_dict,
    ROLES, DTYPES, SOURCES,
)
from contracts.types import FieldDict, FieldType, IngestResult


# Below this per-field confidence a proposal falls back to structural (rule 6).
# Researcher declarations default to confidence 1.0; the knob matters for the
# model source (G-B Source 3).
CONFIDENCE_FLOOR = 0.5


def vet_scenario(
    proposed: ScenarioSpec | None,
    ingest: IngestResult,
    model_columns: Dict[str, ColumnSemantics] | None = None,
) -> Tuple[Dict[str, ColumnSemantics], List[VettingVerdict]]:
    """Merge Source 1 (structural) + Source 3 (model) + Source 2 (researcher).

    Authority order (rule 5): researcher > structural > model. So we lay the
    structural baseline, apply the vetted **model** proposal (lower authority,
    fills gaps / tightens within the data), then apply the vetted **researcher**
    declaration on top (it overrides the model where both speak). Every attribute
    is checked against the observed data by the same rules regardless of source;
    a violator is dropped and logged. Every structural column is in the output.
    """
    fd = ingest.field_dict
    label_col = ingest.label_col
    structural = columns_from_field_dict(fd, label_col)
    verdicts: List[VettingVerdict] = []

    proposed_cols = (proposed.columns if proposed is not None else {})
    model_cols = model_columns or {}

    vetted: Dict[str, ColumnSemantics] = {}
    for name, base in structural.items():
        cur = base
        mprop = model_cols.get(name)
        if mprop is not None and mprop.source != "structural":
            cur, vs = _vet_column(name, cur, mprop, fd.get(name))
            verdicts.extend(vs)
        uprop = proposed_cols.get(name)
        if uprop is not None and uprop.source != "structural":
            cur, vs = _vet_column(name, cur, uprop, fd.get(name))
            verdicts.extend(vs)
        vetted[name] = cur

    # A declared/proposed column that does not exist in the data is dropped +
    # logged (rule 3: data is ground truth — you cannot name a column that isn't
    # there). Covers both the researcher and the model source.
    for src_cols in (proposed_cols, model_cols):
        for name in src_cols:
            if name not in structural:
                verdicts.append(VettingVerdict(
                    field=name, decision="rejected", rule="data_is_ground_truth",
                    source=src_cols[name].source,
                    rationale="declared column is not present in the data",
                ))
    return vetted, verdicts


def _vet_column(
    name: str, base: ColumnSemantics, prop: ColumnSemantics, meta,
) -> Tuple[ColumnSemantics, List[VettingVerdict]]:
    """Apply one researcher-declared column's attributes under the gate rules."""
    out = replace(base)  # start from the structural baseline
    src = prop.source or "user"
    vs: List[VettingVerdict] = []

    # Rule 6: below-threshold confidence → fall back to structural entirely.
    if prop.confidence is not None and prop.confidence < CONFIDENCE_FLOOR:
        vs.append(VettingVerdict(field=name, decision="fallback",
                                 rule="confidence_fallback", source=src,
                                 rationale=f"confidence {prop.confidence} < {CONFIDENCE_FLOOR}"))
        return out, vs

    obs_min = getattr(meta, "min_val", None)
    obs_max = getattr(meta, "max_val", None)
    obs_integer = bool(getattr(meta, "is_integer", False))
    obs_categories = getattr(meta, "categories", None)

    # role — closed vocabulary (rule 2); any valid role is the researcher's call
    # (rule 5). An unknown role is ignored.
    if prop.role and prop.role != base.role:
        if prop.role in ROLES:
            out.role = prop.role
            vs.append(_ok(name, "role", src))
        else:
            vs.append(_reject(name, "role", src, "closed_vocabulary",
                              f"role {prop.role!r} not in {ROLES}"))

    # dtype — closed vocabulary; must not claim integer for non-integral data (rule 3).
    if prop.dtype and prop.dtype != base.dtype:
        if prop.dtype not in DTYPES:
            vs.append(_reject(name, "dtype", src, "closed_vocabulary",
                              f"dtype {prop.dtype!r} not in {DTYPES}"))
        elif prop.dtype == "integer" and not obs_integer:
            vs.append(_reject(name, "dtype", src, "data_is_ground_truth",
                              "claimed integer but observed values are not integral"))
        else:
            out.dtype = prop.dtype
            vs.append(_ok(name, "dtype", src))

    # integer flag — must match observed integrality (rule 3).
    if prop.integer != base.integer:
        if bool(prop.integer) == obs_integer:
            out.integer = bool(prop.integer)
            vs.append(_ok(name, "integer", src))
        else:
            vs.append(_reject(name, "integer", src, "data_is_ground_truth",
                              f"claimed integer={prop.integer} but observed={obs_integer}"))

    # semantic bounds — must CONTAIN the observed range (rules 3 & 4): a proposed
    # min may only widen below the observed min (e.g. currency ≥ 0), never above it
    # (which would clip a value the data actually exhibited).
    if prop.min is not None and obs_min is not None and prop.min != base.min:
        if prop.min <= obs_min:
            out.min = float(prop.min)
            vs.append(_ok(name, "min", src))
        else:
            vs.append(_reject(name, "min", src, "data_is_ground_truth",
                              f"proposed min {prop.min} > observed min {obs_min}"))
    if prop.max is not None and obs_max is not None and prop.max != base.max:
        if prop.max >= obs_max:
            out.max = float(prop.max)
            vs.append(_ok(name, "max", src))
        else:
            vs.append(_reject(name, "max", src, "data_is_ground_truth",
                              f"proposed max {prop.max} < observed max {obs_max}"))

    # category value-set — must be a SUPERSET of observed categories (rule 3).
    if prop.categories is not None and obs_categories is not None:
        if set(map(str, obs_categories)).issubset(set(map(str, prop.categories))):
            out.categories = list(prop.categories)
            vs.append(_ok(name, "categories", src))
        else:
            vs.append(_reject(name, "categories", src, "data_is_ground_truth",
                              "proposed category set omits observed categories"))

    # unit / notes — descriptive metadata, no data contradiction possible; accept.
    if prop.unit is not None and prop.unit != base.unit:
        out.unit = prop.unit
    if prop.notes:
        out.notes = prop.notes

    # Record who filled the column and its confidence (provenance, rule 7/8).
    out.source = src
    out.confidence = prop.confidence if prop.confidence is not None else 1.0
    out.proposal_id = prop.proposal_id
    return out, vs


def _ok(name: str, attr: str, src: str) -> VettingVerdict:
    return VettingVerdict(field=f"{name}.{attr}", decision="accepted",
                          rule="authority_order", source=src,
                          rationale="declared value contains/agrees with observed data")


def _reject(name: str, attr: str, src: str, rule: str, why: str) -> VettingVerdict:
    return VettingVerdict(field=f"{name}.{attr}", decision="rejected", rule=rule,
                          source=src, rationale=why + " → kept structural value")
