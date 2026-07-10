"""
Source 3 — the OPTIONAL advisory model-proposal path (G-B).

A single, cached, provider-agnostic model call that reads the *deterministic
schema profile* (never raw rows) and proposes column meanings/units/bounds/roles.
The proposal is metadata only (Invariant 1/4) and is then run through the SAME
deterministic vetting gate as a researcher declaration (`regen/vetting.py`) — a
proposal that contradicts the data is dropped and logged. So the model can change
*what context is inferred* but never *how numbers are made*.

Guarantees (the parts of the G-B rules specific to a model source):
- **Metadata only, redacted egress.** Only the profile is sent: column names,
  dtypes, cardinalities, observed bounds, and at most `REGEN_SEMANTICS_SAMPLES`
  example values per column — **zero** for identifier-role columns, and none at
  all when `REGEN_SEMANTICS_SAMPLES=0`. Exactly what was sent is persisted so the
  exposure is auditable (rule: what leaves the machine, G-F.3).
- **Advisory by default.** The proposal is applied only when the caller opts in
  (`accept_contract=True`); otherwise it is shown/logged but not applied.
- **Replayable, zero-call.** The raw proposal (text + prompt + model id + payload
  sent) persists in the manifest; a re-run from the persisted spec never calls a
  model. Offline / no key / model error → Sources 1+2 only; generation never
  blocks and never degrades silently (provenance says which sources ran).
- **Cost-bounded.** At most one call per dataset, cached by schema hash, never in
  a loop.

Lives OUTSIDE engine/ (the boundary test stays green). No hard dependency on any
SDK — the default caller uses urllib against an OpenAI-compatible endpoint; tests
inject a fake caller and never touch the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from contracts.scenario import ColumnSemantics, DTYPES, ROLES

logger = logging.getLogger(__name__)

# Process-lifetime cache: schema-hash → proposal, so a repeat within a run reuses
# the single call (cost-bounded, rule 10).
_PROPOSAL_CACHE: Dict[str, "ModelProposal"] = {}


@dataclass
class SemanticsConfig:
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    samples: int = 3

    @classmethod
    def from_env(cls) -> "SemanticsConfig":
        return cls(
            base_url=os.environ.get("REGEN_SEMANTICS_BASE_URL"),
            api_key=os.environ.get("REGEN_SEMANTICS_API_KEY"),
            model=os.environ.get("REGEN_SEMANTICS_MODEL"),
            samples=int(os.environ.get("REGEN_SEMANTICS_SAMPLES", "3")),
        )

    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


@dataclass
class ModelProposal:
    columns: List[ColumnSemantics] = field(default_factory=list)
    raw_text: str = ""
    prompt: str = ""
    model_id: str = ""
    payload_sent: Dict[str, Any] = field(default_factory=dict)  # exactly what left the machine
    proposal_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": [c.to_dict() for c in self.columns],
            "raw_text": self.raw_text, "prompt": self.prompt,
            "model_id": self.model_id, "payload_sent": self.payload_sent,
            "proposal_id": self.proposal_id,
        }


# ── Redacted egress payload ───────────────────────────────────────────────────

def build_model_payload(ingest, samples: int) -> Dict[str, Any]:
    """The ONLY thing sent to a model: the deterministic profile, redacted.

    Per column: name, dtype, cardinality, observed bounds — plus at most
    ``samples`` example values, except identifier-role columns which send **zero**
    values, and none at all when ``samples <= 0``.
    """
    from engine.ingest.profile import column_profiles
    cols = []
    for p in column_profiles(ingest, n_samples=max(0, samples)):
        examples: List[Any] = []
        if samples > 0 and p.get("role_guess") != "identifier":
            examples = list(p.get("sample_values", []))[:samples]
        cols.append({
            "name": p["name"], "field_type": p["field_type"],
            "cardinality": p["cardinality"], "role_guess": p["role_guess"],
            "min": p.get("min"), "max": p.get("max"),
            "example_values": examples,
        })
    return {"columns": cols, "samples_per_column": max(0, samples)}


def _schema_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


_PROMPT = (
    "You are a data-schema annotator. Given ONLY this column profile (no raw "
    "rows), return JSON: a list `columns`, each with name, role (one of "
    f"{list(ROLES)}), dtype (one of {list(DTYPES)}), unit (string or null), min, "
    "max (numbers or null), integer (bool), notes. Propose semantic bounds only "
    "where they are known-safe (e.g. a currency has min 0); never invent values. "
    "Profile:\n"
)


# ── The one call ──────────────────────────────────────────────────────────────

def propose_semantics(
    ingest,
    config: Optional[SemanticsConfig] = None,
    caller: Optional[Callable[[str, Dict[str, Any], SemanticsConfig], str]] = None,
) -> Optional[ModelProposal]:
    """Make (at most) one cached model call and parse a ModelProposal.

    Returns None — and never raises — when the model source is unavailable
    (offline / no key / error), so generation falls back to Sources 1+2. ``caller``
    is injectable for testing; the default hits an OpenAI-compatible endpoint.
    """
    config = config or SemanticsConfig.from_env()
    if caller is None and not config.enabled():
        logger.info("Semantics model source unavailable (no endpoint/key) — Sources 1+2 only.")
        return None

    payload = build_model_payload(ingest, config.samples)
    shash = _schema_hash(payload)
    if shash in _PROPOSAL_CACHE:
        return _PROPOSAL_CACHE[shash]

    prompt = _PROMPT + json.dumps(payload, default=str)
    call = caller or _openai_compatible_call
    try:
        raw = call(prompt, payload, config)
        cols = _parse_columns(raw)
    except Exception as exc:                       # never block generation
        logger.warning("Semantics proposal failed (%s) — Sources 1+2 only.", exc)
        return None

    proposal = ModelProposal(
        columns=cols, raw_text=raw, prompt=prompt,
        model_id=(config.model or "injected"), payload_sent=payload,
        proposal_id=shash,
    )
    _PROPOSAL_CACHE[shash] = proposal
    return proposal


def _parse_columns(raw: str) -> List[ColumnSemantics]:
    data = json.loads(raw)
    out = []
    for c in data.get("columns", []):
        if "name" not in c:
            continue
        out.append(ColumnSemantics(
            name=c["name"],
            role=c.get("role", "feature"),
            dtype=c.get("dtype", "float"),
            unit=c.get("unit"),
            min=c.get("min"), max=c.get("max"),
            integer=bool(c.get("integer", False)),
            notes=c.get("notes", ""),
            confidence=float(c.get("confidence", 0.8)),
            source="model",
        ))
    return out


# ── Full-scenario proposal (intent + gates + columns) — §5.2 ──────────────────

_SCENARIO_PROMPT = (
    "You are configuring a synthetic-data run. Given ONLY this column profile "
    "(no raw rows) and the user's goal, return JSON with three keys:\n"
    "  intent: {task (one of ['detector_training','data_sharing','benchmarking',"
    "'exploration']), label_col (a column name or ''), rare_mode "
    "(['label','percentile','imbalance_ratio']), rare_value, percentile, tail "
    "(['lower','upper']), rare_ratio (0-1 or null), focus_features (list of column "
    "names), mode (['faithful','balanced','boost'])}\n"
    "  gates: {privacy (['floored','none']), delta (0-2)}\n"
    "  columns: list of {name, role, dtype, unit, min, max, integer, notes}\n"
    "Propose only what the profile + goal support; never invent data values. "
    "Bounds must contain the observed range. Goal + profile:\n"
)


def propose_scenario(
    ingest,
    goal: str = "",
    *,
    n_rows: int = 300,
    seed: int = 42,
    config: Optional[SemanticsConfig] = None,
    caller: Optional[Callable[[str, Dict[str, Any], SemanticsConfig], str]] = None,
):
    """Draft a full ScenarioSpec from a plain-language goal + the schema profile.

    Returns (draft_spec, proposal_or_None). Always yields a **valid, editable**
    draft: a structural baseline (columns from the data + safe default intent/
    gates) that a model proposal, when available, refines on top. Offline / no key
    / model error → the structural baseline alone (never blocks). The draft is a
    *suggestion for the user to review and edit* — it is not auto-committed and is
    only vetted against the data when it is later passed to `generate()`.
    """
    from contracts.scenario import (
        ScenarioSpec, ScenarioIntent, ScenarioGates, columns_from_field_dict, TASKS,
    )

    fd = ingest.field_dict
    label = ingest.label_col
    feat_names = [c for c in fd if c != label]

    # Structural baseline (Sources 1 — always valid, always available).
    draft = ScenarioSpec(
        columns=columns_from_field_dict(fd, label),
        intent=ScenarioIntent(label_col=label, n_rows=n_rows, seed=seed),
        gates=ScenarioGates(),  # privacy="floored", delta=0.5 defaults
        notes=(f"Goal: {goal}" if goal else ""),
        provenance={"drafted_by": "structural"},
    )

    config = config or SemanticsConfig.from_env()
    if caller is None and not config.enabled():
        logger.info("Scenario proposer: model unavailable — structural draft only.")
        return draft, None

    payload = build_model_payload(ingest, config.samples)
    payload["goal"] = goal
    key = _schema_hash(payload)
    prompt = _SCENARIO_PROMPT + json.dumps(payload, default=str)
    call = caller or _openai_compatible_call
    try:
        raw = call(prompt, payload, config)
        data = json.loads(raw)
    except Exception as exc:                       # never block
        logger.warning("Scenario proposal failed (%s) — structural draft only.", exc)
        return draft, None

    # Apply the model's proposal onto the baseline, VALIDATED (closed vocabularies,
    # real columns only) — invalid fields fall back silently-to-default but are NOT
    # obeyed. Columns keep source="model"; the vetting gate re-checks them against
    # the data when generate() runs.
    _apply_intent(draft.intent, data.get("intent", {}), fd, label, feat_names, TASKS)
    _apply_gates(draft.gates, data.get("gates", {}))
    for c in _parse_columns(raw):
        if c.name in draft.columns:
            draft.columns[c.name] = c
    draft.provenance["drafted_by"] = "model+structural"
    draft.provenance["model_id"] = config.model or "injected"

    proposal = ModelProposal(
        columns=list(draft.columns.values()), raw_text=raw, prompt=prompt,
        model_id=(config.model or "injected"), payload_sent=payload, proposal_id=key,
    )
    return draft, proposal


def _apply_intent(intent, prop: Dict[str, Any], fd, label, feat_names, TASKS) -> None:
    """Apply a model's proposed intent, honoring only valid values (closed
    vocabularies + real column names); anything else is ignored (fallback kept)."""
    if prop.get("task") in TASKS:
        intent.task = prop["task"]
    if prop.get("label_col") in fd:                       # must be a real column
        intent.label_col = prop["label_col"]
    if prop.get("rare_mode") in ("label", "percentile", "imbalance_ratio"):
        intent.rare_mode = prop["rare_mode"]
    if prop.get("rare_value") is not None:
        intent.rare_value = prop["rare_value"]
    if isinstance(prop.get("percentile"), (int, float)) and 0 < prop["percentile"] < 1:
        intent.percentile = float(prop["percentile"])
    if prop.get("tail") in ("lower", "upper"):
        intent.tail = prop["tail"]
    rr = prop.get("rare_ratio")
    if isinstance(rr, (int, float)) and 0 < rr < 1:
        intent.rare_ratio = float(rr)
    if prop.get("mode") in ("faithful", "balanced", "boost"):
        intent.mode = prop["mode"]
    ff = prop.get("focus_features")
    if isinstance(ff, list):
        intent.focus_features = [c for c in ff if c in feat_names]   # real columns only


def _apply_gates(gates, prop: Dict[str, Any]) -> None:
    if prop.get("privacy") in ("floored", "none"):
        gates.privacy = prop["privacy"]
    d = prop.get("delta")
    if isinstance(d, (int, float)) and 0 < d <= 2:
        gates.delta = float(d)


# ── Target tie-break (Source 3, advisory) — §5.2 hand-off ─────────────────────

_TIEBREAK_PROMPT = (
    "You are selecting the rare-event TARGET column for a synthetic-data run. "
    "Structural scoring already found the `candidates` EQUALLY plausible (a "
    "statistical tie it refuses to guess through). Break the tie using the user's "
    "GOAL, the candidate names + their `example_values`, and `other_columns` (the "
    "rest of the schema — domain context that hints what the dataset is about). Do "
    "not invent anything, and do not second-guess the statistics. Return JSON "
    "{\"label_col\": <exactly one of the candidate names>, \"reason\": <one short "
    "sentence>}. You MUST pick from the candidate list; if nothing clearly favors a "
    "candidate, return {\"label_col\": \"\"} so a human decides. Goal + context:\n"
)


def resolve_ambiguous_target(
    candidates,
    goal: str = "",
    *,
    all_columns: Optional[List[str]] = None,
    candidate_examples: Optional[Dict[str, List[Any]]] = None,
    config: Optional[SemanticsConfig] = None,
    caller: Optional[Callable[[str, Dict[str, Any], SemanticsConfig], str]] = None,
):
    """Break a structural target tie using the user's goal (advisory Source 3).

    ``candidates`` is the ``TargetDetection`` list from ``AmbiguousTargetError`` — the
    columns the rule-based scorer found comparable. ``all_columns`` /
    ``candidate_examples`` are the optional *semantic context* the exception carries
    (the rest of the schema's names + a few example values per candidate). Returns
    ``(chosen_label_col, reason)`` where ``chosen`` is GUARANTEED to be one of the
    candidate names, or ``(None, reason)`` when no model is available / it declines /
    it returns an invalid name.

    NEVER raises. Offline, no key, a bad key, or malformed output all resolve to
    ``(None, ...)`` so the caller keeps the honest behavior — surface the tie and let
    the human choose.

    Egress discipline (same policy as ``build_model_payload``): only NAMES, the tie
    statistics, and — for the tied candidates only — up to ``config.samples`` example
    values are sent; when ``config.samples <= 0`` (``REGEN_SEMANTICS_SAMPLES=0``) no
    example values leave at all. ``other_columns`` sends names only, never values. No
    raw rows are ever sent. The model changes *which* column is chosen, never a value.
    """
    config = config or SemanticsConfig.from_env()
    names = [c.label_col for c in candidates]
    if caller is None and not config.enabled():
        return None, "model unavailable — tie left for the user to resolve"

    examples = candidate_examples or {}
    n_ex = max(0, config.samples)

    def _cand(c):
        d = {"name": c.label_col, "rare_value": c.rare_value,
             "minority_ratio": round(c.minority_ratio, 4), "n_rare": c.n_rare,
             "cardinality": c.cardinality, "score": round(c.score, 4)}
        if n_ex and c.label_col in examples:                 # redacted: capped, opt-out honored
            d["example_values"] = list(examples[c.label_col])[:n_ex]
        return d

    other_columns = [c for c in (all_columns or []) if c not in names]  # names only, no values
    payload = {
        "goal": goal,
        "candidates": [_cand(c) for c in candidates],
        "other_columns": other_columns,
    }
    prompt = _TIEBREAK_PROMPT + json.dumps(payload, default=str)
    call = caller or _openai_compatible_call
    try:
        raw = call(prompt, payload, config)
        data = json.loads(raw)
    except Exception as exc:                       # never block — degrade to human choice
        logger.warning("Target tie-break failed (%s) — tie left for the user.", exc)
        return None, f"model error ({exc}) — tie left for the user"

    chosen = data.get("label_col") or ""
    if chosen not in names:                        # must pick a real candidate (or "")
        if chosen:
            logger.info("Tie-break returned non-candidate %r — ignored.", chosen)
        return None, "model declined — tie left for the user"
    return chosen, str(data.get("reason", ""))[:200]


def _openai_compatible_call(prompt: str, payload: Dict[str, Any],
                            config: SemanticsConfig) -> str:
    """Default caller: POST to an OpenAI-compatible /chat/completions endpoint via
    urllib (no SDK dependency). Returns the assistant message content (JSON text).
    Only ``payload`` (the redacted profile, embedded in ``prompt``) ever leaves."""
    import urllib.request
    body = json.dumps({
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        config.base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {config.api_key}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]
