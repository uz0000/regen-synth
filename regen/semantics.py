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
