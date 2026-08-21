# System layout

How the parts connect, what each hands to the next, and which rules the design
refuses to break. [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) explains the mechanisms;
this file explains the shape they sit in. It extends
[`INVARIANTS.md`](../INVARIANTS.md) and does not override it.

---

## 1. What the system is

Two things that are deliberately separate:

**A certifier** that decides whether a declared analysis survives a swap from
real data to synthetic. It generates nothing, it never asks who produced the
data, and it is the core of the repo.

**A generator** that produces synthetic data with a report card attached — which
statistics it got right, which it did not, and by how much. It exists as the
reference implementation the certifier was built to check, and the certifier
refuses its output.

Nothing here is presented as more trustworthy than it has been measured to be.
The asset is the verifiable certificate and the honest diagnosis — a known,
bounded gap — not the rows, and not a promised accuracy lift.

## 2. The one rule the layout exists to enforce

Everything else follows from a single separation:

- **Code that produces values** is deterministic, offline, and free of any model
  call. It turns a configuration into grounded numbers and measures them.
- **Code that configures, certifies, and explains** produces no values. Model
  proposals, the vetting gate, explanation, verification, and the
  decision-support surface all live here.

They meet at exactly one object: the **`ScenarioSpec`**. The configuring side
writes it, the value-producing side reads it, the manifest persists it. That is
what lets context — a researcher, or a model reading a plain-language goal —
shape *what* gets generated while never touching *how* the numbers are made. It
is also why any run collapses to a deterministic, replayable object.

```
CONFIGURE / CERTIFY  (produces no values)
    human intent + real sample
        │
        ▼
    preflight (regen doctor) ──── out of envelope? stop and say so
        │
        ▼
    model proposal ──draft──►  vetting gate  ◄── researcher declaration
    (metadata only)            (fixed rules; a proposal that
                                contradicts the data is dropped + logged)
        │
        ▼
    ScenarioSpec  ── the single object both sides share ──┐
                                                          │  parameterises
PRODUCE VALUES  (deterministic, no model, no network)     │  (never edits)
        ┌─────────────────────────────────────────────────┘
        ▼
    Prior ─► Scout ─► Amplifier ─► constraints ─► privacy floor
        │
        ▼  delivered rows
    Auditor: fidelity + conformance gates (measured on what actually ships)
        │
        ▼
    Examiner: lift · TSTR · privacy distance
        │
        ▼  report objects
BACK TO CONFIGURE / CERTIFY
        │
        ▼
    explanation.json ─► audit bundle ─► regen verify (a skeptic recomputes it)
        │
        ▼
    tradeoffs + diagnosis surfaced ──► the human decides
```

Two walls hold it up. A model never writes a value; it only shapes the
`ScenarioSpec` and reads report objects. And the engine never makes the
value-laden call — privacy against fidelity against utility. It measures; the
human chooses.

The certifier sits outside this loop entirely. It takes a real dataset, any
synthetic dataset, and a declared analysis, and returns a verdict. It does not
need the generator to have been involved.

**Three surfaces drive that loop**, and none of them computes a value: `cli/`
(`regen <command>`), `server/` (FastAPI plus a self-contained web UI), and the Python
API in `regen/api.py` that both call. They sit on the configure-and-certify side of
the wall, and `engine/` imports none of them — `tests/test_boundary.py` enforces the
direction. Note the asymmetry: the server wraps the **generator** and has no certify
endpoint, because certification needs both tables and a declared analysis at once.
See [`../server/README.md`](../server/README.md).

## 3. How a value becomes a certified surrogate

Read top to bottom, this is why the output is defensible. Every step is
deterministic and later independently recomputable.

1. **Grounded** — the Prior's copula draws it from the real marginals and
   correlations, never as a copy of a real row.
2. **Corrected** — the Amplifier shapes the rare tail.
3. **Constrained** — folded onto valid support.
4. **Floored** — pushed at least δ away from every real rare row. The resulting
   gap is the price of privacy, not a defect.
5. **Gated** — fidelity and conformance checked on the *delivered* data, so the
   verdict describes what you actually get. It fails loudly if broken.
6. **Measured** — TSTR, lift, and privacy distance computed.
7. **Explained** — every number written to `explanation.json`, cited to a
   versioned metric.
8. **Bundled** — hashed into an audit bundle a stranger recomputes with
   `regen verify`.

Separately, and independently of all eight: **certified** — a declared analysis
is fit on the real and synthetic data and compared coefficient by coefficient.

## 4. Principles

1. The engine produces every value. A model may decide, describe, or
   orchestrate; it never writes a cell.
2. A model informs, the human decides, the engine grounds. No silent commit of a
   value-laden tradeoff — tools recommend, with override.
3. A measured gap is a price; an unmeasured gap is a landmine. Ship consequences
   measured and recomputable, and never present synthetic data as real.
4. Refuse loudly at the edges. Out of envelope, or a guarantee that cannot hold,
   means saying so — never a silent pass.
5. Reproducible from the spec, with zero model calls on replay.
6. Only metrics `regen verify` can recompute. No best-of-many headline.

## 5. Scope

**In scope:** a single flat table; continuous, categorical and binary features;
a rare-event framing; rows that are exchangeable, where order carries no signal.

**Out of scope:** differential privacy; time-series, relational, free-text and
image data; and causal claims — a surrogate may validate a pipeline's
engineering, never serve as evidence for a causal effect. Scarce is not the same
as absent: below a minimum viable sample, the system refuses.

If your problem is temporal, engineer time into per-row features
(`txns_last_24h`, `days_since_last_visit`) first. Once the time dimension lives
in columns rather than row order, rows are exchangeable again and the system
applies. It models the feature-engineered snapshot, not the raw sequence.

## 6. What every new piece has to pass

- [ ] No model output becomes a data value; the engine produces every number.
- [ ] Model code stays outside `engine/`; `test_boundary.py` stays green.
- [ ] Any model decision resolves to a persisted `ScenarioSpec`; replay makes
      zero calls and is bit-identical.
- [ ] Value-laden tradeoffs are surfaced with consequences, not auto-selected.
- [ ] No reported number that `regen verify` cannot recompute.
- [ ] Out of envelope, or a guarantee that cannot hold, fails loudly.
- [ ] A new gate, metric or mechanism ships its `explanation.json` entry and its
      documentation in the same change.
