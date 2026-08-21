# Documentation

One subject per file. Start with whichever question you actually have.

The finding itself is not in here — it is in [`../FINDINGS.md`](../FINDINGS.md), and the
claims that were revised on the way to it are in [`../CORRECTIONS.md`](../CORRECTIONS.md).
These files explain how the thing works and what it is held to.

## Start here

| You want | File |
|---|---|
| How the certifier and the generator work, end to end | [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) |
| The math behind every number — what problem each piece solves, and what its output means | [`THE_MATH.md`](THE_MATH.md) |
| The same result as a visual walkthrough, no regressions assumed | [`inference-explainer.html`](inference-explainer.html) |
| How the parts connect, and which rules the shape enforces | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Which file implements which method, and why that method | [`COMPONENT_GUIDE.md`](COMPONENT_GUIDE.md) |

## Reference

| You want | File |
|---|---|
| Exact metric definitions and verification tolerances | [`METHODS.md`](METHODS.md) |
| Every field in `explanation.json` | [`EXPLAINABILITY.md`](EXPLAINABILITY.md) |
| What the privacy floor guarantees, and what it does not | [`PRIVACY.md`](PRIVACY.md) |
| Whether your data shape is supported | [`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md) |
| The HTTP endpoints and the built-in web UI | [`SERVER_API.md`](SERVER_API.md) |

## Status

| You want | File |
|---|---|
| What is currently open, numbered so it can be cited | [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) |
| Change history with before-and-after numbers | [`BUILDLOG.md`](BUILDLOG.md) |
| Rules the codebase holds itself to | [`../INVARIANTS.md`](../INVARIANTS.md) |

## Numbers

No document here restates a measured number. Result tables are written by the scripts
that produce them and carry a generated-file header, and prose links to them, so a
document cannot drift from a run.

- The headline per-coefficient tables: [`../examples/certifier_demo/`](../examples/certifier_demo/)
- The generator's own sweeps: [`../benchmark/RESULTS.md`](../benchmark/RESULTS.md)
