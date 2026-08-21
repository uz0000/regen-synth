# The HTTP server

A FastAPI wrapper around `regen.api`, plus a self-contained single-page UI at
[`static/index.html`](static/index.html). Full endpoint reference:
[`../docs/SERVER_API.md`](../docs/SERVER_API.md).

```bash
pip install fastapi uvicorn python-multipart      # already in requirements.txt
uvicorn server.app:app --port 8000
```

Then `http://localhost:8000/` for the UI, or `http://localhost:8000/docs` for Swagger.

## What this is, and what it is not

**It wraps the generator, not the certifier.** Every endpoint here — ingest, screen,
doctor, propose, explore, generate, campaign, verify, download — drives the reference
generation pipeline. There is deliberately no `/api/certify`. The certifier needs the
real table *and* the synthetic one *and* a declared analysis, which is a two-party
operation rather than an upload-and-download one, so it lives in the CLI (`regen
certify`) and the Python API instead.

That means the web UI is a front end for the part of this repository the finding is
against. It is genuinely useful for exercising the pipeline, watching the gates fire,
and downloading a batch with its manifest — but a green result in the UI says the batch
passed *fidelity, conformance and privacy*. It does not say a conclusion survived.
Nothing in this directory can tell you that. Run `regen certify` for that answer, and
read [`../FINDINGS.md`](../FINDINGS.md) for what it will probably say.

## Why it is in a research repository

It predates the reorganisation around the certifier, and it stayed for two reasons.
It is the only place the whole pipeline is driven end to end by something other than a
test, which is a real check on the API surface; and it is covered by
`tests/test_server.py` (11 tests), so it cannot silently rot. It is not part of the
finding, and no number in [`../FINDINGS.md`](../FINDINGS.md) comes from it.

## The rules still apply

`server/` sits on the configure-and-certify side of the boundary in
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) §2. It calls `regen.api`; it never
computes a data value itself, and `engine/` does not import it. `tests/test_boundary.py`
enforces the direction of that dependency.
