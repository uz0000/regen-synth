# Examples

Two different things live here, and it matters which one you want.

## The finding — start here

[`certifier_demo/`](certifier_demo/) holds the five scripts behind
[`FINDINGS.md`](../FINDINGS.md): the seven-source comparison, the standard-check
sweep, the mechanism measurement, the replication on a second dataset, and the
30-seed sweep. If you came here from the README, this is the directory you want.
It has its own [`README.md`](certifier_demo/README.md) explaining what each script
asks and how to read the output.

## The generator demos

These exercise the reference generator — the pipeline the certifier was built to
check, and which it refuses. They are about whether the machinery runs and reports
itself honestly, not about the finding.

| Script | What it stages |
|---|---|
| `run_demo.py` | the full pipeline end to end: ingest, generate, then Scout → Prior → Amplifier → Auditor → Examiner over several passes. Self-contained; it makes its own sample data if none is given. |
| `certified_surrogate_demo.py` | the two-party story: a producer holds the real data and emits a package; a consumer who never sees the real rows trains on the surrogate and re-verifies the certificate independently. See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) §3. |
| `make_sample_data.py` | writes `transactions.csv`, the toy table the two demos above fall back on. |
| `scenario_fraud.yaml` | a researcher-authored `ScenarioSpec`, kept as the worked example of the contract. |

```bash
python examples/run_demo.py                  # the pipeline, on generated sample data
python examples/run_demo.py --data my.csv    # or on your own table
python examples/certified_surrogate_demo.py  # the producer/consumer clean-room story
```

Both are covered by the test suite (`tests/test_cleanroom_demo.py`), so a crash in
either is a build failure rather than something you discover by running it.

## Note on outputs

`_run/` and `transactions.csv` are generated artifacts and are not tracked. Delete
them freely; the demos rebuild what they need.
