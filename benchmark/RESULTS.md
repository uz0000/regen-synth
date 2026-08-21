# Benchmark results — index

Start here. This file says what each results file is, which are current, and what
the numbers mean. The tables themselves live in the files below, **written by the
scripts that produced them** — so they are regenerated rather than hand-edited,
and cannot drift from the run.

The generator's numbers are not this repo's headline finding. That is the
certifier's, in
[`../examples/certifier_demo/README.md`](../examples/certifier_demo/README.md).
These are the "does the generator work, and where does it stop working" numbers.

## Current

| File | What it measures | Produced by |
|---|---|---|
| [`RESULTS_TSTR.md`](RESULTS_TSTR.md) | how much of real-data performance a model trained on synthetic data recovers | `python benchmark/run_tstr_sweep.py` |
| [`RESULTS_PRIVACY.md`](RESULTS_PRIVACY.md) | what the privacy floor costs — floored vs none, per dataset | `python benchmark/run_privacy_sweep.py` |

### What they show

**Synthetic utility is strongly dataset-dependent.** TSTR recovery runs near 1.0
on several tables and 0.65 on `churn.csv`. There is no single "synthetic data is
X% as good" number, and quoting one would be dishonest. Where too few held-out
rare rows exist to trust an estimate, the status is `insufficient_real_test` and
no number is reported — an honest refusal rather than a fabricated figure.

**The privacy floor has a real, measured cost on two shapes.**
`solar_flare.csv` (low-cardinality integer features, where the floor collapses
coverage to 0.04) and `open_payments.csv` (all-categorical, where the floor
cannot apply at all). Both are open issues 1 and 2 in
[`../docs/KNOWN_ISSUES.md`](../docs/KNOWN_ISSUES.md), and both fail **loudly**
rather than shipping a quietly worse batch.

**Amplification lift is near zero on most of these tables**, which is the honest
result: the amplifier helps when a detector is genuinely starved of rare
examples, and not otherwise.

## Superseded

Kept as a record rather than deleted — several are what later measurement
corrected. Date-stamped in [`superseded/`](superseded/):

| File | Run | Why it was superseded |
|---|---|---|
| [`RESULTS_2026-06-18_SINGLE_PASS.md`](superseded/RESULTS_2026-06-18_SINGLE_PASS.md) | 2026-06-18, single pass | predates the full-synthesis change and the privacy layer |
| [`RESULTS_2026-06-18_MULTI_PASS.md`](superseded/RESULTS_2026-06-18_MULTI_PASS.md) | 2026-06-18, 3 datasets, 5-pass loop | superseded by the breadth run, then by the privacy layer |
| [`RESULTS_2026-06-22_BREADTH.md`](superseded/RESULTS_2026-06-22_BREADTH.md) | 2026-06-22, 11 datasets, 5 seeds | predates the full-synthesis change and the privacy layer |

The full change history, with before-and-after numbers for each correction, is in
[`../docs/BUILDLOG.md`](../docs/BUILDLOG.md).

## Standing check

[`run_regression.py`](run_regression.py) is not a sweep — it is the guard. It runs the
canonical datasets at fixed seeds, self-verifies each bundle with `regen verify`, and
compares every scored quantity against the provenance-stamped baselines in
[`BASELINES/`](BASELINES/) within explicit tolerances. **It exits non-zero on any
regression**: a fidelity or coverage drop, a correlation increase, a gate flip, a lift
drop, a runtime blow-up, or a bundle that fails to verify.

```bash
python benchmark/run_regression.py                    # check against baselines
python benchmark/run_regression.py --degrade          # prove it catches drift
python benchmark/run_regression.py --update-baselines # after an intended change
```

## Everything else in this directory

Exploratory and historical runners, kept because their outputs are cited in
[`../CORRECTIONS.md`](../CORRECTIONS.md) and [`superseded/`](superseded/) and a frozen
table with no script behind it is not reproducible. **None of these produce a current
number.** The two runners in the table above, plus the standing check, are what is live.

| Script | What it was for | Leaves behind |
|---|---|---|
| `run_multi.py` | 3 datasets x 5 seeds vs SMOTE — the first honest multi-dataset run | `RESULTS.json` |
| `run_multipass.py` | the full Scout-driven loop vs SMOTE, same 3 datasets | `RESULTS_MULTIPASS.json` |
| `run_breadth.py` | the 11-dataset breadth run that tested the heterogeneity hypothesis | `RESULTS_BREADTH.json` |
| `breadth_predict.py` | the dataset registry + the a-priori predictions, committed *before* the breadth run | `breadth_predictions.json` |
| `find_breadth.py` | one-off OpenML scan that picked those 11 datasets | `dataset_candidates.csv` |
| `run_satellite.py` | a single-dataset re-run of the above, for the satellite case | `RESULTS_SATELLITE.json` |
| `run_benchmark.py` | the original single-dataset credit-card-fraud campaign | `regen-output/benchmark_summary.json` |
| `run_unified.py` | the same campaign in one process, to remove subprocess overhead | `regen-output/benchmark_summary.json` |
| `noise_sweep.py` | swept the Prior's `noise_scale` to pick a default | `noise_sweep_results.json` |
| `compare_backends.py` | compared the copula Prior against the since-removed PFN backend; prints only | — |

**Read the lift numbers in the JSON files with the correction attached.** Everything
produced before the leakage-free protocol overstates amplification lift — that is
[`../CORRECTIONS.md`](../CORRECTIONS.md) §1, where satellite falls from +39% to about
+4%. The `RESULTS_TSTR.md` figures above are the ones that survived re-measurement.

## Naming

- `RESULTS_<TOPIC>.md` — a current sweep, written by its script.
- `RESULTS_<TOPIC>.json` — the machine-readable output beside it, or, with no `.md`
  beside it, the residue of an exploratory run in the table above.
- `superseded/RESULTS_<date>_<TOPIC>.md` — a historical run, frozen.

## Data

`data/` holds the benchmark tables. They are downloaded from OpenML by the runners
rather than redistributed here, so the directory is untracked; the provenance record
[`data/PROVENANCE.md`](data/PROVENANCE.md) is tracked and says where each one came from.
