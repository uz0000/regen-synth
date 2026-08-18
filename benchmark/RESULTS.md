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

## Naming

- `RESULTS_<TOPIC>.md` — a current sweep, written by its script.
- `superseded/RESULTS_<date>_<TOPIC>.md` — a historical run, frozen.
