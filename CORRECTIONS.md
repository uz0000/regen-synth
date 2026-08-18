# Corrections

Claims this project published and then revised after re-measuring them. Each
entry states what was claimed, what the re-run showed, and what changed as a
result.

This file exists because a repository that only shows results which survived is
not showing its method. The corrections below were all found by re-running
something at a larger scale, under a stricter protocol, or against a check that
had not previously been run at all.

---

## 1. Rare-event amplification lift: +39% became roughly +4%

**Claimed.** An early benchmark table reported large detection gains from
amplifying rare examples, headlined by +39.1% on the satellite dataset and
+12.6% on hypothyroid.

**What was wrong.** The evaluation was not leakage-free. Synthetic rows derived
from the real data were being scored against a test split that was not properly
quarantined, so the measurement partly reflected information the model had
already seen.

**Re-measured.** Under a leakage-free protocol, where a real test slice is held
out before generation and never touched again, the satellite gain fell from +39%
to about +4%.

**What changed.** The benchmark tables were replaced with leakage-free
train-on-synthetic test-on-real recovery figures, in
[`benchmark/RESULTS_TSTR.md`](benchmark/RESULTS_TSTR.md). The headline claim was
narrowed to a conditional one: amplification helps when a detector is genuinely
starved of rare examples, and reports approximately zero when the baseline is
already strong. Weak results were kept in the table rather than dropped, and
cases with too few held-out rare rows now report a refusal instead of a number.

---

## 2. Estimand-preserving generator: "roughly 7 of 8 seeds" became 11 of 30

**Claimed.** The v2 generator was reported as certifying the full declared
analysis on approximately 7 of 8 seeds.

**What was wrong.** That rate came from a validation sample of 8 to 9 seeds, run
without pinning BLAS thread count. The sample was too small to estimate a rate,
and floating-point summation order varied between runs, which is enough to flip a
borderline coefficient.

**Re-measured.** 30 seeds, single-threaded BLAS, deterministic
(`python examples/certifier_demo/seed_sweep.py`): **11 of 30, or 37%.**

The shortfall is not seed noise. Mean bias against standard deviation across the
30 seeds:

```
predictor        theta_real   mean bias       std
pay_delay_1         +0.7141     -0.0133    0.0382
utilization         -0.3693     +0.1689    0.1018
log_limit           -0.3145     +0.0587    0.0342
age                 +0.0100     -0.0010    0.0039
```

`utilization` and `log_limit` carry bias larger than their spread, which is
systematic distortion from how a Gaussian mixture approximates the real predictor
joint, not variance.

**What changed.** Every statement of the rate became 37%. The generator is now
described as a partial fix throughout. The residual bias is documented as an open
issue rather than presented as noise, and thread pinning is stated wherever a
reproduction instruction appears.

---

## 3. "The standard checks miss it" was asserted, not measured

**Claimed.** Several documents stated that the failing generators pass the
ordinary fidelity and prediction checks while breaking the coefficient, so
nothing in a normal workflow would catch it.

**What was wrong.** Nothing in the repository computed those checks on the seven
demo sources. The train-on-synthetic sweep in `benchmark/` measures this repo's
own generator on unrelated datasets, which does not support a statement about
what the copula, SMOTE, or additive noise pass on the credit data. The claim was
plausible and unverified, which given the subject of this project is the specific
failure mode it exists to argue against.

**Re-measured.** `python examples/certifier_demo/fidelity_check.py` now runs
Kolmogorov-Smirnov distance, correlation-matrix shift, and train-on-synthetic
recovery on all seven sources, with results in
[`examples/certifier_demo/FIDELITY.md`](examples/certifier_demo/FIDELITY.md).

At strict thresholds the claim is false: every failing source is caught by at
least one standard check, so there is no silent failure. The margins are thin
enough to matter, though. SMOTE fails only on a KS of 0.107 against a 0.10 line
and REGEN on 0.112. Loosening KS to 0.15 makes both pass every standard check
while still moving the coefficient.

**What changed.** The claim was replaced with the narrower one the evidence
supports, in [`FINDINGS.md`](FINDINGS.md) section 3: the standard checks do not
measure the coefficient, and whether they flag a coefficient failure depends on
threshold choices unrelated to it. The threshold sensitivity is reported as a
table rather than a single number, because the single number was the problem.

---

## How these get found

Each of the three came from the same move: taking a number that had been
established once and running it again at larger scale, under a stricter
protocol, or for the first time. The first two were found that way after
publication. The third was found by asking which sentence in the repository had
no script behind it.
