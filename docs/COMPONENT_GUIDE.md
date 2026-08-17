# Component guide

What each part of the system does, which standard method it uses, where it lives
in the code, and why that method was chosen.

For how the system works end to end, read [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md)
first. This file is the lookup table under it.

---

## The certifier

The core. It does not generate anything — it decides whether a conclusion
survived.

| Component | What it does | Method | Where | Why this method |
|---|---|---|---|---|
| **Certifier** | fits your declared analysis on the real and synthetic data, reports per-coefficient agreement | two-sample Wald consistency test on regression coefficients | `regen/certifier.py`, CLI `regen certify` | generator-agnostic, per-coefficient rather than one blurred score, and portable — recomputable from the disclosed θ_real ± SE without the real rows |
| **Estimand fits** | the regressions themselves | OLS (closed-form) and logistic regression (IRLS), numpy + scipy | `regen/estimand.py` | no dependency on a stats library whose solver behaviour could drift between versions |
| **v2 generator** | a generator built specifically to pass the certifier | Gaussian-mixture model of the predictor joint + a calibrated model of the real conditional P(y\|x) | `regen/estimand_preserving.py` | closes part of the gap the reference generator cannot; two of four coefficients recover unbiased and the rest carry a diagnosed, quantified bias — see [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) |

## The reference generator

REGEN's own synthetic-data pipeline — the system the certifier was built to
check, and which it refuses.

| Component | What it does | Method | Where | Why this method |
|---|---|---|---|---|
| **Ingest** | load, clean, split into normal and rare, profile columns | dtype/cardinality inference, median/mode imputation | `engine/ingest/loader.py` | a deterministic, model-free reading of the data |
| **Prior** | draws base rows grounded in the real distribution | mixed-data Gaussian copula, inverse-CDF sampling; Gaussian Naive Bayes density scorer | `engine/prior/grounded.py` | preserves each column's own distribution *and* how the columns move together, without copying a real row |
| **Amplifier** | corrects and densifies the rare tail | Gaussian-process regression with an ARD kernel (GPy) | `engine/amplifier/tail_corrector.py` | the tail residual is smoother and cheaper to learn than regenerating the whole distribution; ARD learns which features matter |
| **Scout** | picks which rare region to synthesise next | active-learning acquisition score + explored-region memory | `engine/scout/targeting.py` | focuses budget on the most informative region. Its incremental value is unproven — treat it as optional, not a headline |
| **Auditor — fidelity** | rejects batches that break real structure | total variation distance, Wasserstein-1, Pearson correlation delta, coverage radius | `engine/auditor/fidelity.py` | catches the failure that matters: right marginals, scrambled joint structure |
| **Auditor — conformance** | batch must obey its declared contract | bounds, type, category and uniqueness checks | `engine/auditor/conformance.py` | a batch that violates its own declared meaning is not shippable |
| **Examiner — lift** | does adding synthetic data help a detector? | random forest + leakage-free train/test recall on held-out real rare rows | `engine/examiner/detector.py` | measures augmentation benefit honestly; real only when the baseline is weak |
| **Examiner — TSTR** | does the synthetic set stand in for real data? | train-on-synthetic / test-on-real, three-model panel, ROC-AUC and PR-AUC | `engine/examiner/surrogate.py` | gives a recovery number against a real-data ceiling instead of an unanchored score |
| **Privacy floor** | prevents near-copy re-identification | σ-normalised nearest-neighbour distance floor (scipy `cKDTree`), verbatim guard, k-anonymity on discrete tuples | `engine/privacy.py` | pushes every released rare row off the real ones. **Not differential privacy** |
| **Constraint layer** | folds impossible values back onto reality | bounds and support enforcement | `engine/constraints.py` | no negative amounts, no fractional counts; never invents values the data never showed |

## Configuration and assurance

| Component | What it does | Method | Where | Why it exists |
|---|---|---|---|---|
| **ScenarioSpec** | the whole use case as one typed object | dataclasses + JSON/YAML, persisted in the manifest | `contracts/scenario.py` | single source of truth; a batch replays bit-for-bit from it |
| **Vetting gate** | merges three context sources under fixed rules | deterministic rule engine (researcher > structural > model; a proposal contradicting the data is dropped and logged) | `regen/vetting.py` | lets context parameterise the maths without ever overriding it |
| **Explanation** | every batch explains itself | computed report: gate statistics, provenance, feature informativeness via Fisher class-separation score | `regen/explain.py` | numbers cited to versioned metric IDs, never narrated by a model |
| **Audit bundle + verify** | a skeptic recomputes the numbers | SHA-256 artifact hashing plus independent recomputation from disclosure-bounded aggregates | `regen/audit_bundle.py`, `regen/metrics.py` | turns reported claims into checkable facts |
| **Preflight** | is this dataset in the envelope? | rule checks on rare count, all-categorical shape, time-series shape, dimensionality | `regen/preflight.py` | refuses out-of-scope data *before* generating, so failures are named up front |
| **Semantic layer** | optional: a model proposes column meanings and breaks target ties | one cached, provider-agnostic call, vetted by the gate | `regen/semantics.py` | lowers the expertise barrier. Metadata only — never a value |
| **Manifest** | reproducibility | seed + config + schema hash + code version + artifact hashes | `engine/manifest.py` | same manifest produces an identical batch |

---

## What is standard, and what is not

**Standard techniques used here**, none invented for this project: Gaussian
copula (Sklar's theorem); Gaussian-process regression with an ARD kernel
(Rasmussen–Williams, via GPy); Gaussian Naive Bayes; total variation distance;
Wasserstein-1; Pearson correlation; random forest, logistic regression and
gradient boosting (scikit-learn); ROC-AUC and PR-AUC; k-anonymity (Sweeney);
nearest-neighbour distance (scipy `cKDTree`); active-learning acquisition;
TSTR/TRTR as an evaluation protocol; inverse-CDF sampling; Laplace-smoothed
frequency tables.

**What this repo composed**: the `ScenarioSpec` contract and the deterministic
three-source vetting gate; conformance as a hard gate; the computed
`explanation.json`; the audit bundle and `regen verify` with versioned metrics
and a disclosure policy; the δ-floor enforced as the final step and cross-checked
against TSTR to catch memorisation; and the certifier itself — a
generator-agnostic, per-coefficient, recomputable verdict on whether a declared
analysis survived.

The papers in `docs/papers/` informed framing rather than implementation: active
residual learning suggested the Amplifier, Prior-Fitted Networks suggested the
"Prior" name, and a structured-semantic-control paper contributed an idea that
was ultimately deferred. An earlier TabPFN-style backend was tried and removed;
the current Prior is empirical copula sampling.

## What this does not do

- **It is not differential privacy.** It enforces a per-record distance floor so
  no released row is a near-copy of a real one, plus a verbatim and k-anonymity
  guard. There is no ε/δ bound, and it does not stop membership-inference or
  aggregate attacks.
- **Certification requires the real data.** It addresses sharing, not scarcity.
  See [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) §1.
- **The v2 generator does not solve the problem.** It certifies on 37% of seeds.
  The number belongs in any description of it.
- **Amplification lift is conditional.** It is real when a detector is starved of
  rare examples and approximately zero when the baseline is already strong. An
  early +39% headline was leakage-inflated and became +4.4% under leakage-free
  evaluation; that corrected number is the one to use.
- **It preserves correlation, not causation.** A synthetic surrogate can validate
  a pipeline's engineering. It is never evidence for a causal effect.
- **Single-table tabular only** — not time-series, relational, free text, or
  images. See [`CAPABILITY_MATRIX.md`](CAPABILITY_MATRIX.md).

## Common questions

**What is it, in one line?** A tool that checks whether a conclusion you would
draw from synthetic data would also be true of the real data — run against a
generator this repo also ships, and which it catches failing that check.

**How is it different from Gretel or Mostly AI?** Those lead with generation
quality and differential privacy. This leads with independent verifiability: a
batch you can check rather than trust. Different emphasis, not a superiority
claim.

**Does it actually make models better?** Only when the detector is data-starved.
On an already-good detector it honestly reports approximately zero. The headline
generator metric is TSTR, read alongside the privacy distance.

**What was hardest?** The honest evaluation — finding and removing the leakage
and selection bias that had inflated the early results, including a copula that
dropped discrete-to-continuous correlation and a fidelity gate that scored
pre-privacy-floor data instead of what actually shipped.

## Where to look for proof

- Certify any generator's output: `regen certify <real> <synthetic> --outcome <col> --predictors <a,b,c>`.
- Run the generator: `regen doctor <data>` → `regen generate <data>` → `regen verify <out>`.
- The change history with before/after numbers: [`BUILDLOG.md`](BUILDLOG.md).
- Formal metric definitions and thresholds: [`METHODS.md`](METHODS.md).
- Privacy guarantee and its limits: [`PRIVACY.md`](PRIVACY.md).
