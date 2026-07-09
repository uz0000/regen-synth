# REGEN — Component Guide (what's used where, and how to talk about it)

A reference for going back and answering "what technique is used where?" and for
speaking about REGEN confidently and *honestly* when someone asks. It maps each
component to the method it uses (by its real, standard name), where that lives in
the code, and why — then separates **standard techniques** from **REGEN's own
composition**, and closes with talking points + an FAQ.

Rule of thumb for speaking about it: **the individual math is standard and you
should name it as such; the original thing is how it's composed into a
*verifiable, contract-driven* pipeline.** Owning that distinction reads as rigor;
blurring it reads as overclaim.

---

## 1. Describe it in one breath / one minute

- **One breath:** "A synthetic-data system that turns a scarce real sample into a
  usable dataset *and a certificate a third party can independently recompute* —
  so you can act on the synthetic data defensibly."
- **One minute:** "Most synthetic-data tools just generate plausible rows. REGEN
  grounds every value in the real data's statistics with a deterministic engine,
  then wraps it in an assurance layer: a typed use-case contract, a computed
  explanation of why the batch passed, a privacy floor, and an audit bundle a
  skeptic can re-verify. It also measures honestly how well the surrogate stands
  in for real data (TSTR) and refuses — loudly — when it can't help. The math is
  standard; the verifiability is the point."

## 2. Component map — what · method · where · why

| Component | What it does (plain) | Method used (standard name) | Where | Why this method |
|---|---|---|---|---|
| **Ingest** | load, clean, split into normal/rare, profile columns | dtype/cardinality inference, median/mode imputation | `engine/ingest/loader.py`, `profile.py` | deterministic, model-free baseline understanding of the data |
| **Prior — value generator** | draws base rows grounded in the real distribution | **Gaussian copula** (mixed continuous+discrete), inverse-CDF sampling; **Gaussian Naive Bayes** density scorer | `engine/prior/grounded.py` | copula preserves each marginal *and* the correlations without copying a real row |
| **Amplifier — tail model** | corrects/densifies the rare tail | **Gaussian Process regression** with an **ARD kernel** (via GPy) | `engine/amplifier/tail_corrector.py` (`TailCorrector`) | the tail residual is smoother/cheaper to learn than regenerating the whole distribution; ARD learns which features matter |
| **Scout — targeting** | picks which rare region to synthesize next | **active-learning acquisition** (information-gain style score) + explored-region memory | `engine/scout/targeting.py` | focus budget on the most informative tail region (note: its incremental value is unproven — optional, not a headline) |
| **Auditor — fidelity gate** | rejects batches that break real structure | **Total Variation Distance**, **Wasserstein-1** distance, **Pearson correlation** delta, coverage radius | `engine/auditor/fidelity.py` | catches "right marginals, scrambled joint structure" — the failure that matters |
| **Auditor — conformance gate** | batch must obey the declared contract | bounds/type/category/uniqueness checks | `engine/auditor/conformance.py` | a batch that violates its own declared meaning isn't shippable |
| **Examiner — lift** | does adding synthetic help a detector? | **RandomForest** + leakage-free train/test recall (held-out real rare) | `engine/examiner/detector.py` | honest measure of augmentation benefit (conditional — real only when the baseline is weak) |
| **Examiner — TSTR** | does the surrogate *stand in* for real data? | **train-on-synthetic / test-on-real**, model panel (LogReg/RF/GBDT), **ROC-AUC** + **PR-AUC** | `engine/examiner/surrogate.py` (`measure_tstr`) | the headline "how much real performance is recovered" number, with a real-data ceiling |
| **Privacy — δ-floor + guard** | prevents near-copy re-identification | per-record **σ-normalized nearest-neighbour distance** floor (scipy `cKDTree`), verbatim guard, **k-anonymity** for discrete tuples | `engine/privacy.py` | pushes every released rare row off the real ones; **NOT differential privacy** |
| **ScenarioSpec — the contract** | the whole use case as one typed object | dataclasses + JSON/YAML; persisted in the manifest | `contracts/scenario.py` | single source of truth; a batch replays bit-for-bit from it |
| **Vetting gate** | merges 3 context sources under rules | deterministic rule engine (authority order, data-is-ground-truth, …) | `regen/vetting.py` | lets context parameterize the math without ever violating it |
| **Explanation** | every batch explains itself | computed report (gate stats, provenance, feature informativeness via class-separation Fisher score) | `regen/explain.py` | legibility — numbers, cited to versioned metric IDs, never narrated by a model |
| **Audit bundle + verify** | a skeptic recomputes the numbers | SHA-256 artifact hashing + independent recomputation, disclosure-bounded reference aggregates | `regen/audit_bundle.py`, `regen/metrics.py` | turns claims into checkable facts (the moat) |
| **Preflight** | is this dataset in the envelope? | rule checks (rare-count, all-categorical, time-series, dimensionality…) | `regen/preflight.py` | refuse out-of-scope shapes *before* generating |
| **Advisory model layer** | optional: LLM proposes column *meanings* | one cached, provider-agnostic call; vetted by the gate | `regen/semantics.py` | lowers expertise barrier; **metadata only — never a value** |
| **Manifest** | reproducibility | seed + config + schema hash + code version + artifact hashes | `engine/manifest.py` | same manifest → identical batch (Invariant 2) |

## 3. Methods & prior art (say this exactly)

**Standard, off-the-shelf techniques REGEN uses** (name them freely — they're
community/textbook methods, not anyone's proprietary IP, and not invented here):
Gaussian copula (Sklar's theorem); Gaussian Process regression + ARD kernel
(Rasmussen–Williams; GPy); Gaussian Naive Bayes; Total Variation Distance;
Wasserstein-1 / optimal transport; Pearson correlation; RandomForest / Logistic
Regression / Gradient Boosting (scikit-learn); ROC-AUC and PR-AUC; k-anonymity
(Sweeney); nearest-neighbour distance (scipy `cKDTree`); active-learning
information-gain acquisition; TSTR/TRTR (an established synthetic-data evaluation
protocol); inverse-CDF / rank-based sampling; Laplace-smoothed frequency tables.

**REGEN's own composition** (this is the original part — an *engineering synthesis
+ assurance layer*, not new mathematics): the **ScenarioSpec** contract and the
deterministic **3-source vetting gate**; **conformance-as-a-gate**; the computed
**`explanation.json`**; the **audit bundle + `regen verify`** (independent
recomputation, versioned metrics, a disclosure policy); the **δ-floor enforced as
the final step and cross-checked against TSTR** to catch memorization; the
leakage-free, honestly-reported evaluation discipline; and the **"certified
surrogate" framing** overall.

**The research papers in `docs/papers/`** informed the *framing* (active residual
learning → the Amplifier; Prior-Fitted Networks → the "Prior" name; structured
semantic control → a deferred, unused idea). REGEN implements **standard techniques
informed by** them — not verbatim reimplementations. An earlier RDB-PFN/TabPFN
backend was tried and **removed** (see git history); the current Prior is empirical
grounded/copula sampling.

## 4. How to talk about it — claims to make, and to avoid

**Make these (all true and demonstrable):**
- "It grounds every value in the real data statistically and never copies a real record."
- "Every batch ships a certificate a third party can independently recompute (`regen verify`)."
- "It measures honestly whether the synthetic stands in for real (TSTR) and refuses when it can't help."
- "The math is standard; the original part is the verifiable, contract-driven composition."

**Avoid these (they don't survive scrutiny):**
- ❌ "I invented [copula / GP / …]." → No; name them as standard.
- ❌ "It improves fraud detection by 39%." → Conditional and was inflated; lead with TSTR + honesty instead.
- ❌ "It's differential privacy." → It isn't; say what it is (near-copy floor).
- ❌ "It's a million-dollar product." → Unproven demand; frame as a rigorous capability.

## 5. FAQ — the questions people actually ask

- **"What is it, in one line?"** A synthetic-data generator that ships a
  machine-checkable certificate of how well the data stands in and how it's
  protected.
- **"Did you invent the math?"** No — it composes standard techniques (copulas,
  Gaussian processes, standard classifiers/metrics). What's mine is the
  verification/assurance layer and how it's all composed.
- **"Is this differential privacy?"** No. It enforces a per-record distance floor
  so no released row is a near-copy of a real one, plus a verbatim/k-anonymity
  guard. It does **not** give an ε/δ-DP bound or stop aggregate/membership attacks.
- **"How is it different from Gretel / Mostly AI?"** They lead with generation
  quality and DP. REGEN leads with **independent verifiability, a use-case
  contract, and honest measurement** — a batch you can *check*, not just trust.
  (Different emphasis, not a superiority claim.)
- **"Does it actually make models better?"** Only when the detector is
  data-starved (low baseline recall) — then yes, measurably. On an already-good
  detector it honestly reports ~0. The headline isn't lift; it's TSTR (how much
  real performance a model trained on the surrogate recovers), read alongside the
  privacy distance.
- **"What can't it do?"** Single-table tabular only — not time-series,
  relational, free text, or images; it's not differential privacy; scarce ≠
  absent (there's a minimum viable sample); and it preserves *correlation, not
  causation* (never present a surrogate as evidence for a causal effect).
- **"What was the hardest / most interesting part?"** The honest evaluation:
  finding and removing the leakage and selection bias that had inflated the
  results — e.g. a copula that dropped discrete↔continuous correlation, and a
  fidelity gate that scored pre-privacy-floor data instead of what actually ships
  — and building `regen verify` so nobody has to take my word for a number.

## 6. Where to point someone who wants proof
- Run it: `regen doctor <data>` → `regen generate <data>` → `regen verify <out>`.
- The honesty trail with before/after numbers: `docs/BUILDLOG.md`.
- Formal metric definitions + thresholds: `docs/METHODS.md`.
- Architecture and rationale: `docs/PRODUCT_SPEC.md`.
- Privacy guarantee and its limits: `docs/PRIVACY.md`.
