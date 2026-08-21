# The math behind the finding

This repository asks one question:

> **If you swap real data for synthetic data, do you reach the same conclusion?**

Answering it takes four pieces of arithmetic: a way to get the number a decision
rests on, a way to say how sure you are of it, a way to decide whether two such
numbers are the same answer, and a way to tell a lucky run from a real result.
This file walks through each one — what problem it solves, how it works, and what
its output means.

**How to read this.** Every symbol is named in words before it is used, and every
formula is followed by what it means in practice. If you skip the formulas
entirely and read only the prose, the argument still holds together. Sections 1,
3 and 6 are the ones that carry the finding.

Formal definitions and tolerances for the *generator's* internal metrics are a
separate, more technical concern, kept in [`METHODS.md`](METHODS.md).

---

## 1. The number a decision actually rests on

**The problem.** "Is this synthetic data good?" cannot be answered as asked. Data
doesn't have a quality on its own — a *use* of it does. So the question has to be
narrowed to something specific enough to measure.

**The narrowing.** Tables like this are mostly used one way: someone fits a
regression and acts on one of its numbers. A regression is a rule that predicts an
outcome from several inputs at once, and it produces one number per input, called
a **coefficient**. Each coefficient answers a narrow question:

> If this one input goes up by one unit, and every other input stays exactly where
> it is, how much does the outcome move?

Written out, for a straight-line model:

```
predicted outcome  =  β₀  +  β₁·(input 1)  +  β₂·(input 2)  +  …
```

`β₁` is the coefficient on input 1 — the size of the move in the outcome per unit
of that input. `β₀` is just the starting level when every input is zero.

When the outcome is yes-or-no rather than a quantity — did this account default,
yes or no — the same idea applies to the **odds** of yes instead of to the outcome
directly. The model is called logistic regression, and its coefficient says how
much the odds shift per unit of the input.

**What the number means in practice.** In this repository's example, the
coefficient on `pay_delay_1` — how many periods an account is already behind — is
**+0.714**. For a logistic model you turn a coefficient into a plain statement by
exponentiating it: e^0.714 ≈ 2.04. So:

> One more period behind roughly **doubles the odds** of default, holding
> utilisation, credit limit and age fixed.

That sentence is what a lender acts on. If the number moves, the lending decision
moves — no matter how correct the rest of the table looks.

Hold on to the phrase **"holding everything else fixed."** It sounds like a
technicality. It is the entire reason synthetic data fails, and section 6 is about
why.

---

## 2. How sure are we of that number?

Every coefficient comes with a second number: its **standard error**. Plainly:

> If you collected a fresh sample of the same size from the same population and
> refit the model, how much would this coefficient bounce around?

A small standard error means the data pins the number down. A large one means the
data is consistent with a wide range of answers and you happened to land on one.
You need this because section 3 compares two coefficients that are *both* uncertain,
and "are these different?" is meaningless without knowing how much each one wobbles.

Both numbers come from [`../regen/estimand.py`](../regen/estimand.py), written
directly against numpy and scipy instead of a statistics library, so that a
verification run on someone else's machine can't drift because a third-party solver
changed its behaviour.

### When the outcome is a quantity: one calculation, no guessing

`X` is the table of inputs, `y` the column of outcomes, `n` the number of rows and
`p` the number of coefficients being fit.

```
coefficients:     β̂  =  (XᵀX)⁻¹ Xᵀy
leftover error:   σ̂² =  (sum of squared misses) / (n − p)
uncertainty:      Var(β̂) = σ̂² (XᵀX)⁻¹
standard error:   SE of coefficient j = √( j-th diagonal entry of Var(β̂) )
```

`Xᵀ` means the table flipped on its side, and the whole first line is the standard
recipe for "the line that misses by the least overall." It's a single calculation
with no iteration, so it produces the same answer every time.

### When the outcome is yes-or-no: repeated refinement

There is no one-shot formula, so the fit starts from a guess and improves it. Each
round predicts everyone's probability with the current coefficients, sees who was
predicted badly, and nudges the coefficients toward fixing them. It stops when a
round changes nothing meaningful (a step smaller than 0.0000000001).

The uncertainty then comes from a quantity with a plain reading: **how sharply
peaked the fit is**. Picture the quality of fit as a hill, with the best
coefficients at the summit. If the hill is a sharp spike, only coefficients very
near the top fit the data well, so the answer is pinned down and the standard error
is small. If it's a broad plateau, many different coefficients fit about as well,
and the standard error is large. The math measures that curvature and inverts it —
sharp curvature, small uncertainty.

---

## 3. Deciding whether two numbers are the same answer

This is the piece everything else rests on.

**The problem.** The real data gives +0.714. The Gaussian copula gives +0.464.
Those differ — but *any* two estimates differ a little, just from sampling. The
real question is whether they are far enough apart to be **different answers**
rather than the same answer measured twice.

**The rule that looks obvious, and is wrong.** You could ask: does the synthetic
number fall inside the plausible range of the real one? This quietly rewards bad
generators. A generator producing a *noisy* estimate has a wide plausible range,
so it's *more* likely to overlap and pass. Under that rule, being imprecise helps
you. It is available in the code as `rule="within_ci"` and is deliberately not the
default.

**The rule used here.** Both numbers are uncertain, and the two fits are computed
on completely separate datasets. When two independent uncertain quantities are
subtracted, their uncertainties combine like the sides of a right triangle —
squares add:

```
combined uncertainty  =  √( (real SE)²  +  (synthetic SE)² )
```

Then the coefficient counts as **preserved** when the gap between the two estimates
is no bigger than about two of those combined units:

```
| real − synthetic |   ≤   1.96 × combined uncertainty
```

The 1.96 is the standard 95% cutoff. Dividing one side by the other gives a single
score:

```
z  =  | real − synthetic |  /  combined uncertainty
```

**What z means.** It is the gap measured in units of "how much these numbers wobble
anyway." z = 1 means the two differ by about as much as noise alone would explain —
that's agreement. z = 7 means they differ by seven times more than noise can
explain — that's a different answer.

**Why this rule is the right shape.** It accounts for the uncertainty in *both*
numbers, so it doesn't wrongly fail data that genuinely preserved the estimate.
And it contains the naive rule as a special case: as the synthetic dataset grows,
its own uncertainty shrinks toward zero and this rule collapses back into "is it
inside the real range?" The naive check isn't a different philosophy, just the
limiting case — with the flaw that it applies that limit before it's earned.

### Worked, on the headline row

The Gaussian copula, coefficient `pay_delay_1`, from
[`../examples/certifier_demo/RESULTS.md`](../examples/certifier_demo/RESULTS.md)
and the certificate written beside it:

```
real       = +0.7141      its SE = 0.0148
synthetic  = +0.4637      its SE = 0.0312

gap                   = 0.2503
combined uncertainty  = √(0.0148² + 0.0312²) = 0.0345
z                     = 0.2503 / 0.0345      = 7.25        vs a 1.96 cutoff
```

The two answers are **seven times further apart than sampling noise can explain**.
This is not a close call, and collecting more data would not make it one.

Notice also that the synthetic estimate's own uncertainty (0.0312) is about twice
the real one's (0.0148). Under the naive rule that extra sloppiness would have
worked in the copula's favour. Under this rule it is simply accounted for.

Reproduce every z in the table:

```python
from regen.certifier import certify_dataset
cert = certify_dataset(real_df, synth_df, estimand)
[(t["coefficient"], t["z"], t["preserved"]) for t in cert["targets"]]
```

### Why every coefficient has to pass

A source is certified only if **all** its declared coefficients are preserved. That
is a statement about conclusions, not a scoring preference. An analysis whose third
number is wrong is a wrong analysis, and blending four coefficients into one average
would let a broken one hide behind three intact ones. A conclusion is not partly
true.

### Why a stranger can check the verdict

The certificate carries the real coefficient and its standard error — two summary
numbers, not any rows. Anyone holding only the synthetic table can refit it, get
the synthetic coefficient and its standard error, and recompute z themselves. The
verdict is checkable by someone who never sees a single real record.

---

## 4. Where this test is weak

**It doesn't check whether the real data knew the answer.** Look at the rule again:
the passing threshold is proportional to the *combined* uncertainty, which includes
the real data's own. If the real dataset is small, its estimate is imprecise, the
threshold is loose, and a coefficient the real data never actually established can
sail through. That is preserving a shrug, not preserving a finding.

The repository surfaces this rather than hiding it: every coefficient reports
whether the real estimate was itself distinguishable from zero, so an empty
certification is visible. And if the real coefficient can't be fit at all, the
result is `uncertifiable` — refused rather than faked. See
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) issue 4.

**It's a test for difference, used to argue sameness.** Not finding a difference
isn't proof there is none. Here the asymmetry runs in the safe direction: the
finding is that generators **fail** this test, and a failure is solid evidence. But
a *pass* is weaker evidence than a failure, and the paragraph above is why.

The standard tool for the claim actually wanted is an **equivalence test**: instead
of asking "can I tell these apart," declare a margin — say, a coefficient is close
enough if it is within 0.05 of the real one — and test whether the difference is
provably *inside* it. That turns a failure to find a difference into positive
evidence of agreement. It is not implemented here, and
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) issues 4 and 9 are where the gap is recorded.

**It assumes the two fits are independent, and they never are.** The formula
`SE_Δ = √(SE_real² + SE_synth²)` is the variance of a difference *when the two
quantities are unrelated*. Every synthetic source here is built from the real data,
so the two estimates are correlated, and the full expression subtracts a covariance
term the code does not compute:

```
Var(real − synthetic) = Var(real) + Var(synthetic) − 2·Cov(real, synthetic)
```

Dropping a positive covariance makes the combined uncertainty **too large**, which
makes z **too small**, which biases the check toward reporting "preserved."

Work out what that means for this repository. The finding is that generators
**fail**, and a conservative test understates failures — so the results are, if
anything, milder than the truth. But it does mean the tool's failure mode is false
reassurance, which is precisely the sin this project exists to argue against, and
it is worth saying rather than leaving implicit.

**Each coefficient is a separate 95% test, and they compound.** Certification needs
all four to agree at once, so a faithful generator is refused whenever any one
comparison lands in its 5% tail. Four independent tests do that about 18.5% of the
time (`1 − 0.95⁴`). Measured on the positive control, under the demo's exact
configuration:

```
positive control refused   36 / 300 seeds  = 12.0%   (95% CI 8.3% - 15.7%)
```

Below 18.5%, because of the conservatism described just above — the two effects
push in opposite directions and partly cancel, and the measured rate lands where
that leaves it. Two things follow: a refused positive control is normal rather than
a sign of a broken checker, and a certification rate only means something against a
fixed declared analysis, since a larger estimand is harder to certify at the same
underlying fidelity.

**The controls show both weaknesses directly.** The negative control never
certifies across 200 seeds — the joint structure is destroyed and three of the four
coefficients are always caught. But `age` survives the shuffle about 20% of the
time, because its real effect is +0.010: a coefficient the real data barely
established is one that almost any table can match. That is the power limit above,
visible in a control rather than in a generator. Both rates are pinned in
`tests/test_control_rates.py`.

---

## 5. What the usual quality checks measure instead

People normally judge synthetic data three ways. Each measures something real.
None of them measures the coefficient.

**Does each column look right?** Compare the two versions of a single column and
take the largest gap between them, on a 0-to-1 scale where 0 is identical. This
looks at **one column at a time**, so by construction it cannot see anything about
how columns relate to each other.

**Do the columns move together correctly?** For every pair of columns, measure how
strongly they track each other, and report the biggest change. This *does* look at
relationships — but only at the straight-line part of a relationship, one pair at a
time. That limitation is exactly what section 6 turns on.

**Can you still train a model on it?** Train on the synthetic data, score on real
data held back for the purpose, and divide by the score you'd get training on real
data. 1.0 means it's a full stand-in for training. But **predicting well and
measuring an effect correctly are different jobs.** A model can rank cases in the
right order while every coefficient is wrong, because ranking only needs the
coefficients to be right *relative to each other*, not right in size.

**The point.** There is no rule of the form "if the column gap is under 0.1, then
the coefficient moved less than X." No such guarantee exists — the quantities aren't
connected. A small column gap is perfectly compatible with a badly moved
coefficient. Sometimes they happen to agree; nothing makes them agree. That's why
[`FIDELITY.md`](../examples/certifier_demo/FIDELITY.md) reports a table of how the
answer changes as the thresholds move, rather than one verdict: the thresholds were
never chosen with coefficients in mind, so where they land is arbitrary with respect
to this question.

---

## 6. Why the coefficient moves

**The problem.** What does a coefficient actually depend on? Get those things right
and it survives the swap. Get either wrong and it doesn't.

**The answer, in two parts.** Write the coefficients as a combination of two
ingredients:

```
coefficients  =  (how the inputs relate to each other)⁻¹  ×  (how each input relates to the outcome)
```

Both have to be right.

**Ingredient 1 — how the inputs relate to each other.** This is where "holding
everything else fixed" lives. Holding utilisation fixed while moving payment delay
only means something if you know how those two move together in reality. On the
credit data they overlap at +0.387, so they carry partly the same information, and
splitting the credit between them depends on exactly how much they overlap.

The `⁻¹` matters: it means every input's relationships get mixed into every
coefficient. An error anywhere in this ingredient spreads into all of them, not
just the one you got wrong.

**Ingredient 2 — how each input relates to the outcome.** The rows can sit in
exactly the right places and still carry the wrong outcome rate at each place.

**Separating the two.** [`GENERALITY.md`](../examples/certifier_demo/GENERALITY.md)
runs two sources that use the **same learned outcome rule** and differ only in
where the input rows came from — real rows versus a Gaussian copula. The copula
version fails, losing about a third of one coefficient's size. Since ingredient 2
was held correct by construction, that gap is ingredient 1 on its own. Getting the
outcome right does not rescue you from getting the inputs wrong.

**Why a copula breaks ingredient 1.** A Gaussian copula reproduces each column's
own distribution exactly, then ties the columns together using a single number per
pair for how strongly they move together. That works when the relationship really
is a steady trend. Being behind on payments isn't one. Measured on the real data
([`MECHANISM.md`](../examples/certifier_demo/MECHANISM.md)), the share of accounts
that default runs:

```
not behind      0.128
1 period behind 0.339
2 periods       0.691
```

Flat, then a **jump**. One number describing a straight-line relationship cannot
express a jump, so the copula renders the cliff as a gentle slope. Its correlation
with the outcome falls from +0.325 to +0.218, and the coefficient flattens from
+0.714 to +0.464.

**Why the rare-event amplifier breaks ingredient 2.** It manufactures extra rare
rows on purpose, and rare rows sit where accounts are furthest behind. That region
then fills with defaults at a higher rate than reality: its correlation with the
outcome rises to +0.417, the cliff gets steeper than the real one, and the
coefficient overshoots to +0.932.

Opposite errors, opposite causes, both caught — because the test in section 3
never asks how the data was made, only whether the answer came back the same.

**Why adding noise for privacy breaks it too.** A common way to anonymise data is
to jitter every value. Doing that makes the inputs look more spread out than they
really are, while leaving their connection to the outcome unchanged. The
coefficient is a ratio of the second thing to the first, so inflating the
denominator drags every coefficient **toward zero**. This effect has a name —
attenuation — and for a single input the shrinkage factor is exactly:

```
original spread / (original spread + added noise)
```

At 0.5σ of noise that's 1 / 1.25 = 0.8, predicting 0.714 × 0.8 ≈ **0.571**. The
measured value is **+0.521**. So the direction is right and the size is close, and
the simple formula overshoots — because all four inputs are jittered at once, and
their combined relationships have to be untangled together rather than one at a
time. The mechanism is real; the one-input formula is just its simplest case.

This is also why the privacy tension in
[`../FINDINGS.md`](../FINDINGS.md) section 7 is structural rather than a gap
someone could engineer away. **Privacy is bought by moving synthetic rows away
from real ones, and moving them away is precisely the operation that shrinks the
coefficients.** The two goals pull on the same lever in opposite directions.

---

## 7. Telling a lucky run from a real result

**The problem.** The second generator certifies on some runs and not others. One
run tells you nothing about which behaviour is typical — and a single lucky run is
exactly the kind of number this repository exists to distrust.

**How to separate them.** Run the same procedure many times with different random
starting points, and split the total error into two parts:

- **Bias** — how far the *average* run lands from the truth. A consistent lean.
- **Scatter** — how much individual runs bounce around that average.

Comparing them is the whole diagnostic:

- **Scatter bigger than bias** → the miss is luck. More runs would wash it out.
- **Bias bigger than scatter** → the miss is a real error. It repeats every run in
  the same direction, and more runs would not help.

Over 30 runs (`python examples/certifier_demo/seed_sweep.py`):

| input | bias | scatter | reading |
|---|---|---|---|
| `pay_delay_1` | −0.0133 | 0.0382 | scatter wins — this is luck |
| `age` | −0.0010 | 0.0039 | scatter wins — this is luck |
| `utilization` | +0.1689 | 0.1018 | bias wins — a real, repeating error |
| `log_limit` | +0.0587 | 0.0342 | bias wins — a real, repeating error |

**What this buys.** Two coefficients come back honestly; two are wrong the same way
every single time. That is a **diagnosis**, not a score: it says the weakness is in
how the method describes the input relationships — ingredient 1 from section 6 —
rather than something more data would fix. Which is why the fix is reported as
partial, at 11 of 30 runs, instead of by its best run.

**Why the runs have to be comparable.** Numerical libraries add up long lists of
numbers in an order that depends on how many processor threads they use, and those
tiny differences are enough to tip a borderline coefficient across the line. Set
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` or the spread you
measure includes your machine's scheduling. This was learned the hard way and is
recorded in [`../CORRECTIONS.md`](../CORRECTIONS.md) entry 2.

---

## 8. Building a generator for the two ingredients

**The problem.** Section 6 says a coefficient needs both ingredients right. Can a
generator target both on purpose?

**The construction** ([`../regen/estimand_preserving.py`](../regen/estimand_preserving.py)):

**For ingredient 1**, describe the real rows as a **blend of several overlapping
clouds** rather than one. A single cloud can only be stretched and tilted one way
overall, which is the copula's limitation restated; several overlapping clouds can
bend around a shape that isn't one smooth ellipse. New rows are then drawn from
that fitted shape — genuinely new rows, not real rows with noise added.

**For ingredient 2**, learn how the real outcome depends on where a row sits, and
use that to decide each new row's outcome. Deliberately **not** using the same
model form being graded: fitting a logistic model here would plant the very
coefficient under test. Using a different, flexible method means a coefficient that
comes back correct was genuinely carried by the data, not injected.

One implementation detail that matters: the inputs must be put on a common scale
first, or columns with large numbers dominate the description and the small-scale
ones get described badly.

**What the result means.** It certifies on **11 of 30 runs** — better than anything
else here, and not a solution. Section 7 explains why the shortfall is structural:
a blend of finitely many clouds is still an approximation to the real shape, and
the leftover error shows up as a repeatable lean on the two coefficients most
sensitive to it.

**The trade it exposes.** More clouds → a better description of the inputs → better
coefficients → rows sitting closer to the real ones → **less privacy**. That is
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) issue 8, and it comes from the geometry of the
problem rather than from this particular implementation.

---

## 9. The whole argument on one page

1. Decisions are made from a coefficient, so that's what has to survive the swap
   (§1).
2. A coefficient comes with a standard error saying how much it would wobble on a
   fresh sample (§2).
3. Two coefficients are the same answer when they differ by less than their
   combined wobble. That comparison is one division, and a stranger holding only
   the synthetic data can redo it (§3).
4. A coefficient depends on how the inputs relate to each other and on how they
   relate to the outcome (§6). The usual quality checks measure neither, and no
   rule connects them to how far a coefficient moved (§5).
5. So a generator can pass every usual check and still move the number you act on —
   and it does, in a direction and by an amount the mechanism predicts in advance
   (§6).
6. Repeating a result across many runs separates a lucky one from a real one (§7),
   which is why the one partial fix is reported at 37% rather than at its best run.

The limits of the argument are in §4 and in
[`../FINDINGS.md`](../FINDINGS.md) section 8.
