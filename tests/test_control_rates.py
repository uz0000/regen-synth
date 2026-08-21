"""How often the two controls behave, measured rather than asserted.

FINDINGS section 1 and the demo README used to say the positive control "must
certify" and that a failure meant the certifier was broken. That is false, and
misleading in a way that matters: certification requires all four coefficients to
agree at once, each compared at 95%, so a resample of the real rows is refused on
a noticeable fraction of seeds by chance. A reader who re-ran with a different
seed, saw the control refused, and followed that instruction would have concluded
the tool was broken when it was working exactly as designed. CORRECTIONS.md
entry 4.

**These tests assert properties, not individual verdicts.** An earlier version
pinned four specific seeds as "refusing" and failed on Python 3.10 while passing
on 3.11 and 3.12 — because two of those seeds sat only +0.05 and +0.08 past the
1.96 cutoff, and a different numeric build moves a borderline z by that much.
Pinning a borderline outcome is exactly the fragility this repository documents
elsewhere, so the sweep below asserts that *some* seeds are refused and that most
are not, which no build difference can flip.

Full measured rate: 36/300 = 12.0% (95% CI 8.3-15.7%), which is what four
independent 95% tests produce (1 - 0.95^4 = 18.5%) once the certifier's own
conservatism is accounted for. Reproduce by raising SWEEP below.

The real fit is computed once and reused, so a 300-seed sweep costs under a
second.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.scenario import EstimandSpec
from regen.estimand import certify, fit_estimand

DATA = Path(__file__).resolve().parent.parent / "examples" / "certifier_demo" / "credit_default.csv"
PREDICTORS = ["pay_delay_1", "utilization", "log_limit", "age"]
COLS = ["default"] + PREDICTORS
N_SYNTH = 6000          # what run_demo.py draws
DEMO_SEED = 7           # what run_demo.py uses
SWEEP = 300             # seeds swept in the rate tests


@pytest.fixture(scope="module")
def real():
    if not DATA.exists():
        pytest.skip("credit_default.csv not present")
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def spec():
    return EstimandSpec(outcome="default", predictors=PREDICTORS, family="logit")


@pytest.fixture(scope="module")
def real_fit(real, spec):
    return fit_estimand(real, spec)


def _bootstrap(real, seed):
    """Exactly run_demo.py's g_bootstrap."""
    return real[COLS].sample(N_SYNTH, replace=True, random_state=seed).reset_index(drop=True)


def _independent(real, seed):
    """Exactly run_demo.py's g_independent."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.choice(real[c].to_numpy(), size=N_SYNTH) for c in COLS})


def _verdicts(real, real_fit, spec, builder, n=SWEEP):
    return [certify(real_fit, fit_estimand(builder(real, s), spec), spec) for s in range(n)]


class TestThePositiveControl:

    def test_the_committed_demo_seed_certifies(self, real, real_fit, spec):
        """The published table has to be reproducible, so this one IS pinned."""
        cert = certify(real_fit, fit_estimand(_bootstrap(real, DEMO_SEED), spec), spec)
        assert cert["certified"] is True

    def test_it_is_refused_on_some_seeds(self, real, real_fit, spec):
        """The "must certify" claim is false: four 95% tests flag sometimes.

        Measured 12.0% over 300 seeds. Asserted only as "not zero", because the
        exact count depends on the numeric build.
        """
        refused = sum(1 for c in _verdicts(real, real_fit, spec, _bootstrap)
                      if not c["certified"])
        assert refused > 0, "expected chance refusals; the 'must certify' claim would be true"

    def test_but_it_is_refused_only_a_minority_of_the_time(self, real, real_fit, spec):
        """It is still a working positive control, not a broken one."""
        refused = sum(1 for c in _verdicts(real, real_fit, spec, _bootstrap)
                      if not c["certified"])
        assert refused < 0.30 * SWEEP, f"{refused}/{SWEEP} refused — too high for a positive control"

    def test_chance_refusals_are_near_misses_not_real_failures(self, real, real_fit, spec):
        """What separates a chance refusal from a genuine one is the margin.

        The demo's real failures reach z = 3.4 to 7.3. A control's chance refusal
        sits just past the 1.96 cutoff, which is why it is noise rather than a
        finding.
        """
        worst = []
        for cert in _verdicts(real, real_fit, spec, _bootstrap):
            flagged = [t for t in cert["targets"] if t["preserved"] is False]
            if flagged:
                worst.append(max(abs(t["z"]) for t in flagged))
        assert worst, "expected at least one refusal in the sweep"
        assert max(worst) < 3.4, f"a control refusal reached z={max(worst):.2f}"


class TestTheNegativeControl:

    def test_shuffled_columns_are_never_certified(self, real, real_fit, spec):
        """0 of 200 seeds certified — the one absolute claim here that holds."""
        certified = sum(1 for c in _verdicts(real, real_fit, spec, _independent, n=60)
                        if c["certified"])
        assert certified == 0

    def test_but_it_does_not_fail_on_every_coefficient(self, real, real_fit, spec):
        """`age` survives shuffling on about 20% of seeds.

        Not a defect in the shuffle — the real `age` effect is +0.010, and a
        coefficient the real data barely established is one almost any table can
        match. docs/KNOWN_ISSUES.md issue 4, showing up in a control.
        """
        survived = 0
        for cert in _verdicts(real, real_fit, spec, _independent, n=60):
            by_name = {t["coefficient"]: t for t in cert["targets"]}
            if by_name["age"]["preserved"] is True:
                survived += 1
        assert survived > 0, "expected `age` to survive the shuffle on some seeds"

    def test_the_coefficients_the_data_pinned_down_always_fail(self, real, real_fit, spec):
        """The three with real effects far from zero are never preserved."""
        for seed, cert in enumerate(_verdicts(real, real_fit, spec, _independent, n=30)):
            by_name = {t["coefficient"]: t for t in cert["targets"]}
            for name in ("pay_delay_1", "utilization", "log_limit"):
                assert by_name[name]["preserved"] is False, f"{name} survived at seed {seed}"
