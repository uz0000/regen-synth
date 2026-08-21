"""How often the two controls behave, measured rather than asserted.

FINDINGS section 1 and the demo README used to say the positive control "must
certify" and that a failure meant the certifier was broken. That is false, and
misleading in a way that matters: certification requires all four coefficients to
agree at once, each compared at 95%, so a resample of the real rows is refused on
a noticeable fraction of seeds by chance. A reader who re-ran with a different
seed, saw the control refused, and followed that instruction would have concluded
the tool was broken when it was working exactly as designed.

These tests pin what is actually true, so the prose cannot drift back:

  * the committed demo seed certifies (the published table is reproducible);
  * some seeds do not (the "must certify" claim is wrong);
  * the negative control never certifies (that claim is safe);
  * but it does not fail on every coefficient, because `age`'s real effect is
    +0.010 and a coefficient the real data barely established is one almost
    anything matches — docs/KNOWN_ISSUES.md issue 4, visible in a control.

Specific seeds are used rather than a sweep so the suite stays fast. The full
rates (12% and 20%) come from 300- and 200-seed sweeps; the reproduction command
is in the docstring of each test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.scenario import EstimandSpec
from regen.certifier import certify_dataset

DATA = Path(__file__).resolve().parent.parent / "examples" / "certifier_demo" / "credit_default.csv"
PREDICTORS = ["pay_delay_1", "utilization", "log_limit", "age"]
COLS = ["default"] + PREDICTORS
N_SYNTH = 6000          # what run_demo.py draws
DEMO_SEED = 7           # what run_demo.py uses

# Found by sweeping seeds 0..39 under the demo's exact configuration.
REFUSING_SEEDS = [9, 26, 27, 35]


@pytest.fixture(scope="module")
def real():
    if not DATA.exists():
        pytest.skip("credit_default.csv not present")
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def spec():
    return EstimandSpec(outcome="default", predictors=PREDICTORS, family="logit")


def _bootstrap(real, seed):
    """Exactly run_demo.py's g_bootstrap."""
    return real[COLS].sample(N_SYNTH, replace=True, random_state=seed).reset_index(drop=True)


def _independent(real, seed):
    """Exactly run_demo.py's g_independent."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.choice(real[c].to_numpy(), size=N_SYNTH) for c in COLS})


class TestThePositiveControl:

    def test_the_committed_demo_seed_certifies(self, real, spec):
        """The published table has to be reproducible."""
        assert certify_dataset(real, _bootstrap(real, DEMO_SEED), spec)["certified"] is True

    @pytest.mark.parametrize("seed", REFUSING_SEEDS)
    def test_some_seeds_are_refused_by_chance(self, real, spec, seed):
        """Four coefficients, each a 95% test, so refusal happens without a bug.

        Full rate: 12.0% over 300 seeds (95% CI 8.3-15.7%), which is what four
        independent 95% tests produce (1 - 0.95^4 = 18.5%) once the test's own
        conservatism is accounted for. Reproduce by sweeping `seed` here.
        """
        assert certify_dataset(real, _bootstrap(real, seed), spec)["certified"] is False

    def test_a_refusal_is_a_near_miss_not_a_broken_check(self, real, spec):
        """The flagged coefficient sits just past the line, not wildly off.

        This is what separates a chance refusal from a real failure: the demo's
        genuine failures reach z = 3.4 to 7.3, while a control's chance refusal
        sits close to the 1.96 cutoff.
        """
        cert = certify_dataset(real, _bootstrap(real, REFUSING_SEEDS[0]), spec)
        flagged = [t for t in cert["targets"] if t["preserved"] is False]
        assert flagged, "expected at least one flagged coefficient on this seed"
        assert max(abs(t["z"]) for t in flagged) < 3.0


class TestTheNegativeControl:

    @pytest.mark.parametrize("seed", [0, 1, 2, DEMO_SEED])
    def test_shuffled_columns_are_always_refused(self, real, spec, seed):
        """0 of 200 seeds certified — the one absolute claim here that holds."""
        assert certify_dataset(real, _independent(real, seed), spec)["certified"] is False

    def test_but_it_does_not_fail_on_every_coefficient(self, real, spec):
        """`age` survives shuffling on about 20% of seeds.

        Not a defect in the shuffle — the real `age` effect is +0.010, and a
        coefficient the real data barely established is one almost any table can
        match. docs/KNOWN_ISSUES.md issue 4, showing up in a control.
        """
        survived = 0
        for seed in range(12):
            cert = certify_dataset(real, _independent(real, seed), spec)
            by_name = {t["coefficient"]: t for t in cert["targets"]}
            if by_name["age"]["preserved"] is True:
                survived += 1
        assert survived > 0, "expected `age` to survive the shuffle on some seeds"

    def test_the_coefficients_the_data_pinned_down_always_fail(self, real, spec):
        """The three with real effects far from zero are never preserved."""
        for seed in range(4):
            cert = certify_dataset(real, _independent(real, seed), spec)
            by_name = {t["coefficient"]: t for t in cert["targets"]}
            for name in ("pay_delay_1", "utilization", "log_limit"):
                assert by_name[name]["preserved"] is False, f"{name} survived at seed {seed}"
