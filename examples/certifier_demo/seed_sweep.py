"""
Seed sweep for the v2 estimand-preserving generator — is a single demo run
representative, or does it depend on which seed got picked?

A single run (see run_demo.py) reports CERTIFIED or refused for one seed. That
is not evidence of *reliability* on its own — this script re-runs the same
generator across many seeds and reports, per coefficient, how far the
synthetic estimate lands from the real one on average (bias) and how much
that varies (spread), plus the overall certification rate.

Run from the repo root:  python examples/certifier_demo/seed_sweep.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contracts.scenario import EstimandSpec
from regen.certifier import certify_dataset
from regen.estimand_preserving import generate_estimand_preserving

CSV = __import__("pathlib").Path(__file__).resolve().parent / "credit_default.csv"
PREDICTORS = ["pay_delay_1", "utilization", "log_limit", "age"]
N_SEEDS = 30


def main():
    real = pd.read_csv(CSV)
    spec = EstimandSpec(outcome="default", predictors=PREDICTORS, family="logit")

    deltas = {p: [] for p in PREDICTORS}
    theta_real = {}
    n_certified = 0
    for seed in range(1, N_SEEDS + 1):
        synth = generate_estimand_preserving(real, spec, n_rows=6000, seed=seed)
        cert = certify_dataset(real, synth, spec)
        n_certified += int(cert["certified"])
        for t in cert["targets"]:
            deltas[t["coefficient"]].append(t["theta_synth"] - t["theta_real"])
            theta_real[t["coefficient"]] = t["theta_real"]

    print(f"{N_SEEDS} seeds, logit  default ~ {' + '.join(PREDICTORS)}\n")
    print(f"{'predictor':<15}{'theta_real':>12}{'mean bias':>12}{'std':>10}")
    for p in PREDICTORS:
        d = np.array(deltas[p])
        print(f"{p:<15}{theta_real[p]:>+12.4f}{d.mean():>+12.4f}{d.std():>10.4f}")

    print(f"\n{n_certified}/{N_SEEDS} seeds fully certified "
          f"({n_certified / N_SEEDS:.0%}).")
    print("Bias >> std on a coefficient means it's a systematic distortion, "
          "not seed luck — see docs/KNOWN_ISSUES.md.")


if __name__ == "__main__":
    main()
