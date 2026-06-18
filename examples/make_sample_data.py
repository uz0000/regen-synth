"""
Generate a synthetic fraud-like transaction dataset for the REGEN demo.

This is *demo input* — a stand-in for a real tabular dataset. It is NOT
REGEN output; it exists only so the loop has something to ingest. It has a
clear rare class (fraud) that sits in a distinct region of feature space,
so the Auditor's coverage metric and the Examiner's lift number are
meaningful.

Columns:
  amount        continuous   — transaction amount (fraud skews high)
  n_prior_txns  continuous   — account history depth (fraud skews low)
  hour          continuous   — hour of day (fraud skews to odd hours)
  merchant_risk continuous   — merchant risk score (fraud skews high)
  is_fraud      binary       — label (1 = fraud, ~3% of rows)
"""

import argparse

import numpy as np
import pandas as pd


def make_dataset(n: int = 2000, fraud_rate: float = 0.03, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_normal = n - n_fraud

    # Fraud and normal OVERLAP heavily — fraud is only weakly separated. This
    # is deliberate: trivially separable fraud leaves no headroom for synthetic
    # amplification to help. Real fraud hides in the bulk; the tail is subtle.
    normal = pd.DataFrame({
        "amount":        rng.lognormal(3.0, 0.8, n_normal),
        "n_prior_txns":  rng.poisson(40, n_normal).astype(float),
        "hour":          rng.normal(14, 5, n_normal).clip(0, 23),
        "merchant_risk": rng.beta(2, 6, n_normal),
        "is_fraud":      0,
    })
    fraud = pd.DataFrame({
        "amount":        rng.lognormal(3.6, 0.85, n_fraud),  # modestly higher
        "n_prior_txns":  rng.poisson(25, n_fraud).astype(float),  # somewhat thinner
        "hour":          rng.normal(10, 6, n_fraud).clip(0, 23),  # broad, overlapping
        "merchant_risk": rng.beta(3, 5, n_fraud),            # modestly riskier
        "is_fraud":      1,
    })

    df = pd.concat([normal, fraud], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/transactions.csv")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--fraud-rate", type=float, default=0.03)
    args = parser.parse_args()

    df = make_dataset(n=args.n, fraud_rate=args.fraud_rate)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows ({int(df['is_fraud'].sum())} fraud) → {args.out}")
