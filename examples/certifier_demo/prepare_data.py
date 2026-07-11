"""
Provenance for credit_default.csv — how the committed demo dataset was derived.

Source: UCI "Default of Credit Card Clients" (Yeh & Lien, 2009), 30,000 Taiwanese
credit-card accounts. Public dataset:
  https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls

We keep the real target (default payment next month) and derive a small,
interpretable analyst table of well-scaled risk factors. The committed CSV is the
output of this script; re-run it only to regenerate from the raw source.

Requires: pandas, numpy, xlrd (xls reader — demo-only, NOT a REGEN runtime dep).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
           "default%20of%20credit%20card%20clients.xls")


def build(raw_xls: str, out_csv: str) -> None:
    # The .xls has a two-row header: row 1 is a group label (X1..X23, Y), row 2 is
    # the real column names. header=1 takes the real names.
    df = pd.read_excel(raw_xls, header=1).rename(
        columns={"default payment next month": "default", "PAY_0": "PAY_1"})
    out = pd.DataFrame({
        "default": df["default"].astype(int),                       # 1 = defaulted next month
        "pay_delay_1": df["PAY_1"].astype(int),                     # most recent repayment status
        "pay_delay_2": df["PAY_2"].astype(int),                     # prior month
        "log_limit": np.log(df["LIMIT_BAL"].clip(lower=1)).round(4),  # credit limit (log, well-scaled)
        "utilization": (df["BILL_AMT1"] / df["LIMIT_BAL"].clip(lower=1)).clip(-1, 3).round(4),
        "age": df["AGE"].astype(int),
        "sex": df["SEX"].astype(int),
        "education": df["EDUCATION"].astype(int),
    })
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  shape={out.shape}  default_rate={out['default'].mean():.4f}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    raw = sys.argv[1] if len(sys.argv) > 1 else None
    if raw is None:
        print("Usage: python prepare_data.py <path-to-raw.xls>")
        print(f"Download the raw source first:\n  curl -L -o raw.xls '{RAW_URL}'")
        sys.exit(1)
    build(raw, str(here / "credit_default.csv"))
