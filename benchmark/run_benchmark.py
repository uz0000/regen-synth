"""
REGEN Benchmark — Credit Card Fraud Detection.

Runs a REGEN campaign on the Kaggle Credit Card Fraud dataset
(284,807 transactions, 492 fraud, 30 features) and reports results.

Usage:
    python benchmark/run_benchmark.py

Requires: openml (pip install openml) to download the dataset.
"""

import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def main():
    print("=" * 62)
    print("  REGEN BENCHMARK — Credit Card Fraud Detection")
    print("=" * 62)

    # 1. Download dataset if not present
    data_path = REPO_ROOT / "benchmark" / "creditcard.csv"
    if not data_path.exists():
        print("\n[benchmark] Downloading credit card fraud dataset from OpenML...")
        import openml
        dataset = openml.datasets.get_dataset(42175, download_data=True)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        import pandas as pd
        df = X.copy()
        df["Class"] = y
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(data_path), index=False)
        print(f"[benchmark] Saved {len(df):,} rows, fraud={y.sum()} ({y.sum()/len(y)*100:.4f}%)")
    else:
        import pandas as pd
        df = pd.read_csv(str(data_path))
        fraud = df["Class"].sum()
        print(f"\n[benchmark] Dataset: {len(df):,} rows, fraud={fraud} ({fraud/len(df)*100:.4f}%)")

    # 2. Run the campaign
    out_dir = REPO_ROOT / "benchmark" / "regen-output"
    out_dir.mkdir(exist_ok=True)

    print("\n[benchmark] Starting REGEN campaign...")
    print(f"[benchmark] Params: passes=5, batch=300, max_features=10, coverage=0.70\n")

    t0 = time.time()

    result = subprocess.run(
        [
            sys.executable, "-m", "cli.main", "run",
            str(data_path),
            "--label", "Class",
            "--rare-mode", "label",
            "--rare-value", "1",
            "--passes", "5",
            "--n-rows", "300",
            "--out", str(out_dir),
            "--coverage-threshold", "0.70",
            "--max-features", "10",
            "--seed", "42",
            "--json",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True,
        timeout=900,  # 15 minutes max
    )

    elapsed = time.time() - t0

    # 3. Report
    print()
    print("=" * 62)
    print("  BENCHMARK RESULTS")
    print("=" * 62)

    if result.returncode != 0:
        print(f"  Campaign failed (exit code {result.returncode})")
        print(f"  Stderr: {result.stderr[:2000]}")
        print("=" * 62)
        return

    # Parse the JSON output (last JSON block in stdout)
    lines = result.stdout.strip().split("\n")
    json_start = next((i for i, l in enumerate(lines) if l.startswith("{")), -1)
    if json_start >= 0:
        summary = json.loads("\n".join(lines[json_start:]))
    else:
        print("  Could not parse JSON output")
        print(result.stdout[:2000])
        print("=" * 62)
        return

    accepted = [p for p in summary["passes"] if p["status"] == "accepted"]
    rejected = [p for p in summary["passes"] if p["status"] == "rejected"]

    print(f"  Total time:         {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Passes accepted:    {len(accepted)}/{len(summary['passes'])}")
    print(f"  Passes rejected:    {len(rejected)}")
    print(f"  Best tail lift:     {summary['best_lift']:+.4f}")
    print()

    for p in summary["passes"]:
        if p["status"] == "accepted":
            print(f"  Pass {p['pass']}: ✓ ACCEPTED")
            print(f"      recall     {p['baseline_recall']:.3f} → {p['amplified_recall']:.3f}  "
                  f"(lift {p['tail_lift']:+.3f})")
            print(f"      precision  {p.get('baseline_precision', 0):.3f} → "
                  f"{p.get('amplified_precision', 0):.3f}")
        else:
            print(f"  Pass {p['pass']}: ✗ REJECTED  coverage={p.get('coverage', 0):.4f}")

    print()
    print(f"  Best lift:          {summary['best_lift']:+.4f}")
    print(f"  Regions explored:   {summary.get('memory', {}).get('n_explored', 0)}")
    print(f"  Output directory:   {out_dir}")
    print(f"  Elapsed:            {elapsed:.0f}s")

    # Write summary to file
    summary_path = out_dir / "benchmark_summary.json"
    summary_data = {
        "dataset": "creditcard.csv",
        "elapsed_seconds": elapsed,
        "best_tail_lift": summary["best_lift"],
        "passes": summary["passes"],
        "memory": summary.get("memory"),
    }
    with open(str(summary_path), "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Summary saved:      {summary_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
