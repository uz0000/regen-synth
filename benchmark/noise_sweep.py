"""
Sweep Prior noise_scale across multiple values, datasets, and seeds.
Find the value that maximizes detection lift.
"""
import json, sys, warnings, logging
import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

sys.path.insert(0, ".")
from regen.api import run_campaign
from contracts.types import RareEventDef, RareMode

DATASETS = [
    {"name": "hypothyroid", "label": "Class", "rare_value": 0},
    {"name": "satellite",   "label": "Class", "rare_value": 1},
    {"name": "churn",       "label": "Churn", "rare_value": 1},
    {"name": "ozone",       "label": "Class", "rare_value": 1},
]

NOISE_SCALES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
SEEDS = [42, 43]

results = []

for ds in DATASETS:
    path = f"benchmark/data/{ds['name']}.csv"
    for noise in NOISE_SCALES:
        lifts = []
        n_acc = 0
        for seed in SEEDS:
            try:
                cr = run_campaign(
                    path,
                    label_col=ds["label"],
                    rare_def=RareEventDef(mode=RareMode.LABEL, label_value=ds["rare_value"]),
                    seed=seed,
                    n_rows=200,
                    max_passes=5,
                    coverage_threshold=0.50,
                    noise_scale=noise,
                )
                lifts.append(cr.best_lift)
                n_acc += cr.n_accepted
            except Exception as e:
                lifts.append(0.0)
                print(f"  ERROR {ds['name']} noise={noise} seed={seed}: {e}")

        mean_lift = float(np.mean(lifts))
        results.append({
            "dataset": ds["name"],
            "noise_scale": noise,
            "mean_lift": round(mean_lift, 4),
            "lifts": [round(l, 4) for l in lifts],
            "n_accepted": n_acc,
        })
        print(f"{ds['name']:<15} noise={noise:.2f}  mean_lift={mean_lift:.4f}  accepted={n_acc}/{len(SEEDS)*5}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY — mean lift by noise_scale (averaged across datasets)")
print("=" * 60)
for noise in NOISE_SCALES:
    entries = [r for r in results if r["noise_scale"] == noise]
    avg = np.mean([r["mean_lift"] for r in entries])
    total_acc = sum(r["n_accepted"] for r in entries)
    total_possible = len(DATASETS) * len(SEEDS) * 5
    print(f"  noise={noise:.2f}  avg_lift={avg:.4f}  accepted={total_acc}/{total_possible}")

# Best per dataset
print("\nBEST noise_scale per dataset:")
for ds in DATASETS:
    entries = [r for r in results if r["dataset"] == ds["name"]]
    best = max(entries, key=lambda r: r["mean_lift"])
    print(f"  {ds['name']:<15} best_noise={best['noise_scale']:.2f}  lift={best['mean_lift']:.4f}")

# Save
with open("benchmark/noise_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to benchmark/noise_sweep_results.json")
