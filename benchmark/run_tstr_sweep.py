"""
TSTR sweep (real-vs-synthetic): how well does a REGEN surrogate stand in for real
data, measured leakage-free across the benchmark datasets? Feeds the honest README
(replaces the inflated lift table). Writes RESULTS_TSTR.{json,md} with provenance.

    python benchmark/run_tstr_sweep.py

Uses privacy="none" so the number isolates *surrogate quality* (the δ-floor's cost
is characterized separately in RESULTS_PRIVACY.md). auto-tune OFF for comparability.
"""
import json
import subprocess
import sys
import time
import warnings
import logging
from datetime import date
from pathlib import Path

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from regen.api import evaluate_surrogate  # noqa: E402

DATASETS = [
    ("creditcard_subset.csv", "Class"),   # tiny rare → expect insufficient (honest)
    ("satellite.csv", "Target"),
    ("hypothyroid.csv", "Class"),
    ("wilt.csv", "class"),
    ("ozone.csv", "Class"),
    ("churn.csv", "Class"),
    ("creditcard.csv", "Class"),          # large; kept last
]
SEED = 7


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def main():
    results = []
    for fname, label in DATASETS:
        path = REPO / "benchmark" / "data" / fname
        if not path.exists():
            results.append({"dataset": fname, "error": "missing"})
            print(f"[skip] {fname}"); continue
        t0 = time.time()
        try:
            r = evaluate_surrogate(str(path), label_col=label, privacy="none",
                                   auto=False, seed=SEED, tstr_seeds=(42, 53, 61))
            t = r["tstr"]
            row = {"dataset": fname, "status": t["status"],
                   "n_test_real": r["n_test_real"], "n_test_rare": t["n_real_test_rare"],
                   "recovered_roc_auc": t["recovered_roc_auc_median"],
                   "recovered_pr_auc": t["recovered_pr_auc_median"],
                   "per_model": t.get("per_model", []), "secs": round(time.time() - t0, 1)}
        except Exception as e:
            row = {"dataset": fname, "error": str(e), "secs": round(time.time() - t0, 1)}
        results.append(row)
        print(f"[ok] {fname:22s} status={row.get('status')} "
              f"recoveredROC={row.get('recovered_roc_auc')} "
              f"recoveredPR={row.get('recovered_pr_auc')} "
              f"n_test_rare={row.get('n_test_rare')} {row.get('secs')}s")

    payload = {"run_date": date.today().isoformat(), "code_version": _git_hash(),
               "config": {"seed": SEED, "privacy": "none", "auto": False,
                          "tstr_seeds": [42, 53, 61], "test_size": 0.30},
               "metric": "TSTR recovered = train-on-synthetic / train-on-real, on held-out real",
               "results": results}
    (REPO / "benchmark" / "RESULTS_TSTR.json").write_text(json.dumps(payload, indent=2))
    _write_md(payload)
    print("\nWrote benchmark/RESULTS_TSTR.{json,md}")


def _write_md(payload):
    L = ["# TSTR sweep — how well does a REGEN surrogate stand in for real data?", "",
         f"**Run:** {payload['run_date']} · `{payload['code_version']}` · "
         f"privacy=none, auto-tune off, 3 seeds, 30% held-out real test.", "",
         "`recovered = (model trained on synthetic) / (model trained on real)`, both "
         "scored on the **held-out real** test set. 1.0 = the surrogate stands in fully; "
         "the gap is expected (and, with a healthy privacy min-distance, is the price of "
         "privacy — a perfect match would signal memorization).", "",
         "| Dataset | status | held-out rare | recovered ROC-AUC | recovered PR-AUC |",
         "|---|---|---|---|---|"]
    for r in payload["results"]:
        if r.get("error"):
            L.append(f"| {r['dataset']} | _error/skip_ | | | |"); continue
        L.append(f"| {r['dataset']} | {r['status']} | {r.get('n_test_rare','-')} | "
                 f"{r.get('recovered_roc_auc','-')} | {r.get('recovered_pr_auc','-')} |")
    L += ["", "Notes: `insufficient_real_test` = too few held-out rare rows for a "
          "trustworthy estimate (not a failure — an honest refusal). Numbers are "
          "single-config, indicative; re-run this script to reproduce."]
    (payload and (Path(__file__).resolve().parent.parent / "benchmark" / "RESULTS_TSTR.md")
     ).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
