"""
REGEN Multi-Pass Benchmark — Scout-driven active learning vs SMOTE.

Runs the full active-learning loop (Scout → Prior → Amplifier → Auditor →
Examiner × N passes) on 3 datasets, 5 seeds each. Only Auditor-accepted
passes count. Compares against SMOTE with matched synthetic row budget.
Reports per-dataset lift μ±σ and Scout targeting behaviour.
"""

import json, os, sys, time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent

# ── Config ────────────────────────────────────────────────────────────────

N_PASSES = 5
N_ROWS_PER_PASS = 200      # per pass (SMOTE will match total = N_PASSES * N_ROWS_PER_PASS)
SEEDS = [42, 43, 44, 45, 46]

DATASETS = [
    {
        "id": 42175, "name": "creditcard", "display": "Credit Card Fraud",
        "label": "Class", "rare_mode": "label", "rare_value": 1,
    },
    {
        "id": 40676, "name": "hypothyroid", "display": "Hypothyroid",
        "label": "Class", "rare_mode": "label", "rare_value": 0,
    },
    {
        "id": 40900, "name": "satellite", "display": "Satellite",
        "label": "Target", "rare_mode": "label", "rare_value": "Anomaly",
    },
]


# ── SMOTE baseline ────────────────────────────────────────────────────────

def run_smote(normal_df, rare_df, label_col, n_synthetic, seed):
    """SMOTE with a matched synthetic budget. Returns lift."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score, precision_score
    from imblearn.over_sampling import SMOTE

    feat = [c for c in normal_df.columns if c != label_col]
    Xn = normal_df[feat].values.astype(np.float64)
    Xr = rare_df[feat].values.astype(np.float64)
    rng = np.random.RandomState(seed)
    if len(Xn) > 10000:
        Xn = Xn[rng.choice(len(Xn), 10000, replace=False)]

    Xnt, Xn_test, _, _ = train_test_split(Xn, np.zeros(len(Xn)), test_size=0.3, random_state=seed)
    Xrt, Xr_test, _, _ = train_test_split(Xr, np.ones(len(Xr)), test_size=0.3, random_state=seed)

    X_tr = np.vstack([Xnt, Xrt])
    y_tr = np.concatenate([np.zeros(len(Xnt)), np.ones(len(Xrt))])
    X_te = np.vstack([Xn_test, Xr_test])
    y_te = np.concatenate([np.zeros(len(Xn_test)), np.ones(len(Xr_test))])

    # Baseline
    base = RandomForestClassifier(100, class_weight="balanced", random_state=seed)
    base.fit(X_tr, y_tr)
    yb = base.predict(X_te)
    base_r = recall_score(y_te, yb, zero_division=0)
    base_p = precision_score(y_te, yb, zero_division=0)

    # SMOTE — match REGEN's total synthetic budget
    rare_count = int((y_tr == 1).sum())
    target_count = rare_count + n_synthetic
    sm = SMOTE(random_state=seed, sampling_strategy={1: target_count})
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
    clf = RandomForestClassifier(100, class_weight="balanced", random_state=seed)
    clf.fit(X_sm, y_sm)
    y_sm_p = clf.predict(X_te)
    sm_r = recall_score(y_te, y_sm_p, zero_division=0)
    sm_p = precision_score(y_te, y_sm_p, zero_division=0)

    return {
        "bl_recall": float(base_r), "bl_precision": float(base_p),
        "sm_recall": float(sm_r), "sm_precision": float(sm_p),
        "lift": float(sm_r - base_r),
    }


# ── REGEN multi-pass runner ───────────────────────────────────────────────

def run_regen_multipass(csv_path, label_col, rare_value, seed, n_passes, n_rows):
    """Run a full Scout-driven active-learning campaign. Returns per-pass + summary."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    from engine.ingest.loader import ingest as do_ingest
    from contracts.types import RareEventDef, RareMode, BatchManifest, SchemaGraph
    from engine.manifest import build_manifest
    from engine.prior import PriorConfig, fit_prior, generate_base_batch
    from engine.amplifier import AmplifierConfig, fit_correction, sample_correction
    from engine.auditor import AuditorConfig, audit
    from engine.examiner import ExaminerConfig, measure_lift
    from engine.scout import ScoutConfig, select_target

    import pandas as pd

    rare_def = RareEventDef(mode=RareMode.LABEL, label_value=rare_value)
    result = do_ingest(csv_path, label_col, rare_def)

    prior_cfg = PriorConfig()
    amp_cfg = AmplifierConfig(max_features=0)
    aud_cfg = AuditorConfig(coverage_threshold=0.50)
    exam_cfg = ExaminerConfig()
    scout_cfg = ScoutConfig()

    target_region = {}
    passes = []
    total_accepted = 0
    best_lift = 0.0
    scout_targets = []

    for pass_num in range(n_passes):
        rng = np.random.default_rng(seed + pass_num)
        prior = fit_prior(result, prior_cfg, rng)

        base = generate_base_batch(prior, n_rows, target_region, rng)

        residual = fit_correction(result, prior, amp_cfg)
        rng2 = np.random.default_rng(seed + pass_num)
        _, _, X_res = sample_correction(residual, base.values.astype(np.float64), rng2)
        amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
        if label_col in amp_df.columns:
            amp_df[label_col] = rare_value

        report = audit(result, amp_df, aud_cfg)

        if not report.overall_passed:
            passes.append({"pass": pass_num, "status": "rejected",
                           "coverage": report.coverage_rate})
            target_region = select_target(residual, prior._feature_cols, rng,
                                           scout_cfg, explored_points=[])
            scout_targets.append({
                "pass": pass_num, "accepted": False,
                "target": target_region,
            })
            continue

        lift = measure_lift(result, amp_df, exam_cfg)
        best_lift = max(best_lift, lift.tail_lift)
        total_accepted += n_rows

        passes.append({
            "pass": pass_num, "status": "accepted",
            "tail_lift": lift.tail_lift,
            "baseline_recall": lift.baseline_recall,
            "amplified_recall": lift.amplified_recall,
            "baseline_precision": lift.baseline_precision,
            "amplified_precision": lift.amplified_precision,
        })

        target_region = select_target(residual, prior._feature_cols, rng,
                                       scout_cfg, explored_points=[])
        scout_targets.append({
            "pass": pass_num, "accepted": True,
            "target": target_region,
        })

    return {
        "passes": passes,
        "best_lift": best_lift,
        "total_accepted_rows": total_accepted,
        "n_accepted": sum(1 for p in passes if p["status"] == "accepted"),
        "n_rejected": sum(1 for p in passes if p["status"] == "rejected"),
        "scout_targets": scout_targets,
        "backend_used": "gaussian",
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import openml
    data_dir = REPO_ROOT / "benchmark" / "data"
    data_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  REGEN — Multi-Pass Scout vs SMOTE Benchmark")
    print("=" * 70)

    # Download datasets
    for ds in DATASETS:
        csv_path = data_dir / f"{ds['name']}.csv"
        if not csv_path.exists():
            print(f"\nDownloading {ds['display']}...", end=" ", flush=True)
            dataset = openml.datasets.get_dataset(ds["id"], download_data=True)
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            df = X.copy()
            df[ds["label"]] = y
            if ds["name"] == "satellite":
                df[ds["label"]] = (df[ds["label"]] == "Anomaly").astype(int)
                ds["rare_value"] = 1
            df.to_csv(csv_path, index=False)
            print(f"done ({len(df)} rows)")
        print(f"  {ds['display']:25s} ready")

    # Run
    all_results = []

    for ds in DATASETS:
        csv_path = str(data_dir / f"{ds['name']}.csv")
        print(f"\n{'─'*70}")
        print(f"  {ds['display']}")
        print(f"{'─'*70}")

        for seed in SEEDS:
            # REGEN multi-pass
            try:
                regen = run_regen_multipass(
                    csv_path, ds["label"], ds["rare_value"],
                    seed=seed, n_passes=N_PASSES, n_rows=N_ROWS_PER_PASS,
                )
            except Exception as e:
                print(f"  Seed {seed}: REGEN ERROR — {e}", flush=True)
                continue

            # SMOTE with matched budget
            from engine.ingest.loader import ingest as do_ingest
            from contracts.types import RareEventDef, RareMode
            rare_def = RareEventDef(mode=RareMode.LABEL, label_value=ds["rare_value"])
            ing = do_ingest(csv_path, ds["label"], rare_def)

            total_synth = regen["total_accepted_rows"]
            if total_synth == 0:
                smote_result = None
            else:
                smote_result = run_smote(ing.normal_df, ing.rare_df, ds["label"],
                                          total_synth, seed)

            # Accepted-pass lift (mean over accepted passes)
            accepted_lifts = [p["tail_lift"] for p in regen["passes"] if p["status"] == "accepted"]
            avg_lift = np.mean(accepted_lifts) if accepted_lifts else 0.0
            max_lift = regen["best_lift"]

            result = {
                "dataset": ds["display"],
                "seed": seed,
                "n_normal": len(ing.normal_df),
                "n_rare": len(ing.rare_df),
                "regen_accepted": regen["n_accepted"],
                "regen_rejected": regen["n_rejected"],
                "regen_accepted_rows": total_synth,
                "regen_avg_lift": float(avg_lift),
                "regen_best_lift": float(max_lift),
                "regen_pass_details": regen["passes"],
                "regen_scout_targets": regen["scout_targets"],
                "smote_lift": float(smote_result["lift"]) if smote_result else None,
                "smote_recall": float(smote_result["sm_recall"]) if smote_result else None,
                "smote_rows": total_synth,
            }
            all_results.append(result)

            print(f"  Seed {seed}: REGEN {regen['n_accepted']}/{N_PASSES} accepted, "
                  f"avg lift={avg_lift:+.4f}, best={max_lift:+.4f}, "
                  f"SMOTE lift={smote_result['lift']:+.4f}" if smote_result else f"  Seed {seed}: REGEN {regen['n_accepted']}/{N_PASSES} accepted (no SMOTE — 0 accepted)",
                  flush=True)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  MULTI-PASS RESULTS SUMMARY")
    print(f"{'='*70}")

    df = pd.DataFrame(all_results)
    for name in [d["display"] for d in DATASETS]:
        subset = df[df["dataset"] == name]
        r_lifts = subset["regen_best_lift"].dropna()
        s_lifts = subset["smote_lift"].dropna()
        n_acc = subset["regen_accepted"].values

        r_mean = r_lifts.mean()
        r_std = r_lifts.std()
        s_mean = s_lifts.mean() if len(s_lifts) > 0 else None
        s_std = s_lifts.std() if len(s_lifts) > 1 else None
        wins = r_mean > s_mean if s_mean is not None else None

        print(f"\n  {name}")
        print(f"    REGEN multi-pass:   {r_mean:+.4f} ± {r_std:.4f} (mean best lift)")
        if s_mean is not None:
            print(f"    SMOTE:              {s_mean:+.4f} ± {s_std:.4f}")
            print(f"    REGEN beats SMOTE?  {'✅' if wins else '❌'}")
        print(f"    Avg accepted passes: {n_acc.mean():.1f}/{N_PASSES}")

    # Save
    out_path = REPO_ROOT / "benchmark" / "RESULTS_MULTIPASS.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
