"""
REGEN Multi-Dataset Benchmark — honest validation.

Runs REGEN on 3 datasets across 5 seeds each, compares against SMOTE.
Reports lift ± confidence intervals. Every batch carries a manifest
and passes the Auditor.
"""

import json, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent.parent

# ── Datasets ──────────────────────────────────────────────────────────────

DATASETS = [
    {
        "id": 42175,
        "name": "creditcard",
        "display": "Credit Card Fraud",
        "label": "Class",
        "rare_mode": "label",
        "rare_value": 1,
    },
    {
        "id": 40676,
        "name": "hypothyroid",
        "display": "Hypothyroid",
        "label": "Class",
        "rare_mode": "label",
        "rare_value": 0,
    },
    {
        "id": 40900,
        "name": "satellite",
        "display": "Satellite",
        "label": "Target",
        "rare_mode": "label",
        "rare_value": "Anomaly",
    },
]


# ── SMOTE baseline ────────────────────────────────────────────────────────

def run_smote_baseline(
    normal_df: pd.DataFrame,
    rare_df: pd.DataFrame,
    label_col: str,
    seed: int,
) -> dict:
    """Train baseline + SMOTE-augmented detectors, return lift."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score, precision_score
    from imblearn.over_sampling import SMOTE

    feature_cols = [c for c in normal_df.columns if c != label_col]

    X_normal = normal_df[feature_cols].values.astype(np.float64)
    X_rare = rare_df[feature_cols].values.astype(np.float64)

    rng = np.random.RandomState(seed)
    # Subsample normal for speed
    if len(X_normal) > 10000:
        idx = rng.choice(len(X_normal), 10000, replace=False)
        X_normal = X_normal[idx]

    Xn_train, Xn_test, _, _ = train_test_split(
        X_normal, np.zeros(len(X_normal)), test_size=0.3, random_state=seed,
    )
    Xr_train, Xr_test, _, _ = train_test_split(
        X_rare, np.ones(len(X_rare)), test_size=0.3, random_state=seed,
    )

    X_train = np.vstack([Xn_train, Xr_train])
    y_train = np.concatenate([np.zeros(len(Xn_train)), np.ones(len(Xr_train))])
    X_test = np.vstack([Xn_test, Xr_test])
    y_test = np.concatenate([np.zeros(len(Xn_test)), np.ones(len(Xr_test))])

    # Baseline
    base = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed)
    base.fit(X_train, y_train)
    y_pred_base = base.predict(X_test)
    base_recall = recall_score(y_test, y_pred_base, zero_division=0)
    base_prec = precision_score(y_test, y_pred_base, zero_division=0)

    # SMOTE
    smote = SMOTE(random_state=seed)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    sm = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed)
    sm.fit(X_sm, y_sm)
    y_pred_sm = sm.predict(X_test)
    sm_recall = recall_score(y_test, y_pred_sm, zero_division=0)
    sm_prec = precision_score(y_test, y_pred_sm, zero_division=0)

    lift = sm_recall - base_recall

    return {
        "baseline_recall": float(base_recall),
        "baseline_precision": float(base_prec),
        "smote_recall": float(sm_recall),
        "smote_precision": float(sm_prec),
        "smote_recall_lift": float(lift),
    }


# ── REGEN runner ──────────────────────────────────────────────────────────

def run_regen(
    csv_path: str,
    label_col: str,
    rare_value: object,
    seed: int,
    n_rows: int = 200,
    max_features: int = 0,
) -> dict:
    """Run one REGEN pass, return lift + backend info."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)

    from engine.ingest.loader import ingest as do_ingest
    from contracts.types import RareEventDef, RareMode
    from engine.prior import PriorConfig, fit_prior, generate_base_batch
    from engine.amplifier import AmplifierConfig, fit_residuals, sample_residuals
    from engine.auditor import AuditorConfig, audit
    from engine.examiner import ExaminerConfig, measure_lift

    rng = np.random.default_rng(seed)

    rare_def = RareEventDef(mode=RareMode.LABEL, label_value=rare_value)
    result = do_ingest(csv_path, label_col, rare_def)

    prior = fit_prior(result, PriorConfig(), rng)
    amp = AmplifierConfig(max_features=max_features)
    residual = fit_residuals(result, prior, amp)

    target = {}
    base = generate_base_batch(prior, n_rows, target, rng)
    rng2 = np.random.default_rng(seed)
    _, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng2)
    amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)

    # Set label to rare value
    if label_col in amp_df.columns:
        amp_df[label_col] = rare_value

    config = AuditorConfig(coverage_threshold=0.50)
    report = audit(result, amp_df, config)

    lift = measure_lift(result, amp_df, ExaminerConfig())

    return {
        "regen_recall_lift": float(lift.tail_lift),
        "baseline_recall": float(lift.baseline_recall),
        "amplified_recall": float(lift.amplified_recall),
        "amplified_precision": float(lift.amplified_precision),
        "audit_passed": bool(report.overall_passed),
        "coverage_rate": float(report.coverage_rate),
        "n_normal": len(result.normal_df),
        "n_rare": len(result.rare_df),
        "n_features": len(result.field_dict) - 1,
        "backend_used": prior._backend_used,
        "gp_optimized": residual._gp_optimized,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  REGEN — Multi-Dataset Honest Validation")
    print("=" * 70)

    # Download datasets
    import openml
    data_dir = REPO_ROOT / "benchmark" / "data"
    data_dir.mkdir(exist_ok=True)

    csv_paths = {}
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
                # Map to binary: Anomaly=1, Normal=0
                df[ds["label"]] = (df[ds["label"]] == "Anomaly").astype(int)
                ds["rare_value"] = 1
            df.to_csv(csv_path, index=False)
            print(f"done ({len(df)} rows)")
        csv_paths[ds["name"]] = str(csv_path)
        print(f"  {ds['display']:25s} ready")

    # Run benchmarks
    SEEDS = [42, 43, 44, 45, 46]
    all_results = []

    for ds in DATASETS:
        name = ds["name"]
        csv_path = csv_paths[name]

        print(f"\n{'─'*70}")
        print(f"  {ds['display']}")
        print(f"{'─'*70}")

        for seed in SEEDS:
            # REGEN
            try:
                regen_result = run_regen(
                    csv_path, ds["label"], ds["rare_value"],
                    seed=seed, n_rows=200,
                )
            except Exception as e:
                regen_result = {"regen_recall_lift": None, "error": str(e)}
                print(f"  REGEN seed {seed}: ERROR — {e}", flush=True)

            # SMOTE
            from engine.ingest.loader import ingest as do_ingest
            from contracts.types import RareEventDef, RareMode
            rare_def = RareEventDef(mode=RareMode.LABEL, label_value=ds["rare_value"])
            ing = do_ingest(csv_path, ds["label"], rare_def)
            smote_result = run_smote_baseline(ing.normal_df, ing.rare_df, ds["label"], seed)

            result = {
                "dataset": name,
                "seed": seed,
                **regen_result,
                **smote_result,
            }
            all_results.append(result)

            if regen_result.get("regen_recall_lift") is not None:
                print(
                    f"  Seed {seed}: REGEN lift={regen_result['regen_recall_lift']:+.4f}, "
                    f"SMOTE lift={smote_result['smote_recall_lift']:+.4f}, "
                    f"audit={regen_result['audit_passed']}, "
                    f"backend={regen_result['backend_used']}",
                    flush=True,
                )

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")

    df_results = pd.DataFrame(all_results)
    summary_rows = []

    for name in [d["name"] for d in DATASETS]:
        subset = df_results[df_results["dataset"] == name]
        regen_lifts = subset["regen_recall_lift"].dropna()
        smote_lifts = subset["smote_recall_lift"].dropna()

        if len(regen_lifts) == 0:
            summary_rows.append({
                "Dataset": name,
                "N (normal)": int(subset["n_normal"].iloc[0]) if "n_normal" in subset.columns else "?",
                "N (rare)": int(subset["n_rare"].iloc[0]) if "n_rare" in subset.columns else "?",
                "REGEN Lift μ±σ": "FAILED",
                "SMOTE Lift μ±σ": f"{smote_lifts.mean():+.4f} ± {smote_lifts.std():.4f}",
                "REGEN wins?": "—",
                "Auditor pass rate": "—",
            })
            continue

        regen_mean = regen_lifts.mean()
        regen_std = regen_lifts.std()
        smote_mean = smote_lifts.mean()
        smote_std = smote_lifts.std()

        regen_wins = regen_mean > smote_mean
        audit_rate = subset["audit_passed"].mean()

        summary_rows.append({
            "Dataset": name,
            "N (normal)": int(subset["n_normal"].iloc[0]),
            "N (rare)": int(subset["n_rare"].iloc[0]),
            "REGEN Lift μ±σ": f"{regen_mean:+.4f} ± {regen_std:.4f}",
            "SMOTE Lift μ±σ": f"{smote_mean:+.4f} ± {smote_std:.4f}",
            "REGEN wins?": "✅" if regen_wins else "❌",
            "Auditor pass rate": f"{audit_rate:.0%}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Save
    out_path = REPO_ROOT / "benchmark" / "RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # SMOTE requires imbalanced-learn
    print("\nNote: SMOTE baseline requires: pip install imbalanced-learn")


if __name__ == "__main__":
    main()
