"""
Breadth benchmark — runs REGEN multi-pass vs SMOTE on all 11 datasets.
Tests the heterogeneity hypothesis.
"""
import json, os, sys, warnings, logging, importlib
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).parent.parent
SEEDS = [42, 43, 44, 45, 46]
N_PASSES = 5
N_ROWS = 200

# Load dataset registry
with open(REPO_ROOT / "benchmark" / "breadth_predictions.json") as f:
    DATASETS = json.load(f)


def download_dataset(ds):
    """Download a dataset from OpenML, save to CSV. Returns csv_path."""
    import openml
    data_dir = REPO_ROOT / "benchmark" / "data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / f"{ds['name']}.csv"
    if csv_path.exists():
        return str(csv_path)

    oid = ds["id"]
    dataset = openml.datasets.get_dataset(oid, download_data=True, download_qualities=False)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    if y is None:
        raise ValueError(f"No target for {ds['name']}")
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    label_col = ds["label"]
    df[label_col] = y

    # Map string labels to binary if needed
    rare_val = ds["rare_value"]
    if isinstance(rare_val, str) and rare_val not in ("0", "1", 0, 1):
        df[label_col] = (df[label_col] == rare_val).astype(int)
        ds["rare_value"] = 1

    df.to_csv(csv_path, index=False)
    print(f"  Downloaded {ds['display']} — {len(df)} rows", flush=True)
    return str(csv_path)


def run_regen_multipass(csv_path, label_col, rare_value, seed):
    """Run 5-pass Scout loop. Returns lift info."""
    from engine.ingest.loader import ingest as do_ingest
    from contracts.types import RareEventDef, RareMode
    from engine.prior import PriorConfig, fit_prior, generate_base_batch
    from engine.amplifier import AmplifierConfig, fit_residuals, sample_residuals
    from engine.auditor import AuditorConfig, audit
    from engine.examiner import ExaminerConfig, measure_lift
    from engine.scout import ScoutConfig, select_target

    rare_def = RareEventDef(mode=RareMode.LABEL, label_value=rare_value)
    result = do_ingest(csv_path, label_col, rare_def)

    prior_cfg = PriorConfig()
    amp_cfg = AmplifierConfig(max_features=0)
    aud_cfg = AuditorConfig(coverage_threshold=0.50)
    exam_cfg = ExaminerConfig()
    scout_cfg = ScoutConfig()

    explored_points = []
    total_accepted = 0
    best_lift = 0.0
    all_lifts = []
    accepted_count = 0
    rejected_count = 0
    pass_details = []
    scout_log = []
    gp_ard_spread = None

    for pn in range(N_PASSES):
        rng = np.random.default_rng(seed + pn)
        prior = fit_prior(result, prior_cfg, rng)
        residual = fit_residuals(result, prior, amp_cfg)

        # Scout: R-EPIG target selection with cross-pass memory.
        # Pass 1: explored_points is empty, Scout picks globally best region.
        # Passes 2+: explored penalty down-weights already-mapped anchors.
        target = select_target(
            residual, prior._feature_cols, rng, scout_cfg,
            explored_points=explored_points or None,
        )
        if target.get("candidate_point"):
            explored_points.append(target["candidate_point"])

        base = generate_base_batch(prior, N_ROWS, target, rng)
        rng2 = np.random.default_rng(seed + pn)
        _, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng2)
        amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
        if label_col in amp_df.columns:
            amp_df[label_col] = rare_value

        report = audit(result, amp_df, aud_cfg)
        if not report.overall_passed:
            rejected_count += 1
            pass_details.append({"pass": pn, "status": "rejected", "coverage": report.coverage_rate})
            scout_log.append({"pass": pn, "accepted": False, "target": str(target.get("feature_name", ""))})
            continue

        lift = measure_lift(result, amp_df, exam_cfg)
        best_lift = max(best_lift, lift.tail_lift)
        total_accepted += N_ROWS
        all_lifts.append(lift.tail_lift)
        accepted_count += 1
        pass_details.append({
            "pass": pn, "status": "accepted",
            "tail_lift": lift.tail_lift,
            "baseline_recall": lift.baseline_recall,
        })

        # Capture ARD spread from first accepted pass
        if gp_ard_spread is None and residual._gp_optimized:
            try:
                ls = residual._gp.kern.lengthscale.values.copy()
                gp_ard_spread = float(ls.std() / (ls.mean() + 1e-8))
            except Exception:
                gp_ard_spread = None

        scout_log.append({"pass": pn, "accepted": True, "target": str(target.get("feature_name", ""))})

    avg_lift = float(np.mean(all_lifts)) if all_lifts else 0.0
    return {
        "best_lift": best_lift,
        "avg_lift": avg_lift,
        "n_accepted": accepted_count,
        "n_rejected": rejected_count,
        "total_accepted_rows": total_accepted,
        "pass_details": pass_details,
        "gp_ard_spread": gp_ard_spread,
    }


def run_smote(normal_df, rare_df, label_col, n_synthetic, seed):
    """SMOTE with matched synthetic budget. Returns lift."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score
    from imblearn.over_sampling import SMOTE

    feat = [c for c in normal_df.columns if c != label_col]

    # Encode categorical columns to numeric (same approach as engine._encode_features)
    def _encode(df):
        out = df[feat].copy()
        for col in out.columns:
            if out[col].dtype == object or str(out[col].dtype) == "category":
                out[col] = pd.Categorical(out[col]).codes.astype(np.float64)
            elif out[col].dtype == bool:
                out[col] = out[col].astype(np.float64)
            else:
                out[col] = out[col].astype(np.float64)
        return out.values

    Xn = _encode(normal_df)
    Xr = _encode(rare_df)
    rng = np.random.RandomState(seed)
    if len(Xn) > 10000:
        Xn = Xn[rng.choice(len(Xn), 10000, replace=False)]

    Xnt, Xnet, _, _ = train_test_split(Xn, np.zeros(len(Xn)), test_size=0.3, random_state=seed)
    Xrt, Xret, _, _ = train_test_split(Xr, np.ones(len(Xr)), test_size=0.3, random_state=seed)

    Xtr = np.vstack([Xnt, Xrt])
    ytr = np.concatenate([np.zeros(len(Xnt)), np.ones(len(Xrt))])
    Xte = np.vstack([Xnet, Xret])
    yte = np.concatenate([np.zeros(len(Xnet)), np.ones(len(Xret))])

    base = RandomForestClassifier(100, class_weight="balanced", random_state=seed)
    base.fit(Xtr, ytr)
    base_r = float(recall_score(yte, base.predict(Xte), zero_division=0))

    rare_c = int((ytr == 1).sum())
    if n_synthetic == 0:
        return {"bl_recall": base_r, "sm_recall": base_r, "lift": 0.0}
    sm = SMOTE(random_state=seed, sampling_strategy={1: rare_c + n_synthetic})
    Xsm, ysm = sm.fit_resample(Xtr, ytr)
    clf = RandomForestClassifier(100, class_weight="balanced", random_state=seed)
    clf.fit(Xsm, ysm)
    sm_r = float(recall_score(yte, clf.predict(Xte), zero_division=0))
    return {"bl_recall": base_r, "sm_recall": sm_r, "lift": sm_r - base_r}


def main():
    print("=" * 70)
    print("  REGEN — Breadth Benchmark (n=11)")
    print("=" * 70)
    print(f"  Seeds: {SEEDS}")
    print(f"  Passes per seed: {N_PASSES}")
    print(f"  Rows per pass: {N_ROWS}")
    print()

    all_results = []

    for ds in DATASETS:
        print(f"\n{'─'*70}")
        print(f"  {ds['display']}  [{ds['regime']}]  pred={ds['prediction']}")
        print(f"{'─'*70}")

        try:
            csv_path = download_dataset(ds)
        except Exception as e:
            print(f"  ❌ Download failed: {e}", flush=True)
            continue

        for seed in SEEDS:
            # REGEN multi-pass
            try:
                regen = run_regen_multipass(csv_path, ds["label"], ds["rare_value"], seed)
            except Exception as e:
                print(f"  Seed {seed}: REGEN ERROR — {e}", flush=True)
                continue

            # SMOTE with matched budget
            from engine.ingest.loader import ingest as do_ingest
            from contracts.types import RareEventDef, RareMode
            rare_def = RareEventDef(mode=RareMode.LABEL, label_value=ds["rare_value"])
            ing = do_ingest(csv_path, ds["label"], rare_def)
            smote = run_smote(ing.normal_df, ing.rare_df, ds["label"],
                               regen["total_accepted_rows"], seed)

            result = {
                "dataset": ds["display"],
                "regime": ds["regime"],
                "prediction": ds["prediction"],
                "seed": seed,
                "n_normal": len(ing.normal_df),
                "n_rare": len(ing.rare_df),
                "n_features": len(ing.field_dict) - 1,
                "regen_best_lift": regen["best_lift"],
                "regen_avg_lift": regen["avg_lift"],
                "regen_accepted": regen["n_accepted"],
                "regen_rejected": regen["n_rejected"],
                "smote_lift": smote["lift"],
                "baseline_recall": smote["bl_recall"],
                "gp_ard_spread": regen["gp_ard_spread"],
            }
            all_results.append(result)

            print(f"  Seed {seed}: REGEN best={regen['best_lift']:+.4f} "
                  f"({regen['n_accepted']}/{N_PASSES} acc), "
                  f"SMOTE lift={smote['lift']:+.4f}", flush=True)

        # Per-dataset summary
        subset = [r for r in all_results if r["dataset"] == ds["display"]]
        r_lifts = [r["regen_best_lift"] for r in subset]
        s_lifts = [r["smote_lift"] for r in subset]
        if r_lifts:
            rm, rs = np.mean(r_lifts), np.std(r_lifts)
            sm, ss = np.mean(s_lifts), np.std(s_lifts)
            wins = rm > sm
            print(f"  → REGEN: {rm:+.4f} ± {rs:.4f} | SMOTE: {sm:+.4f} ± {ss:.4f} | "
                  f"{'✅ REGEN' if wins else '❌ SMOTE'} (pred: {ds['prediction']})", flush=True)

    # ── Master summary ──────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  BREADTH BENCHMARK — MASTER RESULTS")
    print(f"{'='*70}")

    df = pd.DataFrame(all_results)
    summary_rows = []

    for ds in DATASETS:
        name = ds["display"]
        subset = df[df["dataset"] == name]
        if len(subset) == 0:
            continue
        r_l = subset["regen_best_lift"].dropna()
        s_l = subset["smote_lift"].dropna()
        if len(r_l) == 0:
            continue
        rm, rs = r_l.mean(), r_l.std()
        sm, ss = s_l.mean(), s_l.std()
        actual = "REGEN" if rm > sm else "SMOTE"
        pred = ds["prediction"]
        match = "✅" if actual == pred else "❌"
        ratio = f"{rm/sm:.2f}x" if sm != 0 else "N/A"
        n_acc = subset["regen_accepted"].mean()

        summary_rows.append({
            "name": name, "regime": ds["regime"], "pred": pred, "actual": actual,
            "match": match, "regen_m": f"{rm:+.4f}", "regen_s": f"{rs:.4f}",
            "smote_m": f"{sm:+.4f}", "ratio": ratio,
            "acc_passes": f"{n_acc:.1f}/{N_PASSES}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Confusion matrix
    correct = sum(1 for r in summary_rows if r["match"] == "✅")
    total = len(summary_rows)
    print(f"\n  Prediction accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"  REGEN wins: {sum(1 for r in summary_rows if r['actual']=='REGEN')}/{total}")
    print(f"  SMOTE wins: {sum(1 for r in summary_rows if r['actual']=='SMOTE')}/{total}")

    # ARD spread correlation
    print(f"\n  ARD spread correlation coming from full analysis...")

    # Save
    out = REPO_ROOT / "benchmark" / "RESULTS_BREADTH.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
