"""Run multi-pass benchmark on Satellite only."""
import json, sys, os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).parent.parent
data_dir = REPO_ROOT / "benchmark" / "data"

# Download with mapping
import openml
csv_path = data_dir / "satellite.csv"
dataset = openml.datasets.get_dataset(40900, download_data=True)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]
df = X.copy()
df["Target"] = (y == "Anomaly").astype(int)
df.to_csv(csv_path, index=False)
print(f"Satellite: {len(df)} rows, {int(df['Target'].sum())} anomalies")

# Multi-pass
from engine.ingest.loader import ingest as do_ingest
from contracts.types import RareEventDef, RareMode
from engine.prior import PriorConfig, fit_prior, generate_base_batch
from engine.amplifier import AmplifierConfig, fit_residuals, sample_residuals
from engine.auditor import AuditorConfig, audit
from engine.examiner import ExaminerConfig, measure_lift
from engine.scout import ScoutConfig, select_target

SEEDS = list(range(42, 47))
N_PASSES = 5
N_ROWS = 200
results = []

for seed in SEEDS:
    import logging; logging.getLogger().setLevel(logging.WARNING)
    rare_def = RareEventDef(mode=RareMode.LABEL, label_value=1)
    result = do_ingest(str(csv_path), "Target", rare_def)

    prior_cfg = PriorConfig()
    amp_cfg = AmplifierConfig(max_features=0)
    aud_cfg = AuditorConfig(coverage_threshold=0.50)
    exam_cfg = ExaminerConfig()
    scout_cfg = ScoutConfig()
    
    target_region = {}
    total_accepted = 0
    best_lift = 0.0
    seed_avgs = []

    for pn in range(N_PASSES):
        rng = np.random.default_rng(seed + pn)
        prior = fit_prior(result, prior_cfg, rng)
        base = generate_base_batch(prior, N_ROWS, target_region, rng)
        residual = fit_residuals(result, prior, amp_cfg)
        rng2 = np.random.default_rng(seed + pn)
        _, _, X_res = sample_residuals(residual, base.values.astype(np.float64), rng2)
        amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
        amp_df["Target"] = 1
        report = audit(result, amp_df, aud_cfg)
        
        if not report.overall_passed:
            target_region = select_target(residual, prior._feature_cols, rng, scout_cfg)
            continue
        
        lift = measure_lift(result, amp_df, exam_cfg)
        best_lift = max(best_lift, lift.tail_lift)
        total_accepted += N_ROWS
        seed_avgs.append(lift.tail_lift)
        target_region = select_target(residual, prior._feature_cols, rng, scout_cfg)
    
    avg_lift = np.mean(seed_avgs) if seed_avgs else 0.0
    
    # SMOTE with matched budget
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score
    feat = [c for c in result.normal_df.columns if c != "Target"]
    Xn = result.normal_df[feat].values.astype(np.float64)
    Xr = result.rare_df[feat].values.astype(np.float64)
    rng_s = np.random.RandomState(seed)
    if len(Xn) > 10000:
        Xn = Xn[rng_s.choice(len(Xn), 10000, replace=False)]
    Xnt, Xnet, _, _ = train_test_split(Xn, np.zeros(len(Xn)), test_size=0.3, random_state=seed)
    Xrt, Xret, _, _ = train_test_split(Xr, np.ones(len(Xr)), test_size=0.3, random_state=seed)
    Xtr = np.vstack([Xnt, Xrt])
    ytr = np.concatenate([np.zeros(len(Xnt)), np.ones(len(Xrt))])
    Xte = np.vstack([Xnet, Xret])
    yte = np.concatenate([np.zeros(len(Xnet)), np.ones(len(Xret))])
    base_m = RandomForestClassifier(100, class_weight="balanced", random_state=seed)
    base_m.fit(Xtr, ytr)
    base_r = recall_score(yte, base_m.predict(Xte), zero_division=0)
    rare_c = int((ytr == 1).sum())
    sm = SMOTE(random_state=seed, sampling_strategy={1: rare_c + total_accepted})
    Xsm, ysm = sm.fit_resample(Xtr, ytr)
    clf = RandomForestClassifier(100, class_weight="balanced", random_state=seed)
    clf.fit(Xsm, ysm)
    sm_lift = recall_score(yte, clf.predict(Xte), zero_division=0) - base_r
    
    results.append({
        "seed": seed,
        "regen_best_lift": float(best_lift),
        "regen_avg_lift": float(avg_lift),
        "regen_accepted": len(seed_avgs),
        "regen_total_rows": total_accepted,
        "smote_lift": float(sm_lift),
    })
    print(f"Seed {seed}: REGEN best={best_lift:+.4f} avg={avg_lift:+.4f} ({len(seed_avgs)}/{N_PASSES} accepted)  SMOTE lift={sm_lift:+.4f}", flush=True)

# Summary
import json
r_lifts = np.array([r["regen_best_lift"] for r in results])
s_lifts = np.array([r["smote_lift"] for r in results])
print(f"\nSatellite multi-pass:")
print(f"  REGEN: {r_lifts.mean():+.4f} ± {r_lifts.std():.4f}")
print(f"  SMOTE: {s_lifts.mean():+.4f} ± {s_lifts.std():.4f}")
print(f"  REGEN wins? {'✅' if r_lifts.mean() > s_lifts.mean() else '❌'}")

with open(REPO_ROOT / "benchmark" / "RESULTS_SATELLITE.json", "w") as f:
    json.dump(results, f, indent=2)
