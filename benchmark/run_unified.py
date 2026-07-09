"""
Unified REGEN campaign runner — single-process, no subprocess overhead.

Each pass runs all stages in a single Python process. Stages scripts
are called in-process via import rather than subprocess. This eliminates
Python startup + import overhead (~5-10s per pass on large datasets).

Usage:
    python benchmark/run_unified.py
"""

import json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def main():
    data_path = REPO_ROOT / "benchmark" / "creditcard.csv"
    out_dir = REPO_ROOT / "benchmark" / "regen-output"
    out_dir.mkdir(exist_ok=True)
    ingest_path = str(out_dir / "data")

    print("=" * 62)
    print("  REGEN BENCHMARK — Unified Runner")
    print("=" * 62)
    print(f"  Dataset: {data_path}")
    print(f"  Params:  5 passes, 300 rows, max_features=6, coverage=0.70")
    print()

    # ── Imports (once) ──────────────────────────────────────────────────────
    import numpy as np
    import pandas as pd
    from contracts.types import RareEventDef, RareMode, BatchManifest, SchemaGraph
    from engine.ingest.loader import ingest as do_ingest, persist_ingest
    from engine.manifest import build_manifest, seed_rng
    from engine.prior import PriorConfig, fit_prior, generate_base_batch
    from engine.amplifier import AmplifierConfig, fit_correction, sample_correction
    from engine.auditor import AuditorConfig, audit
    from engine.examiner import ExaminerConfig, measure_lift
    from engine.scout import ScoutConfig, select_target

    # ── Ingest ──────────────────────────────────────────────────────────────
    t0 = time.time()
    result = do_ingest(str(data_path), "Class", RareEventDef(mode=RareMode.LABEL, label_value=1))
    persist_ingest(result, ingest_path)
    print(f"  Ingest: {time.time()-t0:.1f}s ({len(result.normal_df)} normal, {len(result.rare_df)} rare)")

    # ── Configs ─────────────────────────────────────────────────────────────
    prior_cfg = PriorConfig(device="cpu", max_train_rows=5000)
    amp_cfg = AmplifierConfig(max_features=6)
    auditor_cfg = AuditorConfig(coverage_threshold=0.70)
    exam_cfg = ExaminerConfig()
    scout_cfg = ScoutConfig()

    # ── Loop ────────────────────────────────────────────────────────────────
    seed = 42
    n_rows = 300
    max_passes = 5
    target_region = {}
    history = []
    best_lift = 0.0

    for pass_num in range(max_passes):
        pass_start = time.time()
        print(f"\n  Pass {pass_num + 1}/{max_passes} ...", end=" ", flush=True)

        # 1. Seed + prior fit
        rng = np.random.default_rng(seed + pass_num)
        prior = fit_prior(result, prior_cfg, rng)

        # 2. Base batch
        base = generate_base_batch(prior, n_rows, target_region, rng)

        # 3. Amplify
        residual = fit_correction(result, prior, amp_cfg)
        rng2 = np.random.default_rng(seed + pass_num)
        _, _, X_res = sample_correction(residual, base.values.astype(np.float64), rng2)
        amp_df = pd.DataFrame(base.values + X_res, columns=base.columns)
        # Set label to rare value
        if result.label_col and result.label_col in amp_df.columns:
            rare_value = result.rare_df[result.label_col].iloc[0]
            amp_df[result.label_col] = rare_value

        # 4. Audit
        manifest = build_manifest(seed + pass_num, SchemaGraph(), {}, target_region, {}, n_rows)
        report = audit(result, amp_df, auditor_cfg, manifest=manifest)

        if not report.overall_passed:
            print(f"REJECTED (coverage={report.coverage_rate:.3f}) — {time.time()-pass_start:.1f}s")
            history.append({"pass": pass_num + 1, "status": "rejected",
                             "coverage": report.coverage_rate})
            # Still run scout for next pass
            target_region = select_target(residual, prior._feature_cols, rng,
                                           scout_cfg, explored_points=[])
            continue

        # 5. Examine
        lift_report = measure_lift(result, amp_df, exam_cfg, manifest=manifest)
        tail_lift = lift_report.tail_lift
        best_lift = max(best_lift, tail_lift)

        print(f"ACCEPTED lift={tail_lift:.4f} (best={best_lift:.4f}) — {time.time()-pass_start:.1f}s")
        history.append({
            "pass": pass_num + 1, "status": "accepted",
            "tail_lift": tail_lift,
            "baseline_recall": lift_report.baseline_recall,
            "amplified_recall": lift_report.amplified_recall,
            "baseline_precision": lift_report.baseline_precision,
            "amplified_precision": lift_report.amplified_precision,
        })

        # Save the best batch
        amp_df.to_parquet(str(out_dir / "data.prior_batch.parquet.amplified.parquet"), index=False)

        # 6. Scout for next target
        target_region = select_target(residual, prior._feature_cols, rng,
                                       scout_cfg, explored_points=[])

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  BENCHMARK RESULTS")
    print("=" * 62)
    accepted = [p for p in history if p["status"] == "accepted"]
    rejected = [p for p in history if p["status"] == "rejected"]
    print(f"  Accepted: {len(accepted)}/{len(history)}")
    print(f"  Rejected: {len(rejected)}/{len(history)}")
    print(f"  Best tail lift: {best_lift:+.4f}")
    print()
    for p in history:
        if p["status"] == "accepted":
            print(f"  Pass {p['pass']}: ✓ ACCEPTED")
            print(f"      recall     {p['baseline_recall']:.3f} → {p['amplified_recall']:.3f}  "
                  f"(lift {p['tail_lift']:+.3f})")
        else:
            print(f"  Pass {p['pass']}: ✗ REJECTED  coverage={p.get('coverage', 0):.3f}")
    print("=" * 62)

    # Save summary
    summary = {"best_lift": best_lift, "passes": history}
    with open(str(out_dir / "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {out_dir / 'benchmark_summary.json'}")


if __name__ == "__main__":
    main()
