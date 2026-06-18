"""
End-to-end REGEN demo.

Runs the full active-learning loop on the sample fraud dataset and prints
the per-pass tail-lift history. This is the runnable proof that the loop
closes: ingest → (Scout → Prior → Amplifier → Auditor → Examiner) × N.

Usage:
    python examples/make_sample_data.py        # writes examples/transactions.csv
    python examples/run_demo.py                # runs the campaign

Everything here is deterministic given the seed. The only thing the control
plane decides is *which region to amplify next*; every value is engine output.
"""

import logging
import sys
from pathlib import Path

# Make the repo root importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.types import RareEventDef, RareMode  # noqa: E402
from engine.ingest import ingest, persist_ingest     # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def main():
    csv_path = "examples/transactions.csv"
    if not Path(csv_path).exists():
        print(f"Missing {csv_path}. Run: python examples/make_sample_data.py")
        sys.exit(1)

    # ── Ingest: isolate the fraud rows, persist the on-disk layout ────────────
    result = ingest(
        filepath=csv_path,
        label_col="is_fraud",
        rare_def=RareEventDef(mode=RareMode.LABEL, label_value=1),
    )
    print(
        f"Ingested: {len(result.normal_df)} normal, {len(result.rare_df)} rare, "
        f"label='{result.label_col}'"
    )

    work_dir = Path("examples/_run")
    work_dir.mkdir(exist_ok=True)
    ingest_path = str(work_dir / "txns")
    persist_ingest(result, ingest_path)

    # ── Run the campaign via the the agent runtime loop skill ────────────────────────────
    # The skill lives in a hyphenated dir (regen-loop) so load it by path.
    import importlib.util
    skill_path = Path(__file__).parent.parent / "agent-runtime" / "skills" / "regen-loop" / "skill.py"
    spec = importlib.util.spec_from_file_location("regen_loop_skill", skill_path)
    skill = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skill)
    run_campaign = skill.run_campaign

    print("\nRunning REGEN active-learning campaign...\n")
    summary = run_campaign(
        ingest_path=ingest_path,
        seed=42,
        n_rows=300,
        max_passes=3,
        label_col=result.label_col,
        prior_config={"device": "cpu"},
        amplifier_config={"gp_noise_variance": 0.1},
        auditor_config={"coverage_threshold": 0.60},
        examiner_config={"n_estimators": 80},
        scout_config={"num_candidates": 80},
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CAMPAIGN SUMMARY")
    print("=" * 60)
    for p in summary["passes"]:
        if p["status"] == "accepted":
            print(
                f"  Pass {p['pass']}: ACCEPTED\n"
                f"      recall    {p['baseline_recall']:.3f} → {p['amplified_recall']:.3f}  "
                f"(lift {p['tail_lift']:+.3f})\n"
                f"      precision {p['baseline_precision']:.3f} → {p['amplified_precision']:.3f}"
            )
        else:
            print(f"  Pass {p['pass']}: REJECTED  coverage={p['coverage']:.3f}")
    print(f"\n  Best tail lift across campaign: {summary['best_lift']:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
