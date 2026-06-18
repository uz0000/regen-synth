"""
REGEN CLI — entry point and subcommands.

Usage:
    regen run <data> --label <col> [options]
    regen ingest <data> [--label <col>]
    regen test
    regen --version
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Version ─────────────────────────────────────────────────────────────────

__version__ = "0.1.0"


# ── Main entry point ───────────────────────────────────────────────────────

def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.command == "test":
        _cmd_test()
    elif args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "run":
        _cmd_run(args)


# ── Argument parser ────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="regen",
        description="REGEN — statistically grounded synthetic data for rare-event detection.",
    )
    p.add_argument("--version", "-V", action="version", version=f"regen-synth {__version__}",
                    help="Show version and exit")
    p.add_argument("--verbose", "-v", action="store_true",
                    help="Verbose output (info-level logging)")

    sub = p.add_subparsers(dest="command", required=False)

    # ── regen test ──────────────────────────────────────────────────────────
    test_p = sub.add_parser("test", help="Run the REGEN test suite")
    test_p.set_defaults(command="test")

    # ── regen ingest ────────────────────────────────────────────────────────
    ingest_p = sub.add_parser("ingest", help="Inspect a dataset without running a campaign")
    ingest_p.add_argument("data", type=str, help="Path to input data (CSV/JSON/Parquet)")
    ingest_p.add_argument("--label", type=str, default="",
                          help="Label column name (auto-detect if omitted)")
    ingest_p.add_argument("--rare-mode", type=str, default="percentile",
                          choices=["label", "percentile", "imbalance_ratio"],
                          help="How to identify rare events (default: percentile)")
    ingest_p.add_argument("--rare-value", type=str, default=None,
                          help="Value for label mode (e.g. '1' for is_fraud=1)")
    ingest_p.add_argument("--percentile", type=float, default=0.05,
                          help="Percentile threshold (default: 0.05 = bottom 5%%)")
    ingest_p.add_argument("--imbalance-ratio", type=float, default=0.01,
                          help="Imbalance ratio threshold (default: 0.01)")
    ingest_p.set_defaults(command="ingest")

    # ── regen run ───────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run a full REGEN amplification campaign")
    run_p.add_argument("data", type=str, help="Path to input data (CSV/JSON/Parquet)")
    run_p.add_argument("--label", type=str, default="",
                       help="Label column name (auto-detect if omitted)")
    run_p.add_argument("--rare-mode", type=str, default="percentile",
                       choices=["label", "percentile", "imbalance_ratio"],
                       help="How to identify rare events (default: percentile)")
    run_p.add_argument("--rare-value", type=str, default=None,
                       help="Value for label mode (e.g. '1' for is_fraud=1)")
    run_p.add_argument("--percentile", type=float, default=0.05,
                       help="Percentile threshold (default: 0.05 = bottom 5%%)")
    run_p.add_argument("--imbalance-ratio", type=float, default=0.01,
                       help="Imbalance ratio threshold (default: 0.01)")
    run_p.add_argument("--seed", type=int, default=42,
                       help="RNG seed for reproducibility (default: 42)")
    run_p.add_argument("--n-rows", type=int, default=300,
                       help="Synthetic batch size per pass (default: 300)")
    run_p.add_argument("--passes", type=int, default=5,
                       help="Maximum number of amplification passes (default: 5)")
    run_p.add_argument("--out", type=str, default="regen-output",
                       help="Output directory (default: ./regen-output)")
    run_p.add_argument("--coverage-threshold", type=float, default=0.80,
                       help="Auditor coverage threshold (default: 0.80)")
    run_p.add_argument("--gp-noise", type=float, default=0.1,
                       help="GP noise variance (default: 0.1)")
    run_p.add_argument("--json", action="store_true",
                       help="Output campaign summary as JSON")
    run_p.set_defaults(command="run")

    return p


# ── Command: test ──────────────────────────────────────────────────────────

def _cmd_test():
    """Run the REGEN test suite as a quick sanity check."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        cwd=str(_repo_root()),
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)


# ── Command: ingest ────────────────────────────────────────────────────────

def _cmd_ingest(args):
    """Load and inspect a dataset without running a campaign."""
    from engine.ingest.loader import ingest as do_ingest
    from contracts.types import RareEventDef, RareMode

    rare_def = _build_rare_def(args)
    result = do_ingest(
        filepath=args.data,
        label_col=args.label,
        rare_def=rare_def,
    )

    print(f"Dataset:    {args.data}")
    print(f"Label col:  {result.label_col}")
    print(f"Normal rows: {len(result.normal_df)}")
    print(f"Rare rows:   {len(result.rare_df)}")
    print(f"Features:    {len(result.field_dict) - 1}")
    print()
    print("Columns:")
    for name, meta in result.field_dict.items():
        sym = "★" if name == result.label_col else " "
        print(f"  {sym} {name}  ({meta.field_type.value})", end="")
        if meta.cardinality:
            print(f"  [{meta.cardinality} unique]", end="")
        if meta.min_val is not None:
            print(f"  [{meta.min_val:.2f}–{meta.max_val:.2f}]", end="")
        print()


# ── Command: run ───────────────────────────────────────────────────────────

def _cmd_run(args):
    """Run a full REGEN amplification campaign."""
    from engine.ingest.loader import ingest as do_ingest, persist_ingest
    from contracts.types import RareEventDef, RareMode
    import importlib.util
    skill_path = _repo_root() / "agent-runtime" / "skills" / "regen-loop" / "skill.py"
    spec = importlib.util.spec_from_file_location("regen_loop_skill", str(skill_path))
    skill_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skill_mod)
    run_campaign = skill_mod.run_campaign

    # 1. Build the rare-event definition
    rare_def = _build_rare_def(args)

    # 2. Ingest the data
    print(f"[regen] Loading {args.data} ...", file=sys.stderr)
    result = do_ingest(
        filepath=args.data,
        label_col=args.label,
        rare_def=rare_def,
    )
    print(f"[regen] {len(result.normal_df)} normal, {len(result.rare_df)} rare, label='{result.label_col}'",
          file=sys.stderr)

    # 3. Persist the ingest layout for stages
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ingest_path = str(out_dir / "data")
    persist_ingest(result, ingest_path)

    # 4. Run the campaign
    print(f"[regen] Running campaign ({args.passes} passes, {args.n_rows} rows/pass) ...",
          file=sys.stderr)

    summary = run_campaign(
        ingest_path=ingest_path,
        seed=args.seed,
        n_rows=args.n_rows,
        max_passes=args.passes,
        label_col=result.label_col,
        prior_config={"device": "cpu"},
        amplifier_config={"gp_noise_variance": args.gp_noise},
        auditor_config={"coverage_threshold": args.coverage_threshold},
        examiner_config={"n_estimators": 100},
        scout_config={"num_candidates": 100},
    )

    # 5. Report
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_summary(summary, out_dir)


def _print_summary(summary: dict, out_dir: Path):
    """Print a human-readable campaign summary."""
    print()
    print("=" * 62)
    print("  REGEN CAMPAIGN SUMMARY")
    print("=" * 62)

    for p in summary["passes"]:
        if p["status"] == "accepted":
            lift = p.get("tail_lift", 0.0)
            base_r = p.get("baseline_recall", 0.0)
            amp_r  = p.get("amplified_recall", 0.0)
            base_p = p.get("baseline_precision", 0.0)
            amp_p  = p.get("amplified_precision", 0.0)
            print(f"  Pass {p['pass']}: ✓ ACCEPTED")
            print(f"      recall     {base_r:.3f} → {amp_r:.3f}  (lift {lift:+.3f})")
            print(f"      precision  {base_p:.3f} → {amp_p:.3f}")
        else:
            cov = p.get("coverage", 0.0)
            print(f"  Pass {p['pass']}: ✗ REJECTED  coverage={cov:.3f}")

    print()
    print(f"  Best tail lift:  {summary['best_lift']:+.4f}")
    print(f"  Memory:          {summary.get('memory', {}).get('n_explored', 0)} regions explored")
    print(f"  Output:          {out_dir.resolve()}")
    print("=" * 62)

    # Show the best batch path
    best_path = out_dir / "data.prior_batch.parquet.amplified.parquet"
    if best_path.exists():
        print(f"  Accepted synthetic batch: {best_path}")
        import pandas as pd
        df = pd.read_parquet(best_path)
        print(f"    {len(df)} rows, {len(df.columns)} columns")
    print("=" * 62)


# ── Helpers ────────────────────────────────────────────────────────────────

from contracts.types import RareEventDef, RareMode


def _build_rare_def(args) -> RareEventDef:
    """Build a RareEventDef from parsed CLI args."""
    mode = RareMode(args.rare_mode)

    if mode == RareMode.LABEL:
        val = args.rare_value
        if val is None:
            print("Error: --rare-value is required when --rare-mode=label", file=sys.stderr)
            sys.exit(1)
        # Try numeric, fall back to string
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        return RareEventDef(mode=mode, label_value=val)

    if mode == RareMode.PERCENTILE:
        return RareEventDef(mode=mode, percentile=args.percentile)

    if mode == RareMode.IMBALANCE:
        return RareEventDef(mode=mode, imbalance_ratio=args.imbalance_ratio)

    return RareEventDef()


def _repo_root() -> Path:
    """Return the repository root (parent of cli/)."""
    return Path(__file__).parent.parent


if __name__ == "__main__":
    main()
