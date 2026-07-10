"""
REGEN CLI — thin wrappers around regen.api.

Usage:
    regen run <data> --label <col> [options]
    regen ingest <data> [--label <col>]
    regen screen <data> [--label <col>] [options]
    regen test
    regen --version

Every command delegates its logic to regen.api.*. The CLI only parses
flags, calls the API, and formats output.
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
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "explore":
        _cmd_explore(args)
    elif args.command == "propose":
        _cmd_propose(args)
    elif args.command == "doctor":
        _cmd_doctor(args)
    elif args.command == "verify":
        _cmd_verify(args)
    elif args.command == "screen":
        _cmd_screen(args)


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
    run_p.add_argument("--max-features", type=int, default=0,
                       help="Max features for GP input (0=all). Speeds up high-dim data. (default: 0)")
    run_p.add_argument("--privacy", type=str, default="none", choices=["none", "floored"],
                       help="Privacy of persisted pass batches: 'floored' = parametric + "
                            "verbatim guard + δ-floor (NOT differential privacy); 'none' = "
                            "diagnostic default (may contain near-copies). (default: none)")
    run_p.add_argument("--delta", type=float, default=0.5,
                       help="δ-distance floor in σ-units when --privacy floored (default: 0.5)")
    run_p.add_argument("--json", action="store_true",
                       help="Output campaign summary as JSON")
    run_p.set_defaults(command="run")

    # ── regen generate ────────────────────────────────────────────────────────
    gen_p = sub.add_parser("generate", help="Generate a synthetic dataset (primary path)")
    gen_p.add_argument("data", type=str, help="Path to input data (CSV/JSON/Parquet)")
    gen_p.add_argument("--label", type=str, default="",
                       help="Label column name (auto-detect if omitted)")
    gen_p.add_argument("--rare-mode", type=str, default="label",
                       choices=["label", "percentile", "imbalance_ratio"],
                       help="How to identify rare events (default: label, auto rare value)")
    gen_p.add_argument("--rare-value", type=str, default=None,
                       help="Rare value for label mode (auto-detected if omitted)")
    gen_p.add_argument("--percentile", type=float, default=0.05,
                       help="Percentile threshold for percentile mode (default: 0.05)")
    gen_p.add_argument("--imbalance-ratio", type=float, default=0.01,
                       help="Imbalance ratio threshold (default: 0.01)")
    gen_p.add_argument("--n-rows", type=int, default=300,
                       help="Full synthetic dataset size (normal + rare) (default: 300)")
    gen_p.add_argument("--mode", type=str, default="balanced",
                       choices=["faithful", "balanced", "boost"],
                       help="Objective (default: balanced)")
    gen_p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    gen_p.add_argument("--privacy", type=str, default="floored", choices=["floored", "none"],
                       help="Delivered-data privacy: 'floored' (default) = parametric + "
                            "verbatim guard + δ-floor (NOT differential privacy); 'none' = "
                            "legacy grounded sampling.")
    gen_p.add_argument("--delta", type=float, default=0.5,
                       help="δ-distance floor in σ-units when --privacy floored (default: 0.5)")
    gen_p.add_argument("--scenario", type=str, default=None,
                       help="Path to a ScenarioSpec YAML — the saved use case drives "
                            "generation (its intent/gates are authoritative).")
    gen_p.add_argument("--out", type=str, default="regen-output",
                       help="Output directory (default: ./regen-output)")
    gen_p.add_argument("--accept-contract", action="store_true",
                       help="Apply the OPTIONAL advisory model semantic proposal "
                            "(Source 3), vetted by the deterministic gate. Needs "
                            "REGEN_SEMANTICS_* env; offline → Sources 1+2 only.")
    gen_p.add_argument("--json", action="store_true", help="Output summary as JSON")
    gen_p.set_defaults(command="generate")

    # ── regen explore ─────────────────────────────────────────────────────────
    explore_p = sub.add_parser("explore", help="Show the privacy↔fidelity tradeoff frontier (you choose)")
    explore_p.add_argument("data", type=str, help="Path to input data (CSV/JSON/Parquet)")
    explore_p.add_argument("--label", type=str, default="")
    explore_p.add_argument("--rare-mode", type=str, default="label",
                           choices=["label", "percentile", "imbalance_ratio"])
    explore_p.add_argument("--rare-value", type=str, default=None)
    explore_p.add_argument("--percentile", type=float, default=0.05)
    explore_p.add_argument("--imbalance-ratio", type=float, default=0.01)
    explore_p.add_argument("--n-rows", type=int, default=300)
    explore_p.add_argument("--seed", type=int, default=42)
    explore_p.add_argument("--json", action="store_true")
    explore_p.set_defaults(command="explore")

    # ── regen propose ─────────────────────────────────────────────────────────
    propose_p = sub.add_parser("propose", help="Draft a ScenarioSpec YAML from a plain-language goal")
    propose_p.add_argument("data", type=str, help="Path to input data (CSV/JSON/Parquet)")
    propose_p.add_argument("--goal", type=str, default="",
                           help="Plain-language description of what you want (drives the draft)")
    propose_p.add_argument("--label", type=str, default="",
                           help="Label column (auto-detect if omitted)")
    propose_p.add_argument("--rare-mode", type=str, default="label",
                           choices=["label", "percentile", "imbalance_ratio"])
    propose_p.add_argument("--rare-value", type=str, default=None)
    propose_p.add_argument("--percentile", type=float, default=0.05)
    propose_p.add_argument("--imbalance-ratio", type=float, default=0.01)
    propose_p.add_argument("--n-rows", type=int, default=300)
    propose_p.add_argument("--seed", type=int, default=42)
    propose_p.add_argument("--out", type=str, default=None,
                           help="Write the draft ScenarioSpec YAML here (else print to stdout)")
    propose_p.set_defaults(command="propose")

    # ── regen doctor ────────────────────────────────────────────────────────
    doctor_p = sub.add_parser("doctor", help="Preflight a dataset against the supported envelope")
    doctor_p.add_argument("data", type=str, help="Path to input data (CSV/JSON/Parquet)")
    doctor_p.add_argument("--label", type=str, default="",
                          help="Label column name (auto-detect if omitted)")
    doctor_p.add_argument("--rare-mode", type=str, default="label",
                          choices=["label", "percentile", "imbalance_ratio"])
    doctor_p.add_argument("--rare-value", type=str, default=None)
    doctor_p.add_argument("--percentile", type=float, default=0.05)
    doctor_p.add_argument("--imbalance-ratio", type=float, default=0.01)
    doctor_p.add_argument("--json", action="store_true")
    doctor_p.set_defaults(command="doctor")

    # ── regen verify ──────────────────────────────────────────────────────────
    verify_p = sub.add_parser("verify", help="Independently verify an audit bundle (a run dir)")
    verify_p.add_argument("bundle", type=str,
                          help="Path to the run directory (the audit bundle)")
    verify_p.add_argument("--json", action="store_true", help="Emit the full report as JSON")
    verify_p.set_defaults(command="verify")

    # ── regen screen ────────────────────────────────────────────────────────
    screen_p = sub.add_parser("screen", help="Predict whether REGEN or SMOTE will win on your data")
    screen_p.add_argument("data", type=str, help="Path to input data (CSV/JSON/Parquet)")
    screen_p.add_argument("--label", type=str, default="",
                          help="Label column name (auto-detect if omitted)")
    screen_p.add_argument("--rare-mode", type=str, default="percentile",
                          choices=["label", "percentile", "imbalance_ratio"],
                          help="How to identify rare events (default: percentile)")
    screen_p.add_argument("--rare-value", type=str, default=None,
                          help="Value for label mode (e.g. '1' for is_fraud=1)")
    screen_p.add_argument("--percentile", type=float, default=0.05,
                          help="Percentile threshold (default: 0.05 = bottom 5%%)")
    screen_p.add_argument("--imbalance-ratio", type=float, default=0.01,
                          help="Imbalance ratio threshold (default: 0.01)")
    screen_p.add_argument("--seed", type=int, default=42,
                          help="RNG seed (default: 42)")
    screen_p.add_argument("--quick-campaign", action="store_true",
                          help="Run a single campaign pass to sharpen the estimate")
    screen_p.set_defaults(command="screen")

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
    """Load and inspect a dataset via the API."""
    from regen.api import ingest as api_ingest
    rare_def = _build_rare_def(args)
    result = api_ingest(
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
        sym = "\u2605" if name == result.label_col else " "
        print(f"  {sym} {name}  ({meta.field_type.value})", end="")
        if meta.cardinality:
            print(f"  [{meta.cardinality} unique]", end="")
        if meta.min_val is not None:
            print(f"  [{meta.min_val:.2f}\u2013{meta.max_val:.2f}]", end="")
        print()


# ── Command: run ───────────────────────────────────────────────────────────

def _cmd_run(args):
    """Run a full REGEN campaign via the API."""
    from regen.api import run_campaign
    rare_def = _build_rare_def(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[regen] Running campaign on {args.data} ...", file=sys.stderr)

    result = run_campaign(
        filepath=args.data,
        label_col=args.label,
        rare_def=rare_def,
        seed=args.seed,
        n_rows=args.n_rows,
        max_passes=args.passes,
        out_dir=str(out_dir),
        coverage_threshold=args.coverage_threshold,
        gp_noise=args.gp_noise,
        max_features=args.max_features,
        n_estimators=100,
        num_candidates=100,
        privacy=args.privacy,
        delta=args.delta,
    )

    if args.json:
        print(json.dumps(_cr_to_dict(result), indent=2))
    else:
        _print_summary(result, out_dir)
        print(f"  Privacy:         {args.privacy}"
              + (f" (δ={args.delta}σ, NOT differential privacy)"
                 if args.privacy == "floored" else " (diagnostic; may contain near-copies)"))
        print("=" * 62)


# ── Command: generate ────────────────────────────────────────────────────────

def _cmd_generate(args):
    """Generate a synthetic dataset via the primary generate() path."""
    from regen.api import generate
    # A saved ScenarioSpec YAML is authoritative when supplied. Otherwise
    # generate() auto-detects the rare class when rare_def is None; only build an
    # explicit one when the user gave enough to pin it. This lets `generate
    # --label y` and `generate --scenario s.yaml` both just work.
    spec = None
    if args.scenario:
        from contracts.scenario import ScenarioSpec
        spec = ScenarioSpec.load_yaml(args.scenario)
    if args.rare_mode == "label" and args.rare_value is None:
        rare_def = None
    else:
        rare_def = _build_rare_def(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[regen] Generating synthetic dataset from {args.data} ...", file=sys.stderr)
    try:
        summary = generate(
            filepath=args.data,
            label_col=args.label,
            rare_def=rare_def,
            n_rows=args.n_rows,
            mode=args.mode,
            seed=args.seed,
            privacy=args.privacy,
            delta=args.delta,
            out_dir=str(out_dir),
            scenario=spec,
            accept_contract=args.accept_contract,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    fid = summary["fidelity"]
    print()
    print("=" * 62)
    print("  REGEN — SYNTHETIC DATASET")
    print("=" * 62)
    print(f"  Rows:        {summary['n_rows']}  "
          f"({summary['n_synthetic_normal']} normal + {summary['n_synthetic_rare']} rare)")
    print(f"  Fidelity:    score {fid['score']}  coverage {fid['coverage']}  "
          f"[{'PASS' if fid['passed'] else 'FAIL'}]")
    pv = summary.get("privacy")
    if pv:
        floor = ("δ-floor applied" if pv["floor_applied"]
                 else f"δ-floor skipped ({pv['floor_skip_reason']})")
        print(f"  Privacy:     {pv['mode']}  min-dist {pv['min_distance']}  "
              f"{pv['n_verbatim_duplicates']} verbatim  [{'PASS' if pv['passed'] else 'FAIL'}]")
        print(f"               {floor}; NOT differential privacy")
    else:
        print(f"  Privacy:     none")
    lift = summary.get("lift")
    if lift and lift.get("status") == "ok" and lift.get("tail_lift") is not None:
        print(f"  Detection lift: {lift['tail_lift']:+.4f}")
    elif lift and lift.get("status") == "insufficient_rare_rows":
        print(f"  Detection lift: n/a (only {lift['n_test_rare']} held-out rare "
              f"rows — too few to measure)")
    print(f"  Shippable:   {'PASS' if summary['passed'] else 'FAIL'}  (fidelity AND privacy)")
    print(f"  Output:      {out_dir.resolve()}")
    print("=" * 62)


def _print_summary(result, out_dir: Path):
    """Print a human-readable campaign summary from CampaignResult."""
    print()
    print("=" * 62)
    print("  REGEN CAMPAIGN SUMMARY")
    print("=" * 62)

    for p in result.passes:
        if p.status == "accepted":
            print(f"  Pass {p.pass_num}: \u2713 ACCEPTED")
            print(f"      recall     {p.baseline_recall:.3f} \u2192 {p.amplified_recall:.3f}"
                  f"  (lift {p.tail_lift:+.3f})")
            print(f"      precision  {p.baseline_precision:.3f} \u2192 {p.amplified_precision:.3f}")
        else:
            print(f"  Pass {p.pass_num}: \u2717 REJECTED  coverage={p.coverage:.3f}")

    print()
    print(f"  Best tail lift:  {result.best_lift:+.4f}")
    print(f"  Accepted:        {result.n_accepted}/{len(result.passes)}")
    print(f"  Output:          {out_dir.resolve()}")
    print("=" * 62)

    if result.best_batch_path:
        print(f"  Accepted synthetic batch: {result.best_batch_path}")
        import pandas as pd
        df = pd.read_parquet(result.best_batch_path)
        print(f"    {len(df)} rows, {len(df.columns)} columns")
    print("=" * 62)


# ── Command: explore ─────────────────────────────────────────────────────────

def _cmd_explore(args):
    """Print the privacy↔fidelity tradeoff frontier — the user picks."""
    from regen.api import explore_options
    rare_def = None if (args.rare_mode == "label" and args.rare_value is None) \
        else _build_rare_def(args)
    rep = explore_options(args.data, label_col=args.label, rare_def=rare_def,
                          n_rows=args.n_rows, seed=args.seed)
    if args.json:
        print(json.dumps(rep, indent=2))
        return
    print()
    print("=" * 74)
    print(f"  REGEN — OPTIONS  ({args.data})")
    print("=" * 74)
    print(f"  {'#':<2} {'privacy':<9} {'δ':<5} {'fid':<5} {'cov':<6} {'ship':<5} diagnosis")
    for i, o in enumerate(rep["options"]):
        star = "→" if i == rep["recommended"] else " "
        d = "-" if o["delta"] is None else f"{o['delta']}"
        print(f"{star} {i:<2} {o['privacy']:<9} {d:<5} {o['fidelity_score']:<5} "
              f"{o['coverage']:<6} {'yes' if o['shippable'] else 'NO':<5} {o['diagnosis'][:80]}")
    print("-" * 74)
    print(f"  {rep['note']}")
    print("  (→ = recommended default; you decide. This does not pick for you.)")
    print("=" * 74)


# ── Command: propose ─────────────────────────────────────────────────────────

def _cmd_propose(args):
    """Draft a ScenarioSpec from a plain-language goal for the user to review/edit."""
    from regen.api import draft_scenario
    rare_def = None if (args.rare_mode == "label" and args.rare_value is None) \
        else _build_rare_def(args)
    try:
        draft, proposal = draft_scenario(
            args.data, label_col=args.label, rare_def=rare_def,
            goal=args.goal, n_rows=args.n_rows, seed=args.seed,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    yaml_text = draft.to_yaml()
    if args.out:
        Path(args.out).write_text(yaml_text)
        print(f"[regen] wrote draft ScenarioSpec → {args.out}", file=sys.stderr)
    src = "model + structural" if proposal is not None else "structural only (no model configured)"
    print(f"[regen] drafted from: {src}", file=sys.stderr)
    tb = draft.provenance.get("target_tiebreak")
    if tb:  # the LLM broke a structural target tie — say so, and that it's overridable
        print(f"[regen] target tie among {tb['candidates']} → auto-selected "
              f"'{tb['chosen']}' ({tb['reason']}). Override with --label.", file=sys.stderr)
    print(f"[regen] review/edit, then: regen generate {args.data} --scenario "
          f"{args.out or '<saved>.yaml'}", file=sys.stderr)
    if not args.out:
        print()
        print(yaml_text)


# ── Command: doctor ────────────────────────────────────────────────────────

_LEVEL_MARK = {"ok": "✓", "warn": "!", "degraded": "~", "unsupported": "✗", "error": "✗"}


def _cmd_doctor(args):
    """Preflight a dataset and print the envelope verdicts."""
    from regen.api import preflight
    rare_def = None if (args.rare_mode == "label" and args.rare_value is None) \
        else _build_rare_def(args)
    rep = preflight(args.data, label_col=args.label, rare_def=rare_def)
    if args.json:
        print(json.dumps(rep, indent=2))
        sys.exit(0 if rep["ok_to_generate"] else 1)

    print()
    print("=" * 62)
    print(f"  REGEN — PREFLIGHT  ({args.data})")
    print("=" * 62)
    if "n_rare" in rep:
        print(f"  label: {rep['label_col']}   rare: {rep['n_rare']}   total: {rep['n_total']}")
    for c in rep["checks"]:
        print(f"  {_LEVEL_MARK.get(c['level'], '?')} [{c['level']}] {c['check']}: {c['message']}")
        if c["recommendation"]:
            print(f"       → {c['recommendation']}")
    print("-" * 62)
    print(f"  OK to generate: {'yes' if rep['ok_to_generate'] else 'NO'}")
    print("=" * 62)
    sys.exit(0 if rep["ok_to_generate"] else 1)


# ── Command: verify ──────────────────────────────────────────────────────────

def _cmd_verify(args):
    """Recompute an audit bundle's statistics and report PASS/FAIL — exit
    non-zero on any integrity or value mismatch."""
    from regen.audit_bundle import verify_bundle
    rep = verify_bundle(args.bundle)
    if args.json:
        print(json.dumps(rep, indent=2))
        sys.exit(0 if rep["passed"] else 1)

    print()
    print("=" * 62)
    print(f"  REGEN — AUDIT VERIFY  ({args.bundle})")
    print("=" * 62)
    print("  Integrity (artifact hashes vs manifest):")
    for a in rep["integrity"]:
        print(f"    {'✓' if a['passed'] else '✗'} {a['artifact']}")
    print("  Statistics (recomputed from delivered data + reference aggregates):")
    for s in rep["stats"]:
        if s["status"] == "uncheckable":
            print(f"    – {s['metric']}: UNCHECKABLE ({s['note']})")
        else:
            mark = "✓" if s["passed"] else "✗"
            print(f"    {mark} {s['metric']}: reported={s['reported']} recomputed={s['recomputed']}")
    for n in rep.get("notes", []):
        print(f"  note: {n}")
    print("-" * 62)
    print(f"  RESULT: {'VERIFIED' if rep['passed'] else 'FAILED'}")
    print("=" * 62)
    sys.exit(0 if rep["passed"] else 1)


# ── Command: screen ────────────────────────────────────────────────────────

def _cmd_screen(args):
    """Predict REGEN vs SMOTE win boundary for a dataset."""
    from regen.api import screen as api_screen
    rare_def = _build_rare_def(args)

    print(f"[regen] Screening {args.data} ...", file=sys.stderr)

    result = api_screen(
        filepath=args.data,
        label_col=args.label,
        rare_def=rare_def,
        seed=args.seed,
        quick_campaign=args.quick_campaign,
    )

    print()
    print("=" * 62)
    print("  REGEN — WIN-BOUNDARY SCREEN")
    print("=" * 62)
    print(f"  Recommended method:  {result.recommended_method}")
    print(f"  Heterogeneity score: {result.heterogeneity_score:.4f}")
    print(f"  Confidence:          {result.confidence:.4f}")
    print(f"  Predicted lift band: {result.predicted_lift_band}")
    print()
    print(f"  {result.rationale}")
    print()
    print(f"  Data: {result.n_rare} rare rows, {result.n_features} features")
    print(f"  Prediction ~75% accurate (benchmark/RESULTS_BREADTH.md)")
    print(f"  Two known misclassifications are conservative")
    print("=" * 62)


# ── Helpers ────────────────────────────────────────────────────────────────

from contracts.types import RareEventDef, RareMode  # noqa: E402


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


def _cr_to_dict(result) -> dict:
    """Convert CampaignResult to a plain JSON-serialisable dict."""
    return {
        "best_lift": result.best_lift,
        "passes": [
            {
                "pass": p.pass_num,
                "status": p.status,
                "tail_lift": p.tail_lift,
                "baseline_recall": p.baseline_recall,
                "amplified_recall": p.amplified_recall,
                "baseline_precision": p.baseline_precision,
                "amplified_precision": p.amplified_precision,
                "coverage": p.coverage,
            }
            for p in result.passes
        ],
        "n_accepted": result.n_accepted,
        "n_rejected": result.n_rejected,
        "n_normal": result.n_normal,
        "n_rare": result.n_rare,
        "n_features": result.n_features,
        "n_rows_per_pass": result.n_rows_per_pass,
        "output_dir": result.output_dir,
        "best_batch_path": result.best_batch_path,
    }


def _repo_root() -> Path:
    """Return the repository root (parent of cli/)."""
    return Path(__file__).parent.parent


if __name__ == "__main__":
    main()