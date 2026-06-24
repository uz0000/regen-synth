"""
End-to-end REGEN demo (CLI, no UI framework).

This is the runnable proof that the pipeline works:
    ingest → generate a full synthetic dataset (normal part + amplified rare part)
           → (Scout → Prior → Amplifier → Auditor → Examiner) × N

It is fully self-contained: if the sample dataset is missing it generates one,
then (1) screens REGEN vs SMOTE, (2) generates a full synthetic dataset via the
primary generate() path, and (3) runs a multi-pass amplification campaign.
Everything is deterministic given the seed — every value is engine output; the
only thing the loop decides is *which region to amplify next*.

Usage:
    python examples/run_demo.py                 # auto-generate data + run
    python examples/run_demo.py --data my.csv   # use your own data
    python examples/run_demo.py --label is_fraud --rare-value 1
"""

import argparse
import contextlib
import io
import logging
import sys
from pathlib import Path

# Make the repo root importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.types import RareEventDef, RareMode  # noqa: E402
from regen.api import generate, run_campaign, screen          # noqa: E402

# Silence library logging (paramz/GPy warn "reconstraining parameters" on every
# GP fit). The demo drives all output via print(), so logging is noise here.
logging.basicConfig(level=logging.CRITICAL)

DEFAULT_DATA = Path(__file__).parent / "transactions.csv"


def _ensure_data(path: Path) -> None:
    """Generate the sample fraud dataset if it isn't already on disk."""
    if path.exists():
        return
    from examples.make_sample_data import make_dataset

    df = make_dataset(n=2000, fraud_rate=0.03)
    df.to_csv(path, index=False)
    print(f"[demo] generated sample data → {path} "
          f"({len(df)} rows, {int(df['is_fraud'].sum())} fraud)")


@contextlib.contextmanager
def _quiet():
    """Suppress noisy library output (GPy prints to stderr) during engine work.

    Exceptions raised inside still surface: the context restores the real
    streams on exit, before Python prints the traceback.
    """
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def run(data: Path, label_col: str, rare_value, seed: int, passes: int) -> int:
    print("\n" + "=" * 64)
    print("  REGEN — rare-event amplification demo")
    print("=" * 64)

    rare_def = RareEventDef(
        mode=RareMode.LABEL,
        label_value=int(rare_value) if str(rare_value).lstrip("-").isdigit() else rare_value,
    )

    # ── 1. Screen: quick REGEN vs SMOTE head-to-head on this data ─────────────
    print("\n[1/3] Screening: REGEN vs SMOTE (1 pass each, matched budget)...")
    with _quiet():
        sr = screen(
            filepath=str(data),
            label_col=label_col,
            rare_def=rare_def,
            seed=seed,
        )
    print(f"      recommended: {sr.recommended_method}   "
          f"(confidence {sr.confidence:.2f}, Fisher CV {sr.heterogeneity_score:.2f})")
    print(f"      {sr.rationale}")

    # ── 2. Generate a full synthetic dataset (the primary deliverable) ────────
    #    generate() returns a FULL dataset: an amplified rare part concatenated
    #    with a synthetic normal part, at a rare:normal ratio that reflects the
    #    amplification. This is what changed — it no longer returns rare rows only.
    print("\n[2/3] Generating full synthetic dataset (normal part + amplified rare part)...")
    with _quiet():
        gs = generate(
            filepath=str(data),
            label_col=label_col,
            rare_def=rare_def,
            n_rows=400,
            mode="balanced",
            seed=seed,
        )
    print(f"      rows: {gs['n_rows']}  =  {gs['n_synthetic_normal']} normal  +  "
          f"{gs['n_synthetic_rare']} rare")
    print(f"      rare ratio: {gs['rare_ratio']:.2f}  "
          f"(natural prevalence {gs['natural_prevalence']:.2f} → amplified)")
    print(f"      fidelity: rare {gs['fidelity']['score']:.2f} "
          f"(coverage {gs['fidelity']['coverage']:.2f}), "
          f"normal {gs['normal_fidelity']['score']:.2f}  "
          f"[overall {'PASS' if gs['fidelity']['passed'] else 'FAIL'}]")
    if gs["lift"]:
        print(f"      detection lift: recall {gs['lift']['baseline_recall']:.3f} → "
              f"{gs['lift']['amplified_recall']:.3f}  ({gs['lift']['tail_lift']:+.3f})")
    print(f"      full dataset:  {gs['best_batch_path']}")

    # ── 3. Full multi-pass campaign ───────────────────────────────────────────
    print(f"\n[3/3] Running {passes}-pass REGEN campaign...")
    with _quiet():
        cr = run_campaign(
            filepath=str(data),
            label_col=label_col,
            rare_def=rare_def,
            seed=seed,
            n_rows=300,
            max_passes=passes,
            coverage_threshold=0.60,
            gp_noise=0.1,
            n_estimators=80,
            num_candidates=80,
        )

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f"\n  ingested: {cr.n_normal} normal, {cr.n_rare} rare, {cr.n_features} features")
    print("-" * 64)
    for p in cr.passes:
        if p.status == "accepted":
            print(
                f"  pass {p.pass_num}: ACCEPTED   "
                f"recall {p.baseline_recall:.3f} → {p.amplified_recall:.3f}  "
                f"(lift {p.tail_lift:+.3f})   "
                f"precision {p.baseline_precision:.3f} → {p.amplified_precision:.3f}"
            )
        else:
            print(f"  pass {p.pass_num}: REJECTED   coverage {p.coverage:.3f}")
    print("-" * 64)
    print(f"  best tail lift: {cr.best_lift:+.4f}   "
          f"accepted {cr.n_accepted}/{cr.n_accepted + cr.n_rejected} passes")
    if cr.best_batch_path:
        print(f"  best batch:     {cr.best_batch_path}")
    print("=" * 64 + "\n")
    return 0


def main():
    p = argparse.ArgumentParser(description="Run the REGEN end-to-end demo.")
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="Path to input CSV (default: auto-generated sample).")
    p.add_argument("--label", default="is_fraud", help="Label column name.")
    p.add_argument("--rare-value", default=1, help="Label value marking a rare event.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--passes", type=int, default=3, help="Number of amplification passes.")
    args = p.parse_args()

    data = Path(args.data)
    if data == DEFAULT_DATA:
        _ensure_data(data)
    elif not data.exists():
        print(f"Missing data file: {data}", file=sys.stderr)
        return 1

    return run(data, args.label, args.rare_value, args.seed, args.passes)


if __name__ == "__main__":
    sys.exit(main())
