"""
Standing regression harness — catches quality or speed drift automatically.

Runs the canonical datasets × (privacy on/off) at fixed seeds, self-verifies each
produced bundle with `regen verify`, and compares every scored quantity
against committed, provenance-stamped baselines in benchmark/BASELINES/ within
explicit tolerances. **Exits non-zero on any regression** — a fidelity/coverage
drop, a correlation increase, a gate flip, a lift drop, a runtime blow-up past
budget, or a bundle that fails to verify.

    python benchmark/run_regression.py                  # check against baselines
    python benchmark/run_regression.py --update-baselines
    python benchmark/run_regression.py --degrade        # prove it catches drift

This is minutes, not seconds, so it is a pre-push / CI step — the pre-commit hook
stays tests-only.
"""
import argparse
import json
import logging
import subprocess
import sys
import time
import tempfile
from datetime import date
from pathlib import Path

logging.disable(logging.CRITICAL)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "benchmark" / "BASELINES"

from regen.api import generate                    # noqa: E402
from regen.audit_bundle import verify_bundle       # noqa: E402

# Canonical datasets — fast, gate-passing, cover distinct regimes.
CANONICAL = [
    ("creditcard_subset.csv", "Class"),   # homogeneous numeric
    ("hypothyroid.csv", "Class"),         # heterogeneous mixed
    ("wilt.csv", "class"),                # small numeric
]
SEED = 11
N_ROWS = 400

# Tolerances (drift beyond these = regression). Generous enough for cross-machine
# float/BLAS noise, tight enough to catch a real quality drop.
TOL = {"fidelity_score": 0.05, "coverage": 0.05, "corr_delta": 0.05, "tail_lift": 0.08}
# Runtime budget: fail if a run exceeds baseline * MULT + PAD seconds. Headroom
# is wide (CI machines vary a lot); this catches a gross blow-up, not jitter.
TIME_MULT, TIME_PAD = 4.0, 5.0


def _measure(path, label, privacy, degrade=False):
    noise = 0.9 if degrade else None   # cranked noise deliberately breaks fidelity
    t0 = time.time()
    with tempfile.TemporaryDirectory() as out:
        s = generate(str(path), label_col=label, rare_def=None, n_rows=N_ROWS,
                     auto=(not degrade), noise_scale=noise, seed=SEED,
                     privacy=privacy, out_dir=out)
        verified = verify_bundle(out)["passed"]
    dt = time.time() - t0
    fid = s["fidelity"]
    lift = s.get("lift")
    return {
        "fidelity_score": fid["score"],
        "coverage": fid["coverage"],
        "corr_delta": fid["correlation"]["delta"],
        "gate_passed": bool(fid["passed"]),
        "tail_lift": (lift["tail_lift"] if lift and lift.get("status") == "ok" else None),
        "verified": bool(verified),
        "wall_time_s": round(dt, 2),
    }


def _run_all(degrade=False):
    rows = {}
    for fname, label in CANONICAL:
        path = REPO / "benchmark" / "data" / fname
        if not path.exists():
            print(f"[skip] {fname} missing")
            continue
        for privacy in ("none", "floored"):
            key = f"{fname}__{privacy}"
            rows[key] = _measure(path, label, privacy, degrade=degrade)
            r = rows[key]
            print(f"[run] {key:34s} fid={r['fidelity_score']} cov={r['coverage']} "
                  f"corrΔ={r['corr_delta']} gate={r['gate_passed']} "
                  f"verified={r['verified']} {r['wall_time_s']}s")
    return rows


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _compare(cur, base):
    """Return a list of regression strings (empty = no regression)."""
    issues = []
    for key, c in cur.items():
        b = base.get(key)
        if b is None:
            issues.append(f"{key}: no baseline (run --update-baselines)")
            continue
        if not c["verified"]:
            issues.append(f"{key}: bundle failed `regen verify`")
        if b["gate_passed"] and not c["gate_passed"]:
            issues.append(f"{key}: fidelity gate flipped PASS→FAIL")
        if b["fidelity_score"] - c["fidelity_score"] > TOL["fidelity_score"]:
            issues.append(f"{key}: fidelity {b['fidelity_score']}→{c['fidelity_score']}")
        if b["coverage"] - c["coverage"] > TOL["coverage"]:
            issues.append(f"{key}: coverage {b['coverage']}→{c['coverage']}")
        if (b["corr_delta"] is not None and c["corr_delta"] is not None
                and c["corr_delta"] - b["corr_delta"] > TOL["corr_delta"]):
            issues.append(f"{key}: corr_delta {b['corr_delta']}→{c['corr_delta']}")
        if (b["tail_lift"] is not None and c["tail_lift"] is not None
                and b["tail_lift"] - c["tail_lift"] > TOL["tail_lift"]):
            issues.append(f"{key}: tail_lift {b['tail_lift']}→{c['tail_lift']}")
        budget = b["wall_time_s"] * TIME_MULT + TIME_PAD
        if c["wall_time_s"] > budget:
            issues.append(f"{key}: runtime {c['wall_time_s']}s > budget {budget:.1f}s")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baselines", action="store_true")
    ap.add_argument("--degrade", action="store_true",
                    help="crank noise to prove the harness catches drift")
    args = ap.parse_args()

    cur = _run_all(degrade=args.degrade)
    baseline_path = BASELINE_DIR / "regression_baseline.json"

    if args.update_baselines:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"run_date": date.today().isoformat(), "code_version": _git_hash(),
                   "config": {"seed": SEED, "n_rows": N_ROWS}, "results": cur}
        baseline_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote baselines → {baseline_path}")
        return 0

    if not baseline_path.exists():
        print("No baselines; run --update-baselines first.")
        return 2
    base = json.loads(baseline_path.read_text())["results"]
    issues = _compare(cur, base)
    print()
    if issues:
        print("REGRESSION DETECTED:")
        for i in issues:
            print("  ✗ " + i)
        return 1
    print("✓ No regression — all metrics within tolerance, all bundles verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
