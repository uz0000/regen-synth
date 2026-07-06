"""
Privacy sweep (P1-6): measure what privacy="floored" costs vs "none" across the
benchmark datasets, at a fixed config and seed so the two modes are directly
comparable. Writes RESULTS_PRIVACY.json + RESULTS_PRIVACY.md with the run date
and the code version (git hash), per the audit's provenance rule (§6.6).

Run from the repo root:  python benchmark/run_privacy_sweep.py

Fixed config (auto-tuning OFF) so floored-vs-none isolates the privacy cost, not
a different noise_scale the tuner might have picked per mode.
"""
import json
import os
import subprocess
import sys
import time
import tempfile
import logging
from datetime import date
from pathlib import Path

logging.disable(logging.CRITICAL)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from regen.api import generate  # noqa: E402

# (filename, label_col). Rare value is auto-detected (minority class).
DATASETS = [
    ("creditcard_subset.csv", "Class"),
    ("satellite.csv", "Target"),
    ("hypothyroid.csv", "Class"),
    ("wilt.csv", "class"),
    ("ozone.csv", "Class"),
    ("bank_marketing.csv", "y"),
    ("churn.csv", "Class"),
    ("solar_flare.csv", "Class"),
    ("open_payments.csv", "Class"),   # all-categorical → floor skipped (P2-9)
    ("amazon.csv", "ACTION"),         # all-categorical
    ("creditcard.csv", "Class"),      # large (284k) — kept last
]

N_ROWS = 400
SEED = 7


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _row_for(path, label_col, privacy):
    t0 = time.time()
    with tempfile.TemporaryDirectory() as out:
        s = generate(str(path), label_col=label_col, rare_def=None, n_rows=N_ROWS,
                     auto=False, seed=SEED, privacy=privacy, out_dir=out)
    dt = time.time() - t0
    fid = s["fidelity"]
    lift = s.get("lift")
    pv = s.get("privacy")
    return {
        "privacy": privacy,
        "fidelity_score": fid["score"],
        "coverage": fid["coverage"],
        "corr_delta": fid["correlation"]["delta"],
        "fidelity_passed": fid["passed"],
        "shippable_passed": s["passed"],
        "tail_lift": (lift["tail_lift"] if lift else None),
        "lift_status": (None if lift else "insufficient_or_gate_failed"),
        "privacy_min_distance": (pv["min_distance"] if pv else None),
        "floor_applied": (pv["floor_applied"] if pv else None),
        "floor_skip_reason": (pv["floor_skip_reason"] if pv else None),
        "n_verbatim": (pv["n_verbatim_duplicates"] if pv else None),
        "privacy_passed": (pv["passed"] if pv else None),
        "n_synth_rare": s["n_synthetic_rare"],
        "wall_time_s": round(dt, 2),
    }


def main():
    data_dir = REPO / "benchmark" / "data"
    results = []
    for fname, label in DATASETS:
        path = data_dir / fname
        entry = {"dataset": fname, "label_col": label}
        if not path.exists():
            entry["error"] = "missing (gitignored; run a benchmark downloader first)"
            results.append(entry)
            print(f"[skip] {fname}: missing")
            continue
        for privacy in ("none", "floored"):
            try:
                entry[privacy] = _row_for(path, label, privacy)
                r = entry[privacy]
                print(f"[ok]  {fname:22s} {privacy:8s} fid={r['fidelity_score']} "
                      f"cov={r['coverage']} corrΔ={r['corr_delta']} "
                      f"lift={r['tail_lift']} floor={r['floor_applied']} "
                      f"minD={r['privacy_min_distance']} {r['wall_time_s']}s")
            except Exception as e:
                entry[privacy] = {"privacy": privacy, "error": str(e)}
                print(f"[ERR] {fname:22s} {privacy}: {e}")
        results.append(entry)

    payload = {
        "run_date": date.today().isoformat(),
        "code_version": _git_hash(),
        "config": {"n_rows": N_ROWS, "seed": SEED, "auto": False},
        "command": "python benchmark/run_privacy_sweep.py",
        "results": results,
    }
    out_json = REPO / "benchmark" / "RESULTS_PRIVACY.json"
    out_json.write_text(json.dumps(payload, indent=2))
    _write_md(payload, REPO / "benchmark" / "RESULTS_PRIVACY.md")
    print(f"\nWrote {out_json} and RESULTS_PRIVACY.md")


def _fmt(v):
    return "—" if v is None else v


def _write_md(payload, path):
    lines = [
        "# Privacy sweep — floored vs none",
        "",
        f"**Run date:** {payload['run_date']}  |  **Code version:** "
        f"`{payload['code_version']}`  |  **Command:** `{payload['command']}`",
        "",
        f"Fixed config: n_rows={payload['config']['n_rows']}, seed="
        f"{payload['config']['seed']}, auto-tune OFF (so floored-vs-none isolates "
        "the privacy cost). Rare class auto-detected. Privacy is NOT differential "
        "privacy (near-copy re-identification floor + verbatim guard).",
        "",
        "Per dataset, `none` (grounded sampling) vs `floored` (parametric + "
        "δ-floor + verbatim guard). ΔX = floored − none.",
        "",
        "| Dataset | mode | fid | cov | corrΔ | gate | lift | floor | minDist | verbatim | t(s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in payload["results"]:
        if "error" in e:
            lines.append(f"| {e['dataset']} | — | _skipped: {e['error']}_ |||||||||")
            continue
        for mode in ("none", "floored"):
            r = e.get(mode, {})
            if "error" in r:
                lines.append(f"| {e['dataset']} | {mode} | _error: {r['error']}_ |||||||||")
                continue
            lines.append(
                f"| {e['dataset']} | {mode} | {_fmt(r['fidelity_score'])} | "
                f"{_fmt(r['coverage'])} | {_fmt(r['corr_delta'])} | "
                f"{'PASS' if r['fidelity_passed'] else 'FAIL'} | "
                f"{_fmt(r['tail_lift'])} | {_fmt(r['floor_applied'])} | "
                f"{_fmt(r['privacy_min_distance'])} | {_fmt(r['n_verbatim'])} | "
                f"{_fmt(r['wall_time_s'])} |"
            )
    lines += [
        "",
        "## Reading this",
        "",
        "- **gate** is the Auditor fidelity verdict (coverage + per-column + "
        "correlation). A floored row that flips PASS→FAIL is a privacy cost worth "
        "flagging.",
        "- **floor** = whether the δ-distance floor was enforced. `False` on "
        "all-categorical datasets (no continuous features) is expected and "
        "honest (P2-9); the verbatim guard still applies.",
        "- **minDist** ≥ delta (0.5) when the floor is applied and passes; `inf` "
        "when no continuous features exist.",
        "- **lift** `—` means the batch failed the gate (no lift measured) or the "
        "held-out rare fold was too small for a lift estimate.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
