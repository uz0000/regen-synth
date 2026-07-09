"""
Certified-surrogate clean-room demo (PRODUCT_SPEC §5.5).

Stages the two-party story that ties REGEN together:

  PRODUCER (holds the real data)  →  emits a data package  →  CONSUMER (never
  sees the real data) trains a model on the surrogate and independently re-verifies
  the certificate; the PRODUCER/auditor (who holds a quarantined real test slice)
  measures how well the surrogate stands in (TSTR).

Headline artifacts: **TSTR** ("stands in this well") + **VERIFIED** ("and you can
prove it"), read alongside the privacy min-distance (the memorization cross-check).

Run:  python examples/certified_surrogate_demo.py
"""

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split


@contextlib.contextmanager
def _quiet():
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def run(data: str, label_col: str, rare_value=1, seed: int = 7) -> int:
    import logging
    logging.disable(logging.CRITICAL)
    from regen.api import ingest, generate
    from regen.audit_bundle import verify_bundle
    from engine.examiner import measure_tstr
    from contracts.types import RareEventDef, RareMode
    from sklearn.ensemble import RandomForestClassifier
    from engine.prior.grounded import _encode_features

    print("\n" + "=" * 68)
    print("  REGEN — certified synthetic surrogate (clean-room demo)")
    print("=" * 68)

    rd = RareEventDef(mode=RareMode.LABEL, label_value=rare_value)
    res = ingest(data, label_col, rd)
    real = pd.concat([res.normal_df, res.rare_df], ignore_index=True)
    strat = (real[label_col] == rare_value).astype(int)

    # ── PRODUCER: quarantine a real test slice, generate a leakage-free surrogate
    #    from the TRAIN fold only, and emit the data package (the bundle). ────────
    train_real, test_real = train_test_split(real, test_size=0.30,
                                             random_state=seed, stratify=strat)
    work = tempfile.mkdtemp(prefix="regen_cleanroom_")
    train_csv = os.path.join(work, "train.csv")
    train_real.to_csv(train_csv, index=False)
    print(f"\n[PRODUCER] real data: {len(real)} rows  →  train {len(train_real)} / "
          f"quarantined test {len(test_real)}")
    with _quiet():
        s = generate(train_csv, label_col=label_col, rare_def=rd, n_rows=len(train_real),
                     auto=False, seed=seed, privacy="floored", out_dir=os.path.join(work, "pkg"))
    pkg = s["output_dir"]
    print(f"[PRODUCER] emitted data package → {pkg}")
    print(f"           (surrogate.parquet + manifest + explanation + reference_aggregates"
          f" + scenario.yaml)")
    print(f"           privacy: floored, nearest real row {s['privacy']['min_distance']}σ, "
          f"{s['privacy']['n_verbatim_duplicates']} verbatim copies")

    surrogate = pd.read_parquet(s["best_batch_path"])

    # ── CONSUMER: never sees the real data. Re-verifies the certificate and builds
    #    a model on the surrogate alone. ─────────────────────────────────────────
    print(f"\n[CONSUMER] (never sees the real data)")
    with _quiet():
        vr = verify_bundle(pkg)
    checked = sum(1 for x in vr["stats"] if x["status"] == "checked")
    print(f"           `regen verify` → {'VERIFIED' if vr['passed'] else 'FAILED'} "
          f"({checked} statistics recomputed + {len(vr['integrity'])} artifact hashes)")
    feat = [c for c in surrogate.columns if c != label_col]
    with _quiet():
        clf = RandomForestClassifier(50, class_weight="balanced", random_state=seed)
        clf.fit(_encode_features(surrogate[feat], res.field_dict),
                (surrogate[label_col] == rare_value).astype(int))
    print(f"           trained a model on the surrogate alone (never touched real rows)")

    # ── AUDITOR (holds the real test slice): how well does the surrogate stand in?
    print(f"\n[AUDITOR]  (holds the quarantined real test slice)")
    with _quiet():
        tstr = measure_tstr(surrogate, train_real.reset_index(drop=True),
                            test_real.reset_index(drop=True), label_col, res.field_dict,
                            rare_value=rare_value, seeds=(seed,))
    if tstr.status == "ok":
        print(f"           TSTR — a model trained ONLY on the surrogate recovers "
              f"{tstr.recovered_roc_auc_median:.0%} of real-data ROC-AUC "
              f"({tstr.recovered_pr_auc_median:.0%} PR-AUC), across {len(tstr.per_model)} models")
        if tstr.note:
            print(f"           note: {tstr.note}")
    else:
        print(f"           TSTR: {tstr.status} ({tstr.note})")

    print("\n" + "-" * 68)
    print("  The real data never moved. The consumer built + independently")
    print("  re-verified without it. TSTR says how much it stands in; the privacy")
    print("  floor says it isn't a copy. That's a *certified* surrogate.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    data = str(Path(__file__).resolve().parent / "transactions.csv")
    if not Path(data).exists():
        from examples.make_sample_data import main as _mk  # type: ignore
        _mk()
    raise SystemExit(run(data, "is_fraud", rare_value=1))
