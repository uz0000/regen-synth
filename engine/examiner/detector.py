"""
Downstream detector training and tail-lift measurement.

We use a lightweight RandomForest as the detector — easy to train, stable
across seeds, interpretable. The choice of classifier is not load-bearing;
what matters is that the *same* classifier is used for baseline vs amplified
so the lift number isolates the effect of synthetic data.

Evaluation is done on held-out real rare events only, so the lift number
reflects detection improvement on the actual tail, not on synthetic proxies.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from contracts.types import BatchManifest, IngestResult, LiftReport
from engine.prior.grounded import _encode_features

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

# Minimum held-out real rare rows for a non-degenerate lift estimate. Below this,
# recall is computed over so few positives that it can only take a few discrete
# values and a 0.0 is indistinguishable from "no benefit" — so we report a status
# instead of a bare number (P2-7). Chosen at 10: recall then resolves to steps of
# ≤0.1, enough to distinguish a real lift from noise.
MIN_TEST_RARE = 10


@dataclass
class ExaminerConfig:
    n_estimators: int = 100
    test_size: float = 0.30      # fraction of real rare events held out
    random_state: int = 42
    max_train_rows: int = 10000  # subsample normal training rows (Examiner doesn't need all 284K)


# ── Public API ─────────────────────────────────────────────────────────────────

def measure_lift(
    ingest: IngestResult,
    config: ExaminerConfig,
    generate_synth_fn=None,
    manifest: Optional[BatchManifest] = None,
) -> LiftReport:
    """
    Train baseline and amplified detectors on a held-out split; return the lift.

    The estimate is leakage-free: the real rare rows are split into a train fold
    and a held-out test fold FIRST, then the amplified model's synthetic data is
    generated from the train fold only (via `generate_synth_fn`). Both detectors
    are evaluated on the same held-out real test set. This is the whole point —
    generating synthetic from all rare rows and then testing on a subset of them
    (the previous behavior) tests the amplified model on near-copies of its own
    training data and inflates lift, worst when noise_scale is small.

    Args:
        ingest:            IngestResult — supplies real normal + rare rows.
        config:            ExaminerConfig.
        generate_synth_fn: callable(train_ingest: IngestResult) -> synthetic
                           DataFrame. Receives an IngestResult restricted to the
                           TRAIN fold (no held-out rare rows), so the synthetic it
                           returns is leakage-free. If None, no augmentation is
                           done (amplified == baseline) and lift is 0.
        manifest:          BatchManifest to embed in the report.

    Returns:
        LiftReport with tail_lift = amplified_recall - baseline_recall.
    """
    label_col    = ingest.label_col
    feature_cols = [c for c in ingest.normal_df.columns if c != label_col]
    field_dict   = ingest.field_dict

    normal_df = ingest.normal_df
    rare_df   = ingest.rare_df

    # Subsample normal rows for speed (representative sample is enough). Done on
    # the raw frame, before the split, so it doesn't bias baseline-vs-amplified.
    if len(normal_df) > config.max_train_rows:
        normal_df = normal_df.sample(config.max_train_rows, random_state=config.random_state)

    if len(rare_df) < 4:
        logger.warning("Too few rare events (%d) for reliable lift estimate", len(rare_df))

    # Split the RAW frames into train/test. Holding out both normal and rare rows
    # keeps negatives in the test set so precision penalizes false positives.
    n_train, n_test = train_test_split(
        normal_df, test_size=config.test_size, random_state=config.random_state,
    )
    r_train, r_test = train_test_split(
        rare_df, test_size=config.test_size, random_state=config.random_state,
    )

    Xn_train = _encode_features(n_train[feature_cols], field_dict)
    Xn_test  = _encode_features(n_test[feature_cols], field_dict)
    Xr_train = _encode_features(r_train[feature_cols], field_dict)
    Xr_test  = _encode_features(r_test[feature_cols], field_dict)

    X_train_real = np.vstack([Xn_train, Xr_train])
    y_train_real = np.concatenate([
        np.zeros(len(Xn_train), dtype=np.int64),
        np.ones(len(Xr_train), dtype=np.int64),
    ])
    X_test = np.vstack([Xn_test, Xr_test])
    y_test = np.concatenate([
        np.zeros(len(Xn_test), dtype=np.int64),
        np.ones(len(Xr_test), dtype=np.int64),
    ])

    # Baseline: train on real train fold only
    baseline = _train(X_train_real, y_train_real, config)
    base_recall, base_prec = _evaluate(baseline, X_test, y_test)

    # Amplified: real train fold + synthetic generated FROM THE TRAIN FOLD ONLY.
    X_synth = None
    if generate_synth_fn is not None:
        train_ingest = IngestResult(
            normal_df=n_train.reset_index(drop=True),
            rare_df=r_train.reset_index(drop=True),
            schema_graph=ingest.schema_graph,
            field_dict=field_dict,
            label_col=label_col,
            detection=ingest.detection,
        )
        synth_df = generate_synth_fn(train_ingest)
        if synth_df is not None and len(synth_df):
            # Align to feature_cols by NAME (missing → 0); never right-pad, which
            # would shift values onto the wrong feature axis.
            synth_feats = synth_df.reindex(columns=feature_cols)
            X_synth = _encode_features(synth_feats, field_dict).astype(np.float32)
            X_synth = np.nan_to_num(X_synth, nan=0.0)

    if X_synth is not None:
        X_train_aug = np.vstack([X_train_real, X_synth])
        y_train_aug = np.concatenate([y_train_real, np.ones(len(X_synth), dtype=np.int64)])
        n_synth_used = len(X_synth)
    else:
        X_train_aug, y_train_aug, n_synth_used = X_train_real, y_train_real, 0

    augmented = _train(X_train_aug, y_train_aug, config)
    amp_recall, amp_prec = _evaluate(augmented, X_test, y_test)

    tail_lift = amp_recall - base_recall

    # Degeneracy guard (P2-7): the lift is measured on the held-out rare fold. With
    # too few test-fold rare rows the estimate can only take a few discrete values
    # and a 0.0 is an artifact, not "no benefit". Flag it rather than emit a bare
    # number. The measurement is NOT weakened — the honest leakage-free protocol is
    # unchanged (git 57a45fc); we only annotate its reliability.
    n_test_rare = int(len(r_test))
    status = "ok" if n_test_rare >= MIN_TEST_RARE else "insufficient_rare_rows"

    logger.info(
        "Examiner: baseline recall=%.3f, amplified recall=%.3f, lift=%.3f "
        "(held-out, leakage-free; n_test_rare=%d, status=%s)",
        base_recall, amp_recall, tail_lift, n_test_rare, status,
    )

    return LiftReport(
        baseline_recall=base_recall,
        baseline_precision=base_prec,
        amplified_recall=amp_recall,
        amplified_precision=amp_prec,
        tail_lift=tail_lift,
        n_synthetic_used=n_synth_used,
        manifest=manifest,
        n_test_rare=n_test_rare,
        status=status,
    )


# ── Internals ──────────────────────────────────────────────────────────────────

def _train(X: np.ndarray, y: np.ndarray, config: ExaminerConfig) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        class_weight="balanced",
    )
    clf.fit(X, y)
    return clf


def _evaluate(clf: RandomForestClassifier, X: np.ndarray, y: np.ndarray):
    if len(X) == 0:
        return 0.0, 0.0
    y_pred = clf.predict(X)
    recall = float(recall_score(y, y_pred, zero_division=0))
    prec   = float(precision_score(y, y_pred, zero_division=0))
    return recall, prec
