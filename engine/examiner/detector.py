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
from engine.prior.rdbpfn import _encode_features

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ExaminerConfig:
    n_estimators: int = 100
    test_size: float = 0.30      # fraction of real rare events held out
    random_state: int = 42
    max_train_rows: int = 10000  # subsample normal training rows (Examiner doesn't need all 284K)


# ── Public API ─────────────────────────────────────────────────────────────────

def measure_lift(
    ingest: IngestResult,
    synthetic_df: pd.DataFrame,
    config: ExaminerConfig,
    manifest: Optional[BatchManifest] = None,
) -> LiftReport:
    """
    Train baseline and amplified detectors; return the lift report.

    Args:
        ingest:       IngestResult — supplies real normal + rare rows.
        synthetic_df: Accepted synthetic batch (Auditor must have passed it).
        config:       ExaminerConfig.
        manifest:     BatchManifest to embed in the report.

    Returns:
        LiftReport with tail_lift = amplified_recall - baseline_recall.
    """
    label_col    = ingest.label_col
    feature_cols = [c for c in ingest.normal_df.columns if c != label_col]

    normal_df = ingest.normal_df
    rare_df   = ingest.rare_df

    # Build real dataset with binary rare label
    real_df = pd.concat([normal_df, rare_df], ignore_index=True)
    X_normal_all = _encode_features(normal_df[feature_cols])
    X_rare_all   = _encode_features(rare_df[feature_cols])

    # Subsample normal training rows for speed. The Examiner only needs a
    # representative sample to estimate lift — 10K rows is plenty and keeps
    # RandomForest training under 10s instead of 2+ minutes on 284K rows.
    if len(X_normal_all) > config.max_train_rows:
        rng_state = np.random.RandomState(config.random_state)
        idx = rng_state.choice(len(X_normal_all), size=config.max_train_rows, replace=False)
        X_normal_all = X_normal_all[idx]

    if len(X_rare_all) < 4:
        logger.warning("Too few rare events (%d) for reliable lift estimate", len(X_rare_all))

    # Hold out BOTH normal and rare rows. The test set must contain negatives
    # so precision genuinely penalizes false positives — otherwise a model that
    # simply predicts "rare" everywhere would score perfect recall and the lift
    # number would be meaningless.
    Xn_train, Xn_test = train_test_split(
        X_normal_all, test_size=config.test_size, random_state=config.random_state,
    )
    Xr_train, Xr_test = train_test_split(
        X_rare_all, test_size=config.test_size, random_state=config.random_state,
    )

    # Real training set: normal_train (label 0) + rare_train (label 1)
    X_train_real = np.vstack([Xn_train, Xr_train])
    y_train_real = np.concatenate([
        np.zeros(len(Xn_train), dtype=np.int64),
        np.ones(len(Xr_train), dtype=np.int64),
    ])

    # Test set: held-out normal (0) + held-out rare (1)
    X_test = np.vstack([Xn_test, Xr_test])
    y_test = np.concatenate([
        np.zeros(len(Xn_test), dtype=np.int64),
        np.ones(len(Xr_test), dtype=np.int64),
    ])
    X_rare_train = Xr_train  # used below for the augmented training set

    # Baseline: train on real only
    baseline = _train(X_train_real, y_train_real, config)
    base_recall, base_prec = _evaluate(baseline, X_test, y_test)

    # Amplified: train on real + synthetic
    synth_cols = [c for c in feature_cols if c in synthetic_df.columns]
    if synth_cols:
        X_synth = _encode_features(synthetic_df[synth_cols]).astype(np.float32)
        # Pad missing columns with zeros if needed
        if X_synth.shape[1] < len(feature_cols):
            pad = np.zeros((len(X_synth), len(feature_cols) - X_synth.shape[1]), dtype=np.float32)
            X_synth = np.hstack([X_synth, pad])
        y_synth = np.ones(len(X_synth), dtype=np.int64)
        X_train_aug = np.vstack([X_train_real, X_synth])
        y_train_aug = np.concatenate([y_train_real, y_synth])
    else:
        X_train_aug = X_train_real
        y_train_aug = y_train_real

    augmented = _train(X_train_aug, y_train_aug, config)
    amp_recall, amp_prec = _evaluate(augmented, X_test, y_test)

    tail_lift = amp_recall - base_recall

    logger.info(
        "Examiner: baseline recall=%.3f, amplified recall=%.3f, lift=%.3f",
        base_recall, amp_recall, tail_lift,
    )

    return LiftReport(
        baseline_recall=base_recall,
        baseline_precision=base_prec,
        amplified_recall=amp_recall,
        amplified_precision=amp_prec,
        tail_lift=tail_lift,
        n_synthetic_used=len(synthetic_df),
        manifest=manifest,
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
