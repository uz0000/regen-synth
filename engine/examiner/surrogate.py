"""
Surrogate quality — TSTR (Train on Synthetic, Test on Real).

The headline actionable metric (PRODUCT_SPEC §5.1): does a model trained on
*nothing but the synthetic surrogate* recover the performance it would get from
real data? We report the ratio against the real-data ceiling —

    recovered = TSTR_score / TRTR_score

for a small model panel, averaged over seeds. This is a **pure evaluation**: it
does not generate anything. Leakage-freedom is the caller's responsibility — the
surrogate must have been generated from data disjoint from `real_test_df`
(`regen.api.evaluate_surrogate` enforces that split). Feeding a surrogate built
from the full data would inflate TSTR exactly like the lift-leakage bug did.

Metrics are chosen for rare events: ROC-AUC (stable, threshold-free) and PR-AUC
(average precision, rare-class-sensitive), on the natural imbalanced test set.
A `recovered` above ~1 is flagged, not celebrated — with a healthy privacy
min-distance it's noise; with a low one it's memorization masquerading as quality.

Pure Python (numpy/sklearn). No LLM/network — the engine boundary holds.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from contracts.types import FieldDict, TSTRReport
from engine.prior.grounded import _encode_features

logger = logging.getLogger(__name__)

# Below this many held-out real rare rows the TSTR estimate is degenerate — same
# floor and honesty as the lift metric (P2-7).
MIN_REAL_TEST_RARE = 10

PANEL = ("logreg", "random_forest", "gradient_boosting")


def _model(name: str, seed: int):
    """A fresh classifier from the panel. logreg is scaled (it's scale-sensitive);
    trees are not. class_weight balances the rare class where supported."""
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                      random_state=seed)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(name)


def measure_tstr(
    synth_df: pd.DataFrame,
    real_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    label_col: str,
    field_dict: FieldDict,
    *,
    rare_value=None,
    seeds=(42, 53, 61),
) -> TSTRReport:
    """Train-on-synthetic vs train-on-real, both scored on held-out real.

    Returns a TSTRReport. `real_test_df` must be disjoint from whatever produced
    `synth_df` (leakage-freedom is the caller's job).
    """
    feat = [c for c in real_train_df.columns if c != label_col]
    if rare_value is None:
        rare_value = real_train_df[label_col].value_counts().idxmin()

    def X(df):
        return np.nan_to_num(_encode_features(df[feat], field_dict).astype(np.float64), nan=0.0)

    def y(df):
        return (df[label_col] == rare_value).astype(int).to_numpy()

    Xs, ys = X(synth_df), y(synth_df)
    Xrt, yrt = X(real_train_df), y(real_train_df)
    Xte, yte = X(real_test_df), y(real_test_df)

    n_test_rare = int(yte.sum())
    if n_test_rare < MIN_REAL_TEST_RARE or len(np.unique(yte)) < 2:
        return TSTRReport(
            status="insufficient_real_test", n_real_test=len(yte),
            n_real_test_rare=n_test_rare, seeds=list(seeds),
            note=(f"held-out real test has {n_test_rare} rare rows "
                  f"(< {MIN_REAL_TEST_RARE}) or a single class — TSTR would be degenerate"),
        )
    if len(np.unique(ys)) < 2 or len(np.unique(yrt)) < 2:
        return TSTRReport(
            status="insufficient_train_classes", n_real_test=len(yte),
            n_real_test_rare=n_test_rare, n_synth_train=len(ys), seeds=list(seeds),
            note="a training set (synthetic or real) has only one class",
        )

    per_model: List[Dict] = []
    for name in PANEL:
        tstr_roc, trtr_roc, tstr_ap, trtr_ap = [], [], [], []
        for s in seeds:
            m_s, m_r = _model(name, s), _model(name, s)
            m_s.fit(Xs, ys)
            m_r.fit(Xrt, yrt)
            ps = m_s.predict_proba(Xte)[:, 1]
            pr = m_r.predict_proba(Xte)[:, 1]
            tstr_roc.append(roc_auc_score(yte, ps))
            trtr_roc.append(roc_auc_score(yte, pr))
            tstr_ap.append(average_precision_score(yte, ps))
            trtr_ap.append(average_precision_score(yte, pr))
        m_tstr_roc, m_trtr_roc = float(np.mean(tstr_roc)), float(np.mean(trtr_roc))
        m_tstr_ap, m_trtr_ap = float(np.mean(tstr_ap)), float(np.mean(trtr_ap))
        per_model.append({
            "model": name,
            "tstr_roc_auc": round(m_tstr_roc, 4),
            "trtr_roc_auc": round(m_trtr_roc, 4),
            "tstr_pr_auc": round(m_tstr_ap, 4),
            "trtr_pr_auc": round(m_trtr_ap, 4),
            "recovered_roc_auc": round(m_tstr_roc / max(m_trtr_roc, 1e-9), 4),
            "recovered_pr_auc": round(m_tstr_ap / max(m_trtr_ap, 1e-9), 4),
            "tstr_roc_auc_sd": round(float(np.std(tstr_roc)), 4),  # cross-seed spread
        })

    med_roc = float(np.median([m["recovered_roc_auc"] for m in per_model]))
    med_pr = float(np.median([m["recovered_pr_auc"] for m in per_model]))
    note = ""
    if med_roc > 1.05 or med_pr > 1.05:
        note = ("recovered > 1 — suspicious; check the privacy min-distance "
                "(high recovery + low min-distance = memorization, not quality)")
    logger.info("TSTR: recovered ROC-AUC median=%.3f, PR-AUC median=%.3f over %d models × %d seeds",
                med_roc, med_pr, len(PANEL), len(seeds))
    return TSTRReport(
        status="ok", n_real_test=len(yte), n_real_test_rare=n_test_rare,
        n_synth_train=len(ys), seeds=list(seeds), per_model=per_model,
        recovered_roc_auc_median=round(med_roc, 4),
        recovered_pr_auc_median=round(med_pr, 4), note=note,
    )
