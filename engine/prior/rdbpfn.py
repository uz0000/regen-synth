"""
Prior Engine — statistical base data generator.

Role in REGEN:
  The Prior characterizes average-case behaviour (what "normal" looks like)
  and provides a baseline for the Amplifier to correct. It is intentionally
  strong on the bulk distribution and weak on the tail — that is the
  Amplifier's job.

Two components:
  1. Scoring — how "normal" does a row look? A Gaussian Naive Bayes
     estimator (GaussianPrior) computes P(normal | x) in O(1) per row.
  2. Base batch generation — produce new rows in the rare-event region
     by perturbing real rare rows (grounded sampling). No generative
     model required because the Amplifier handles the tail.

Optional upgrade: TabPFN or RDB-PFN can replace the GaussianPrior scorer
for relational schemas or structural feature interactions. Install
'pip install regen-synth[pfn]' and set PriorConfig(backend='pfn') to
enable. Without it, the GaussianPrior fallback runs everywhere, fully
air-gapped, with no API keys.

Key design constraint: this file must not import any LLM client, agent
framework, or network library. All randomness flows through the passed
Generator.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from contracts.types import FieldDict, FieldType, IngestResult, SchemaGraph

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PriorConfig:
    """Prior Engine configuration.

    Attributes:
        backend:          Which prior backend to use.
                          'gaussian' (default) — fast, air-gapped, always
                          available. 'pfn' — TabPFN/RDB-PFN for relational
                          data and structural feature interactions (requires
                          `pip install regen-synth[pfn]`).
        device:           'cpu' or 'cuda' (only used with 'pfn' backend).
        gnn_layers:       GNN message-passing rounds for relational schemas
                          (only used with 'pfn' backend).
        latent_dim:       Per-row latent vector (only used with 'pfn' backend).
        max_train_rows:   Subsample normal rows when dataset exceeds this.
                          5000 is the TabPFN scale limit. Not used with
                          'gaussian' backend (GNB handles any size).
    """
    backend: str = "gaussian"    # 'gaussian' or 'pfn'
    device: str = "cpu"
    gnn_layers: int = 3
    latent_dim: int = 64
    max_train_rows: int = 5000


# ── Fitted model ──────────────────────────────────────────────────────────────

@dataclass
class PriorModel:
    """
    Fitted prior. Exposes .score(df) and stores training state
    needed by the Amplifier to compute residuals.

    Scoring uses a Gaussian Naive Bayes estimator (O(1) per row) that
    captures the "normal vs rare" contrast. The optional PFN backend is
    used only when PriorConfig(backend='pfn') is set.
    """
    _scorer: object              # GaussianPrior or TabPFN instance
    _feature_cols: List[str]
    _label_col: str
    _X_train: np.ndarray         # encoded normal training features (float32)
    _X_train_std: np.ndarray     # per-feature std of normal training data
    _X_rare: np.ndarray          # encoded rare-event covariate support
    _X_rare_std: np.ndarray      # per-feature std of rare support
    _is_continuous: np.ndarray   # bool mask: which feature columns are continuous
    schema_graph: SchemaGraph
    _backend_used: str = "gaussian"  # 'gaussian' or 'pfn' — which backend actually ran

    def score(self, df: pd.DataFrame) -> pd.Series:
        """
        Return predicted probability of being a *normal* event per row.
        Higher = more normal; lower = more anomalous (rare).

        Uses the fitted scorer (GaussianPrior by default).
        """
        X = _encode_features(df[self._feature_cols])
        proba = self._scorer.predict_proba(X)
        # Returns [P(class=0), P(class=1)]; class 1 = normal
        normal_proba = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
        return pd.Series(normal_proba, index=df.index)


# ── Default prior: Gaussian Naive Bayes ────────────────────────────────────────

class GaussianPrior:
    """
    Deterministic, offline prior — Gaussian Naive Bayes.

    A class-conditional diagonal-Gaussian scorer. It captures the average-case
    structure of the normal class well and is intentionally weak in the tail —
    which is precisely the prior's role. The Amplifier's ResidualGP corrects
    the tail.

    Exposes .fit(X, y) / .predict_proba(X) interface, same as TabPFN, so
    downstream code is agnostic to which backend is in use.
    """

    def __init__(self):
        self._classes = None
        self._means = {}      # class → per-feature mean
        self._vars = {}       # class → per-feature variance
        self._priors = {}     # class → prior probability

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianPrior":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self._classes = np.unique(y)
        n = len(y)
        for c in self._classes:
            Xc = X[y == c]
            self._means[c] = Xc.mean(axis=0)
            # Floor the variance so a constant feature doesn't blow up the density
            self._vars[c] = Xc.var(axis=0) + 1e-6
            self._priors[c] = len(Xc) / n
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        # Log-likelihood per class under a diagonal Gaussian, plus log prior
        log_probs = []
        for c in self._classes:
            mean, var = self._means[c], self._vars[c]
            ll = -0.5 * (np.log(2 * np.pi * var) + (X - mean) ** 2 / var).sum(axis=1)
            log_probs.append(ll + np.log(self._priors[c]))
        log_probs = np.vstack(log_probs).T              # (n, n_classes)
        # Softmax over classes for normalised probabilities
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


# ── Public API ─────────────────────────────────────────────────────────────────

def fit_prior(
    ingest: IngestResult,
    config: PriorConfig,
    rng: np.random.Generator,
) -> PriorModel:
    """
    Fit the prior model on normal events.

    The Prior has two responsibilities:
      1. Provide a .score() that distinguishes normal from rare (used by the
         Amplifier to compute residuals). This is a Gaussian Naive Bayes
         estimator — fast, deterministic, no auth needed.
      2. Store the rare-event covariate support for base-batch generation
         (used by generate_base_batch). This is just the real rare rows —
         no generative model needed.

    Args:
        ingest: IngestResult from the ingestion step.
        config: PriorConfig (backend='gaussian' by default).
        rng:    Seeded Generator — all randomness must flow through here.

    Returns:
        PriorModel with .score() interface and support for base batch
        generation.
    """
    normal_df = ingest.normal_df
    label_col = ingest.label_col
    schema_graph = ingest.schema_graph

    feature_cols = [c for c in normal_df.columns if c != label_col]
    X = _encode_features(normal_df[feature_cols])

    # Subsample normal rows for speed. The prior only needs to characterize
    # the average-case distribution — 5000 rows is plenty.
    if len(X) > config.max_train_rows:
        idx = rng.choice(len(X), size=config.max_train_rows, replace=False)
        X = X[idx]
        logger.info("Subsampled normal training set: %d → %d", len(normal_df), config.max_train_rows)

    # Build the training set for the scorer: normal rows (class 1) plus a
    # small set of random out-of-distribution points (class 0). This gives
    # the scorer a contrast class so its probabilities are meaningful.
    n_synthetic_rare = max(5, int(len(X) * 0.05))
    X_fake = rng.standard_normal((n_synthetic_rare, X.shape[1])).astype(np.float32)
    X_all = np.vstack([X, X_fake])
    y_all = np.array([1] * len(X) + [0] * n_synthetic_rare, dtype=np.int64)

    # Fit the scorer
    if config.backend == "pfn":
        scorer, backend_actual = _load_pfn_backend(config, X_all, y_all)
    else:
        scorer = GaussianPrior()
        scorer.fit(X_all, y_all)
        backend_actual = "gaussian"
        logger.info("Prior fitted (Gaussian backend) on %d normal rows, %d features",
                     len(X), len(feature_cols))

    X_std = X.std(axis=0)
    X_std = np.where(X_std < 1e-8, 1.0, X_std)

    # Encode the rare-event covariate support. This anchors base-batch
    # generation — every value is grounded in observed data.
    if len(ingest.rare_df) > 0:
        X_rare = _encode_features(ingest.rare_df[feature_cols]).astype(np.float64)
    else:
        X_rare = X.astype(np.float64)
    X_rare_std = X_rare.std(axis=0)
    X_rare_std = np.where(X_rare_std < 1e-8, 1.0, X_rare_std)

    # Build continuous-feature mask
    from contracts.types import FieldType
    is_continuous = np.array([
        ingest.field_dict[c].field_type == FieldType.CONTINUOUS
        if c in ingest.field_dict else True
        for c in feature_cols
    ], dtype=bool)
    logger.info("Continuous features: %d/%d", int(is_continuous.sum()), len(feature_cols))

    return PriorModel(
        _scorer=scorer,
        _backend_used=backend_actual,
        _feature_cols=feature_cols,
        _label_col=label_col,
        _X_train=X,
        _X_train_std=X_std,
        _X_rare=X_rare,
        _X_rare_std=X_rare_std,
        _is_continuous=is_continuous,
        schema_graph=schema_graph,
    )


def generate_base_batch(
    prior: PriorModel,
    n: int,
    target_region: Dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate a base batch of n rows in the rare-event region.

    REGEN amplifies rare events, so base rows are anchored on the rare-event
    covariate support (real rare rows) with Gaussian perturbation scaled to
    the rare support's own spread. This densifies the targeted region; the
    Amplifier's ResidualGP then refines the tail values. When Scout supplies
    a target_region, generation is focused further within that region.

    Anchoring on real rare rows is grounded sampling, not invention — every
    value is produced deterministically by the engine from observed support.
    No generative model (PFN, GAN, VAE, or LLM) is involved.

    Args:
        prior:         Fitted PriorModel.
        n:             Number of rows to generate.
        target_region: Dict from Scout describing the rare-event target.
                       Empty dict = densify the full rare region.
        rng:           Seeded Generator.

    Returns:
        DataFrame with columns matching prior._feature_cols.
    """
    feature_cols = prior._feature_cols
    X_anchor = prior._X_rare
    anchor_std = prior._X_rare_std

    # When Scout specifies a target, bias WHICH real rare rows we anchor on
    # toward the target's feature band — rather than overwriting feature values.
    weights = _target_sampling_weights(X_anchor, target_region, feature_cols)

    idx = rng.choice(len(X_anchor), size=n, replace=True, p=weights)
    X_base = X_anchor[idx].copy().astype(np.float64)

    # Only perturb continuous features. Binary/categorical features keep
    # their anchor values — perturbing a binary column (e.g. on_thyroxine)
    # produces meaningless intermediate values and causes the Auditor
    # to reject with TVD=1.0 on that column.
    continuous = prior._is_continuous
    noise = np.zeros_like(X_base)
    noise[:, continuous] = rng.standard_normal((n, int(continuous.sum()))) * anchor_std[continuous] * 0.25
    X_base += noise

    return pd.DataFrame(X_base, columns=feature_cols)


# ── Optional PFN backend ──────────────────────────────────────────────────────

def _load_pfn_backend(
    config: PriorConfig,
    X_all: np.ndarray,
    y_all: np.ndarray,
) -> Tuple[object, str]:
    """
    Load and fit the PFN backend (TabPFN or RDB-PFN).

    Requires 'pip install regen-synth[pfn]'. Falls back to GaussianPrior
    if the package is not installed or authentication fails. Logs a loud
    warning on any fallback.

    Returns:
        (scorer, backend_name) where backend_name is 'pfn', 'rdbpfn',
        or 'gaussian' (fallback).
    """
    # Try RDB-PFN first (relational)
    try:
        import rdbpfn
        logger.info("Using rdbpfn (relational PFN)")
        model = rdbpfn.RDBPFNClassifier(device=config.device, seed=42)
        model.fit(X_all, y_all)
        return model, "rdbpfn"
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("rdbpfn loaded but failed: %s", exc)

    # Fall back to TabPFN (flat table)
    try:
        from tabpfn import TabPFNClassifier
        logger.info("Using TabPFN (flat-table PFN)")
        model = TabPFNClassifier(device=config.device, ignore_pretraining_limits=True)
        model.fit(X_all, y_all)
        return model, "pfn"
    except ImportError:
        logger.warning(
            "PFN backend requested but TabPFN not installed. "
            "Install with: pip install regen-synth[pfn]"
        )
    except Exception as exc:
        logger.warning(
            "PFN backend requested but TabPFN failed: %s. "
            "Falling back to GaussianPrior.",
            exc,
        )

    # Fallback
    scorer = GaussianPrior()
    scorer.fit(X_all, y_all)
    logger.warning(
        "Prior backend fell back to GaussianPrior (requested: '%s'). "
        "PFN features (ARD lengthscales, relational structure) unavailable.",
        config.backend,
    )
    return scorer, "gaussian"


# ── Internals ──────────────────────────────────────────────────────────────────

def _encode_features(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to float32 array. Categorical → label-encoded."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype) == "category":
            out[col] = pd.Categorical(out[col]).codes.astype(np.float32)
        elif out[col].dtype == bool:
            out[col] = out[col].astype(np.float32)
        else:
            out[col] = out[col].astype(np.float32)
    return out.values.astype(np.float32)


def _target_sampling_weights(
    X_anchor: np.ndarray,
    target_region: Dict,
    feature_cols: List[str],
) -> Optional[np.ndarray]:
    """
    Build a probability vector over the rare anchor rows that favours rows in
    Scout's target band along the targeted feature.

    target_region (from Scout):
      {"feature_idx": int | "feature_name": str,
       "percentile_low": float, "percentile_high": float}

    Rows whose target-feature value falls inside [p_low, p_high] of the rare
    support get up-weighted; others keep a small baseline weight so the tails
    of the rare region are never fully starved. Returns None (→ uniform) when
    no usable target is given.
    """
    if not target_region:
        return None

    feat_idx = target_region.get("feature_idx")
    if feat_idx is None and "feature_name" in target_region:
        name = target_region["feature_name"]
        if name in feature_cols:
            feat_idx = feature_cols.index(name)
    if feat_idx is None or feat_idx >= X_anchor.shape[1]:
        return None

    pct_low  = target_region.get("percentile_low",  0.50)
    pct_high = target_region.get("percentile_high", 1.00)

    col = X_anchor[:, feat_idx]
    lo = np.percentile(col, pct_low * 100)
    hi = np.percentile(col, pct_high * 100)

    in_band = (col >= lo) & (col <= hi)
    w = np.where(in_band, 2.0, 1.0)
    return w / w.sum()
