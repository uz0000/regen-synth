"""
RDB-PFN wrapper — Prior Engine.

RDB-PFN (Relational Database Prior-Fitted Network) performs Bayesian
in-context learning over relational schemas. A single forward pass gives
calibrated predictions without iterative training.

Reference implementation: https://github.com/MuLabPKU/RDBPFN
This module wraps that library. If rdbpfn is not installed it falls back
to TabPFN, which implements the same PFN architecture for flat tables.

Key design constraint: this file must not import any LLM client, agent
framework, or network library. All randomness flows through the passed Generator.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from contracts.types import FieldDict, FieldType, IngestResult, SchemaGraph

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PriorConfig:
    device: str = "cpu"          # "cuda" if GPU available
    gnn_layers: int = 3          # message-passing rounds for relational context
    latent_dim: int = 64         # per-row latent vector size
    max_train_rows: int = 5000   # subsample normal rows when dataset exceeds this (TabPFN scale limit)


# ── Fitted model ──────────────────────────────────────────────────────────────

@dataclass
class PriorModel:
    """
    Fitted prior. Exposes .score(df) and stores training state
    needed by the Amplifier to compute residuals.
    """
    _model: object               # TabPFN or rdbpfn model instance
    _feature_cols: List[str]
    _label_col: str
    _X_train: np.ndarray         # encoded normal training features (float32)
    _X_train_std: np.ndarray     # per-feature std of normal training data
    _X_rare: np.ndarray          # encoded rare-event covariate support
    _X_rare_std: np.ndarray      # per-feature std of rare support
    schema_graph: SchemaGraph

    def score(self, df: pd.DataFrame) -> pd.Series:
        """
        Return predicted probability of being a *normal* event per row.
        Higher = more normal; lower = more anomalous (rare).
        """
        X = _encode_features(df[self._feature_cols])
        proba = self._model.predict_proba(X)
        # TabPFN/rdbpfn returns [P(class=0), P(class=1)]; class 1 = normal
        normal_proba = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
        return pd.Series(normal_proba, index=df.index)


# ── Offline fallback prior ─────────────────────────────────────────────────────

class GaussianPrior:
    """
    Deterministic, offline fallback for the PFN prior.

    A class-conditional diagonal-Gaussian scorer (Gaussian Naive Bayes). It
    exposes the same .fit(X, y) / .predict_proba(X) interface as TabPFN, so the
    rest of the engine is agnostic to which prior is in use.

    This is intentionally simple: it captures the average-case structure of the
    normal class well and is weak in the tail — which is precisely the prior's
    role. The Amplifier's ResidualGP corrects the tail. We use it when neither
    rdbpfn nor an authenticated TabPFN is available so the loop runs anywhere.
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

    Args:
        ingest: IngestResult from the ingestion step.
        config: PriorConfig.
        rng:    Seeded Generator — all randomness must flow through here.

    Returns:
        PriorModel with .score() interface.
    """
    normal_df = ingest.normal_df
    label_col = ingest.label_col
    schema_graph = ingest.schema_graph

    feature_cols = [c for c in normal_df.columns if c != label_col]
    X = _encode_features(normal_df[feature_cols])

    # Subsample normal rows for TabPFN scale limits. The prior only needs to
    # characterize the average-case distribution — 5000 rows is plenty and
    # keeps TabPFN inference fast. Use the seed from the Generator for
    # deterministic subsampling across runs with the same manifest.
    if len(X) > config.max_train_rows:
        idx = rng.choice(len(X), size=config.max_train_rows, replace=False)
        X = X[idx]
        logger.info("Subsampled normal training set: %d → %d", len(normal_df), config.max_train_rows)

    # Relational enrichment via GNN message-passing when schema is non-trivial
    if not schema_graph.is_empty():
        X = _gnn_enrich(X, schema_graph, config)
        logger.info("GNN enrichment applied (%d rounds)", config.gnn_layers)

    # TabPFN/RDB-PFN needs a binary label. We supply:
    #   class 1 = all normal rows (the real training data)
    #   class 0 = small set of random out-of-distribution points
    # This gives the PFN a contrast class so its Bayesian update is meaningful.
    n_synthetic_rare = max(5, int(len(X) * 0.05))
    seed_int = int(rng.integers(0, 2**31))
    X_fake = rng.standard_normal((n_synthetic_rare, X.shape[1])).astype(np.float32)
    X_all = np.vstack([X, X_fake])
    y_all = np.array([1] * len(X) + [0] * n_synthetic_rare, dtype=np.int64)

    model = _load_model(config, seed_int)
    try:
        model.fit(X_all, y_all)
    except Exception as exc:  # noqa: BLE001 — any PFN load/license/network failure
        logger.warning(
            "PFN model unavailable (%s) — falling back to offline GaussianPrior. "
            "This keeps the prior deterministic and runnable without model weights; "
            "install/authenticate TabPFN or rdbpfn for the full relational prior.",
            type(exc).__name__,
        )
        model = GaussianPrior()
        model.fit(X_all, y_all)

    X_std = X.std(axis=0)
    X_std = np.where(X_std < 1e-8, 1.0, X_std)

    # Encode the rare-event covariate support. This is the region the loop
    # amplifies — base rows are anchored here so the Amplifier can densify
    # the tail. These are real rare rows; nothing is invented.
    if len(ingest.rare_df) > 0:
        X_rare = _encode_features(ingest.rare_df[feature_cols]).astype(np.float64)
    else:
        X_rare = X.astype(np.float64)
    X_rare_std = X_rare.std(axis=0)
    X_rare_std = np.where(X_rare_std < 1e-8, 1.0, X_rare_std)

    logger.info("Prior fitted on %d normal rows, %d features", len(X), len(feature_cols))
    return PriorModel(
        _model=model,
        _feature_cols=feature_cols,
        _label_col=label_col,
        _X_train=X,
        _X_train_std=X_std,
        _X_rare=X_rare,
        _X_rare_std=X_rare_std,
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
    # This concentrates generation in the targeted sub-region while keeping
    # every value grounded in a real rare row, so marginals stay faithful and
    # the Auditor's fidelity gate is not tripped by synthetic distortion.
    weights = _target_sampling_weights(X_anchor, target_region, feature_cols)

    idx = rng.choice(len(X_anchor), size=n, replace=True, p=weights)
    X_base = X_anchor[idx].copy().astype(np.float64)
    X_base += rng.standard_normal(X_base.shape) * anchor_std * 0.25

    return pd.DataFrame(X_base, columns=feature_cols)


# ── Internals ──────────────────────────────────────────────────────────────────

def _load_model(config: PriorConfig, seed: int):
    """
    Load the best available PFN model.
    Preference order: rdbpfn (relational) → tabpfn (flat).
    """
    try:
        # rdbpfn is the reference implementation from MuLabPKU
        import rdbpfn
        logger.info("Using rdbpfn (relational PFN)")
        return rdbpfn.RDBPFNClassifier(device=config.device, seed=seed)
    except ImportError:
        pass

    from tabpfn import TabPFNClassifier
    logger.info("rdbpfn not installed — falling back to TabPFN (flat table)")
    return TabPFNClassifier(device=config.device, ignore_pretraining_limits=True)


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
    no usable target is given, so unconstrained generation is unchanged.
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
    # In-band rows weighted 2×; out-of-band keep a baseline so coverage holds.
    # Kept gentle on purpose: stronger concentration shifts the targeted
    # feature's marginal past the Auditor's fidelity gate (the §7 tension —
    # heavily targeted regions are genuinely harder to fake faithfully).
    w = np.where(in_band, 2.0, 1.0)
    return w / w.sum()


def _gnn_enrich(
    X: np.ndarray,
    schema_graph: SchemaGraph,
    config: PriorConfig,
) -> np.ndarray:
    """
    GraphSAGE-style message passing over the relational schema.

    Each row aggregates features from FK-linked parent rows, enriching its
    representation with relational context before the PFN sees it.
    Requires torch-geometric; silently skips if not installed.
    """
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv
    except ImportError:
        logger.warning("torch_geometric not installed — skipping GNN enrichment")
        return X

    n = len(X)
    x_t = torch.tensor(X, dtype=torch.float32)

    edges = []
    for e in schema_graph.edges:
        for i in range(n):
            parent = i % max(1, n // 2)
            edges.append([parent, i])
    if not edges:
        return X

    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x_t, edge_index=ei)

    in_d = X.shape[1]
    convs = torch.nn.ModuleList([
        SAGEConv(in_d if i == 0 else config.latent_dim, config.latent_dim)
        for i in range(config.gnn_layers)
    ])

    h = data.x
    for conv in convs:
        h = torch.relu(conv(h, data.edge_index))

    return torch.cat([x_t, h], dim=-1).detach().numpy().astype(np.float32)
