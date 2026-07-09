"""
Prior — grounded-sampling base generator + normal-density scorer.

Role in REGEN:
  The Prior characterizes average-case behaviour (what "normal" looks like)
  and provides a baseline for the Amplifier to correct. It is intentionally
  strong on the bulk distribution and weak on the tail — that is the
  Amplifier's job.

Two components, both empirical (no learned generative model):
  1. Normal-density scoring — how "normal" does a row look? A class-conditional
     diagonal-Gaussian scorer (GaussianPrior) computes P(normal | x) in O(1)
     per row. This score is consumed by the Amplifier to weight residual
     relevance; it is NOT used to generate rows.
  2. Base batch generation — produce new rows by *grounded sampling*: draw a
     real anchor row and add Gaussian noise scaled to that region's observed
     spread, perturbing only continuous features. Anchors come from the rare
     support (generate_base_batch) or the normal support (generate_normal_batch).

This is deliberately not a deep/relational generative model. An earlier design
proposed wrapping RDB-PFN/TabPFN for relational schemas; that path was dropped
as unused — REGEN is single-table, and grounded sampling plus the Amplifier's
residual GP cover the need without it.

Key design constraint: this file must not import any LLM client, agent
framework, or network library. All randomness flows through the passed
Generator.
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
    """Prior configuration.

    Attributes:
        max_train_rows:   Subsample normal rows when the dataset exceeds this
                          (the prior only needs to characterize the average case).
        noise_scale:      Fraction of the anchor region's std-dev used as Gaussian
                          perturbation on continuous features during grounded
                          sampling. Lower = tighter to real anchor rows (better
                          distribution match, less exploration). Higher = more
                          exploration, wider coverage.
    """
    max_train_rows: int = 5000
    noise_scale: float = 0.10  # fraction of std-dev for continuous-feature perturbation


# ── Fitted model ──────────────────────────────────────────────────────────────

@dataclass
class PriorModel:
    """
    Fitted prior. Exposes .score(df) and stores training state needed by the
    Amplifier to compute residuals, plus the anchor support used by grounded
    sampling.

    Scoring uses a class-conditional diagonal-Gaussian estimator (O(1) per row)
    that captures the "normal vs rare" contrast.

    Privacy (parametric generation): per-class discrete frequency tables and the
    continuous-column indices are stored so ``generate_parametric_batch`` can
    draw fresh rows from a Gaussian-copula fit of the class distribution instead
    of perturbing real anchor rows. ``_cont_idx``/``_disc_idx`` split the encoded
    feature columns into copula-sampled (continuous) and frequency-sampled
    (categorical + binary) groups.
    """
    _scorer: object              # GaussianPrior — P(normal|x) density scorer
    _feature_cols: List[str]
    _label_col: str
    _X_train: np.ndarray         # encoded normal training features (float32)
    _X_train_std: np.ndarray     # per-feature std of normal training data
    _X_rare: np.ndarray          # encoded rare-event covariate support
    _X_rare_std: np.ndarray      # per-feature std of rare support
    _is_continuous: np.ndarray   # bool mask: which feature columns are continuous
    schema_graph: SchemaGraph
    _field_dict: object = None   # ingest field_dict → canonical categorical encoding
    # ── Parametric (privacy) parameters ───────────────────────────────────────
    _cont_idx: object = None     # np.ndarray of continuous column indices
    _disc_idx: object = None     # np.ndarray of discrete column indices
    _disc_freq: object = None    # {class: {col: np.ndarray freq over codes}}

    def score(self, df: pd.DataFrame) -> pd.Series:
        """
        Return predicted probability of being a *normal* event per row.
        Higher = more normal; lower = more anomalous (rare).

        Uses the fitted scorer (GaussianPrior by default).
        """
        X = _encode_features(df[self._feature_cols], self._field_dict)
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
    which is precisely the prior's role. The Amplifier's TailCorrector corrects
    the tail.
    """

    def __init__(self):
        self._classes = None
        self._means = {}      # class → per-feature mean (in standardized space)
        self._vars = {}       # class → per-feature variance (in standardized space)
        self._priors = {}     # class → prior probability
        self._feat_mean = None  # global per-feature mean (standardization)
        self._feat_std = None   # global per-feature std (standardization)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feat_mean) / self._feat_std

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianPrior":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        # Standardize features to unit scale before fitting. Without this, a
        # low-variance or small-unit feature contributes a disproportionately
        # large (X-mean)^2/var term and dominates the diagonal-Gaussian score,
        # swamping the other features. Standardizing puts every feature on equal
        # footing. Scoring only — generation uses raw anchor values, untouched.
        self._feat_mean = X.mean(axis=0)
        std = X.std(axis=0)
        self._feat_std = np.where(std < 1e-8, 1.0, std)
        Xs = self._standardize(X)
        self._classes = np.unique(y)
        n = len(y)
        for c in self._classes:
            Xc = Xs[y == c]
            self._means[c] = Xc.mean(axis=0)
            # Floor the variance so a constant feature doesn't blow up the density
            self._vars[c] = Xc.var(axis=0) + 1e-6
            self._priors[c] = len(Xc) / n
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._standardize(np.asarray(X, dtype=np.float64))
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
    field_dict = ingest.field_dict
    X = _encode_features(normal_df[feature_cols], field_dict)

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
    scorer = GaussianPrior()
    scorer.fit(X_all, y_all)
    logger.info("Prior scorer fitted on %d normal rows, %d features",
                 len(X), len(feature_cols))

    X_std = X.std(axis=0)
    X_std = np.where(X_std < 1e-8, 1.0, X_std)

    # Encode the rare-event covariate support. This anchors base-batch
    # generation — every value is grounded in observed data.
    if len(ingest.rare_df) > 0:
        X_rare = _encode_features(ingest.rare_df[feature_cols], field_dict).astype(np.float64)
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

    # ── Parametric (privacy) parameters ───────────────────────────────────────
    # Column-index split + per-class discrete (categorical+binary) frequency
    # tables. The privacy path samples continuous features from a Gaussian copula
    # built on demand from the stored class arrays (_X_train/_X_rare) at
    # generation time — no per-class moments are precomputed here.
    cont_idx = np.where(is_continuous)[0]
    disc_idx = np.where(~is_continuous)[0]
    disc_freq = {
        cls_name: _fit_discrete_freq(X_cls, disc_idx, feature_cols, field_dict)
        for cls_name, X_cls in (("normal", X), ("rare", X_rare))
    }

    return PriorModel(
        _scorer=scorer,
        _feature_cols=feature_cols,
        _label_col=label_col,
        _X_train=X,
        _X_train_std=X_std,
        _X_rare=X_rare,
        _X_rare_std=X_rare_std,
        _is_continuous=is_continuous,
        schema_graph=schema_graph,
        _field_dict=field_dict,
        _cont_idx=cont_idx,
        _disc_idx=disc_idx,
        _disc_freq=disc_freq,
    )


def generate_base_batch(
    prior: PriorModel,
    n: int,
    target_region: Dict,
    rng: np.random.Generator,
    noise_scale: float = 0.10,
) -> pd.DataFrame:
    """
    Generate a base batch of n rows in the rare-event region.

    REGEN amplifies rare events, so base rows are anchored on the rare-event
    covariate support (real rare rows) with Gaussian perturbation scaled to
    the rare support's own spread. This densifies the targeted region; the
    Amplifier's TailCorrector then refines the tail values. When Scout supplies
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
        noise_scale:   Fraction of rare-event std-dev for continuous feature
                       perturbation. Default 0.25.

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
    noise[:, continuous] = rng.standard_normal((n, int(continuous.sum()))) * anchor_std[continuous] * noise_scale
    X_base += noise

    return pd.DataFrame(X_base, columns=feature_cols)


def generate_normal_batch(
    prior: PriorModel,
    n: int,
    rng: np.random.Generator,
    noise_scale: float = 0.10,
) -> pd.DataFrame:
    """Generate n synthetic *normal*-class rows.

    Mirror of generate_base_batch but anchored on the normal covariate support
    (prior._X_train) instead of the rare region, with no Scout targeting — the
    normal class has no tail to densify. Same grounded-sampling discipline: real
    normal rows + Gaussian perturbation scaled to the normal spread; only
    continuous features are perturbed. Used to synthesize the majority half of a
    full synthetic dataset; its fidelity is gated by the Auditor against the
    normal reference (the Amplifier is not involved — it corrects the rare tail).
    """
    feature_cols = prior._feature_cols
    X_anchor = prior._X_train
    anchor_std = prior._X_train_std

    idx = rng.choice(len(X_anchor), size=n, replace=True)
    X_base = X_anchor[idx].copy().astype(np.float64)

    continuous = prior._is_continuous
    noise = np.zeros_like(X_base)
    noise[:, continuous] = rng.standard_normal((n, int(continuous.sum()))) * anchor_std[continuous] * noise_scale
    X_base += noise

    return pd.DataFrame(X_base, columns=feature_cols)


def generate_parametric_batch(
    prior: PriorModel,
    n: int,
    rng: np.random.Generator,
    which_class: str = "rare",
) -> pd.DataFrame:
    """Generate n rows by sampling from the fitted per-class distribution.

    This is the privacy-safe generator: unlike ``generate_base_batch`` /
    ``generate_normal_batch`` (which perturb a *real anchor row* and so emit
    near-copies of real individuals), this never touches a real row. Continuous
    features are drawn from a **mixed-data Gaussian copula** fit on demand from
    the stored class array (``_X_rare``/``_X_train``): every feature column
    (continuous *and* discrete) is mapped to standard-normal scores, one latent
    correlation is estimated across all of them, fresh latent rows are drawn from
    that correlation, then each column is mapped back to its own marginal —
    continuous columns through their empirical quantiles, categorical/binary
    columns through the inverse-CDF of their per-class frequency table. This
    preserves each marginal exactly (values lie on the real support) **and** the
    full correlation structure — including correlation *between* discrete and
    continuous features — which the Auditor gates on, without ever copying a real
    row. (Drawing the discrete columns independently, as an earlier version did,
    reproduced their marginals but erased that cross-correlation and failed the
    gate whenever a binary/categorical feature was correlated with the continuous
    ones — the P0-2 defect.) Discrete values are never copied verbatim from a
    real row, which was the strongest re-id signal.

    Returns a feature-only DataFrame in *encoded* space — the same contract as
    ``generate_base_batch`` — so the downstream TailCorrector tail correction,
    constraint layer, and categorical decode apply unchanged. The privacy
    δ-distance floor (engine.privacy) is enforced by the caller after any GP
    correction, against the full real set.

    Args:
        prior:      Fitted PriorModel (carries the per-class arrays + freq tables).
        n:          Number of rows to generate.
        rng:        Seeded Generator.
        which_class: "rare" or "normal" — which class distribution to sample.

    Returns:
        DataFrame with columns matching prior._feature_cols (encoded space).
    """
    feature_cols = prior._feature_cols
    p = len(feature_cols)
    X = np.zeros((n, p), dtype=np.float64)

    disc_idx = prior._disc_idx
    disc_freq = (prior._disc_freq or {}).get(which_class, {})

    # The real class array the copula + frequency tables were derived from.
    class_X = (prior._X_rare if which_class == "rare" else prior._X_train)
    class_X = np.asarray(class_X, dtype=np.float64)
    if class_X.shape[0] == 0:
        return pd.DataFrame(X, columns=feature_cols)

    # ONE joint Gaussian copula over ALL feature columns (continuous + discrete),
    # not a continuous-only copula with discrete columns drawn independently.
    # Sampling discrete columns independently (the previous behaviour) preserved
    # each marginal but erased every correlation *between* a discrete feature and
    # the continuous ones. When a binary/categorical feature is correlated with
    # the continuous features in the rare tail (e.g. is_fraud ↔ n_prior_txns), the
    # Auditor's correlation gate then failed on the whole batch — the P0-2 defect,
    # which only surfaced once such a feature was gated (in LABEL mode the binary
    # is the label and is excluded from the gate). The joint copula ties every
    # column to a shared latent correlation, so cross-correlation is restored
    # while each marginal is still reproduced exactly (continuous via
    # empirical-quantile inverse; discrete via inverse-CDF on the frequency table).
    # No real row is emitted: every coordinate is an interpolation/lookup selected
    # by an independently drawn latent rank. When there are no discrete features
    # this reduces exactly (same RNG draws, same values) to the continuous copula.
    U = _copula_uniforms(class_X, n, rng)                     # (n, p) correlated uniforms

    disc_set = set(int(j) for j in disc_idx) if disc_idx is not None else set()
    for j in range(p):
        if j in disc_set:
            freq = disc_freq.get(feature_cols[j])
            if freq is None or freq.size == 0:
                X[:, j] = class_X[0, j]      # degenerate: single observed code
            else:
                X[:, j] = _discrete_inverse_cdf(freq, U[:, j])
        else:
            X[:, j] = _quantile_inverse(class_X[:, j], U[:, j])

    return pd.DataFrame(X, columns=feature_cols)


# ── Internals ──────────────────────────────────────────────────────────────────

def _normal_scores(x: np.ndarray) -> np.ndarray:
    """Map a 1-D sample to standard-normal scores via the empirical CDF
    (rank → uniform → Φ⁻¹). Constant columns map to 0. This is the latent-
    Gaussian representation a Gaussian copula samples from."""
    from scipy.stats import norm, rankdata
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    if np.allclose(x.std(), 0):
        return np.zeros_like(x)
    u = rankdata(x, method="average") / (x.size + 1.0)
    return norm.ppf(u)


def _quantile_inverse(real_col: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Map uniform values ``u`` back to the scale of ``real_col`` via linear
    interpolation over its sorted empirical quantiles. Preserves the marginal
    distribution exactly (output values lie on the real support)."""
    sv = np.sort(np.asarray(real_col, dtype=np.float64))
    if sv.size == 0:
        return np.zeros_like(u)
    if sv.size == 1:
        return np.full_like(u, sv[0])
    pos = np.clip(u, 0.0, 1.0) * (sv.size - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, sv.size - 1)
    frac = pos - lo
    return sv[lo] * (1.0 - frac) + sv[hi] * frac


def _copula_uniforms(
    source: np.ndarray, n: int, rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` correlated uniform rows from a Gaussian copula fit on
    ``source`` (shape (m, p)), for ALL columns jointly — continuous and discrete.

    Returns an (n, p) matrix of uniforms in (0, 1); mapping each uniform back to
    its column's marginal (continuous → empirical-quantile inverse; discrete →
    inverse-CDF on the frequency table) is the caller's job. Handling every
    column jointly here is what lets a mixed continuous/discrete batch keep the
    cross-correlation between the two kinds of feature.

    The copula factors a joint distribution into (marginals) × (dependence):
      1. Map each source column to standard-normal scores (``_normal_scores``) →
         the latent Gaussian space where dependence is a plain correlation. Rank
         scores work for continuous *and* ordinal-coded discrete columns, and a
         discrete column with no real association leaves the estimated latent
         correlation ≈ 0, so no spurious dependence is manufactured.
      2. Estimate that latent correlation and draw ``n`` fresh latent rows from
         it. With <2 rows or a degenerate/non-finite correlation, fall back to
         the identity (independent columns): marginals still match exactly, only
         cross-column correlation is dropped.
      3. Push each latent column through Φ to a uniform.

    No real row is ever emitted: the uniforms are drawn from a fitted latent, and
    the caller's marginal lookup selects sorted real values / frequency-table
    codes by an independently drawn latent rank.
    """
    from scipy.stats import norm

    m, p = source.shape
    if p == 0:
        return np.zeros((n, 0), dtype=np.float64)

    # 1. Latent normal scores per column.
    Z = np.column_stack([_normal_scores(source[:, c]) for c in range(p)])

    # 2. Latent correlation + fresh draws. np.corrcoef needs ≥2 rows and varying
    #    columns; guard both and ridge-regularize so the Cholesky/eigendecomp in
    #    multivariate_normal stays well-conditioned.
    corr = np.eye(p)
    if m >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(Z, rowvar=False)
        c = np.atleast_2d(c)
        if c.shape == (p, p) and np.all(np.isfinite(c)):
            corr = 0.999 * c + 0.001 * np.eye(p)  # pull toward identity
    L = rng.multivariate_normal(np.zeros(p), corr, size=n)

    # 3. Latent → uniform.
    return norm.cdf(L)


def _discrete_inverse_cdf(freq: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Map uniforms ``u`` ∈ [0,1] to discrete codes via the inverse CDF of the
    frequency table ``freq`` (probabilities over codes 0..K-1).

    ``code = min{ k : cumsum(freq)[k] ≥ u }``. This reproduces the marginal
    frequency exactly (same distribution the old independent sampler produced),
    while the *ordering* of the draw is governed by the shared copula latent — so
    a discrete column now co-varies with the continuous ones it was correlated
    with, instead of being sampled in isolation.
    """
    cum = np.cumsum(np.asarray(freq, dtype=np.float64))
    if cum.size:
        cum[-1] = 1.0  # guard fp drift so u≈1 lands on the last code, not past it
    codes = np.searchsorted(cum, np.clip(u, 0.0, 1.0), side="left")
    return np.clip(codes, 0, freq.size - 1).astype(np.float64)


def _fit_discrete_freq(
    X: np.ndarray, disc_idx: np.ndarray, feature_cols: List[str], field_dict,
) -> Dict[str, np.ndarray]:
    """Per-class empirical frequency table for each discrete (categorical/binary)
    column, over its canonical code range.

    Sampling from these tables (instead of copying an anchor's value) is what
    stops categorical values being reproduced verbatim — the strongest
    re-identification signal in the old grounded sampler. Unseen codes get a
    small epsilon so the sampler never starves a real category. Length matches
    the canonical category count where known, so sampled codes decode cleanly.
    """
    out: Dict[str, np.ndarray] = {}
    if disc_idx.size == 0:
        return out
    Xn = np.asarray(X, dtype=np.float64)
    for j in disc_idx:
        col = feature_cols[j]
        codes = np.rint(Xn[:, j]).astype(int)
        meta = field_dict.get(col) if field_dict is not None else None
        if meta is not None and getattr(meta, "categories", None) is not None:
            n_codes = len(meta.categories)
        elif meta is not None and meta.field_type == FieldType.BINARY:
            n_codes = 2
        else:
            n_codes = int(codes.max()) + 1 if codes.size else 1
        n_codes = max(n_codes, 1)
        counts = np.bincount(
            np.clip(codes, 0, n_codes - 1), minlength=n_codes,
        ).astype(np.float64)
        # Floor every category so a class that never shows a value can still
        # produce it rarely (smooths the tail, avoids zero-probability traps).
        counts = counts + 0.5
        out[col] = counts / counts.sum()
    return out


def _encode_features(df: pd.DataFrame, field_dict=None) -> np.ndarray:
    """Convert DataFrame to float32 array. Categorical → label-encoded.

    When field_dict is provided, categorical columns are encoded against their
    canonical category order (computed once from the full dataset at ingest), so
    the same string maps to the same code whether it appears in the normal rows,
    the rare rows, or a synthetic batch. Without it, codes are derived per-call
    from whatever values are present — fine for a single self-consistent frame,
    but not comparable across subsets (the old categorical-decode bug).
    """
    out = df.copy()
    for col in out.columns:
        is_cat = out[col].dtype == object or str(out[col].dtype) == "category"
        if is_cat:
            cats = None
            if field_dict is not None and col in field_dict:
                cats = getattr(field_dict[col], "categories", None)
            if cats is not None:
                # Unseen values → code -1; the GP treats it as an out-of-vocab code.
                out[col] = pd.Categorical(out[col], categories=cats).codes.astype(np.float32)
            else:
                out[col] = pd.Categorical(out[col]).codes.astype(np.float32)
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
