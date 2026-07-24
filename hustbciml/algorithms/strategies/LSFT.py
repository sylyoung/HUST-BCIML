# ===========================================================================
# LSFT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Zhang2023,
#     author  = {Zhang, Wen and Wu, Dongrui},
#     journal = {IEEE Transactions on Cognitive and Developmental Systems},
#     title   = {Lightweight Source-Free Transfer for Privacy-Preserving Motor Imagery Classification},
#     year    = {2023},
#     volume  = {15},
#     number  = {2},
#     pages   = {938-949},
#     doi     = {10.1109/TCDS.2022.3193731},
#   }
# ===========================================================================
"""LSFT — Lightweight Source-Free Transfer (Zhang & Wu, 2023, IEEE TCDS).

Task: offline unsupervised cross-subject motor-imagery (MI) classification with
labeled EEG from several source subjects but only UNLABELED EEG from the target
subject (Abstract; Sec. III-A). "Source-free / privacy-preserving": each source
subject's classifier is pre-trained locally and exposed only as a model API; at
transfer time LSFT queries just those APIs' output probabilities and never
touches source raw data, features, labels, model parameters, or model type
(contributions 1-3, Sec. I). "Lightweight": no deep source data/model is
transferred and the adaptation learns only a small mapping matrix A, so it is
computationally cheap (Table X reports the lowest runtime among the compared
transfer methods) with few parameters.

LSFT has three sequential steps (Fig. 2, Sec. III):

1. CA-based source-model pre-training (Sec. III-B): centroid-align (CA, Eq. 2)
   each source subject's covariances, map to the Riemannian tangent space (TSM,
   Eq. 1/3), and train Z source classifiers -- combined-data voting or Bagging
   (Eq. 4) -- encapsulated as APIs. SVM/LDA/LR are the suggested base learners,
   with at least two trained (Sec. III-F).
2. Virtual Intermediate Domain (VID) construction (Sec. III-C): query the APIs
   for target prediction probabilities, score each target trial by its Source
   Inconsistency SI (variance of the predictions across the Z APIs, Eq. 5), keep
   the trials with SI < eps (eps = maximum acceptable inconsistency), and
   soft-vote their labels (y = argmax of the averaged probabilities). These
   selected, pseudo-labeled target trials form the virtual intermediate domain
   D_v; because they come from the target, D_v is already close to the target
   distribution.
3. Feature-adaptation learning (Sec. III-D): reduce the conditional-distribution
   discrepancy between D_v and the target D_t via a mapping matrix A into a
   lower p-dimensional subspace, using joint-probability MMD (JP-MMD, Eq. 6;
   couples transferability and discriminability with trade-off mu) inside an
   unsupervised joint-probability domain-adaptation objective with a
   regularizer and PCA constraint (Eq. 7). Setting the gradient to zero gives a
   generalized eigen-problem (Eq. 10); A is the p trailing eigenvectors, and the
   pseudo-labels/A are refined for T iterations.

This file implements the experimental LDA+LR voting configuration of the
authors' reference code (``LSFT/main_demo_LSFT_voting.py``). It maps onto
hustbciml's fit/predict as:

* ``fit``  -- Step 1: CA-align each source subject, tangent-map, train the two
  base classifiers (shrinkage LDA + logistic regression). Source raw data is
  then discarded (source-free).
* ``predict`` -- CA-align/tangent-map the target, query the two classifiers for
  probabilities, select the low-SI target trials into the VID (``_virtual_source``,
  Eq. 5), and iterate ``feature_adaptation`` (JP-MMD subspace, Eq. 6/10) with
  shrinkage-LDA re-labeling (Step 3).

``mode='fit'`` (classical, deep-learning-free -- no gradient loop or neural
backbone, by design); LSFT does its own CA/Riemannian alignment, so the preset's
aligner stage is Identity. Robustness guards, kept minimal and disclosed in the
card, cover degenerate VID selection on tiny or easy target sets. Requires
pyriemann + scikit-learn (imported lazily).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._lsft import feature_adaptation, ra_align, tangent_features


def _shrinkage_lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def _source_features(source: EEGEpochs) -> Tuple[np.ndarray, np.ndarray]:
    """Step 1 features: per-subject centroid-align (CA), concatenate, tangent-map.

    Riemannian centroid alignment (CA, Eq. 2) is applied to each source subject
    separately, then all subjects' tangent-space vectors (TSM, Eq. 3) are
    stacked into the single source feature matrix on which the base classifiers
    are trained (Sec. III-B).
    """
    aligned = [ra_align(source.X[source.domain == d].astype(np.float64))
               for d in source.domains()]
    labels = [source.y[source.domain == d] for d in source.domains()]
    X = np.concatenate(aligned, axis=0)
    y = np.concatenate(labels, axis=0)
    return tangent_features(X), y


def _virtual_source(feats: np.ndarray, votes: np.ndarray, n_classes: int,
                    si_th: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Step 2 (Sec. III-C): select low-inconsistency target trials into the VID.

    Score each target trial by its Source Inconsistency SI (Eq. 5) -- the
    variance of the source-API predictions across the voting models -- keep the
    trials with SI < ``si_th`` (the paper's maximum acceptable inconsistency
    eps), and label them by soft voting (argmax of the averaged probabilities).
    They form the virtual intermediate domain D_v = (Xs_mid, Ys_mid).

    Guard (disclosed): if the threshold leaves too few trials or misses a class,
    fall back to the lowest-SI trials until each class appears and at least
    ``max(2*n_classes, 10)`` trials are kept -- so tiny/easy sets do not collapse
    D_v to a single class.
    """
    vote_mean = votes.mean(axis=0)                       # (nt, C) averaged API probabilities
    ins_std = votes.std(axis=0).mean(axis=1)             # (nt,) source inconsistency SI (Eq. 5)
    y_vote = vote_mean.argmax(axis=1)                    # soft-voted pseudo-labels
    keep = np.where(ins_std < si_th)[0]

    min_keep = max(2 * n_classes, 10)
    order = np.argsort(ins_std)
    if len(keep) < min_keep:
        keep = order[:min(min_keep, len(order))]
    present = set(np.unique(y_vote[keep]).tolist())
    for c in range(n_classes):                           # ensure every class is represented
        if c not in present:
            c_idx = order[y_vote[order] == c]
            if len(c_idx):
                keep = np.union1d(keep, c_idx[:max(2, min_keep // n_classes)])
    return feats[keep], y_vote[keep]


class LSFT(Strategy):
    mode = "fit"

    def __init__(self, dim: int = 20, mu: float = 0.1, n_iter: int = 10, **_):
        self.dim = dim         # p: feature-subspace dimensionality (paper default p = 20)
        self.mu = mu           # mu: JP-MMD transferability/discriminability trade-off (default 0.1)
        self.n_iter = n_iter   # T: feature-adaptation / relabeling iterations (default 10)
        self._voters = None

    def fit(self, model, source: EEGEpochs, ctx: RunContext):
        from sklearn.linear_model import LogisticRegression
        hp = ctx.cfg.hp
        self.dim = int(hp.get("lsft_dim", self.dim))          # p: subspace dimensionality (Eq. 7)
        self.mu = float(hp.get("lsft_mu", self.mu))           # mu: JP-MMD trade-off (Eq. 6)
        self.n_iter = int(hp.get("lsft_niter", self.n_iter))  # T: adaptation iterations (Algorithm 1)
        # Step 1 (Sec. III-B): train the source-API base classifiers on the
        # CA-aligned tangent features. Here the two suggested learners LDA + LR,
        # whose outputs are later voted (Sec. III-F, Table VII LDA & LR row).
        Xs, Ys = _source_features(source)
        lda = _shrinkage_lda().fit(Xs, Ys)
        lr = LogisticRegression(penalty="l2", max_iter=500).fit(Xs, Ys)
        self._voters = [lda, lr]                          # source model APIs = [LDA, LR]
        self._n_classes = source.n_classes
        return model                                      # deep backbone unused (deep-learning-free)

    def predict(self, model, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        # CA-align + tangent-map the target, then query the source-model APIs.
        Xt = tangent_features(ra_align(target.X.astype(np.float64)))
        votes = np.array([clf.predict_proba(Xt) for clf in self._voters])   # (2, nt, C) API probabilities

        # Step 2: build the virtual intermediate domain D_v from low-SI trials (Eq. 5).
        Xs_mid, Ys_mid = _virtual_source(Xt, votes, self._n_classes)
        y_pseudo = votes.mean(axis=0).argmax(axis=1)       # initial target pseudo-labels (soft vote)
        dim = min(self.dim, Xt.shape[1] - 1)

        # Step 3: iterate JP-MMD feature adaptation (Eq. 6/10) + LDA re-labeling.
        y_score = votes.mean(axis=0)                       # fallback if adaptation degenerates
        for _ in range(self.n_iter):
            Z = feature_adaptation(Xs_mid, Ys_mid, Xt, y_pseudo, dim=dim, mu=self.mu)
            ns = len(Xs_mid)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T      # A^T x for D_v and D_t
            clf = _shrinkage_lda().fit(Xs_new, Ys_mid)
            y_pseudo = clf.predict(Xt_new)                 # refine target pseudo-labels
            y_score = clf.predict_proba(Xt_new)
        return np.asarray(y_pseudo, dtype=np.int64), y_score
