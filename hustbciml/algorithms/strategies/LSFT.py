# ===========================================================================
# LSFT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Zhang2023,
#     author  = {Zhang, Wen and Wu, Dongrui},
#     journal = {IEEE Trans. Cognitive and Developmental Systems},
#     title   = {Lightweight Source-Free Transfer for Privacy-Preserving Motor Imagery Classification},
#     year    = {2023},
#     number  = {2},
#     pages   = {938-949},
#     volume  = {15},
#     doi     = {10.1109/TCDS.2022.3193731},
#   }
# ===========================================================================
"""LSFT — Lightweight Source-Free Transfer (Zhang & Wu, IEEE TCDS 2022).

A lab source-free, privacy-preserving transfer method. No source raw data is
kept at transfer time: pre-trained source classifiers vote to pseudo-label the
target, the confident (low-disagreement) target trials form a *virtual
intermediate source*, and an iterative JPDA/DJP-MMD subspace adaptation with
pseudo-label refinement produces the final target predictions. Everything runs
on classical Riemannian tangent-space features — the neural backbone is unused.

Ported from the authors' ``LSFT/main_demo_LSFT_voting.py`` (the CA+LSFT lda+lr
configuration). It maps onto hustbciml's fit/predict as:

* ``fit``  — Riemannian-align each source subject, tangent-space map, and train
  the two voting classifiers (shrinkage LDA + logistic regression). The source
  raw data is then discarded (source-free).
* ``predict`` — Riemannian-align and tangent-map the target, vote to soft
  labels, build the virtual intermediate source from low-std target trials, and
  iterate ``feature_adaptation`` (DJP-MMD subspace) + shrinkage-LDA relabeling.

``mode='fit'`` (classical, no gradient loop); LSFT does its own Riemannian
alignment, so the preset's aligner stage is Identity. Robustness guards, kept
minimal and disclosed in the card, cover degenerate virtual-source selection on
tiny or easy sets. Requires pyriemann + scikit-learn (imported lazily).
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
    """Per-subject Riemannian-align the source, concatenate, tangent-space map."""
    aligned = [ra_align(source.X[source.domain == d].astype(np.float64))
               for d in source.domains()]
    labels = [source.y[source.domain == d] for d in source.domains()]
    X = np.concatenate(aligned, axis=0)
    y = np.concatenate(labels, axis=0)
    return tangent_features(X), y


def _virtual_source(feats: np.ndarray, votes: np.ndarray, n_classes: int,
                    std_th: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Confident target trials -> virtual intermediate source (Xs_mid, Ys_mid).

    Select trials whose per-class std across the voting models is below
    ``std_th``. Guard (disclosed): if that leaves too few trials or misses a
    class, fall back to the lowest-std trials until each class appears and at
    least ``max(2*n_classes, 10)`` trials are kept — so tiny/easy sets don't
    collapse the intermediate domain to one class.
    """
    vote_mean = votes.mean(axis=0)                       # (nt, C)
    ins_std = votes.std(axis=0).mean(axis=1)             # (nt,) mean per-class disagreement
    y_vote = vote_mean.argmax(axis=1)
    keep = np.where(ins_std < std_th)[0]

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
        self.dim = dim
        self.mu = mu
        self.n_iter = n_iter
        self._voters = None

    def fit(self, model, source: EEGEpochs, ctx: RunContext):
        from sklearn.linear_model import LogisticRegression
        hp = ctx.cfg.hp
        self.dim = int(hp.get("lsft_dim", self.dim))          # JPDA subspace dimension
        self.mu = float(hp.get("lsft_mu", self.mu))           # DJP-MMD discriminability weight
        self.n_iter = int(hp.get("lsft_niter", self.n_iter))  # relabel/adapt iterations
        Xs, Ys = _source_features(source)
        lda = _shrinkage_lda().fit(Xs, Ys)
        lr = LogisticRegression(penalty="l2", max_iter=500).fit(Xs, Ys)
        self._voters = [lda, lr]                          # mdl_idx = [LDA, LR]
        self._n_classes = source.n_classes
        return model                                      # neural model unused

    def predict(self, model, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        Xt = tangent_features(ra_align(target.X.astype(np.float64)))
        votes = np.array([clf.predict_proba(Xt) for clf in self._voters])   # (2, nt, C)

        Xs_mid, Ys_mid = _virtual_source(Xt, votes, self._n_classes)
        y_pseudo = votes.mean(axis=0).argmax(axis=1)
        dim = min(self.dim, Xt.shape[1] - 1)

        y_score = votes.mean(axis=0)                       # fallback if adaptation degenerates
        for _ in range(self.n_iter):
            Z = feature_adaptation(Xs_mid, Ys_mid, Xt, y_pseudo, dim=dim, mu=self.mu)
            ns = len(Xs_mid)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
            clf = _shrinkage_lda().fit(Xs_new, Ys_mid)
            y_pseudo = clf.predict(Xt_new)
            y_score = clf.predict_proba(Xt_new)
        return np.asarray(y_pseudo, dtype=np.int64), y_score
