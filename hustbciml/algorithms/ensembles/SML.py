# ===========================================================================
# SML.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Binary Spectral Meta-Learner. Original authors' code:
# https://github.com/sylyoung/TestEnsemble
#
# References (IEEE BibTeX):
#   @Article{Parisi2014,
#     author  = {Parisi, Fabio and Strino, Francesco and Nadler, Boaz and Kluger, Yuval},
#     journal = {Proceedings of the National Academy of Sciences},
#     title   = {Ranking and Combining Multiple Predictors Without Labeled Data},
#     year    = {2014},
#     number  = {4},
#     pages   = {1253-1258},
#     volume  = {111},
#     doi     = {10.1073/pnas.1219097111},
#   }
# ===========================================================================
"""Binary Spectral Meta-Learner (Parisi et al. 2014).

An unsupervised way to weight base models with no labels, valid for two classes.
Map each model's ±1 votes to a mean-centered vector, form the K x K vote
covariance, and take its leading eigenvector as the per-model weight vector: for
conditionally independent models better than chance, that eigenvector is
proportional to their balanced accuracies, so more accurate models get more say.
The weighted sign of the votes is the consensus. This is the binary base that the
lab's SML-OVR generalizes to more classes; on two-class tasks the two coincide.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import Combiner

from ._common import principal_eigvec


class SML(Combiner):
    """Binary spectral meta-learner: weight = principal eigenvector of the ±1
    vote covariance. Defined for two classes only (``binary_only``)."""

    name = "SML"
    binary_only = True

    def combine(self, scores: np.ndarray) -> np.ndarray:
        preds = scores.argmax(axis=2)                   # (K, N) in {0,1}
        # Map {0,1} class votes to {-1,+1} so the covariance below measures
        # agreement in sign, the form Parisi et al. (2014) derive the eigenvector
        # relation for.
        pred = np.where(preds == 1, 1.0, -1.0)
        mu = pred.mean(axis=1)                           # per-model vote bias
        dev = pred - mu[:, None]
        Q = dev @ dev.T / (pred.shape[1] - 1)            # (K, K) vote covariance
        # Leading eigenvector of Q ~ the models' balanced accuracies (the SML weights).
        v = principal_eigvec(Q)
        if np.any(v < 0):                                # fix global sign: weights are accuracies (>=0)
            v = np.abs(v)
        # Consensus = sign of the accuracy-weighted vote sum, mapped back to {0,1}.
        return np.where(np.einsum("a,ab->b", v, pred) >= 0, 1, 0)
