# ===========================================================================
# SMLOVR.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# SML one-vs-rest — the lab's multi-class spectral meta-learner.
# Original authors' code: https://github.com/sylyoung/TestEnsemble
#
# References (IEEE BibTeX):
#   @Article{Li2026b,
#     author  = {Li, Siyang and Wang, Ziwei and Liu, Chenhao and Wu, Dongrui},
#     journal = {IEEE Computational Intelligence Magazine},
#     title   = {Black-Box Test-Time Ensemble},
#     year    = {2026},
#     number  = {1},
#     pages   = {57-68},
#     volume  = {21},
#     doi     = {10.1109/MCI.2025.3624194},
#   }
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
"""SML one-vs-rest — the lab's multi-class generalization of the binary SML.

The binary Spectral Meta-Learner (Parisi et al. 2014) only handles two classes.
This lab-proposed method (Li et al. 2026, "Black-Box Test-Time Ensemble",
Algorithm 1 / Eqs. 12-13) extends it to any number of classes by a one-vs-rest
decomposition: for each class it forms the one-vs-rest ±1 votes, takes the leading
eigenvector of the model vote-covariance as the per-class model weights, and
averages the per-class weightings into one reliability vector, then predicts the
argmax of the reliability-weighted votes. For two classes it reduces exactly to the
binary SML, which is why the two report identical accuracy on the two-class tasks;
the multi-class advantage only appears on native multi-class data.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import Combiner

from ._common import onehot, principal_eigvec


class SMLOVR(Combiner):
    """The lab's one-vs-rest spectral meta-learner (multi-class SML).

    For each class ``i`` it runs the binary-SML weight estimation on the one-hot
    votes and sums the per-class weightings, so it also handles more than two
    classes (for example the native four-class BNCI2014001, which the code still
    supports). On two-class tasks it reduces exactly to the binary ``SML``, so the
    two report identical accuracy there.
    """

    name = "SML-OVR"
    lab_proposed = True

    def combine(self, scores: np.ndarray) -> np.ndarray:
        preds = scores.argmax(axis=2)                   # (K, N)
        C = scores.shape[2]
        oh = onehot(preds, C)                            # (K, N, C)
        weights_all = []
        for i in range(C):                               # one binary SML per class (one-vs-rest)
            pred = np.where(oh.argmax(-1) == i, 1.0, -1.0)  # (K, N): did each model vote class i?
            mu = pred.mean(axis=1)
            dev = pred - mu[:, None]
            Q = dev @ dev.T / (pred.shape[1] - 1)        # (K, K) one-vs-rest vote covariance
            v = principal_eigvec(Q)                      # per-class model weights
            if v[0] <= 0:                                # fix global sign (assume model 0 > chance)
                v = -v
            weights_all.append(v / np.sum(v))            # normalize so the per-class weights sum to 1
        wf = np.sum(np.array(weights_all), axis=0)       # v-bar: average reliability across classes
        # Predict argmax over classes of the reliability-weighted one-hot votes.
        return np.argmax(np.einsum("a,abc->bc", wf, oh), axis=1)
