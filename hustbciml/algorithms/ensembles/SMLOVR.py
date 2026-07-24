# ===========================================================================
# SMLOVR.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# SML-OVR — the lab's one-vs-rest multi-class spectral meta-learner (label-free, hyperparameter-free).
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
"""SML-OVR — one-vs-rest multi-class Spectral Meta-Learner (Li et al., 2026).

The binary Spectral Meta-Learner (SML; Parisi et al., 2014) scores each base
classifier's reliability with NO labels: under conditional independence, the
leading eigenvector of the K x K covariance of the classifiers' +/-1 predictions
has entries proportional to (2*BCA - 1), where BCA is a classifier's balanced
classification accuracy, so the eigenvector doubles as a weight vector that favors
the more accurate models. Plain SML is defined only for two classes.

"Black-Box Test-Time Ensemble" (Li et al., 2026, IEEE CIM; Algorithm 1) lifts SML
to K > 2 classes with a one-vs-rest split: for each class k the K models' votes are
recoded to +/-1 (class k vs. the rest), the leading eigenvector v_k of that
subtask's vote-covariance is taken (Lemma 2 holds per class), and the per-class
eigenvectors are each sum-normalized and averaged into a single reliability vector
v-bar (Eq. 12). The prediction is the argmax over classes of the v-bar-weighted
one-hot votes (Eq. 13). It is hyperparameter-free and needs no ground truth.

For K = 2 the two one-vs-rest subtasks are mirror images, so SML-OVR collapses to
the binary SML -- which is why SML and SML-OVR post identical numbers on the
two-class motor-imagery datasets, and SML-OVR only pulls ahead on natively
multi-class data.

The leading eigenvector is taken with ``eigh`` (argmax over eigenvalues), as
Algorithm 1 specifies. This was cross-checked against the TestEnsemble reference's
first-returned ``np.linalg.eig`` eigenvector and the two agree on the benchmark
predictions, so only this paper-faithful version is kept.
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
        wf = np.sum(np.array(weights_all), axis=0)       # v-bar (Eq. 12): sum of the per-class normalized eigenvectors; the 1/K averaging factor is a global scale that does not change the argmax below
        # Predict argmax over classes of the reliability-weighted one-hot votes.
        return np.argmax(np.einsum("a,abc->bc", wf, oh), axis=1)
