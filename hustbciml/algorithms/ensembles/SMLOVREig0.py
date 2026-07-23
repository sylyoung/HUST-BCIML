# ===========================================================================
# SMLOVREig0.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Verification-only cross-check for SML-OVR's eigenvector selection.
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
# ===========================================================================
"""SML-OVR computed with the reference's first-returned eigenvector (diagnostic).

This is ``SMLOVR`` with one line changed: instead of the *leading* eigenvector
(``eigh`` then argmax of the eigenvalues, which is what the paper's Algorithm 1
specifies and what ``SMLOVR`` uses), it takes the first-returned eigenvector of
``np.linalg.eig(Q)`` exactly as the offline TestEnsemble reference wrote it. Its
sole purpose is verification: run it alongside ``SMLOVR`` on the real predictions
to confirm the two eigenvector selections agree. It is not part of the default
combiner suite and is not a benchmark row.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import Combiner

from ._common import onehot


class SMLOVREig0(Combiner):
    """Diagnostic twin of ``SMLOVR`` using ``eig(Q)[1][:, 0]`` (first-returned,
    not sorted) instead of the leading eigenvector. Verification-only."""

    name = "SML-OVR-eig0"

    def combine(self, scores: np.ndarray) -> np.ndarray:
        preds = scores.argmax(axis=2)
        C = scores.shape[2]
        oh = onehot(preds, C)
        weights_all = []
        for i in range(C):
            pred = np.where(oh.argmax(-1) == i, 1.0, -1.0)
            mu = pred.mean(axis=1)
            dev = pred - mu[:, None]
            Q = dev @ dev.T / (pred.shape[1] - 1)
            v = np.linalg.eig(Q)[1][:, 0].real          # reference: first-returned eigenvector
            if v[0] <= 0:
                v = -v
            weights_all.append(v / np.sum(v))
        wf = np.sum(np.array(weights_all), axis=0)
        return np.argmax(np.einsum("a,abc->bc", wf, oh), axis=1)
