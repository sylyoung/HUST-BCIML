# ===========================================================================
# RA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Reference implementation: https://github.com/pyRiemann/pyRiemann
#
# Reference (IEEE BibTeX):
#   @Article{Zanini2018,
#     author  = {Zanini, Paolo and Congedo, Marco and Jutten, Christian and Said, Salem and Berthoumieu, Yannick},
#     journal = {IEEE Trans. Biomedical Engineering},
#     title   = {Transfer Learning: A {R}iemannian Geometry Framework with Applications to Brain-Computer Interfaces},
#     year    = {2018},
#     number  = {5},
#     pages   = {1107-1116},
#     volume  = {65},
#     doi     = {10.1109/TBME.2017.2742541},
#   }
# ===========================================================================
"""Riemannian Alignment (Zanini et al., 2018, IEEE TBME).

Like Euclidean Alignment, whiten each subject's trials by the inverse square
root of a reference covariance so that the subject's average covariance is
re-centred to the identity — but the reference is the **affine-invariant
Riemannian geometric mean** of the per-trial covariances (the Fréchet mean on
the SPD manifold) rather than their arithmetic mean. Recentring by the
Riemannian mean respects the curved geometry of covariance matrices.

Whitening the raw signal by ``R^{-1/2}`` maps each trial covariance
``C_i -> R^{-1/2} C_i R^{-1/2}``, whose Riemannian mean is the identity — the
same recentring the classical Riemannian-alignment pipeline applies to
covariances directly, here expressed on the signal so the rest of the pipeline
is unchanged.

The Riemannian mean is computed self-contained (no pyriemann dependency): the
standard fixed-point iteration on the tangent space (``R <- R^{1/2}
exp(mean_i log(R^{-1/2} C_i R^{-1/2})) R^{1/2}``).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm, fractional_matrix_power, logm

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.stages import Aligner


def _sqrtm(M: np.ndarray, power: float) -> np.ndarray:
    return np.real(fractional_matrix_power(M, power))


def _riemann_mean(covs: np.ndarray, tol: float = 1e-4, max_iter: int = 30) -> np.ndarray:
    """Affine-invariant Riemannian mean of SPD matrices ``covs:(n, C, C)``."""
    R = np.mean(covs, axis=0)                 # arithmetic mean is a good init
    for _ in range(max_iter):
        R_isqrt = _sqrtm(R, -0.5)
        R_sqrt = _sqrtm(R, 0.5)
        # mean of the trials mapped to R's tangent space
        S = np.zeros_like(R)
        for c in covs:
            S += np.real(logm(R_isqrt @ c @ R_isqrt))
        S /= len(covs)
        R = R_sqrt @ np.real(expm(S)) @ R_sqrt
        if np.linalg.norm(S, ord="fro") < tol:
            break
    return R


def _reference_inv_sqrt(X: np.ndarray) -> np.ndarray:
    """R^{-1/2} for one subject's trials X:(n, C, T), R = Riemannian mean."""
    C = X.shape[1]
    covs = np.empty((X.shape[0], C, C))
    ridge = None
    for i in range(X.shape[0]):
        cov = np.cov(X[i])
        if ridge is None:                     # light ridge keeps logm well-defined
            ridge = 1e-6 * np.trace(cov) / C * np.eye(C)
        covs[i] = cov + ridge
    R = _riemann_mean(covs)
    return _sqrtm(R, -0.5)


class RA(Aligner):
    requires_labels = False
    supports_online = False          # offline reference only (Riemannian mean)

    def __init__(self, **_):
        self._inv_sqrt = {}          # domain id -> R^{-1/2}

    def fit(self, epochs: EEGEpochs) -> "RA":
        self._inv_sqrt = {}
        for d in epochs.domains():
            Xd = epochs.X[epochs.domain == d]
            self._inv_sqrt[int(d)] = _reference_inv_sqrt(Xd)
        return self

    def transform(self, epochs: EEGEpochs) -> EEGEpochs:
        X = epochs.X.copy()
        for d in epochs.domains():
            mask = epochs.domain == d
            W = self._inv_sqrt.get(int(d))
            if W is None:                      # unseen domain -> fit on the fly
                W = _reference_inv_sqrt(epochs.X[mask])
            Xd = epochs.X[mask]
            X[mask] = np.real(np.matmul(W[None, :, :], Xd))
        return epochs.with_X(X)
