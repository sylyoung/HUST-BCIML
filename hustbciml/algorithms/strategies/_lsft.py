# ===========================================================================
# _lsft.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Zhang2023,
#     author  = {Zhang, Wen and Wu, Dongrui},
#     journal = {IEEE Transactions on Cognitive and Developmental Systems},
#     title   = {Lightweight Source-Free Transfer for Privacy-Preserving Motor Imagery Classification},
#     year    = {2023},
#     number  = {2},
#     pages   = {938-949},
#     volume  = {15},
#     doi     = {10.1109/TCDS.2022.3193731},
#   }
# ===========================================================================
"""LSFT helpers — Riemannian alignment, tangent-space features, and the
JPDA/DJP-MMD subspace adaptation (Zhang & Wu, IEEE TCDS 2022).

Vendored and refactored from the authors' ``LSFT/{data_align,dataloader,
feature_adaptation}.py``. All three pieces operate on classical Riemannian
features, no neural network:

* ``ra_align`` — Riemannian centroid alignment (``centroid_align`` with
  ``center_type='riemann'``): re-reference every trial by R^{-1/2}, R the
  Riemannian mean covariance of the (single-subject) set. LSFT's own alignment,
  so the hustbciml aligner stage is Identity.
* ``tangent_features`` — Ledoit-Wolf covariances mapped to the Riemannian
  tangent space, giving one C(C+1)/2 vector per trial.
* ``feature_adaptation`` — a JDA/JPDA subspace: solve the generalized
  eigenproblem ``(K M Kᵀ + λI) A = (K H Kᵀ) A Λ`` with the DJP-MMD joint
  matrix ``M = R_min − μ R_max``, keep the ``dim`` smallest-eigenvalue vectors,
  and project both domains into the shared subspace.

Disclosed numerical fixes vs the source (see the card): (1) the generalized
eigenproblem is solved with ``scipy.linalg.eig`` exactly as the source, but the
selected eigenvectors are cast to their real part before projecting — the pair
(a, b) is real-symmetric so the eigenpairs are real up to numerical noise, and
the source relied on that implicitly; (2) eigenvalues are ordered by real part.

Underscore-prefixed so the registry auto-scan skips it (a helper, not a plug-in).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from sklearn.preprocessing import OneHotEncoder


# ----------------------------------------------------------------- features ---
def ra_align(X: np.ndarray, cov_type: str = "lwf") -> np.ndarray:
    """Riemannian centroid alignment of one subject's trials ``(N, C, T)``.

    Re-reference each trial by R^{-1/2}, R the Riemannian mean of the trial
    covariances — the ``centroid_align(center_type='riemann')`` of the source.
    """
    from pyriemann.utils.covariance import covariances
    from pyriemann.utils.mean import mean_covariance
    from scipy.linalg import fractional_matrix_power

    cov = covariances(X, estimator=cov_type)
    ref = fractional_matrix_power(mean_covariance(cov, metric="riemann"), -0.5)
    ref = np.real(ref)
    return np.stack([ref @ X[j] for j in range(len(X))], axis=0)


def tangent_features(X: np.ndarray, cov_type: str = "lwf") -> np.ndarray:
    """Ledoit-Wolf covariances -> Riemannian tangent-space vectors ``(N, C(C+1)/2)``."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    cov = Covariances(estimator=cov_type).transform(X)
    return TangentSpace().fit_transform(cov)


# ------------------------------------------------------------- DJP-MMD matrix ---
def _matrix_M(Ys: np.ndarray, Yt_pseudo, ns: int, nt: int, C: int, mu: float) -> np.ndarray:
    """DJP-MMD joint matrix ``M = R_min - mu * R_max`` (each block Frobenius-normalized).

    ``R_min`` (transferability) couples same-class source/target; ``R_max``
    (discriminability) couples each class against the other C-1. Faithful to the
    authors' ``get_matrix_M(..., mmd_type='djp-mmd')``.
    """
    ohe = OneHotEncoder()
    ohe.fit(np.unique(Ys).reshape(-1, 1))
    Ns = ohe.transform(Ys.reshape(-1, 1)).toarray().astype(np.float64) / ns
    Nt = np.zeros([nt, C])
    if Yt_pseudo is not None:
        Nt = ohe.transform(Yt_pseudo.reshape(-1, 1)).toarray().astype(np.float64) / nt

    Rmin = np.r_[np.c_[Ns @ Ns.T, -Ns @ Nt.T], np.c_[-Nt @ Ns.T, Nt @ Nt.T]]
    Rmin = Rmin / np.linalg.norm(Rmin, "fro")

    Ms = np.zeros([ns, (C - 1) * C])
    Mt = np.zeros([nt, (C - 1) * C])
    for i in range(C):
        idx = np.arange((C - 1) * i, (C - 1) * (i + 1))
        Ms[:, idx] = np.tile(Ns[:, i], (C - 1, 1)).T
        other = np.arange(C)
        Mt[:, idx] = Nt[:, other[other != i]]
    Rmax = np.r_[np.c_[Ms @ Ms.T, -Ms @ Mt.T], np.c_[-Mt @ Ms.T, Mt @ Mt.T]]
    Rmax = Rmax / np.linalg.norm(Rmax, "fro")
    return Rmin - mu * Rmax


def feature_adaptation(Xs: np.ndarray, Ys: np.ndarray, Xt: np.ndarray, Yt_pseudo,
                       dim: int = 20, lamb: float = 1.0, mu: float = 0.1) -> np.ndarray:
    """Learn the shared subspace and return ``Z`` (dim, ns+nt), column-normalized.

    Split downstream as ``Xs_new = Z[:, :ns].T``, ``Xt_new = Z[:, ns:].T``.
    ``kernel_type='primal'`` (K = X), matching the LSFT demo.
    """
    X = np.hstack((Xs.T, Xt.T))
    X = X @ np.diag(1.0 / np.linalg.norm(X, axis=0))
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)
    C = len(np.unique(Ys))
    H = np.eye(n) - 1.0 / n * np.ones((n, n))

    M = _matrix_M(Ys, Yt_pseudo, ns, nt, C, mu)
    K = X                                              # primal kernel
    a = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(m)
    b = np.linalg.multi_dot([K, H, K.T])
    w, V = scipy.linalg.eig(a, b)
    order = np.argsort(w.real)
    A = np.real(V[:, order[:dim]])                     # real subspace (see module docstring)
    Z = A.T @ K
    Z = Z @ np.diag(1.0 / np.linalg.norm(Z, axis=0))
    return Z
