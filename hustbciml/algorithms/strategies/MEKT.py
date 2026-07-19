# ===========================================================================
# MEKT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/chamwen/MEKT
# Ported via: https://github.com/TBC-TJU/MetaBCI
#
# Reference (IEEE BibTeX):
#   @Article{Zhang2020a,
#     author  = {Zhang, Wen and Wu, Dongrui},
#     journal = {IEEE Trans. Neural Systems and Rehabilitation Engineering},
#     title   = {Manifold Embedded Knowledge Transfer for Brain-Computer Interfaces},
#     year    = {2020},
#     number  = {5},
#     pages   = {1117-1127},
#     volume  = {28},
#     doi     = {10.1109/TNSRE.2020.2985996},
#   }
# ===========================================================================
"""MEKT — Manifold Embedded Knowledge Transfer (Zhang & Wu, IEEE TNSRE 2020).

A classical, network-free vanilla-transfer method (mode='fit'): no neural
network is trained. Offline, unsupervised, cross-subject transfer on Riemannian
tangent-space features. This is now the FULL method (paper Algorithm 1), both
parts:

  FEATURE EXTRACTION (steps 1-2)
    1. Covariance-matrix Centroid Alignment (CA): per source subject, align each
       covariance P by its own Riemannian reference mean, P' = M^{-1/2} P M^{-1/2}
       (Eq. 14, M = Riemannian mean -> MEKT-R, the paper's best variant); the
       target is CA'd by its own mean. Whitens each domain to a common identity
       centroid.
    2. Tangent-space features: x = upper(log P') (Eqs. 19/20), dim d = c(c+1)/2
       (253 for 22 channels). Source subjects are CA+tangent'd individually then
       pooled into one source matrix.

  DOMAIN-ADAPTATION PROJECTION (step 3, Sec. III-C)  <-- previously omitted
    Learn two low-dimensional projections A (source) and B (target) that jointly
      * minimise the joint-probability distribution shift between source and
        (pseudo-labelled) target class-conditional means  [R],
      * preserve source separability            [alpha * within/between scatter],
      * preserve target locality                [beta * heat-kernel graph Laplacian],
      * couple A~B for parameter transfer       [rho * Q],
    solved as a symmetric-definite generalised eigenproblem eigh(Emin, Emax); the
    d SMALLEST generalised eigenvectors give [A;B]. Pseudo-labels are refined for
    max_iter EM rounds (init = shrinkage-LDA on the raw tangent features, i.e. the
    method's own feature-extraction core). Source is classified by a shrinkage-LDA
    trained on Xs@A and applied to Xt@B.

Provenance: the step-3 projection is ported faithfully from the authors' source
(chamwen/MEKT) via TBC-TJU/MetaBCI's ``mekt_kernel`` (MIT). Defaults match that
reference: subspace_dim=10, max_iter=5, alpha=0.01, beta=0.1, rho=20, k=10, t=1,
Ledoit-Wolf ('lwf') covariance. Two hustbciml specifics: source CA is done
per-subject (each source subject aligned to its own identity before pooling — a
multi-source refinement over MetaBCI's single pooled mean), and the projection
uses target features that are only ever unlabelled (pseudo-labels come from the
classifier, never target.y — no label leakage). Hyperparameters are overridable
from ``ctx.cfg.hp`` (mekt_dim / mekt_alpha / mekt_beta / mekt_rho / mekt_iter /
mekt_k / mekt_cov) for the tuning campaign. Requires pyriemann + scikit-learn +
scipy.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy


def _tangent_features(X: np.ndarray, covariance_type: str = "lwf") -> np.ndarray:
    """Covariances -> per-domain CA (Riemannian mean) -> tangent space (Eqs. 14/19)."""
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.tangentspace import tangent_space
    covs = Covariances(estimator=covariance_type).transform(X.astype(np.float64))  # (n, c, c) SPD
    M = mean_riemann(covs)
    W = invsqrtm(M)                                                        # M^{-1/2}
    aligned = np.stack([W @ P @ W for P in covs])                         # CA (Eq. 14)
    c = covs.shape[-1]
    return tangent_space(aligned, np.eye(c))                              # (n, d), d=c(c+1)/2


def _onehot(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """One-hot over a FIXED class set (columns stay consistent even if a
    pseudo-label round misses a class). Version-agnostic (no sklearn encoder)."""
    Y = np.zeros((len(y), len(classes)), dtype=np.float64)
    for j, c in enumerate(classes):
        Y[y == c, j] = 1.0
    return Y


def _source_discriminability(Xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Within- (Sw) and between-class (Sb) scatter of the source features (Sec. III-C)."""
    classes = np.unique(ys)
    n_features = Xs.shape[1]
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))
    mean_total = Xs.mean(axis=0, keepdims=True)
    for c in classes:
        Xi = Xs[ys == c, :]
        mean_class = Xi.mean(axis=0, keepdims=True)
        Sw += (Xi - mean_class).T @ (Xi - mean_class)
        Sb += len(Xi) * (mean_class - mean_total).T @ (mean_class - mean_total)
    return Sw, Sb


def _graph_laplacian(Xt: np.ndarray, k: int = 10, t: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Heat-kernel kNN graph Laplacian of the target features (target locality)."""
    from scipy.spatial.distance import pdist, squareform
    pair = squareform(pdist(Xt, metric="euclidean"))
    kk = min(k + 1, Xt.shape[0])                       # self-connection included (guard tiny sets)
    ix = np.argsort(pair, axis=-1)[:, :kk]
    heat = np.exp(-np.square(pair) / (2.0 * t * t))
    W = np.zeros_like(pair)
    for i, ind in enumerate(ix):
        W[i, ind] = heat[i, ind]
    W = np.maximum(W, W.T)                              # symmetrise
    D = np.diag(W.sum(axis=-1))
    return D - W, D


def _mekt_kernel(Xs: np.ndarray, Xt: np.ndarray, ys: np.ndarray, *, d: int = 10,
                 max_iter: int = 5, alpha: float = 0.01, beta: float = 0.1,
                 rho: float = 20.0, k: int = 10, t: float = 1.0,
                 log=lambda *_: None) -> Tuple[np.ndarray, np.ndarray]:
    """Joint source/target projections A, B (ported from MetaBCI ``mekt_kernel``).

    Minimise (joint-MMD R + alpha*Sw + beta*target-locality + rho*coupling)
    subject to (target variance + alpha*Sb + ridge): the d SMALLEST generalised
    eigenvectors of (Emin, Emax). Pseudo-labels are refined for ``max_iter`` EM
    rounds. Returns A:(f,d) for source, B:(f,d) for target.
    """
    from scipy.linalg import block_diag, eigh
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    ns, f = Xs.shape
    nt = Xt.shape[0]
    d = int(min(d, f))

    # source separability (Sw minimised, Sb preserved)
    Sw, Sb = _source_discriminability(Xs, ys)
    P = block_diag(Sw, np.zeros((f, f)))                       # source block only
    P0 = block_diag(Sb, np.zeros((f, f)))

    # target locality: normalised heat-kernel graph Laplacian, embedded in target block
    L, D = _graph_laplacian(Xt, k=k, t=t)
    iD12 = np.diag(1.0 / np.sqrt(np.clip(np.diag(D), 1e-12, None)))
    L = iD12 @ L @ iD12
    L = block_diag(np.zeros((f, f)), Xt.T @ L @ Xt)

    # parameter-transfer coupling A~B
    Q = np.block([[np.eye(f), -np.eye(f)], [-np.eye(f), 2.0 * np.eye(f)]])

    # target variance (centering scatter) on the 'max' side
    Ht = np.eye(nt) - np.ones((nt, nt)) / nt
    S = block_diag(np.zeros((f, f)), Xt.T @ Ht @ Xt)

    classes = np.sort(np.unique(ys))
    Ns = _onehot(ys, classes) / len(ys)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    yt = clf.fit(Xs, ys).predict(Xt)                          # init pseudo-labels = raw-tangent core

    Xblk = block_diag(Xs, Xt)                                 # (ns+nt, 2f)
    Emin_base = alpha * P + beta * L + rho * Q

    A = np.zeros((f, d))
    B = np.zeros((f, d))
    for it in range(max_iter):
        Nt = _onehot(yt, classes) / max(len(yt), 1)
        Mmmd = np.block([[Ns @ Ns.T, -Ns @ Nt.T], [-Nt @ Ns.T, Nt @ Nt.T]])   # joint-MMD (ns+nt sq)
        R = Xblk.T @ Mmmd @ Xblk                              # (2f, 2f)
        Emin = Emin_base + R
        Emin = 0.5 * (Emin + Emin.T)

        # symmetric-definite generalised eig; escalate the ridge if Emax is not PD
        V = None
        for ridge in (1e-3, 1e-2, 1e-1, 1.0):
            Emax = S + alpha * P0 + ridge * np.eye(2 * f)
            Emax = 0.5 * (Emax + Emax.T)
            try:
                _, V = eigh(Emin, Emax)                       # ascending eigenvalues
                break
            except np.linalg.LinAlgError:
                continue
        if V is None:
            log(f"  MEKT: eigh failed at iter {it}; keeping previous projection")
            break

        A = V[:f, :d]                                         # d SMALLEST eigenvectors
        B = V[f:, :d]
        Zs, Zt = Xs @ A, Xt @ B
        if not (np.all(np.isfinite(Zs)) and np.all(np.isfinite(Zt))):
            log(f"  MEKT: non-finite projection at iter {it}; keeping previous projection")
            A = np.zeros((f, d)) if it == 0 else A_prev
            B = np.zeros((f, d)) if it == 0 else B_prev
            break
        A_prev, B_prev = A, B
        yt = clf.fit(Zs, ys).predict(Zt)
    return A, B


class MEKT(Strategy):
    mode = "fit"
    uses_target = False        # transductive work happens in predict (target given there)

    def __init__(self, subspace_dim: int = 10, max_iter: int = 5, alpha: float = 0.01,
                 beta: float = 0.1, rho: float = 20.0, k: int = 10, t: float = 1.0,
                 covariance_type: str = "lwf", **_):
        self.subspace_dim = subspace_dim
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.k = k
        self.t = t
        self.covariance_type = covariance_type
        self._source = None

    def fit(self, model, source: EEGEpochs, ctx: RunContext):
        hp = getattr(ctx.cfg, "hp", {}) or {}
        self.subspace_dim = int(hp.get("mekt_dim", self.subspace_dim))
        self.max_iter = int(hp.get("mekt_iter", self.max_iter))
        self.alpha = float(hp.get("mekt_alpha", self.alpha))
        self.beta = float(hp.get("mekt_beta", self.beta))
        self.rho = float(hp.get("mekt_rho", self.rho))
        self.k = int(hp.get("mekt_k", self.k))
        self.t = float(hp.get("mekt_t", self.t))
        self.covariance_type = str(hp.get("mekt_cov", self.covariance_type))
        self._source = source            # stash; MEKT is transductive (projects at predict)
        return model

    def predict(self, model, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        source = self._source
        cov = self.covariance_type

        # per-source-subject CA + tangent, then pool
        ts_list, y_list = [], []
        for kdom in [int(dd) for dd in source.domains()]:
            e = source.select(source.domain == kdom)
            ts_list.append(_tangent_features(e.X, cov))
            y_list.append(e.y)
        ts_S = np.concatenate(ts_list, 0)                     # (n_S, f)
        y_S = np.concatenate(y_list, 0)
        ts_T = _tangent_features(target.X, cov)               # (n_T, f)  target CA'd to its own mean

        # step-3 domain-adaptation projection (Sec. III-C)
        A, B = _mekt_kernel(ts_S, ts_T, y_S, d=self.subspace_dim, max_iter=self.max_iter,
                            alpha=self.alpha, beta=self.beta, rho=self.rho, k=self.k, t=self.t,
                            log=ctx.log)
        Zs, Zt = ts_S @ A, ts_T @ B                           # (n_S, d), (n_T, d)

        clf = LDA(solver="lsqr", shrinkage="auto").fit(Zs, y_S)
        y_pred = clf.predict(Zt)
        y_score = clf.predict_proba(Zt)
        return np.asarray(y_pred, dtype=np.int64), y_score
