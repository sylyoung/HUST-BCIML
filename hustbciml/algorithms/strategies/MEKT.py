# ===========================================================================
# MEKT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/chamwen/MEKT
# Ported via: https://github.com/TBC-TJU/MetaBCI
#
# Reference (IEEE BibTeX):
#   @Article{Zhang2020a,
#     author  = {Zhang, Wen and Wu, Dongrui},
#     journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
#     title   = {Manifold Embedded Knowledge Transfer for Brain-Computer Interfaces},
#     year    = {2020},
#     number  = {5},
#     pages   = {1117-1127},
#     volume  = {28},
#     doi     = {10.1109/TNSRE.2020.2985996},
#   }
# ===========================================================================
"""MEKT — Manifold Embedded Knowledge Transfer (Zhang & Wu, 2020).

Offline, unsupervised, cross-subject EEG classification: labeled trials from one
or more source subjects, only unlabeled trials from the target subject. No neural
network is trained. MEKT is the three-step procedure of Sec. III / Algorithm 1
(here in the paper: c = #channels, d = c(c+1)/2 = tangent-space dimensionality,
p = shared-subspace dimensionality with p << d; this code names the subspace
dimensionality ``subspace_dim`` and the tangent-space dimensionality ``f``):

  1. Covariance matrix Centroid Alignment (CA, Sec. III-A, Eq. 14). Each SPD
     covariance P is aligned by a reference matrix M_ref = M^{-1/2},
     P' = M_ref P M_ref, where M is a per-domain mean. Taking M as the Riemannian
     mean (Eq. 6) is the MEKT-R variant, which the paper reports as its best of
     three (MEKT-R / MEKT-E / MEKT-L, using the Riemannian / Euclidean / Log-
     Euclidean mean, Sec. IV-C); this file uses the Riemannian mean. CA drives
     each domain's centroid toward the identity, minimizing the marginal
     distribution shift (Eq. 15) and approximately whitening each trial (Sec.
     III-A).
  2. Tangent space feature extraction (Sec. III-B, Eqs. 19-20). Each aligned
     covariance is mapped to a vector x = upper(log P') of dimensionality
     d = c(c+1)/2. Source subjects are CA'd and mapped individually, then their
     feature vectors are pooled into one source matrix (multi-source assembly,
     Sec. III-C footnote / Algorithm 1).
  3. Mapping matrices identification (Sec. III-C). Learn two projection matrices,
     A for the source and B for the target, mapping the d-dimensional tangent
     features to a p-dimensional subspace so that A^T X_S and B^T X_T are close.
     The overall loss (Eq. 30) combines four terms:
       * joint probability distribution shift minimization (Eqs. 21-24), the
         paper's joint probability MMD between the source class-conditional means
         and the pseudo-labeled target class-conditional means (its proposed
         alternative to the traditional marginal+conditional MMD);
       * source domain discriminability (Eq. 25): minimize the within-class
         scatter S_w subject to fixing the between-class scatter, A^T S_b A = I;
       * target domain locality preservation (Eqs. 26-28): a heat-kernel kNN
         graph regularizer built on the target tangent features, via the
         normalized graph Laplacian L = I - D^{-1/2} S D^{-1/2};
       * parameter transfer and regularization (Eq. 29): keep B close to A and
         small, min(||B - A||_F^2 + ||B||_F^2).
     Stacking W = [A; B], Eq. 30 is solved by generalized eigen-decomposition
     (Eq. 35); W is the p trailing (smallest-generalized-eigenvalue) eigenvectors.
     Because the target term needs pseudo-labels, they are refined over several
     iterations by an EM-like procedure (Algorithm 1): a classifier trained on the
     projected source relabels the projected target each round. A final classifier
     is trained on A^T X_S with the source labels and applied to B^T X_T to obtain
     the target predictions.

  An optional fourth step, Domain Transferability Estimation (DTE, Sec. III-F),
  selects the most beneficial source subjects to reduce negative transfer and
  computational cost; it is not part of this file.

Implementation notes. The tangent features are extracted with pyriemann; the
step-3 projection and its EM-style pseudo-label refinement follow the authors'
reference implementation (chamwen/MEKT), reached here via the port in
TBC-TJU/MetaBCI. Defaults follow that reference and the paper's experimental
settings (Sec. IV-D): subspace_dim (paper's p) = 10, max_iter = 5, alpha = 0.01,
beta = 0.1, rho = 20, kNN k = 10, heat-kernel width t = 1, Ledoit-Wolf ('lwf')
covariance estimation. CA is applied per source subject here (each source subject
aligned to its own reference before pooling), matching the multi-source recipe of
Sec. III-C. The target features are always unlabeled: pseudo-labels come only from
the classifier, never from the target labels, so there is no label leakage.
Hyperparameters can be overridden from ``ctx.cfg.hp`` (mekt_dim / mekt_alpha /
mekt_beta / mekt_rho / mekt_iter / mekt_k / mekt_cov). Requires pyriemann,
scikit-learn, and scipy.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy


def _tangent_features(X: np.ndarray, covariance_type: str = "lwf") -> np.ndarray:
    """Steps 1-2 for one domain: SPD covariances -> CA with the Riemannian mean
    (Eq. 14) -> tangent-space feature vectors (Eqs. 19-20)."""
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.tangentspace import tangent_space
    covs = Covariances(estimator=covariance_type).transform(X.astype(np.float64))  # (n, c, c) SPD
    M = mean_riemann(covs)                                                # Riemannian mean (Eq. 6)
    W = invsqrtm(M)                                                        # reference M_ref = M^{-1/2}
    aligned = np.stack([W @ P @ W for P in covs])                         # CA: P' = M_ref P M_ref (Eq. 14)
    c = covs.shape[-1]
    return tangent_space(aligned, np.eye(c))                              # x=upper(log P'), dim d=c(c+1)/2


def _onehot(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """One-hot over a FIXED class set (columns stay consistent even if a
    pseudo-label round misses a class). Version-agnostic (no sklearn encoder)."""
    Y = np.zeros((len(y), len(classes)), dtype=np.float64)
    for j, c in enumerate(classes):
        Y[y == c, j] = 1.0
    return Y


def _source_discriminability(Xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Source domain discriminability term (Eq. 25): within-class scatter S_w and
    between-class scatter S_b of the source tangent features."""
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
    """Target domain locality preservation (Eqs. 26-28): heat-kernel kNN
    similarity S (Eq. 26) on the target tangent features and its unnormalized
    graph Laplacian D - S (D degree matrix)."""
    from scipy.spatial.distance import pdist, squareform
    pair = squareform(pdist(Xt, metric="euclidean"))
    kk = min(k + 1, Xt.shape[0])                       # self-connection included (guard tiny sets)
    ix = np.argsort(pair, axis=-1)[:, :kk]
    heat = np.exp(-np.square(pair) / (2.0 * t * t))    # heat-kernel weights (Eq. 26)
    W = np.zeros_like(pair)
    for i, ind in enumerate(ix):
        W[i, ind] = heat[i, ind]
    W = np.maximum(W, W.T)                              # symmetrise the kNN graph
    D = np.diag(W.sum(axis=-1))
    return D - W, D


def _mekt_kernel(Xs: np.ndarray, Xt: np.ndarray, ys: np.ndarray, *, d: int = 10,
                 max_iter: int = 5, alpha: float = 0.01, beta: float = 0.1,
                 rho: float = 20.0, k: int = 10, t: float = 1.0,
                 log=lambda *_: None) -> Tuple[np.ndarray, np.ndarray]:
    """Mapping matrices identification (Sec. III-C): learn source/target
    projections A, B by solving the overall loss (Eq. 30) as a generalized
    eigenproblem.

    ``d`` is the shared-subspace dimensionality (the paper's p); ``f`` is the
    tangent-space dimensionality (the paper's d = c(c+1)/2). The generalized
    eigenproblem eigh(Emin, Emax) is assembled with the four Eq.-30 terms on the
    minimized side Emin -- joint probability MMD (Eqs. 21-24), alpha * within-
    class scatter S_w (Eq. 25), beta * target locality Laplacian (Eqs. 26-28), and
    rho * parameter-transfer coupling (Eq. 29) -- and the between-class scatter
    constraint plus target-variance normalization on the constraint side Emax
    (A^T S_b A = I, B^T X_T H X_T^T B = I; a ridge is added for numerical
    stability). W = [A; B] is taken from the d smallest generalized eigenvalues
    (the paper's p trailing eigenvectors, Eq. 35). Pseudo-labels for the target
    are refined over ``max_iter`` EM-like rounds (Algorithm 1). Returns A:(f, d)
    for the source and B:(f, d) for the target.
    """
    from scipy.linalg import block_diag, eigh
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    ns, f = Xs.shape
    nt = Xt.shape[0]
    d = int(min(d, f))

    # source domain discriminability (Eq. 25): min S_w s.t. A^T S_b A = I
    Sw, Sb = _source_discriminability(Xs, ys)
    P = block_diag(Sw, np.zeros((f, f)))                       # S_w in the source block (P, Eq. 32)
    P0 = block_diag(Sb, np.zeros((f, f)))                      # S_b for the Emax constraint

    # target domain locality preservation (Eqs. 26-28): normalized graph
    # Laplacian L = I - D^{-1/2} S D^{-1/2}, embedded in the target block (Eq. 32)
    L, D = _graph_laplacian(Xt, k=k, t=t)
    iD12 = np.diag(1.0 / np.sqrt(np.clip(np.diag(D), 1e-12, None)))
    L = iD12 @ L @ iD12
    L = block_diag(np.zeros((f, f)), Xt.T @ L @ Xt)

    # parameter transfer and regularization (Eq. 29): coupling of A and B (U, Eq. 33)
    Q = np.block([[np.eye(f), -np.eye(f)], [-np.eye(f), 2.0 * np.eye(f)]])

    # target-variance normalization B^T X_T H X_T^T B = I (H centering matrix, Eq. 28)
    Ht = np.eye(nt) - np.ones((nt, nt)) / nt
    S = block_diag(np.zeros((f, f)), Xt.T @ Ht @ Xt)

    classes = np.sort(np.unique(ys))
    Ns = _onehot(ys, classes) / len(ys)                       # source class indicator N_S (Eq. 24)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    yt = clf.fit(Xs, ys).predict(Xt)                          # initial target pseudo-labels

    Xblk = block_diag(Xs, Xt)                                 # (ns+nt, 2f)
    Emin_base = alpha * P + beta * L + rho * Q                # alpha*S_w + beta*L + rho*U (Eq. 31)

    A = np.zeros((f, d))
    B = np.zeros((f, d))
    for it in range(max_iter):
        Nt = _onehot(yt, classes) / max(len(yt), 1)          # pseudo-labeled target indicator N_T (Eq. 24)
        Mmmd = np.block([[Ns @ Ns.T, -Ns @ Nt.T], [-Nt @ Ns.T, Nt @ Nt.T]])   # joint probability MMD (Eqs. 23-24)
        R = Xblk.T @ Mmmd @ Xblk                              # R (Eq. 34), (2f, 2f)
        Emin = Emin_base + R                                  # alpha*P + beta*L + rho*U + R (Eq. 35)
        Emin = 0.5 * (Emin + Emin.T)

        # generalized eigen-decomposition of (Emin, Emax); escalate the ridge on
        # the constraint side until Emax is positive definite
        V = None
        for ridge in (1e-3, 1e-2, 1e-1, 1.0):
            Emax = S + alpha * P0 + ridge * np.eye(2 * f)     # V (Eq. 33) + ridge, right-hand side of Eq. 35
            Emax = 0.5 * (Emax + Emax.T)
            try:
                _, V = eigh(Emin, Emax)                       # ascending generalized eigenvalues
                break
            except np.linalg.LinAlgError:
                continue
        if V is None:
            log(f"  MEKT: eigh failed at iter {it}; keeping previous projection")
            break

        A = V[:f, :d]                                         # p trailing eigenvectors (Eq. 35); A source
        B = V[f:, :d]                                         #                                     B target
        Zs, Zt = Xs @ A, Xt @ B
        if not (np.all(np.isfinite(Zs)) and np.all(np.isfinite(Zt))):
            log(f"  MEKT: non-finite projection at iter {it}; keeping previous projection")
            A = np.zeros((f, d)) if it == 0 else A_prev
            B = np.zeros((f, d)) if it == 0 else B_prev
            break
        A_prev, B_prev = A, B
        yt = clf.fit(Zs, ys).predict(Zt)                     # refine target pseudo-labels (Algorithm 1)
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

        # steps 1-2 (CA + tangent features) per source subject, then pool
        ts_list, y_list = [], []
        for kdom in [int(dd) for dd in source.domains()]:
            e = source.select(source.domain == kdom)
            ts_list.append(_tangent_features(e.X, cov))
            y_list.append(e.y)
        ts_S = np.concatenate(ts_list, 0)                     # source X_S, (n_S, d)
        y_S = np.concatenate(y_list, 0)
        ts_T = _tangent_features(target.X, cov)               # target X_T, (n_T, d), CA'd to its own mean

        # step 3: mapping matrices identification (Sec. III-C)
        A, B = _mekt_kernel(ts_S, ts_T, y_S, d=self.subspace_dim, max_iter=self.max_iter,
                            alpha=self.alpha, beta=self.beta, rho=self.rho, k=self.k, t=self.t,
                            log=ctx.log)
        Zs, Zt = ts_S @ A, ts_T @ B                           # A^T X_S, B^T X_T; (n_S, p), (n_T, p)

        # classifier trained on the projected source, applied to the projected target (Algorithm 1)
        clf = LDA(solver="lsqr", shrinkage="auto").fit(Zs, y_S)
        y_pred = clf.predict(Zt)
        y_score = clf.predict_proba(Zt)
        return np.asarray(y_pred, dtype=np.int64), y_score
