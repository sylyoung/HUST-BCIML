# ===========================================================================
# _ensembles.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Majority voting, SML/SML-OVR and StackingNet combiners.
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
#   @Article{Li2026a,
#     author  = {Li, Siyang and Liu, Chenhao and Wu, Dongrui and Zeng, Zhigang and Ding, Lieyun},
#     journal = {arXiv preprint arXiv:2602.13792},
#     title   = {{S}tacking{N}et: Collective Inference Across Independent {AI} Foundation Models},
#     year    = {2026},
#     doi     = {10.48550/arXiv.2602.13792},
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
"""Post-hoc black-box ensemble combiners over K base models' predictions.

Each combiner takes ``scores`` of shape ``(K, N, C)`` — K base models (here K
random seeds of one algorithm), N trials, C class scores — and returns hard
predictions ``(N,)``. "Black-box" = it sees only the models' outputs, no labels
and no model internals.

Every combiner here aggregates only the base models' HARD votes (each internally
argmaxes the scores). There is deliberately no soft-score averaging combiner: a
soft mean would give one method a strict information advantage over all the hard-
label crowdsourcing aggregators it is compared against, so the comparison would no
longer be apples-to-apples. Hard majority ``voting`` is the baseline every other
combiner must beat.

Ported from github.com/sylyoung/TestEnsemble (``ensemble.py`` SML / SML_OVR /
pred_voting_hard, ``algs/StackingNet_classification.py``) and the lab's
``tl/ttime_ensemble.py``. The crowdsourcing baselines (Dawid-Skene, Wawa, M-MSR,
MACE, GLAD, ZenCrowd, PM, LA, LAA, EBCC) live in ``_ensemble_baselines.py``.
"""
from __future__ import annotations

import numpy as np

from hustbciml.scripts import _ensemble_baselines as _bl


def _onehot(preds: np.ndarray, C: int) -> np.ndarray:
    """(K, N) hard labels -> (K, N, C) one-hot."""
    K, N = preds.shape
    oh = np.zeros((K, N, C), dtype=np.float64)
    for k in range(K):
        oh[k, np.arange(N), preds[k]] = 1
    return oh


def voting(scores: np.ndarray) -> np.ndarray:
    """Hard majority vote; ties broken uniformly at random with a *local* fixed
    seed so the result is reproducible regardless of call order (the global NumPy
    RNG state would otherwise make the reported accuracy drift between runs)."""
    preds = scores.argmax(axis=2)                       # (K, N)
    K, N = preds.shape
    C = scores.shape[2]
    votes = np.zeros((C, N))
    for k in range(K):
        for n in range(N):
            votes[preds[k, n], n] += 1
    rng = np.random.RandomState(0)
    out = [rng.choice(np.flatnonzero(votes[:, n] == votes[:, n].max())) for n in range(N)]
    return np.array(out)


def _principal_eigvec(Q: np.ndarray) -> np.ndarray:
    """Leading eigenvector (largest eigenvalue) of a symmetric matrix Q.

    This is the SML weight vector: the leading eigenvector of the base models'
    prediction-covariance is proportional to their balanced accuracies (Parisi et
    al. 2014; Li et al. 2026, "Black-Box Test-Time Ensemble", Algorithm 1, which
    specifies the "leading eigenvector"). ``eigh`` returns real eigenvalues in
    ascending order for a symmetric matrix, so ``argmax`` of the eigenvalues yields
    the leading eigenvector deterministically — matching Algorithm 1 and the method's
    online variant. The TestEnsemble reference's ``eig(Q)[1][:, 0]`` selects the same
    vector whenever ``eig`` returns the dominant eigenpair first."""
    w, V = np.linalg.eigh(Q)
    return V[:, int(np.argmax(w))]


def sml(scores: np.ndarray) -> np.ndarray:
    """Binary Spectral Meta-Learner (Parisi et al. 2014). Weights = principal
    eigenvector of the models' {-1,+1} prediction covariance."""
    preds = scores.argmax(axis=2)                       # (K, N) in {0,1}
    pred = np.where(preds == 1, 1.0, -1.0)
    mu = pred.mean(axis=1)
    dev = pred - mu[:, None]
    Q = dev @ dev.T / (pred.shape[1] - 1)
    v = _principal_eigvec(Q)
    if np.any(v < 0):
        v = np.abs(v)
    return np.where(np.einsum("a,ab->b", v, pred) >= 0, 1, 0)


def sml_ovr(scores: np.ndarray) -> np.ndarray:
    """SML one-vs-rest — the MULTI-CLASS (K>2) extension of binary SML (Li et al.
    2026, "Black-Box Test-Time Ensemble", Algorithm 1 / Eqs. 12-13). For each class
    k, form the one-vs-rest {-1,+1} votes, take the leading eigenvector of the M x M
    vote-covariance as the per-class model weights (``_principal_eigvec``), normalize
    it by its entry sum, then average the K per-class weightings into one reliability
    vector v-bar and predict argmax_k sum_j f_hat_{j,k} * v-bar_j. For K=2 this
    reduces to binary SML, which is used directly instead (callers skip SML-OVR when
    there are 2 classes)."""
    preds = scores.argmax(axis=2)                       # (K, N)
    C = scores.shape[2]
    oh = _onehot(preds, C)                              # (K, N, C)
    weights_all = []
    for i in range(C):
        pred = np.where(oh.argmax(-1) == i, 1.0, -1.0)  # (K, N)
        mu = pred.mean(axis=1)
        dev = pred - mu[:, None]
        Q = dev @ dev.T / (pred.shape[1] - 1)
        v = _principal_eigvec(Q)
        if v[0] <= 0:                                   # fix global sign (assume model 0 > chance)
            v = -v
        weights_all.append(v / np.sum(v))
    wf = np.sum(np.array(weights_all), axis=0)
    return np.argmax(np.einsum("a,abc->bc", wf, oh), axis=1)


def sml_ovr_eig0(scores: np.ndarray) -> np.ndarray:
    """SML-OVR computed with the offline TestEnsemble reference's eigenvector
    selection ``np.linalg.eig(Q)[1][:, 0]`` (first-returned, not sorted) instead of
    the leading eigenvector. VERIFICATION-ONLY: run alongside ``sml_ovr`` on the real
    predictions to confirm the two selections agree (the paper's Algorithm 1 specifies
    the leading eigenvector). Not part of the default combiner set."""
    preds = scores.argmax(axis=2)
    C = scores.shape[2]
    oh = _onehot(preds, C)
    weights_all = []
    for i in range(C):
        pred = np.where(oh.argmax(-1) == i, 1.0, -1.0)
        mu = pred.mean(axis=1)
        dev = pred - mu[:, None]
        Q = dev @ dev.T / (pred.shape[1] - 1)
        v = np.linalg.eig(Q)[1][:, 0].real            # reference: first-returned eigenvector
        if v[0] <= 0:
            v = -v
        weights_all.append(v / np.sum(v))
    wf = np.sum(np.array(weights_all), axis=0)
    return np.argmax(np.einsum("a,abc->bc", wf, oh), axis=1)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> float:
    """Macro-recall (balanced accuracy) of ``y_pred`` against ``y_true``, averaged
    over the classes present in ``y_true``. Used only to seed StackingNet's weights
    from each model's agreement with the majority-vote consensus (no ground truth)."""
    recalls = []
    for c in range(C):
        mask = y_true == c
        if mask.any():
            recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def stackingnet(scores: np.ndarray, epochs: int = 200, lr: float = 0.001,
                unsupervised_weight: float = 0.001, reg_weight: float = 100.0,
                seed: int = 0) -> np.ndarray:
    """Unsupervised transductive StackingNet (Li, Liu & Wu, 2026). Learns one
    non-negative weight per base model with NO labels, transductively over the test
    predictions. Faithful to the authors' ``StackingNet_classification`` (unsupervised
    ``golden_num == 0`` path with ``use_weight_init``):

    * weights are INITIALISED from each model's balanced-accuracy agreement with the
      majority-vote consensus (not uniform 1/K), which anchors the optimisation;
    * they are then refined by the PM indicator loss (each model pulled toward the
      current weighted-ensemble consensus), scaled DOWN by ``unsupervised_weight``
      (0.001), plus a STRONG ``reg_weight * (1 - sum w)^2`` (100) term that keeps the
      weights near a convex combination; weights are clamped >= 0 after each step.

    The small PM weight + strong regulariser + informed init are what keep the
    weights near a sensible convex combination instead of collapsing to zero (the
    earlier uniform-init / unit-PM / weak-reg port collapsed to the constant class,
    i.e. chance)."""
    import torch

    K, N, C = scores.shape
    preds = scores.argmax(axis=2)                       # (K, N)
    oh = _onehot(preds, C).astype(np.float32)           # (K, N, C)

    # weight init = each model's balanced accuracy vs the majority-vote consensus
    consensus = voting(scores)                          # (N,) label-free consensus
    w0 = np.array([_balanced_accuracy(consensus, preds[k], C) for k in range(K)])
    w0 = w0 / w0.sum() if w0.sum() > 0 else np.ones(K) / K

    X = torch.from_numpy(oh.transpose(1, 0, 2))         # (N, K, C)
    torch.manual_seed(seed)
    w = torch.nn.Parameter(torch.tensor(w0.reshape(K, 1), dtype=torch.float32))
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(epochs):
        out = (X * w.unsqueeze(0)).sum(dim=1)           # (N, C) weighted-ensemble scores
        ens = torch.zeros_like(out)
        ens.scatter_(1, out.argmax(1, keepdim=True), 1)  # hard consensus one-hot (detached by argmax)
        pm = sum(((X[:, i, :] != ens).any(dim=1).float().sum()) * w[i] for i in range(K))
        loss = unsupervised_weight * pm + reg_weight * (1 - w.sum()) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w.clamp_(min=0)                             # non-negative weights
    with torch.no_grad():
        out = (X * w.unsqueeze(0)).sum(dim=1)
    return out.argmax(1).numpy()


def _hard(fn):
    """Adapt a vendored baseline that consumes hard votes ``(K, N)`` to the combiner
    interface ``(K, N, C) -> (N,)`` by argmaxing the scores into hard labels first."""
    return lambda scores: fn(scores.argmax(axis=2))


# name -> combiner; binary-only ones are filtered by the caller on class count.
# All aggregate HARD votes only (no soft averaging). "(lab)" marks lab-proposed
# methods; the rest are black-box crowdsourcing baselines from TestEnsemble.
COMBINERS = {
    "voting": voting,                    # hard majority-vote baseline
    "Dawid-Skene": _hard(_bl.dawid_skene),
    "Wawa": _hard(_bl.wawa),
    "M-MSR": _hard(_bl.mmsr),
    "MACE": _hard(_bl.mace),
    "GLAD": _hard(_bl.glad),
    "ZenCrowd": _hard(_bl.zencrowd),
    "PM": _hard(_bl.pm),
    "LA": _hard(_bl.la),
    "LAA": _hard(_bl.laa),
    "EBCC": _hard(_bl.ebcc),
    "SML": sml,                          # binary only
    "SML-OVR": sml_ovr,                  # (lab)
    "SML-OVR-eig0": sml_ovr_eig0,        # verification-only: eig()[:,0] cross-check
    "StackingNet": stackingnet,          # (lab)
}
