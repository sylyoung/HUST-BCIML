# _common.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Shared helpers for the ensemble combiners.

The combiner files in this folder each hold one method and keep their own math
self-contained. What they share is small and mechanical: turning hard votes into
one-hot tables, running a stochastic aggregator under a fixed local seed so its
output is reproducible, feeding votes to the ``crowdkit`` library, and two tiny
numeric utilities used by the spectral and StackingNet combiners. Those live here
so no combiner has to import another combiner.

Nothing here is a plug-in. The file name starts with an underscore, so the
registry's ``available`` scan skips it and it never shows up as a combiner name.
"""
from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np


def onehot(preds: np.ndarray, C: int) -> np.ndarray:
    """``(K, N)`` hard labels -> ``(K, N, C)`` {0,1} one-hot.

    K base models, N trials, C classes. Row ``[k, n]`` is the unit vector that
    marks the class model k voted for on trial n. Combiners that reason over
    per-class vote mass (SML-OVR, PM, LAA, EBCC) build on this representation.
    """
    K, N = preds.shape
    oh = np.zeros((K, N, C), dtype=np.float64)
    for k in range(K):
        oh[k, np.arange(N), preds[k]] = 1
    return oh


def majority_vote(scores: np.ndarray) -> np.ndarray:
    """Hard majority vote over ``(K, N, C)`` scores -> ``(N,)`` labels.

    Each base model's vote is the argmax of its scores; the consensus is the class
    with the most votes per trial. Ties are broken uniformly at random with a
    *local* fixed seed so the result is reproducible regardless of call order (the
    global NumPy RNG state would otherwise make the reported accuracy drift between
    runs). This is both the ``Voting`` baseline and the label-free consensus that
    ``StackingNet`` initializes its per-model weights from.
    """
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


@contextmanager
def fixed_seed(seed: int = 0):
    """Run a stochastic aggregator under a fixed numpy+random seed, then restore
    the caller's RNG state so the call is order-independent and side-effect free.

    Several aggregators here draw random initial states or break ties at random.
    Pinning the seed makes each combiner's reported accuracy reproducible and
    independent of how many combiners ran before it, and restoring the previous
    RNG state on exit means these calls never perturb the rest of the experiment.
    """
    np_state, py_state = np.random.get_state(), random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def to_long_df(preds: np.ndarray):
    """``(K, N)`` -> crowdkit long-format DataFrame with columns task/worker/label.

    ``crowdkit`` expects one row per (trial, model) vote. Here a "worker" is a
    base model and a "task" is a trial, so the K x N vote matrix is unrolled into
    K*N rows. pandas is imported lazily because only the crowdkit-backed combiners
    need it.
    """
    import pandas as pd

    K, N = preds.shape
    task = np.repeat(np.arange(N), K)
    worker = np.tile(np.arange(K), N)
    label = preds.T.reshape(-1)
    return pd.DataFrame({"task": task, "worker": worker, "label": label})


def crowdkit_predict(preds: np.ndarray, method) -> np.ndarray:
    """Fit a crowdkit aggregator on the votes and return task-ordered labels.

    Every task carries all K votes, so no task is dropped; we still reindex to the
    original ``0..N-1`` order and fall back to the majority vote for any task the
    aggregator failed to score (never silently — the array is validated). The fit
    runs under ``fixed_seed`` so a stochastic crowdkit method is reproducible.
    """
    N = preds.shape[1]
    with fixed_seed(0):
        series = method.fit_predict(to_long_df(preds))
    aligned = series.reindex(range(N))
    if aligned.isna().any():                       # should not happen; guard anyway
        from scipy import stats

        fallback = stats.mode(preds, axis=0, keepdims=False).mode
        aligned = aligned.fillna(dict(enumerate(fallback)))
    return aligned.to_numpy().astype(int)


def principal_eigvec(Q: np.ndarray) -> np.ndarray:
    """Leading eigenvector (largest eigenvalue) of a symmetric matrix ``Q``.

    This is the SML weight vector: the leading eigenvector of the base models'
    prediction-covariance is proportional to their balanced accuracies (Parisi et
    al. 2014; Li et al. 2026, "Black-Box Test-Time Ensemble", Algorithm 1, which
    specifies the "leading eigenvector"). ``eigh`` returns real eigenvalues in
    ascending order for a symmetric matrix, so ``argmax`` of the eigenvalues yields
    the leading eigenvector deterministically — matching Algorithm 1 and the
    method's online variant. The TestEnsemble reference's ``eig(Q)[1][:, 0]``
    selects the same vector whenever ``eig`` returns the dominant eigenpair first.
    """
    w, V = np.linalg.eigh(Q)
    return V[:, int(np.argmax(w))]


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> float:
    """Macro-recall (balanced accuracy) of ``y_pred`` against ``y_true``, averaged
    over the classes present in ``y_true``. Used only to seed StackingNet's weights
    from each model's agreement with the majority-vote consensus (no ground truth).
    """
    recalls = []
    for c in range(C):
        mask = y_true == c
        if mask.any():
            recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0
