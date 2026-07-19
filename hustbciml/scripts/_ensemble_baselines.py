# ===========================================================================
# _ensemble_baselines.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Ten crowd-labelling / truth-discovery aggregators: Dawid-Skene (JRSS-C
# 1979), PM (SIGMOD 2014), ZenCrowd (WWW 2012), EBCC (ICML 2019), LA (ACM
# TKDD 2024), LAA (IJCAI 2017), GLAD (NeurIPS 2009), M-MSR (NeurIPS 2020),
# MACE (NAACL-HLT 2013), Wawa (crowd-kit heuristic). Full citations in
# references.bib and gallery/data/benchmark.yml.
# Ported from: https://github.com/sylyoung/TestEnsemble
# Reference implementation: https://github.com/Toloka/crowd-kit
# ===========================================================================
"""Black-box label-aggregation baselines for the decentralized ensemble.

Every function here takes the base models' HARD predictions ``preds`` of shape
``(K, N)`` (K base models, N trials; integer class ids) and returns hard consensus
labels ``(N,)``. None of them see the soft scores, ground-truth labels, or model
internals — they aggregate votes only, exactly the black-box test-time setting of
the lab's SML-OVR / StackingNet.

Two groups, mirroring ``github.com/sylyoung/TestEnsemble`` (``ensemble.py``):

* Crowdsourcing aggregators from the ``crowdkit`` library — Dawid-Skene, Wawa,
  M-MSR, MACE, GLAD — called with the SAME defaults as the lab's ``ensemble.py``.
* Pure-numpy / torch aggregators vendored from ``TestEnsemble/algs`` — ZenCrowd,
  PM (truth discovery), LA (lightweight two-pass), LAA (label-aware autoencoder),
  EBCC (enhanced Bayesian classifier combination). Their core math is transplanted
  verbatim; the only change is that every source of randomness is driven by a LOCAL
  fixed seed so the reported accuracy is reproducible and independent of call order
  (the same discipline as ``voting``'s tie-break), and the global RNG state is
  restored afterwards so these calls never perturb the rest of the experiment.

Each paper is cited in the function that ports it.
"""
from __future__ import annotations

import math
import random
from contextlib import contextmanager

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _onehot(preds: np.ndarray, C: int) -> np.ndarray:
    """(K, N) hard labels -> (K, N, C) {0,1} one-hot."""
    K, N = preds.shape
    oh = np.zeros((K, N, C), dtype=np.float64)
    for k in range(K):
        oh[k, np.arange(N), preds[k]] = 1
    return oh


@contextmanager
def _fixed_seed(seed: int = 0):
    """Run a stochastic aggregator under a fixed numpy+random seed, then restore
    the caller's RNG state so it is call-order independent and side-effect free."""
    np_state, py_state = np.random.get_state(), random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def _to_long_df(preds: np.ndarray):
    """(K, N) -> crowdkit long-format DataFrame with columns task/worker/label."""
    import pandas as pd

    K, N = preds.shape
    task = np.repeat(np.arange(N), K)
    worker = np.tile(np.arange(K), N)
    label = preds.T.reshape(-1)
    return pd.DataFrame({"task": task, "worker": worker, "label": label})


def _crowdkit_predict(preds: np.ndarray, method) -> np.ndarray:
    """Fit a crowdkit aggregator on the votes and return task-ordered labels.
    Every task carries all K votes, so no task is dropped; we still reindex to the
    original 0..N-1 order and fall back to the majority vote for any task the
    aggregator failed to score (never silently — the array is validated)."""
    N = preds.shape[1]
    with _fixed_seed(0):
        series = method.fit_predict(_to_long_df(preds))
    aligned = series.reindex(range(N))
    if aligned.isna().any():                       # should not happen; guard anyway
        from scipy import stats

        fallback = stats.mode(preds, axis=0, keepdims=False).mode
        aligned = aligned.fillna(dict(enumerate(fallback)))
    return aligned.to_numpy().astype(int)


# --------------------------------------------------------------------------- #
# crowdsourcing aggregators (crowdkit) — same defaults as TestEnsemble/ensemble.py
# --------------------------------------------------------------------------- #
def dawid_skene(preds: np.ndarray) -> np.ndarray:
    """Dawid & Skene (1979), EM over per-worker confusion matrices."""
    from crowdkit.aggregation import DawidSkene

    return _crowdkit_predict(preds, DawidSkene(n_iter=10))


def wawa(preds: np.ndarray) -> np.ndarray:
    """Wawa (Worker Agreement With Aggregate): reweight workers by their agreement
    with the majority vote, then re-vote."""
    from crowdkit.aggregation import Wawa

    return _crowdkit_predict(preds, Wawa())


def mmsr(preds: np.ndarray) -> np.ndarray:
    """M-MSR (Ma & Olshevsky 2020), Matrix Mean-Subsequence-Reduced worker skill
    estimation from the pairwise agreement matrix."""
    from crowdkit.aggregation import MMSR

    return _crowdkit_predict(preds, MMSR())


def mace(preds: np.ndarray) -> np.ndarray:
    """MACE (Hovy et al. 2013), Multi-Annotator Competence Estimation — a
    variational model separating competent labelling from per-worker spamming."""
    from crowdkit.aggregation import MACE

    return _crowdkit_predict(preds, MACE())


def glad(preds: np.ndarray) -> np.ndarray:
    """GLAD (Whitehill et al. 2009), jointly infers label, worker ability, and
    per-item difficulty by EM."""
    from crowdkit.aggregation import GLAD

    return _crowdkit_predict(preds, GLAD())


# --------------------------------------------------------------------------- #
# vendored aggregators (pure numpy / torch) — from TestEnsemble/algs
# --------------------------------------------------------------------------- #
def zencrowd(preds: np.ndarray, n_iter: int = 20) -> np.ndarray:
    """ZenCrowd (Demartini et al. 2012). EM with a single per-worker reliability
    scalar; each item's posterior multiplies workers' reliabilities for the class
    they voted and the smoothed complement otherwise. Ported from ``algs/ZC.py``."""
    K, N = preds.shape
    C = int(preds.max()) + 1
    labels = list(range(C))
    with _fixed_seed(0):
        wm = {w: 0.8 for w in range(K)}                    # worker reliabilities
        e2lpd = None
        for _ in range(n_iter):
            # E-step: posterior over labels for each item
            e2lpd = {}
            for t in range(N):
                post = {c: 1.0 for c in labels}
                for w in range(K):
                    lab = preds[w, t]
                    for c in labels:
                        if lab == c:
                            post[c] *= wm[w]
                        else:
                            post[c] *= (1 - wm[w]) / (C - 1)
                s = sum(post.values())
                if s == 0:
                    post = {c: 1.0 / C for c in labels}
                else:
                    post = {c: post[c] / s for c in labels}
                e2lpd[t] = post
            # M-step: reliability = mean posterior mass on each worker's votes
            for w in range(K):
                wm[w] = float(np.mean([e2lpd[t][preds[w, t]] for t in range(N)]))
        rng = np.random.RandomState(0)
        out = []
        for t in range(N):
            best = max(e2lpd[t].values())
            cand = [c for c in labels if e2lpd[t][c] == best]
            out.append(rng.choice(cand))
    return np.array(out, dtype=int)


def pm(preds: np.ndarray, n_iter: int = 3) -> np.ndarray:
    """PM — truth discovery by source reliability (Li et al. 2014, "Resolving
    conflicts in heterogeneous data by truth discovery and source reliability
    estimation"). Worker weight = -log(normalised disagreement with the current
    truth); re-estimate the truth as the weighted one-hot vote. From ``algs/PM.py``."""
    K, N = preds.shape
    C = int(preds.max()) + 1
    with _fixed_seed(0):
        votes = np.zeros((C, N))
        for i in range(K):
            for j in range(N):
                votes[preds[i, j], j] += 1
        rng = np.random.RandomState(0)
        truth = np.array([rng.choice(np.flatnonzero(votes[:, j] == votes[:, j].max()))
                          for j in range(N)])
        oh = _onehot(preds, C)                              # {0,1}
        oh = np.where(oh == 1, 1, -1)                       # {-1,+1} as in PM.py
        weight = np.zeros(K)
        wmax = 0.0
        for _ in range(n_iter):
            for w in range(K):
                dif = float(np.sum(preds[w, :] != truth)) or 1e-8
                weight[w] = dif
                wmax = max(wmax, weight[w])
            weight /= wmax
            weight = -np.log(weight + 1e-7) + 1e-7
            truth = np.argmax(np.einsum("a,abc->bc", weight, oh), axis=1)
    return truth.astype(int)


def la(preds: np.ndarray, alpha: int = 2, beta: int = 2) -> np.ndarray:
    """LA — lightweight two-pass label aggregation (Yang et al. 2024, "A
    Lightweight, Effective, and Efficient Model for Label Aggregation in
    Crowdsourcing", ACM TKDD). One online pass estimates each worker's ability
    a_w (Beta prior alpha,beta), a second pass re-votes with weight (a_w*K - 1).
    From ``algs/LA_twopass.py``."""
    K, N = preds.shape
    C = int(preds.max()) + 1
    labels = list(range(C))
    e2wl = {t: [(w, int(preds[w, t])) for w in range(K)] for t in range(N)}
    with _fixed_seed(0):
        rng = random.Random(0)
        c = {w: alpha - 1 for w in range(K)}
        t_cnt = {w: alpha + beta - 2 for w in range(K)}
        a = {w: c[w] / t_cnt[w] for w in range(K)}
        items = list(e2wl.keys())
        rng.shuffle(items)
        truths = {}
        for item in items:                                 # pass 1: online ability
            votes = {}
            for w, lab in e2wl[item]:
                votes[lab] = votes.get(lab, 0) + a[w]
            best, cand = -1, []
            for cl in labels:
                if cl not in votes:
                    continue
                if votes[cl] > best:
                    best, cand = votes[cl], [cl]
                elif votes[cl] == best:
                    cand.append(cl)
            truths[item] = rng.choice(cand)
            for w, lab in e2wl[item]:
                t_cnt[w] += 1
                if lab == truths[item]:
                    c[w] += 1
                a[w] = c[w] / t_cnt[w]
        out = []
        for item in range(N):                              # pass 2: re-vote
            votes = {}
            for w, lab in e2wl[item]:
                votes[lab] = votes.get(lab, 0) + (a[w] * C - 1)
            best, cand = -1, []
            for cl in labels:
                if cl not in votes:
                    continue
                if votes[cl] > best:
                    best, cand = votes[cl], [cl]
                elif votes[cl] == best:
                    cand.append(cl)
            out.append(rng.choice(cand))
    return np.array(out, dtype=int)


def laa(preds: np.ndarray) -> np.ndarray:
    """LAA — label-aware autoencoder (Yin et al. 2017, "Aggregating crowd wisdoms
    with label-aware autoencoders"). A linear classifier maps the concatenated
    one-hot votes to a label, a source-wise-softmax decoder reconstructs the votes,
    trained to the majority-vote target. Ported from ``algs/LAA.py``."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    K, N = preds.shape
    C = int(preds.max()) + 1
    oh = _onehot(preds, C)                                  # (K, N, C)
    user_labels = np.concatenate([oh[k] for k in range(K)], axis=1)  # (N, K*C)
    # majority vote target (label-free), local tie-break
    votes = np.zeros((C, N))
    for k in range(K):
        for j in range(N):
            votes[preds[k, j], j] += 1
    rng = np.random.RandomState(0)
    majority = np.array([rng.choice(np.flatnonzero(votes[:, j] == votes[:, j].max()))
                         for j in range(N)])

    in_features = K * C
    template = torch.zeros((in_features, in_features))
    for i in range(K):
        template[i * C:(i + 1) * C, i * C:(i + 1) * C] = 1

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.empty(in_features, C))
            self.biases = nn.Parameter(torch.empty(C))
            nn.init.trunc_normal_(self.weights, std=0.01)
            nn.init.zeros_(self.biases)

        def forward(self, x):
            return F.softmax(x @ self.weights + self.biases, dim=1)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.empty(C, in_features))
            self.biases = nn.Parameter(torch.empty(in_features))
            nn.init.trunc_normal_(self.weights, std=0.01)
            nn.init.zeros_(self.biases)

        def forward(self, y):
            e = torch.exp(y @ self.weights + self.biases)
            return e / (e @ template)

    with _fixed_seed(0):
        torch.manual_seed(0)
        clf, dec = Classifier(), Decoder()
        opt = torch.optim.Adam(list(clf.parameters()) + list(dec.parameters()), lr=0.005)
        x = torch.FloatTensor(user_labels)
        y = F.one_hot(torch.LongTensor(majority), num_classes=C).float()
        ce = nn.CrossEntropyLoss()
        for _ in range(50):                                # warm up classifier to MV
            opt.zero_grad()
            ce(clf(x), y).backward()
            opt.step()
        for _ in range(100):                               # autoencoder refinement
            opt.zero_grad()
            yc = clf(x)
            xr = dec(yc)
            loss = (torch.mean(torch.sum(-x * torch.log(xr + 1e-10), dim=1))
                    + 0.0001 * nn.KLDivLoss(reduction="batchmean")(torch.log(yc + 1e-10), y)
                    + 0.005 / (K * C * C) * torch.sum(torch.abs(clf.weights)))
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = clf(x).argmax(1).numpy()
    return pred.astype(int)


def ebcc(preds: np.ndarray, num_groups: int = 10, a_pi: float = 0.1, alpha: float = 1.0,
         a_v: float = 4.0, b_v: float = 1.0, max_iter: int = 500,
         empirical_prior: bool = True) -> np.ndarray:
    """EBCC — enhanced Bayesian classifier combination (Li et al. 2019, "Exploiting
    worker correlation for label aggregation in crowdsourcing"). Variational Bayes
    over worker sub-type groups and per-group confusion matrices. Ported from
    ``algs/EBCC.py``; defaults follow the lab's ``ensemble.py`` call
    (num_groups=10, empirical_prior=True, max_iter=10 is set by the caller)."""
    import scipy.sparse as ssp
    from scipy.special import digamma

    K, N = preds.shape
    num_classes = int(preds.max()) + 1
    with _fixed_seed(0):
        first = np.repeat(np.arange(N), K)
        second = np.tile(np.arange(K), N)
        third = preds.T.flatten()
        tuples = np.vstack((first, second, third)).T

        y_lij, y_lji = [], []
        for k in range(num_classes):
            sel = tuples[:, 2] == k
            coo = ssp.coo_matrix((np.ones(sel.sum()), tuples[sel, :2].T),
                                 shape=(N, K), dtype=bool)
            y_lij.append(coo.tocsr())
            y_lji.append(coo.T.tocsr())

        beta_kl = np.eye(num_classes) * (a_v - b_v) + b_v
        z_ik = np.zeros((N, num_classes))
        for l in range(num_classes):
            z_ik[:, [l]] += y_lij[l].sum(axis=-1)
        z_ik /= z_ik.sum(axis=-1, keepdims=True)
        if empirical_prior:
            alpha = z_ik.sum(axis=0)

        zg_ikm = np.random.dirichlet(np.ones(num_groups), z_ik.shape) * z_ik[:, :, None]
        for _ in range(max_iter):
            eta_km = a_pi / num_groups + zg_ikm.sum(axis=0)
            nu_k = alpha + z_ik.sum(axis=0)
            mu_jkml = np.zeros((K, num_classes, num_groups, num_classes)) + beta_kl[None, :, None, :]
            for l in range(num_classes):
                for k in range(num_classes):
                    mu_jkml[:, k, :, l] += y_lji[l].dot(zg_ikm[:, k, :])
            Eq_log_pi = digamma(eta_km) - digamma(eta_km.sum(axis=-1, keepdims=True))
            Eq_log_tau = digamma(nu_k) - digamma(nu_k.sum())
            Eq_log_v = digamma(mu_jkml) - digamma(mu_jkml.sum(axis=-1, keepdims=True))
            zg_ikm[:] = Eq_log_pi[None, :, :] + Eq_log_tau[None, :, None]
            for l in range(num_classes):
                for k in range(num_classes):
                    zg_ikm[:, k, :] += y_lij[l].dot(Eq_log_v[:, k, :, l])
            zg_ikm = np.exp(zg_ikm)
            zg_ikm /= zg_ikm.reshape(N, -1).sum(axis=-1)[:, None, None]
            last = z_ik
            z_ik = zg_ikm.sum(axis=-1)
            if np.allclose(last, z_ik, atol=1e-3):
                break
    return np.argmax(z_ik, axis=1).astype(int)
