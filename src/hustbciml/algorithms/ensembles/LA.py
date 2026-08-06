# ===========================================================================
# LA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.

# Credit chain († = co-first authors; every node except the integrator carries its GitHub link):
#   Original authors:    Yi Yang, Zhong-Qiu Zhao, Quan Bai, Qing Liu, Weihua Li (2024) — "A Lightweight, Effective, and Efficient Model for Label Aggregation in Crowdsourcing", ACM Trans. Knowl. Discov. Data
#                        Original code: https://github.com/yyang318/LA_onepass (official)
#   Implementation:      Chenhao Liu — Flashingcat/Golden_task-Ensemble (https://github.com/Flashingcat/Golden_task-Ensemble) (LA_twopass.py, the two-pass variant this port follows)
#   Current code:        Siyang Li — sylyoung/TestEnsemble (https://github.com/sylyoung/TestEnsemble) (vendored from)
#   Integrated by:       Siyang Li <lsyyoungll@gmail.com> — HUST-BCIML

# References (IEEE BibTeX):
#   @Article{Yang2024LA,
#     author  = {Yang, Y. and others},
#     journal = {ACM Transactions on Knowledge Discovery from Data},
#     title   = {A Lightweight, Effective, and Efficient Model for Label Aggregation in Crowdsourcing},
#     year    = {2024},
#     doi     = {10.1145/3630102},
#   }
# ===========================================================================
"""LA (Yang et al., 2024): lightweight two-pass label aggregation.

A cheap alternative to EM. Pass 1 walks the trials in random order and maintains a
running estimate of each base model's ability under a Beta(alpha, beta) prior,
updating it online as each trial's provisional truth is decided by ability-weighted
vote. Pass 2 re-votes every trial with the final abilities, weighting each vote by
``a_w * C - 1``. Two linear passes, no matrix work; ties broken with a local RNG.
"""
from __future__ import annotations

import random

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import fixed_seed


class LA(VoteCombiner):
    """Two-pass ability-weighted aggregator with a Beta(alpha, beta) ability prior."""

    name = "LA"

    def __init__(self, alpha: int = 2, beta: int = 2):
        self.alpha = alpha                               # Beta prior pseudo-counts for model ability
        self.beta = beta

    def aggregate(self, votes: np.ndarray, n_classes: int) -> np.ndarray:
        preds = votes                                    # (K, N) integer hard votes
        alpha, beta = self.alpha, self.beta
        K, N = preds.shape
        C = int(n_classes)
        labels = list(range(C))
        e2wl = {t: [(w, int(preds[w, t])) for w in range(K)] for t in range(N)}
        with fixed_seed(0):
            rng = random.Random(0)
            c = {w: alpha - 1 for w in range(K)}         # correct-count pseudo-counts
            t_cnt = {w: alpha + beta - 2 for w in range(K)}
            a = {w: c[w] / t_cnt[w] for w in range(K)}   # each model's current ability estimate
            items = list(e2wl.keys())
            rng.shuffle(items)
            truths = {}
            for item in items:                           # pass 1: online ability estimation
                tally = {}
                for w, lab in e2wl[item]:
                    tally[lab] = tally.get(lab, 0) + a[w]
                # ``-inf``, not ``-1``: vote weights are ``a[w] * C - 1`` and go
                # negative for a below-chance worker, so a class whose whole tally
                # fell below -1 was skipped even when it was the best available —
                # leaving ``cand`` empty and crashing ``rng.choice``, or dropping a
                # legitimate candidate and biasing the output.
                best, cand = -np.inf, []
                for cl in labels:
                    if cl not in tally:
                        continue
                    if tally[cl] > best:
                        best, cand = tally[cl], [cl]
                    elif tally[cl] == best:
                        cand.append(cl)
                truths[item] = rng.choice(cand)
                for w, lab in e2wl[item]:                 # update ability from this trial's truth
                    t_cnt[w] += 1
                    if lab == truths[item]:
                        c[w] += 1
                    a[w] = c[w] / t_cnt[w]
            out = []
            for item in range(N):                        # pass 2: re-vote with final abilities
                tally = {}
                for w, lab in e2wl[item]:
                    tally[lab] = tally.get(lab, 0) + (a[w] * C - 1)
                # ``-inf``, not ``-1``: vote weights are ``a[w] * C - 1`` and go
                # negative for a below-chance worker, so a class whose whole tally
                # fell below -1 was skipped even when it was the best available —
                # leaving ``cand`` empty and crashing ``rng.choice``, or dropping a
                # legitimate candidate and biasing the output.
                best, cand = -np.inf, []
                for cl in labels:
                    if cl not in tally:
                        continue
                    if tally[cl] > best:
                        best, cand = tally[cl], [cl]
                    elif tally[cl] == best:
                        cand.append(cl)
                out.append(rng.choice(cand))
        return np.array(out, dtype=int)
