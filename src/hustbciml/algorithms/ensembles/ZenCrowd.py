# ===========================================================================
# ZenCrowd.py  —  HUST-BCIML EEG-decoding benchmark

# Original authors:    Gianluca Demartini, Djellel Eddine Difallah, Philippe Cudré-Mauroux (2012) — "ZenCrowd: Leveraging Probabilistic Reasoning and Crowdsourcing Techniques for Large-Scale Entity Linking", Proc. WWW
#                      Original code: no official release (verified 2026-08-06)
# Implementation:      Chenhao Liu — Flashingcat/Golden_task-Ensemble (https://github.com/Flashingcat/Golden_task-Ensemble) (Z_C.py, simplified single-coin EM variant, documented as not paper-faithful)
# Current code:        Siyang Li — sylyoung/TestEnsemble (https://github.com/sylyoung/TestEnsemble) (vendored from)

# References (IEEE BibTeX):
#   @InProceedings{Demartini2012,
#     author    = {Demartini, Gianluca and Difallah, Djellel Eddine and Cudr{\'e}-Mauroux, Philippe},
#     booktitle = {Proc. 21st Int. Conf. World Wide Web (WWW)},
#     title     = {{ZenCrowd}: Leveraging Probabilistic Reasoning and Crowdsourcing Techniques for Large-Scale Entity Linking},
#     year      = {2012},
#     doi       = {10.1145/2187836.2187900},
#   }
# ===========================================================================
"""Simplified TestEnsemble single-coin EM baseline labeled ``ZenCrowd``.

This is the lightweight implementation vendored from TestEnsemble, not the full
ZenCrowd model described by Demartini et al. Each base model has one reliability
scalar rather than a full confusion matrix. The E-step multiplies that scalar on the
voted class and a smoothed complement otherwise; the M-step replaces reliability by
the mean posterior mass on the model's votes. It aggregates hard votes without target
labels. The HUST legacy artifacts used 20 passes; TestEnsemble's published driver used
one. The pass count is therefore part of method identity and is serialized by the new
runners.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import fixed_seed


class ZenCrowd(VoteCombiner):
    """EM aggregator with a single per-model reliability scalar (vendored numpy)."""

    name = "ZenCrowd"

    def __init__(self, n_iter: int = 20):
        if int(n_iter) < 1:
            raise ValueError("ZenCrowd n_iter must be at least one")
        self.n_iter = int(n_iter)                        # serialized EM pass count

    def aggregate(self, votes: np.ndarray, n_classes: int) -> np.ndarray:
        preds = votes                                    # (K, N) integer hard votes
        K, N = preds.shape
        C = int(n_classes)
        labels = list(range(C))
        with fixed_seed(0):
            wm = {w: 0.8 for w in range(K)}                    # worker reliabilities
            e2lpd = None
            for _ in range(self.n_iter):
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
