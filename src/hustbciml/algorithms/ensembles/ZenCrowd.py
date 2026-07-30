# ===========================================================================
# ZenCrowd.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# ZenCrowd EM aggregator, vendored (pure numpy) from TestEnsemble/algs/ZC.py
# (https://github.com/sylyoung/TestEnsemble).
#
# References (IEEE BibTeX):
#   @InProceedings{Demartini2012,
#     author    = {Demartini, Gianluca and Difallah, Djellel Eddine and Cudr{\'e}-Mauroux, Philippe},
#     booktitle = {Proc. 21st Int. Conf. World Wide Web (WWW)},
#     title     = {{ZenCrowd}: Leveraging Probabilistic Reasoning and Crowdsourcing Techniques for Large-Scale Entity Linking},
#     year      = {2012},
#     doi       = {10.1145/2187836.2187900},
#   }
# ===========================================================================
"""ZenCrowd (Demartini et al., 2012): EM with one reliability scalar per model.

The lightest EM aggregator here. Each base model has a single reliability scalar
(not a full confusion matrix). E-step: each trial's label posterior multiplies, for
every model, that model's reliability on the class it voted and the smoothed
complement otherwise. M-step: each model's reliability becomes the mean posterior
mass on the classes it voted. Ties in the final posterior are broken with a local
fixed seed. Aggregates the hard votes only, no target labels.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import fixed_seed


class ZenCrowd(VoteCombiner):
    """EM aggregator with a single per-model reliability scalar (vendored numpy)."""

    name = "ZenCrowd"

    def __init__(self, n_iter: int = 20):
        self.n_iter = n_iter                             # EM sweeps (TestEnsemble default)

    def aggregate(self, votes: np.ndarray) -> np.ndarray:
        preds = votes                                    # (K, N) integer hard votes
        K, N = preds.shape
        C = int(preds.max()) + 1
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
