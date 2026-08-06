# ===========================================================================
# PM.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.

# Credit chain († = co-first authors; every node except the integrator carries its GitHub link):
#   Original authors:    Qi Li, Yaliang Li, Jing Gao, Bo Zhao, Wei Fan, Jiawei Han (2014) — "Resolving Conflicts in Heterogeneous Data by Truth Discovery and Source Reliability Estimation (CRH)", Proc. ACM SIGMOD
#                        Original code: no official release (verified 2026-08-06); PM is the hard-label specialization of Participant-Mine/CRH
#   Implementation:      Chenhao Liu — Flashingcat/Golden_task-Ensemble (https://github.com/Flashingcat/Golden_task-Ensemble) (pm.py, implemented following the paper)
#   Current code:        Siyang Li — sylyoung/TestEnsemble (https://github.com/sylyoung/TestEnsemble) (vendored from)
#   Integrated by:       Siyang Li <lsyyoungll@gmail.com> — HUST-BCIML

# References (IEEE BibTeX):
#   @InProceedings{Li2014PM,
#     author    = {Li, Q. and others},
#     booktitle = {Proc. ACM SIGMOD Int. Conf. Management of Data},
#     title     = {Resolving Conflicts in Heterogeneous Data by Truth Discovery and Source Reliability Estimation},
#     year      = {2014},
#     doi       = {10.1145/2588555.2610509},
#   }
# ===========================================================================
"""Three-round TestEnsemble PM/CRH truth-discovery implementation.

The current consensus initializes from majority vote. Each round gives a base model a
weight equal to the negative logarithm of its maximum-normalized disagreement with the
current consensus, then re-estimates the consensus from weighted one-hot votes. This
matches the practical max-normalization scheme in the cited CRH method, but the paper
describes alternating optimization to convergence rather than a fixed round count.
The archived HUST runner uses three rounds while TestEnsemble's published driver used
one; the configured count is part of method identity.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import fixed_seed, onehot


class PM(VoteCombiner):
    """Truth discovery: weight = -log(normalized disagreement with the current truth)."""

    name = "PM"

    def __init__(self, n_iter: int = 3):
        if int(n_iter) < 1:
            raise ValueError("PM n_iter must be at least one")
        self.n_iter = int(n_iter)                        # truth <-> weight refinement rounds

    def aggregate(self, votes: np.ndarray, n_classes: int) -> np.ndarray:
        preds = votes                                    # (K, N) integer hard votes
        K, N = preds.shape
        C = int(n_classes)
        with fixed_seed(0):
            # provisional truth = majority vote (local-seed tie-break)
            counts = np.zeros((C, N))
            for i in range(K):
                for j in range(N):
                    counts[preds[i, j], j] += 1
            rng = np.random.RandomState(0)
            truth = np.array([rng.choice(np.flatnonzero(counts[:, j] == counts[:, j].max()))
                              for j in range(N)])
            oh = onehot(preds, C)                              # {0,1}
            oh = np.where(oh == 1, 1, -1)                      # {-1,+1} as in PM.py
            weight = np.zeros(K)
            for _ in range(self.n_iter):
                for w in range(K):
                    dif = float(np.sum(preds[w, :] != truth)) or 1e-8   # disagreement with current truth
                    weight[w] = dif
                # Normalise by *this* round's largest disagreement. Carrying a
                # running maximum across rounds (which only ever grew) meant that
                # once the estimates improved, every later round was scaled against
                # a stale denominator from an earlier one, making the truth-discovery
                # update depend on the history rather than the current round.
                wmax = weight.max() or 1e-8
                weight = weight / wmax
                weight = -np.log(weight + 1e-7) + 1e-7        # low disagreement -> high weight
                scores = np.einsum("a,abc->bc", weight, oh)
                # Resolve ties the same way the initialisation does — uniformly at
                # random among the tied classes — instead of ``argmax``'s
                # first-maximum rule, which deterministically favours class 0 and
                # shows up as a class-prior artifact on balanced binary MI tasks.
                truth = np.array([
                    rng.choice(np.flatnonzero(row == row.max())) for row in scores])
        return truth.astype(int)
