# ===========================================================================
# PM.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# PM truth-discovery aggregator, vendored (pure numpy) from TestEnsemble/algs/PM.py
# (https://github.com/sylyoung/TestEnsemble).
#
# References (IEEE BibTeX):
#   @InProceedings{Li2014PM,
#     author    = {Li, Q. and others},
#     booktitle = {Proc. ACM SIGMOD Int. Conf. Management of Data},
#     title     = {Resolving Conflicts in Heterogeneous Data by Truth Discovery and Source Reliability Estimation},
#     year      = {2014},
#     doi       = {10.1145/2588555.2610509},
#   }
# ===========================================================================
"""PM (Li et al., 2014): truth discovery by source reliability.

An iterative truth-discovery aggregator. Start from the majority vote as the
provisional truth; give each base model a weight equal to the negative log of its
normalized disagreement with the current truth (so a model that rarely disagrees is
trusted more); then re-estimate the truth as the weighted one-hot vote. A few
iterations suffice. Votes are held in ±1 form, matching the reference PM.py.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import fixed_seed, onehot


class PM(VoteCombiner):
    """Truth discovery: weight = -log(normalized disagreement with the current truth)."""

    name = "PM"

    def __init__(self, n_iter: int = 3):
        self.n_iter = n_iter                             # truth <-> weight refinement rounds

    def aggregate(self, votes: np.ndarray) -> np.ndarray:
        preds = votes                                    # (K, N) integer hard votes
        K, N = preds.shape
        C = int(preds.max()) + 1
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
            wmax = 0.0
            for _ in range(self.n_iter):
                for w in range(K):
                    dif = float(np.sum(preds[w, :] != truth)) or 1e-8   # disagreement with current truth
                    weight[w] = dif
                    wmax = max(wmax, weight[w])
                weight /= wmax
                weight = -np.log(weight + 1e-7) + 1e-7        # low disagreement -> high weight
                truth = np.argmax(np.einsum("a,abc->bc", weight, oh), axis=1)
        return truth.astype(int)
