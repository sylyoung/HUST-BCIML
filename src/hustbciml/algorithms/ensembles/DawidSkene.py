# ===========================================================================
# DawidSkene.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.

# Credit chain († = co-first authors; every node except the integrator carries its GitHub link):
#   Original authors:    A. P. Dawid, A. M. Skene (1979) — "Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm", J. R. Stat. Soc. Ser. C
#                        Original code: https://github.com/Toloka/crowd-kit (reference implementation)
#   Implementation:      Toloka contributors — Toloka/crowd-kit (https://github.com/Toloka/crowd-kit)
#   Current code:        Siyang Li — sylyoung/TestEnsemble (https://github.com/sylyoung/TestEnsemble) (wrapper ported from)
#   Integrated by:       Siyang Li <lsyyoungll@gmail.com> — HUST-BCIML

# References (IEEE BibTeX):
#   @Article{DawidSkene1979,
#     author  = {Dawid, A. P. and Skene, A. M.},
#     journal = {Journal of the Royal Statistical Society: Series C (Applied Statistics)},
#     title   = {Maximum Likelihood Estimation of Observer Error-Rates Using the {EM} Algorithm},
#     year    = {1979},
#     doi     = {10.2307/2346806},
#   }
# ===========================================================================
"""Dawid & Skene (1979): EM over per-worker confusion matrices.

The classic crowd-labelling aggregator. Treats each base model as a noisy
annotator with its own full class-confusion matrix, and alternates (EM) between
estimating the consensus label of each trial and re-estimating every model's
confusion matrix from those labels — all from the hard votes alone, with no target
labels. More reliable models end up with sharper confusion matrices and thus more
influence on the consensus.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import crowdkit_predict


class DawidSkene(VoteCombiner):
    """Dawid-Skene EM confusion-matrix aggregator (crowdkit, ``n_iter=10``)."""

    name = "Dawid-Skene"
    backend = "crowdkit.aggregation.DawidSkene"
    backend_distribution = "crowd-kit"

    def __init__(self, n_iter: int = 10):
        if int(n_iter) < 1:
            raise ValueError("Dawid-Skene n_iter must be at least one")
        self.n_iter = int(n_iter)

    def aggregate(self, votes: np.ndarray, n_classes: int) -> np.ndarray:
        from crowdkit.aggregation import DawidSkene as _DawidSkene

        return crowdkit_predict(votes, _DawidSkene(n_iter=self.n_iter))
