# ===========================================================================
# DawidSkene.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Dawid-Skene EM crowd-labelling aggregator (via the crowdkit library).
# Ported from https://github.com/sylyoung/TestEnsemble ; reference implementation
# https://github.com/Toloka/crowd-kit . Called with the same defaults as the lab's
# TestEnsemble/ensemble.py.
#
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

    def aggregate(self, votes: np.ndarray) -> np.ndarray:
        from crowdkit.aggregation import DawidSkene as _DawidSkene

        # n_iter=10 matches the lab's TestEnsemble/ensemble.py call.
        return crowdkit_predict(votes, _DawidSkene(n_iter=10))
