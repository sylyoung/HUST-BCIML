# ===========================================================================
# Voting.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Plain hard majority voting — the label-only baseline every other combiner is
# measured against. Original authors' code: https://github.com/sylyoung/TestEnsemble
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
# ===========================================================================
"""Hard majority voting over the base models' predictions.

The simplest black-box combiner and the baseline of the ensemble-learning family:
every base model casts one hard vote (the argmax of its class scores) and the
consensus is the most-voted class per trial. It gives every model equal weight, so
it is what a combiner has to beat to justify estimating per-model reliabilities.
Ties are broken with a local fixed seed for reproducibility (see
``_common.majority_vote``).
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import Combiner

from ._common import majority_vote


class Voting(Combiner):
    """Plain hard majority vote over the K base models' hard labels (baseline)."""

    name = "voting"

    def combine(self, scores: np.ndarray) -> np.ndarray:
        # scores: (K, N, C). majority_vote argmaxes each model's scores to a hard
        # vote and returns the per-trial most-voted class (local-seed tie-break).
        return majority_vote(scores)
