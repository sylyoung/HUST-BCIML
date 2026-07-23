# ===========================================================================
# MMSR.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# M-MSR (Matrix Mean-Subsequence-Reduced) worker-skill aggregator (via crowdkit).
# Ported from https://github.com/sylyoung/TestEnsemble ; reference implementation
# https://github.com/Toloka/crowd-kit .
# Citation (see gallery/data/benchmark.yml and references.bib):
#   Q. Ma and A. Olshevsky, NeurIPS, 2020.
# ===========================================================================
"""M-MSR (Ma & Olshevsky, 2020): worker skill from the agreement matrix.

Recovers each base model's skill from the pairwise inter-model agreement matrix by
robust rank-one matrix completion (the mean-subsequence-reduced estimator), then
weights each model's vote by that recovered skill. Unlike the EM aggregators it
never forms per-item posteriors; it reads reliability straight from how often pairs
of models agree, which is robust when a few models are adversarially bad.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import crowdkit_predict


class MMSR(VoteCombiner):
    """Matrix mean-subsequence-reduced worker-skill aggregator (crowdkit)."""

    name = "M-MSR"

    def aggregate(self, votes: np.ndarray) -> np.ndarray:
        from crowdkit.aggregation import MMSR as _MMSR

        return crowdkit_predict(votes, _MMSR())
