# ===========================================================================
# GLAD.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# GLAD (Generative model of Labels, Abilities, and Difficulties) aggregator
# (via crowdkit). Ported from https://github.com/sylyoung/TestEnsemble ; reference
# implementation https://github.com/Toloka/crowd-kit .
# Citation (see gallery/data/benchmark.yml):
#   J. Whitehill, ..., NeurIPS, 2009.
# ===========================================================================
"""GLAD (Whitehill et al., 2009): jointly infer label, ability, and difficulty.

An EM aggregator with a richer generative model than Dawid-Skene: the probability
that a base model votes correctly depends on both that model's ability and the
trial's difficulty. It alternates between estimating consensus labels and
estimating per-model abilities and per-trial difficulties, so an easy trial the
weak models still get right is not treated the same as a hard one.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import crowdkit_predict


class GLAD(VoteCombiner):
    """EM aggregator over per-model ability and per-trial difficulty (crowdkit)."""

    name = "GLAD"

    def aggregate(self, votes: np.ndarray) -> np.ndarray:
        from crowdkit.aggregation import GLAD as _GLAD

        return crowdkit_predict(votes, _GLAD())
