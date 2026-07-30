# ===========================================================================
# MACE.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# MACE (Multi-Annotator Competence Estimation) aggregator (via crowdkit).
# Ported from https://github.com/sylyoung/TestEnsemble ; reference implementation
# https://github.com/Toloka/crowd-kit .
# Citation (see gallery/data/benchmark.yml):
#   D. Hovy, ..., NAACL-HLT, 2013.
# ===========================================================================
"""MACE (Hovy et al., 2013): separate competent labelling from spamming.

A variational Bayesian model that gives each base model two behaviors: label
competently, or "spam" with a model-specific label distribution independent of the
truth. Inferring the mix down-weights models that mostly spam, so a few unreliable
sources do not drag the consensus. Aggregates the hard votes only, no target labels.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import crowdkit_predict


class MACE(VoteCombiner):
    """Multi-annotator competence estimation, down-weighting spamming models (crowdkit)."""

    name = "MACE"

    def aggregate(self, votes: np.ndarray) -> np.ndarray:
        from crowdkit.aggregation import MACE as _MACE

        return crowdkit_predict(votes, _MACE())
