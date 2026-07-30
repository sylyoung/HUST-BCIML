# ===========================================================================
# Wawa.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Wawa (Worker Agreement With Aggregate) — a crowd-kit heuristic with no separate
# paper. Ported from https://github.com/sylyoung/TestEnsemble ; reference
# implementation https://github.com/Toloka/crowd-kit .
# ===========================================================================
"""Wawa — Worker Agreement With Aggregate (crowd-kit heuristic).

A two-pass reweighting heuristic: take the plain majority vote, score each base
model by how often it agrees with that vote, then re-vote with those agreement
weights. There is no separate paper; it is a standard crowd-kit baseline included
so the lab combiners are compared against the obvious agreement-weighted vote.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import crowdkit_predict


class Wawa(VoteCombiner):
    """Reweight base models by their agreement with the majority vote, then re-vote."""

    name = "Wawa"

    def aggregate(self, votes: np.ndarray) -> np.ndarray:
        from crowdkit.aggregation import Wawa as _Wawa

        return crowdkit_predict(votes, _Wawa())
