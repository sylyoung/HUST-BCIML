# ===========================================================================
# Wawa.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.

# Credit chain († = co-first authors; every node except the integrator carries its GitHub link):
#   Original authors:    Worker Agreement With Aggregate — crowd-kit heuristic, no standalone paper.
#   Implementation:      Toloka contributors — Toloka/crowd-kit (https://github.com/Toloka/crowd-kit)
#   Current code:        Siyang Li — sylyoung/TestEnsemble (https://github.com/sylyoung/TestEnsemble) (wrapper ported from)
#   Integrated by:       Siyang Li <lsyyoungll@gmail.com> — HUST-BCIML
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
    backend = "crowdkit.aggregation.Wawa"
    backend_distribution = "crowd-kit"

    def aggregate(self, votes: np.ndarray, n_classes: int) -> np.ndarray:
        from crowdkit.aggregation import Wawa as _Wawa

        return crowdkit_predict(votes, _Wawa())
