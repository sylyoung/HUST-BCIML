# ===========================================================================
# MACE.py  —  HUST-BCIML EEG-decoding benchmark

# Original authors:    Dirk Hovy, Taylor Berg-Kirkpatrick, Ashish Vaswani, Eduard Hovy (2013) — "Learning Whom to Trust with MACE", Proc. NAACL-HLT
#                      Original code: https://github.com/Toloka/crowd-kit (reference implementation)
# Implementation:      Toloka contributors — Toloka/crowd-kit (https://github.com/Toloka/crowd-kit)
# Current code:        Siyang Li — sylyoung/TestEnsemble (https://github.com/sylyoung/TestEnsemble) (wrapper ported from)
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
    backend = "crowdkit.aggregation.MACE"
    backend_distribution = "crowd-kit"

    def __init__(self, n_restarts: int = 10, n_iter: int = 50,
                 method: str = "vb", smoothing: float = 0.1,
                 default_noise: float = 0.5, alpha: float = 0.5,
                 beta: float = 0.5, random_state: int = 0, verbose: int = 0):
        if int(n_restarts) < 1 or int(n_iter) < 1:
            raise ValueError("MACE restart and iteration counts must be at least one")
        if method not in {"vb", "em"}:
            raise ValueError("MACE method must be 'vb' or 'em'")
        if float(smoothing) <= 0 or not 0 <= float(default_noise) <= 1:
            raise ValueError("MACE requires smoothing > 0 and default_noise in [0, 1]")
        if float(alpha) <= 0 or float(beta) <= 0:
            raise ValueError("MACE alpha and beta must be positive")
        self.n_restarts = int(n_restarts)
        self.n_iter = int(n_iter)
        self.method = method
        self.smoothing = float(smoothing)
        self.default_noise = float(default_noise)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.random_state = int(random_state)
        self.verbose = int(verbose)

    def aggregate(self, votes: np.ndarray, n_classes: int) -> np.ndarray:
        from crowdkit.aggregation import MACE as _MACE

        return crowdkit_predict(votes, _MACE(
            n_restarts=self.n_restarts, n_iter=self.n_iter, method=self.method,
            smoothing=self.smoothing, default_noise=self.default_noise,
            alpha=self.alpha, beta=self.beta, random_state=self.random_state,
            verbose=self.verbose,
        ))
