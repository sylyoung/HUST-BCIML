# ===========================================================================
# GLAD.py  —  HUST-BCIML EEG-decoding benchmark

# Original authors:    Jacob Whitehill, Ting-fan Wu, Jacob Bergsma, Javier Movellan, Paul Ruvolo (2009) — "Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown Expertise", Proc. NeurIPS
#                      Original code: https://github.com/Toloka/crowd-kit (reference implementation)
# Implementation:      Toloka contributors — Toloka/crowd-kit (https://github.com/Toloka/crowd-kit)
# Current code:        Siyang Li — sylyoung/TestEnsemble (https://github.com/sylyoung/TestEnsemble) (wrapper ported from)
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
    backend = "crowdkit.aggregation.GLAD"
    backend_distribution = "crowd-kit"

    def __init__(self, n_iter: int = 100, tol: float = 1e-5,
                 m_step_max_iter: int = 25, m_step_tol: float = 0.01):
        if int(n_iter) < 1 or int(m_step_max_iter) < 1:
            raise ValueError("GLAD iteration counts must be at least one")
        if float(tol) <= 0 or float(m_step_tol) <= 0:
            raise ValueError("GLAD tolerances must be positive")
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.m_step_max_iter = int(m_step_max_iter)
        self.m_step_tol = float(m_step_tol)

    def aggregate(self, votes: np.ndarray, n_classes: int) -> np.ndarray:
        from crowdkit.aggregation import GLAD as _GLAD

        model = _GLAD(
            n_iter=self.n_iter, tol=self.tol, silent=True,
            labels_priors=None, alphas_priors_mean=None, betas_priors_mean=None,
            m_step_max_iter=self.m_step_max_iter, m_step_tol=self.m_step_tol,
        )
        return crowdkit_predict(votes, model)
