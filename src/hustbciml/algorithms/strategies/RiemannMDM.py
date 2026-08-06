# ===========================================================================
# RiemannMDM.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.

# Credit chain († = co-first authors; every node except the integrator carries its GitHub link):
#   Original authors:    Alexandre Barachant, Stéphane Bonnet, Marco Congedo, Christian Jutten (2012) — "Multiclass Brain-Computer Interface Classification by Riemannian Geometry", IEEE Trans. Biomed. Eng.
#                        Original code: https://github.com/pyRiemann/pyRiemann (reference implementation)
#   Implementation:      Alexandre Barachant & pyRiemann contributors — pyRiemann/pyRiemann (https://github.com/pyRiemann/pyRiemann)
#   Current code:        Siyang Li — implemented in HUST-BCIML with pyRiemann components
#   Integrated by:       Siyang Li <lsyyoungll@gmail.com> — HUST-BCIML

# References (IEEE BibTeX):
#   @Article{Barachant2012,
#     author  = {Barachant, Alexandre and Bonnet, St\'ephane and Congedo, Marco and Jutten, Christian},
#     journal = {IEEE Transactions on Biomedical Engineering},
#     title   = {Multiclass Brain-Computer Interface Classification by {R}iemannian Geometry},
#     year    = {2012},
#     number  = {4},
#     pages   = {920-928},
#     volume  = {59},
#     doi     = {10.1109/TBME.2011.2172210},
#   }
# ===========================================================================
"""Riemannian MDM — Minimum Distance to Riemannian Mean (Barachant et al., 2012).

A classical, network-free covariance-space classifier: estimate each trial's
spatial covariance matrix, compute the geometric (Riemannian) mean covariance of
each class on the source, and classify a target trial by the class whose mean is
closest under the affine-invariant Riemannian metric.

Not present in the DeepTransferEEG repo (which ships CSP+LDA); implemented here
following the official pyriemann reference (``pyriemann.estimation.Covariances``
+ ``pyriemann.classification.MDM``). ``mode='fit'`` — fit on the EA-aligned
source, predict the aligned target; the neural model is unused. Requires
pyriemann (imported lazily).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy


class RiemannMDM(Strategy):
    mode = "fit"

    def __init__(self, **_):
        self.cov = None
        self.mdm = None

    def fit(self, model, source: EEGEpochs, ctx: RunContext):
        from pyriemann.classification import MDM
        from pyriemann.estimation import Covariances
        self.cov = Covariances(estimator="oas")             # shrinkage — robust, well-conditioned
        covs = self.cov.fit_transform(source.X.astype(np.float64))
        self.mdm = MDM(metric="riemann").fit(covs, source.y)
        return model                                        # neural model unused

    def predict(self, model, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        covs = self.cov.transform(target.X.astype(np.float64))
        y_pred = self.mdm.predict(covs)
        # Only the "this pyriemann version has no predict_proba" case is a
        # legitimate fallback. A blanket ``except Exception`` also swallowed shape
        # mismatches and numerical failures, substituting a fabricated softmax for
        # a real bug in a measurement path — the AUC would still print.
        if hasattr(self.mdm, "predict_proba"):
            y_score = self.mdm.predict_proba(covs)
        else:                                               # older pyriemann: softmax of -distances
            d = self.mdm.transform(covs)
            e = np.exp(-(d - d.min(axis=1, keepdims=True)))
            y_score = e / e.sum(axis=1, keepdims=True)
        y_score = np.asarray(y_score, dtype=np.float64)
        if y_score.shape != (len(covs), ctx.cfg.n_classes) or not np.all(np.isfinite(y_score)):
            raise ValueError(
                f"RiemannMDM produced scores of shape {y_score.shape} "
                f"(expected {(len(covs), ctx.cfg.n_classes)}) or containing non-finite values."
            )
        return np.asarray(y_pred, dtype=np.int64), y_score
