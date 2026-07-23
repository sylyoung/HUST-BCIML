# ===========================================================================
# RiemannMDM.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Reference implementation: https://github.com/pyRiemann/pyRiemann
#
# Reference (IEEE BibTeX):
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
        try:
            y_score = self.mdm.predict_proba(covs)
        except Exception:                                   # older pyriemann: softmax of -distances
            d = self.mdm.transform(covs)
            e = np.exp(-(d - d.min(axis=1, keepdims=True)))
            y_score = e / e.sum(axis=1, keepdims=True)
        return np.asarray(y_pred, dtype=np.int64), y_score
