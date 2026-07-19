# ===========================================================================
# CSP_LDA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Reference implementation: https://scikit-learn.org
#
# Reference (IEEE BibTeX):
#   @Article{Ramoser2000,
#     author  = {Ramoser, Herbert and M\"uller-Gerking, Johannes and Pfurtscheller, Gert},
#     journal = {IEEE Trans. Rehabilitation Engineering},
#     title   = {Optimal Spatial Filtering of Single Trial {EEG} During Imagined Hand Movement},
#     year    = {2000},
#     number  = {4},
#     pages   = {441-446},
#     volume  = {8},
#     doi     = {10.1109/86.895946},
#   }
# ===========================================================================
"""CSP + LDA — the classical motor-imagery baseline (Ramoser et al., 2000;
Blankertz et al., 2008), as used in DeepTransferEEG ``ml/feature.py``.

Common Spatial Patterns learns supervised spatial filters that maximize the
band-power variance ratio between the two classes; the log-power of the top
components feeds a Linear Discriminant Analysis classifier. No neural network,
no gradient loop — ``mode='fit'``: fit CSP+LDA on the (EA-aligned) source, then
transform-and-predict the aligned target. The neural ``model`` argument is
unused. Requires MNE + scikit-learn (imported lazily).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy


class CSP_LDA(Strategy):
    mode = "fit"

    def __init__(self, **_):
        self.csp = None
        self.lda = None

    def fit(self, model, source: EEGEpochs, ctx: RunContext):
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        n_comp = min(10, source.n_channels)                 # DeepTransferEEG uses 10
        self.csp = CSP(n_components=n_comp)
        feats = self.csp.fit_transform(source.X.astype(np.float64), source.y)
        self.lda = LinearDiscriminantAnalysis().fit(feats, source.y)
        return model                                        # neural model unused

    def predict(self, model, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        feats = self.csp.transform(target.X.astype(np.float64))
        y_score = self.lda.predict_proba(feats)
        return y_score.argmax(1), y_score
