# ===========================================================================
# Linear.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Linear classification head. No external reference.
# ===========================================================================
"""Linear classification head (gradient head)."""
from __future__ import annotations

import torch
import torch.nn as nn

from hustbciml.core.stages import Head


class Linear(Head):
    is_gradient = True

    def __init__(self, in_features: int, n_classes: int, **_):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)
        # Xavier init, matching DeepTransferEEG feat_classifier
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats)
