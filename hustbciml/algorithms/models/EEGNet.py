# ===========================================================================
# EEGNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/vlawhern/arl-eegmodels
#
# Reference (IEEE BibTeX):
#   @Article{Lawhern2018,
#     author  = {Lawhern, Vernon J. and Solon, Amelia J. and Waytowich, Nicholas R. and Gordon, Stephen M. and Hung, Chou P. and Lance, Brent J.},
#     journal = {Journal of Neural Engineering},
#     title   = {{EEGNet}: A Compact Convolutional Neural Network for {EEG}-Based Brain-Computer Interfaces},
#     year    = {2018},
#     number  = {5},
#     pages   = {056013},
#     volume  = {15},
#     doi     = {10.1088/1741-2552/aace8c},
#   }
# ===========================================================================
"""EEGNet backbone (Lawhern et al., 2018), the DeepTransferEEG configuration.

Feature-extractor only: block1 (temporal conv + depthwise spatial conv) +
block2 (separable conv). ``out_features`` = F2 * (T // 32), so the Head can be
sized generically. Defaults (F1=4, D=2, F2=8, kernLength = sfreq // 2,
dropout=0.25) match ``tl/utils/network.backbone_net``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hustbciml.core.stages import Backbone


class EEGNet(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 F1: int = 4, D: int = 2, F2: int = 8,
                 kern_length: int = None, dropout: float = 0.25, **_):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        kern_length = int(sfreq // 2) if kern_length is None else kern_length
        self.kern_length = kern_length

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((kern_length // 2 - 1, kern_length - kern_length // 2, 0, 0)),
            nn.Conv2d(1, F1, (1, kern_length), stride=1, bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False),  # depthwise
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=dropout),
        )
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False),  # separable pt1
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),                      # separable pt2
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=dropout),
        )
        self.out_features = F2 * (n_times // (4 * 8))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        out = self.block2(out)
        return out.reshape(out.size(0), -1)
