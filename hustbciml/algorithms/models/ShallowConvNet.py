# ===========================================================================
# ShallowConvNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Reference implementation: https://github.com/braindecode/braindecode
#
# Reference (IEEE BibTeX):
#   @Article{Schirrmeister2017,
#     author  = {Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer, Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
#     journal = {Human Brain Mapping},
#     title   = {Deep Learning with Convolutional Neural Networks for {EEG} Decoding and Visualization},
#     year    = {2017},
#     number  = {11},
#     pages   = {5391-5420},
#     volume  = {38},
#     doi     = {10.1002/hbm.23730},
#   }
# ===========================================================================
"""ShallowConvNet (Schirrmeister et al., 2017) — a strong, compact MI baseline.

Temporal conv -> spatial conv -> square -> average-pool -> log -> dropout, i.e.
a learnable band-power feature. Feature-extractor only; ``out_features`` is
sized by a dummy forward so it adapts to any (C, T).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hustbciml.core.stages import Backbone


class _Square(nn.Module):
    def forward(self, x):
        return x * x


class _Log(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(Backbone):
    task_name = "classification"

    def __init__(self, n_chans, n_times, n_classes, sfreq,
                 n_filters=40, drop=0.5, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, 25)),
            nn.Conv2d(n_filters, n_filters, (n_chans, 1), bias=False),
            nn.BatchNorm2d(n_filters),
            _Square(),
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            _Log(),
            nn.Dropout(drop),
        )
        with torch.no_grad():
            self.out_features = self._feat(torch.zeros(1, 1, n_chans, n_times)).shape[1]

    def _feat(self, x):
        return self.net(x).flatten(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._feat(x)
