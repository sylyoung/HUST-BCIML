# ===========================================================================
# DeepConvNet.py  —  HUST-BCIML EEG-decoding benchmark
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
"""DeepConvNet (Schirrmeister et al., 2017) — a deeper conv baseline.

Four conv/BN/ELU/max-pool blocks. Feature-extractor only; ``out_features`` is
sized by a dummy forward. Needs a reasonably long window (works at BNCI
T=1001 and toy T=128).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hustbciml.core.stages import Backbone


def _block(cin, cout, kt, spatial=None, drop=0.5):
    # One conv-pool stage of the paper: a temporal conv over width `kt`, then
    # BN, ELU, a width-2 max-pool that halves the time axis, and dropout.
    layers = [nn.Conv2d(cin, cout, (1, kt))]
    if spatial is not None:  # first block folds in the spatial conv
        # Only the first stage adds a spatial conv (one (n_chans, 1) kernel per
        # map) so that later stages operate on a single spatial row.
        layers.append(nn.Conv2d(cout, cout, (spatial, 1), bias=False))
    layers += [nn.BatchNorm2d(cout), nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout(drop)]
    return nn.Sequential(*layers)


class DeepConvNet(Backbone):
    task_name = "classification"

    def __init__(self, n_chans, n_times, n_classes, sfreq, drop=0.5, **_):
        super().__init__()
        # The four conv-pool stages of the paper, with the channel count growing
        # 25 -> 50 -> 100 -> 200. The first stage also carries the spatial conv
        # over electrodes. Each stage halves the time axis via its max-pool.
        self.net = nn.Sequential(
            _block(1, 25, 5, spatial=n_chans, drop=drop),   # stage 1: temporal + spatial conv
            _block(25, 50, 5, drop=drop),                   # stage 2: temporal conv
            _block(50, 100, 5, drop=drop),                  # stage 3: temporal conv
            _block(100, 200, 5, drop=drop),                 # stage 4: temporal conv
        )
        # Four conv-and-pool stages shrink the time axis by a data-dependent
        # amount, so size the flat feature width with a dummy forward.
        with torch.no_grad():
            self.out_features = self._feat(torch.zeros(1, 1, n_chans, n_times)).shape[1]

    def _feat(self, x):
        return self.net(x).flatten(1)             # stacked conv-pool maps -> flat feature vector

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._feat(x)
