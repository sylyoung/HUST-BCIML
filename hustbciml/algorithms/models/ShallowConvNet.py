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
    # Squaring nonlinearity. On a band-limited signal the square is proportional
    # to instantaneous power, so this is the "square" step of the band-power
    # pipeline the paper designed to mimic filter-bank common spatial patterns.
    def forward(self, x):
        return x * x


class _Log(nn.Module):
    # Log nonlinearity, the final step of the band-power pipeline. Taking the log
    # of the pooled power compresses its dynamic range. The clamp keeps the
    # argument strictly positive so the log is finite.
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(Backbone):
    task_name = "classification"

    def __init__(self, n_chans, n_times, n_classes, sfreq,
                 n_filters=40, drop=0.5, **_):
        super().__init__()
        # The band-power pipeline of the paper, in order: temporal conv, spatial
        # conv, square, average-pool, log. Input is (B, 1, C, T).
        self.net = nn.Sequential(
            # Temporal conv: `n_filters` width-25 kernels over time, learning
            # band-pass-like filters (the analogue of the filter bank).
            nn.Conv2d(1, n_filters, (1, 25)),
            # Spatial conv: one (n_chans, 1) kernel per filter spanning all
            # electrodes, the learnable analogue of a CSP spatial filter. BN
            # follows and there is no nonlinearity yet, so the next step squares
            # a still-band-limited signal.
            nn.Conv2d(n_filters, n_filters, (n_chans, 1), bias=False),
            nn.BatchNorm2d(n_filters),
            _Square(),                                # -> instantaneous power
            # Average-pool over a 75-sample window (stride 15) estimates mean
            # power in each sliding window, the temporal smoothing of band power.
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            _Log(),                                   # log of pooled power
            nn.Dropout(drop),
        )
        # Length after the strided pool depends on T, so size the flat feature
        # width with a dummy forward instead of a closed form.
        with torch.no_grad():
            self.out_features = self._feat(torch.zeros(1, 1, n_chans, n_times)).shape[1]

    def _feat(self, x):
        return self.net(x).flatten(1)             # band-power maps -> flat feature vector

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._feat(x)
