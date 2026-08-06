# ===========================================================================
# Implementation of: Deep4Net architecture transfer

# Credit chain († = co-first authors; every node except the integrator carries its GitHub link):
#   Original authors:    Robin Tibor Schirrmeister, Jost Tobias Springenberg, Lukas Dominique Josef Fiederer, Martin Glasstetter, Katharina Eggensperger, Michael Tangermann, Frank Hutter, Wolfram Burgard, Tonio Ball (2017) — "Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization", Hum. Brain Mapp.
#                        Original code: https://github.com/braindecode/braindecode (pinned f7562e9)
#   Implementation:      Robin T. Schirrmeister & Braindecode contributors — braindecode/braindecode (https://github.com/braindecode/braindecode) (official)
#   Current code:        Siyang Li — braindecode/braindecode (https://github.com/braindecode/braindecode), pinned commit f7562e9 (architecture transfer)
#   Integrated by:       Siyang Li <lsyyoungll@gmail.com> — HUST-BCIML

# References (IEEE BibTeX):
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
"""Braindecode Deep4Net feature architecture under the HUST benchmark protocol.

The temporal kernels, pooling, bias choices, dropout placement, batch
normalization, and feature-layer initialization follow the pinned Deep4Net
source. The original cropped-training classifier is intentionally replaced by
the benchmark's shared linear head, so this is an architecture transfer rather
than a reproduction of Schirrmeister et al.'s complete training protocol.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from hustbciml.core.stages import Backbone
from hustbciml.utils.shapes import probe


class Deep4NetAT(Backbone):
    """Four-block Deep4Net feature extractor for ``(B, 1, C, T)`` input."""

    task_name = "classification"

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_classes: int,
        sfreq: float,
        drop_prob: float = 0.5,
        **_,
    ):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.features = nn.Sequential(OrderedDict([
            ("conv_time", nn.Conv2d(1, 25, (1, 10), bias=True)),
            ("conv_spat", nn.Conv2d(25, 25, (n_chans, 1), bias=False)),
            ("bnorm_1", nn.BatchNorm2d(25, momentum=0.1, eps=1e-5)),
            ("elu_1", nn.ELU()),
            ("pool_1", nn.MaxPool2d((1, 3), stride=(1, 3))),
            ("drop_2", nn.Dropout(self.drop_prob)),
            ("conv_2", nn.Conv2d(25, 50, (1, 10), bias=False)),
            ("bnorm_2", nn.BatchNorm2d(50, momentum=0.1, eps=1e-5)),
            ("elu_2", nn.ELU()),
            ("pool_2", nn.MaxPool2d((1, 3), stride=(1, 3))),
            ("drop_3", nn.Dropout(self.drop_prob)),
            ("conv_3", nn.Conv2d(50, 100, (1, 10), bias=False)),
            ("bnorm_3", nn.BatchNorm2d(100, momentum=0.1, eps=1e-5)),
            ("elu_3", nn.ELU()),
            ("pool_3", nn.MaxPool2d((1, 3), stride=(1, 3))),
            ("drop_4", nn.Dropout(self.drop_prob)),
            ("conv_4", nn.Conv2d(100, 200, (1, 10), bias=False)),
            ("bnorm_4", nn.BatchNorm2d(200, momentum=0.1, eps=1e-5)),
            ("elu_4", nn.ELU()),
            ("pool_4", nn.MaxPool2d((1, 3), stride=(1, 3))),
        ]))
        self._initialize_features()

        with probe(self):
            features = self.forward_features(torch.zeros(1, 1, n_chans, n_times))
        self.out_features = int(features.shape[1])

    def _initialize_features(self) -> None:
        for module in self.features.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)
