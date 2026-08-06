# ===========================================================================
# Implementation of: ShallowFBCSPNet architecture transfer

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
"""Braindecode ShallowFBCSPNet features under the HUST benchmark protocol.

The feature layers and their initialization follow the pinned reference. The
reference full-width convolutional classifier is replaced by the benchmark's
shared linear head, so this is an architecture transfer rather than a complete
reproduction of the original cropped-training protocol.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from hustbciml.core.stages import Backbone
from hustbciml.utils.shapes import probe


class _Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class _SafeLog(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.epsilon))


class ShallowFBCSPNetAT(Backbone):
    """Shallow log-band-power feature extractor for ``(B, 1, C, T)`` input."""

    task_name = "classification"

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_classes: int,
        sfreq: float,
        n_filters: int = 40,
        drop_prob: float = 0.5,
        **_,
    ):
        super().__init__()
        self.n_filters = int(n_filters)
        self.drop_prob = float(drop_prob)
        self.features = nn.Sequential(OrderedDict([
            ("conv_time", nn.Conv2d(1, self.n_filters, (1, 25), bias=True)),
            ("conv_spat", nn.Conv2d(
                self.n_filters,
                self.n_filters,
                (n_chans, 1),
                bias=False,
            )),
            ("bnorm", nn.BatchNorm2d(self.n_filters, momentum=0.1, eps=1e-5)),
            ("square", _Square()),
            ("pool", nn.AvgPool2d((1, 75), stride=(1, 15))),
            ("safe_log", _SafeLog(1e-6)),
            ("drop", nn.Dropout(self.drop_prob)),
        ]))
        self._initialize_features()

        with probe(self):
            features = self.forward_features(torch.zeros(1, 1, n_chans, n_times))
        self.out_features = int(features.shape[1])

    def _initialize_features(self) -> None:
        nn.init.xavier_uniform_(self.features.conv_time.weight, gain=1.0)
        nn.init.zeros_(self.features.conv_time.bias)
        nn.init.xavier_uniform_(self.features.conv_spat.weight, gain=1.0)
        nn.init.ones_(self.features.bnorm.weight)
        nn.init.zeros_(self.features.bnorm.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)
