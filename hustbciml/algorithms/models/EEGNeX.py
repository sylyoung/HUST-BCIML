# ===========================================================================
# EEGNeX.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/chenxiachan/EEGNeX
#
# Reference (IEEE BibTeX):
#   @Article{Chen2024,
#     author  = {Chen, Xia and Teng, Xiangbin and Chen, Han and Pan, Yafeng and Geyer, Philipp},
#     journal = {Biomedical Signal Processing and Control},
#     title   = {Toward Reliable Signals Decoding for Electroencephalogram: A Benchmark Study to {EEGNeX}},
#     year    = {2024},
#     pages   = {105475},
#     volume  = {87},
#     doi     = {10.1016/j.bspc.2023.105475},
#   }
# ===========================================================================
"""EEGNeX backbone (Chen et al., 2024), a purely convolutional EEG decoder.

EEGNeX is an EEGNet-style network that replaces the separable temporal
convolutions of the block after the spatial filter with a stack of dilated
convolutions, so a compact model reaches a wider temporal receptive field
without extra pooling. The paper (Section 2.3, "Model architecture", and Fig. 2)
describes three stages, which this port keeps one to one:

  * Block 1 (temporal feature expansion). Two standard 2-D temporal convolutions
    with kernel length 64 along time only, each followed by batch normalization,
    lifting the single input map to 8 then to 32 feature maps. This is the paper's
    temporal front end that learns frequency-selective filters.
  * Block 2 (depthwise spatial filter). A depthwise convolution with kernel
    (n_chans, 1) and groups=32 mixes all electrodes into one spatial map per
    feature (the paper's spatial-filter stage, analogous to EEGNet's depthwise
    conv), then batch norm, ELU, average pooling by 4, and dropout.
  * Block 3 (dilated temporal convolutions). Two grouped convolutions with time
    kernel 16 and dilation 2 then 4, batch norm, ELU, average pooling by 8, and
    dropout. The dilations are the defining EEGNeX change, widening the temporal
    context cheaply (paper Section 2.3).

The paper's final Dense classifier is removed. ``forward_features`` returns the
flattened output of Block 3 as the pre-logit feature vector, so the shared
hustbciml ``Linear`` head produces the logits. ``out_features`` is inferred by a
dummy forward in ``__init__`` (the reference hardcodes it from a fixed
``time_step``), which keeps the backbone dataset-agnostic for any (C, T).

Source: github.com/chenxiachan/EEGNeX, ported via the PyTorch reproduction in
wzwvv/DBConformer (``models/EEGNeX.py``). The only deviation is behaviour
preserving. The reference threads the flat width from a hand-set ``time_step``
into a final Linear layer that is commented out. Here that Linear is dropped and
the width is measured by the dummy forward instead. All layer sizes, kernel
lengths, groups, dilations, the 0.5 dropout, and the ELU nonlinearities are
identical to the reference.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hustbciml.core.stages import Backbone


class EEGNeX(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 drop_out: float = 0.5, **_):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times

        # Block 1 (paper Section 2.3, temporal feature expansion): two temporal
        # convs over time only (kernel 64), padding keeps the channel axis intact
        # since 1 // 2 == 0. Batch norm after each; ELU between the two convs.
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), bias=False, padding=(1 // 2, 64 // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8, 32, (1, 64), bias=False, padding=(1 // 2, 64 // 2)),
            nn.BatchNorm2d(32),
        )
        # Block 2 (spatial filter): depthwise (n_chans, 1) conv with groups=32
        # collapses the electrode axis to 1, then BN, ELU, average pool by 4,
        # and dropout. This is EEGNeX's spatial-filtering stage.
        self.block_2 = nn.Sequential(
            nn.Conv2d(32, 64, (n_chans, 1), groups=32, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_out),
        )
        # Block 3 (dilated temporal convs, the defining EEGNeX change): two
        # grouped time convs (kernel 16) with dilation 2 then 4 widen the temporal
        # receptive field, then BN, ELU, average pool by 8, and dropout.
        self.block_3 = nn.Sequential(
            nn.Conv2d(64, 32, (1, 16), groups=32, bias=False,
                      padding=(1 // 2, 16 // 2), dilation=(1, 2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 8, (1, 16), groups=8, bias=False,
                      padding=(1 // 2, 16 // 2), dilation=(1, 4)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop_out),
        )

        # Infer the flattened pre-logit width by a dummy forward, replacing the
        # reference's hardcoded 8 * (time_step // 32) so any (C, T) works.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            self.out_features = self._features(dummy).shape[1]

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 1, C, T). Run the three blocks and flatten to (B, feat).
        out = self.block_1(x)
        out = self.block_2(out)
        out = self.block_3(out)
        return out.reshape(out.size(0), -1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, C, T)
        return self._features(x)
