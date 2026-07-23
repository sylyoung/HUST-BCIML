# ===========================================================================
# ADFCNN.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/UM-Tao/ADFCNN-MI
#
# Reference (IEEE BibTeX):
#   @Article{Tao2024,
#     author  = {Tao, Wei and Wang, Ze and Wong, Chi Man and Jia, Ziyu and Li, Chang and Chen, Xun and Chen, C. L. Philip and Wan, Feng},
#     journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
#     title   = {{ADFCNN}: Attention-Based Dual-Scale Fusion Convolutional Neural Network for Motor Imagery Brain-Computer Interface},
#     year    = {2024},
#     pages   = {154-165},
#     volume  = {32},
#     doi     = {10.1109/TNSRE.2023.3342331},
#   }
# ===========================================================================
"""ADFCNN (Tao et al., 2024) - an attention-based dual-scale fusion CNN for
motor-imagery decoding.

The network runs two parallel spectral-spatial pathways at two different
temporal scales, then fuses them with a self-attention module. Mapping to the
paper (IEEE Transactions on Neural Systems and Rehabilitation Engineering,
2024), Section III (Method) and Fig. 2:

  * Spectral convolution (paper Section III-B, "Spectral Convolution Module").
    Two temporal convolutions read the raw signal at two scales. A long kernel
    of 125 samples captures low-frequency rhythm information and a short kernel
    of 30 samples captures higher-frequency detail. Each produces F1 feature
    maps followed by batch normalization.
  * Spatial convolution (paper Section III-C, "Spatial Convolution Module").
    Each scale gets its own spatial branch. Branch 1 uses a depthwise spatial
    convolution over the channel axis, then a pointwise convolution, ELU, and
    average pooling, in the EEGNet-style temporal-compression sense. Branch 2
    uses a spatial convolution followed by the square nonlinearity, average
    pooling, and the log nonlinearity, in the ShallowConvNet log-power sense.
  * Feature fusion (paper Section III-D, "Feature Fusion Module"). The two
    pooled branch outputs are concatenated along the time axis and fed to a
    single-head self-attention block. Query and key are L2-normalized before
    the scaled dot product, then attention is applied to the value and added
    back as a residual. This is the cross-scale attention that gives the
    network its name.

The pre-logit feature is the flattened fused representation of width
F2 * (W1 + W2), where W1 and W2 are the pooled time lengths of the two spatial
branches. That width depends on the input length, so it is inferred by a dummy
forward in ``__init__`` and stored in ``self.out_features``; the backbone is
therefore dataset-agnostic. The paper's final classifier (a convolution that
maps the fused feature to class scores) is removed here so the shared hustbciml
``Linear`` head produces the logits.

Source: github.com/UM-Tao/ADFCNN-MI, as mirrored in DBConformer/models/ADFCNN.py.
Deviations, all behaviour-preserving: the paper hardcodes the classifier kernel
to a fixed input length, so that final classifier is dropped in favour of the
shared head and the fused feature width is measured by a dummy forward. No
global cudnn flags are set. Layer sizes, kernel lengths, pooling, dropout, and
nonlinearities are kept identical to the reference default configuration
(F1=8, D=1, F2=8, dropout=0.25, mean pooling).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.core.stages import Backbone


class _Conv2dWithConstraint(nn.Conv2d):
    """Conv2d whose weight rows are max-norm constrained (paper uses this
    throughout the spectral and spatial modules to bound the filter norm)."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class _ActSquare(nn.Module):
    """Square nonlinearity for the ShallowConvNet-style log-power branch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(x)


class _ActLog(nn.Module):
    """Clamped log nonlinearity for the ShallowConvNet-style log-power branch."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))


class ADFCNN(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 pool_mode: str = "mean", drop_out: float = 0.25, **_):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        # ADFCNN is not an EEGNet-family network: the spectral module emits F1
        # feature maps that the two spatial modules consume directly, so F1 and F2
        # must be equal. The paper's reference configuration fixes F1 = F2 = 8
        # (depth multiplier D = 1). Pin them here and deliberately ignore the
        # pipeline's injected EEGNet F1/D/F2 knobs (absorbed by **_), exactly as
        # the other non-EEGNet backbones (e.g. EEGNeX) ignore them. Letting the
        # EEGNet defaults (F1=4, F2=8) through would build the spatial convolutions
        # for 8 input channels while the spectral module produces only 4.
        F1 = F2 = 8
        self.F2 = F2

        # Spectral convolution module (paper Section III-B): two temporal scales.
        # Long 125-sample kernel for low-frequency content.
        self.spectral_1 = nn.Sequential(
            _Conv2dWithConstraint(1, F1, kernel_size=[1, 125], padding="same", max_norm=2.0),
            nn.BatchNorm2d(F1),
        )
        # Short 30-sample kernel for higher-frequency content.
        self.spectral_2 = nn.Sequential(
            _Conv2dWithConstraint(1, F1, kernel_size=[1, 30], padding="same", max_norm=2.0),
            nn.BatchNorm2d(F1),
        )

        # Spatial convolution module (paper Section III-C), branch 1: EEGNet-style
        # depthwise spatial conv + pointwise conv + ELU + average pooling.
        self.spatial_1 = nn.Sequential(
            _Conv2dWithConstraint(F2, F2, (n_chans, 1), padding=0, groups=F2, bias=False, max_norm=2.0),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(drop_out),
            _Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding="valid", max_norm=2.0),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            pooling_layer((1, 32), stride=32),
            nn.Dropout(drop_out),
        )
        # Spatial convolution module (paper Section III-C), branch 2: ShallowConvNet-style
        # spatial conv + square + average pooling + log (log-power features).
        self.spatial_2 = nn.Sequential(
            _Conv2dWithConstraint(F2, F2, kernel_size=[n_chans, 1], padding="valid", max_norm=2.0),
            nn.BatchNorm2d(F2),
            _ActSquare(),
            pooling_layer((1, 75), stride=25),
            _ActLog(),
            nn.Dropout(drop_out),
        )

        # Feature fusion module (paper Section III-D): single-head self-attention
        # over the concatenated two-scale features. Q/K/V project the channel axis.
        self.drop = nn.Dropout(drop_out)
        self.w_q = nn.Linear(F2, F2)
        self.w_k = nn.Linear(F2, F2)
        self.w_v = nn.Linear(F2, F2)

        # Infer the flattened fused-feature width (dataset-dependent) by a dummy
        # forward, so the shared Linear head can be sized generically.
        with torch.no_grad():
            self.out_features = self.forward_features(torch.zeros(1, 1, n_chans, n_times)).shape[1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, C, T)
        # Two spectral scales.
        x_1 = self.spectral_1(x)
        x_2 = self.spectral_2(x)

        # Two spatial branches, one per scale.
        x_filter_1 = self.spatial_1(x_1)
        x_filter_2 = self.spatial_2(x_2)

        # Concatenate the two pooled branches along the time axis (paper Fig. 2 fusion).
        x_noattention = torch.cat((x_filter_1, x_filter_2), 3)
        B2, C2, H2, W2 = x_noattention.shape
        # Flatten spatial/time into tokens; the channel axis C2 becomes the feature axis.
        x_attention = x_noattention.reshape(B2, C2, H2 * W2).permute(0, 2, 1)
        B, N, C = x_attention.shape

        # Single-head self-attention with L2-normalized query and key (paper Section III-D).
        q = self.w_q(x_attention).permute(0, 2, 1)
        k = self.w_k(x_attention).permute(0, 2, 1)
        v = self.w_v(x_attention).permute(0, 2, 1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        d_k = q.size(-1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)

        # Residual add, reshape back to feature-map form, dropout, then flatten.
        x_attention = x_attention + self.drop(x)
        x_attention = x_attention.reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)
        x = self.drop(x_attention)
        return x.reshape(x.shape[0], -1)
