# ===========================================================================
# IFNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/wzwvv/MVCNet
#
# Reference (IEEE BibTeX):
#   @Article{Wang2023,
#     author  = {Wang, Jiaheng and Yao, Lin and Wang, Yueming},
#     journal = {IEEE Trans. Neural Systems and Rehabilitation Engineering},
#     title   = {{IFN}et: An Interactive Frequency Convolutional Neural Network for Enhancing Motor Imagery Decoding from {EEG}},
#     year    = {2023},
#     pages   = {1900-1911},
#     volume  = {31},
#     doi     = {10.1109/TNSRE.2023.3257319},
#   }
# ===========================================================================
"""IFNet (Wang et al., 2023, IEEE TNSRE) — Interactive Frequency convolutional
network. The CNN backbone of MVCNet, and a standalone MI decoder.

A multi-scale depthwise-conv *Stem* (a pointwise mixer that splits into ``radix``
bands, a depthwise temporal conv per band, summed and GELU'd = the "interactive
frequency" step), then patch-average-pooling; the flattened result is the feature
vector. The paper's final linear layer is the shared hustbciml ``Linear`` head, so
``forward_features`` returns the pre-logit features.

This ports the configuration MVCNet uses (``radix=1``, ``out_planes=64``). Number
of temporal patches is inferred by a dummy forward, so ``patch_size`` defaults to
``n_times//8`` (=125 on BNCI2014001, the paper's value) and works on any dataset.
``timm.trunc_normal_`` -> ``torch.nn.init.trunc_normal_`` (behaviour-preserving).

Source: github.com/wzwvv/MVCNet (``models/IFNet.py``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.core.stages import Backbone


class _Conv(nn.Module):
    def __init__(self, conv, bn=None):
        super().__init__()
        if bn is not None:
            conv.bias = None
        self.conv, self.bn = conv, bn

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class _Stem(nn.Module):
    """Interactive-frequency conv stem -> (B, out_planes, P)."""

    def __init__(self, in_planes, out_planes=64, kernel_size=63, patch_size=125,
                 radix=1, drop=0.5, drop_last_t=True):
        super().__init__()
        self.out_planes = out_planes
        self.drop_last_t = drop_last_t
        mid = out_planes * radix
        self.sconv = _Conv(nn.Conv1d(in_planes, mid, 1, bias=False, groups=radix),
                           bn=nn.BatchNorm1d(mid))
        self.tconv = nn.ModuleList()
        k = kernel_size
        for _ in range(radix):
            self.tconv.append(_Conv(
                nn.Conv1d(out_planes, out_planes, k, 1, groups=out_planes,
                          padding=k // 2, bias=False),
                bn=nn.BatchNorm1d(out_planes)))
            k //= 2
        self.pool = nn.AvgPool1d(patch_size, patch_size)
        self.dp = nn.Dropout(drop)

    def forward(self, x):                                   # (B, in_planes, T)
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)
        out = F.gelu(sum(m(o) for o, m in zip(out, self.tconv)))   # interactive-frequency sum
        if self.drop_last_t:
            out = out[:, :, :-1]
        return self.dp(self.pool(out))                      # (B, out_planes, P)


class IFNet(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 embed_dims: int = 64, kernel_size: int = 63, radix: int = 1,
                 patch_size: int = None, drop: float = 0.5, drop_last_t: bool = True, **_):
        super().__init__()
        if patch_size is None:
            patch_size = max(1, (n_times - (1 if drop_last_t else 0)) // 8)
        self.stem = _Stem(n_chans * radix, embed_dims, kernel_size, patch_size,
                          radix, drop, drop_last_t)
        with torch.no_grad():
            feat = self.stem(torch.zeros(1, n_chans, n_times)).flatten(1)
        self.out_features = feat.shape[1]
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, C, T)
        return self.stem(x.squeeze(1)).flatten(1)
