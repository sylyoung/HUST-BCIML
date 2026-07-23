# ===========================================================================
# FBMSNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/Want2Vanish/FBMSNet
#
# Reference (IEEE BibTeX):
#   @Article{Liu2023,
#     author  = {Liu, Ke and Yang, Mingzhao and Yu, Zhuliang and Wang, Guoyin and Wu, Wei},
#     journal = {IEEE Transactions on Biomedical Engineering},
#     title   = {{FBMSNet}: A Filter-Bank Multi-Scale Convolutional Neural Network for {EEG}-Based Motor Imagery Decoding},
#     year    = {2023},
#     number  = {2},
#     pages   = {436-445},
#     volume  = {70},
#     doi     = {10.1109/TBME.2022.3193277},
#   }
# ===========================================================================
"""FBMSNet (Liu et al., IEEE Transactions on Biomedical Engineering 2023) - a
filter-bank multi-scale convolutional network for motor-imagery EEG decoding.

The architecture (paper Section III, Fig. 1) is a four-stage feature extractor:

  1. Filter bank. The raw single-channel EEG is decomposed into nBands narrow
     sub-bands by a bank of band-pass filters. The paper uses 9 Chebyshev type II
     band-pass filters covering 4-40 Hz in 4 Hz steps (4-8, 8-12, ..., 36-40),
     the same bank as FBCNet. This turns the (B, 1, C, T) input into a
     (B, nBands, C, T) multi-view tensor (paper Section III-A).
  2. Mixed-scale temporal convolution (MixConv). A grouped multi-scale temporal
     convolution with four kernel lengths (1x15, 1x31, 1x63, 1x125) applied to
     disjoint channel groups, then concatenated, followed by batch norm. This is
     the "multi-scale" block that captures several temporal receptive fields at
     once (paper Section III-B, the MS module).
  3. Spatial convolution block (SCB). A depthwise (per feature map) spatial
     convolution of length nChan with a 2-norm max-norm weight constraint,
     followed by batch norm and a swish nonlinearity. It expands each of the
     num_Feat maps by a dilatability factor into spatial filters (paper Section
     III-C).
  4. Temporal variance aggregation. The time axis is split into strideFactor
     equal segments and a log-variance (log band power) is taken within each
     segment, then flattened. This is the variance layer of paper Section III-D
     (LogVarLayer), analogous to the log-power features of FBCSP / FBCNet.

``forward_features`` returns the flattened log-variance feature vector (paper's
pre-classifier features). The paper's final ``LinearWithConstraint`` +
``LogSoftmax`` classifier is removed so the shared hustbciml ``Linear`` head
produces the logits. ``out_features`` (num_Feat * dilatability * strideFactor,
i.e. 36 * 8 * 4 = 1152 at the default config) is inferred by a dummy forward so
the backbone is dataset-agnostic.

Source: github.com/Want2Vanish/FBMSNet (the authors' ``FBMSNet`` network, whose
default config is nBands=9, num_Feat=36, dilatability=8, strideFactor=4,
temporalLayer='LogVarLayer', dropoutP=0.5). Deviations, all self-contained and
behaviour-preserving:

* The authors precompute the filter bank offline (an external transforms module
  that Chebyshev-filters each trial with ``scipy.signal.filtfilt``) and feed the
  9-band tensor in. Here the same 9 Chebyshev type II bands are precomputed at
  init with scipy (already a repo dependency) and applied inside the model as a
  fixed, non-trainable depthwise convolution with SAME padding, so any (C, T)
  input works and no external filter-bank code is needed. Each band conv weight
  is the finite impulse response of that band's zero-phase Chebyshev filter, so
  it reproduces the paper's band decomposition while staying a single conv. This
  mirrors the KDFNet port's data-free filter-bank-as-conv pattern.
* The MixConv multi-scale temporal convolution and the log-variance aggregation
  are kept identical to the reference (same kernels, same num_Feat, same swish,
  same LogVarLayer, same strideFactor). Only the trailing classifier Linear is
  dropped, and the feature width is inferred by a dummy forward instead of the
  reference's hardcoded ``get_size``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.core.stages import Backbone


# ---------------------------------------------------------------------------
# Constrained layers (paper: 2-norm max-norm weight constraint on the spatial
# convolution, following Schirrmeister et al. / FBCNet).
# ---------------------------------------------------------------------------
class _Conv2dWithConstraint(nn.Conv2d):
    """Conv2d whose weights are renormalized to a max L2 norm each forward pass."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class _Swish(nn.Module):
    """Swish activation x * sigmoid(x) (paper Section III-C nonlinearity)."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class _LogVarLayer(nn.Module):
    """Log-variance over ``dim`` (paper Section III-D variance / log-power layer).

    Clamps the variance to a positive range for numerical stability before the
    natural logarithm, exactly as the reference LogVarLayer does."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


# ---------------------------------------------------------------------------
# Mixed-scale temporal convolution (paper Section III-B, MS module).
# Splits the input feature maps into groups, convolves each group with a
# different temporal kernel length, and concatenates. Ported from the authors'
# MixedConv2d (based on the MixNet MDConv), with the same TensorFlow-style SAME
# padding so the time length is preserved.
# ---------------------------------------------------------------------------
def _split_channels(num_chan: int, num_groups: int):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class _MixedConv2d(nn.ModuleDict):
    """Grouped multi-scale temporal convolution with per-group SAME padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size):
        super().__init__()
        self.kernel_size = list(kernel_size)
        num_groups = len(self.kernel_size)
        self.in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        for idx, (k, in_ch, out_ch) in enumerate(zip(self.kernel_size, self.in_splits, out_splits)):
            # even kernel widths (e.g. 1x125 -> effective width 125 is odd, but
            # the reference relies on dynamic SAME padding for all four kernels)
            self.add_module(str(idx), nn.Conv2d(in_ch, out_ch, k, stride=1, padding=0, bias=False))

    @staticmethod
    def _same_pad(x, kernel_size):
        # TensorFlow-style SAME padding on the time (width) axis only; the
        # height (kernel is 1 tall) needs no padding. Matches the reference
        # conv2d_same used by MixedConv2d.
        iw = x.size(-1)
        kw = kernel_size[-1]
        pad_w = max((-(iw // -1) - 1) * 1 + (kw - 1) * 1 + 1 - iw, 0)
        return F.pad(x, [pad_w // 2, pad_w - pad_w // 2, 0, 0])

    def forward(self, x):
        x_split = torch.split(x, self.in_splits, dim=1)
        x_out = [conv(self._same_pad(x_split[i], self.kernel_size[i]))
                 for i, conv in enumerate(self.values())]
        return torch.cat(x_out, dim=1)


class FBMSNet(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 n_bands: int = 9, f_low: float = 4.0, f_high: float = 40.0,
                 num_feat: int = 36, dilatability: int = 8, stride_factor: int = 4,
                 dropout: float = 0.5, filt_order: int = 4, **_):
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.sfreq = float(sfreq)
        self.n_bands = int(n_bands)
        self.num_feat = int(num_feat)
        self.dilatability = int(dilatability)
        self.stride_factor = int(stride_factor)
        # 4 Hz-wide bands from f_low to f_high (default 4-40 Hz -> 9 bands).
        bw = (f_high - f_low) / self.n_bands
        self.bands = [(f_low + i * bw, f_low + (i + 1) * bw) for i in range(self.n_bands)]

        # --- Stage 1: filter bank (paper Section III-A). A fixed non-trainable
        # depthwise conv whose per-band kernel is that band's Chebyshev type II
        # impulse response; SAME padding keeps the length T. ---
        self.filt_order = int(filt_order)
        self.fb_taps = min(int(round(self.sfreq)) | 1, self._max_odd(self.n_times))  # odd length
        self.filter_bank = nn.Conv2d(1, self.n_bands, (1, self.fb_taps), bias=False, padding=0)
        self.filter_bank.weight.requires_grad_(False)
        self._init_filter_bank()

        # --- Stage 2: mixed-scale temporal convolution (paper Section III-B) ---
        self.mix_conv = nn.Sequential(
            _MixedConv2d(self.n_bands, self.num_feat,
                         kernel_size=[(1, 15), (1, 31), (1, 63), (1, 125)]),
            nn.BatchNorm2d(self.num_feat),
        )

        # --- Stage 3: spatial convolution block (paper Section III-C) ---
        self.scb = nn.Sequential(
            _Conv2dWithConstraint(self.num_feat, self.num_feat * self.dilatability,
                                  (self.n_chans, 1), groups=self.num_feat,
                                  max_norm=2, padding=0),
            nn.BatchNorm2d(self.num_feat * self.dilatability),
            _Swish(),
        )

        # --- Stage 4: temporal log-variance aggregation (paper Section III-D) ---
        self.temporal_layer = _LogVarLayer(dim=3)
        self.dropout = nn.Dropout(p=dropout)

        # Infer the flattened feature width with a dummy forward so the backbone
        # is dataset-agnostic (the reference hardcodes this via get_size).
        with torch.no_grad():
            f = self.forward_features(torch.zeros(1, 1, self.n_chans, self.n_times))
        self.out_features = int(f.shape[1])

    @staticmethod
    def _max_odd(n: int) -> int:
        n = max(1, int(n))
        return n if n % 2 == 1 else n - 1

    @torch.no_grad()
    def _init_filter_bank(self) -> None:
        """Precompute the 9 Chebyshev type II band-pass filters (paper's FBCNet
        filter bank) and store each band's finite impulse response as a fixed
        conv kernel.

        The authors design each band with ``scipy.signal.cheb2ord`` +
        ``scipy.signal.cheby2`` (passband 3 dB, stopband 30 dB, 2 Hz transition)
        and apply it with zero-phase ``filtfilt``. A zero-phase IIR filter is not
        a single causal conv, so we take the truncated impulse response of the
        band's ``filtfilt`` and use it as a symmetric FIR kernel. Convolving with
        this kernel reproduces that band's magnitude response, giving the same
        band decomposition the paper feeds into MixConv."""
        from scipy.signal import cheb2ord, cheby2, filtfilt

        nyq = self.sfreq / 2.0
        allowance = 2.0            # transition bandwidth, matches the reference
        a_pass, a_stop = 3.0, 30.0
        taps = self.fb_taps
        impulse = np.zeros(taps, dtype=np.float64)
        impulse[taps // 2] = 1.0   # centered unit impulse
        w = np.zeros((self.n_bands, 1, 1, taps), dtype=np.float32)

        for i, (lo, hi) in enumerate(self.bands):
            hi = min(hi, nyq - 1e-3)
            f_pass = [lo / nyq, hi / nyq]
            f_stop = [max((lo - allowance) / nyq, 1e-4), min((hi + allowance) / nyq, 1.0 - 1e-4)]
            try:
                order, wn = cheb2ord(f_pass, f_stop, a_pass, a_stop)
                b, a = cheby2(order, a_stop, wn, btype="bandpass")
            except Exception:
                # fall back to a fixed-order design if order estimation fails
                b, a = cheby2(self.filt_order, a_stop, f_pass, btype="bandpass")
            # zero-phase impulse response -> symmetric FIR kernel for this band
            kernel = filtfilt(b, a, impulse).astype(np.float32)
            w[i, 0, 0, :] = kernel

        self.filter_bank.weight.copy_(torch.from_numpy(w))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, C, T)
        # Stage 1: filter bank -> (B, nBands, C, T) with SAME padding on time.
        pad = self.fb_taps // 2
        x = F.pad(x, [pad, pad, 0, 0])
        x = self.filter_bank(x)                       # (B, nBands, C, T)

        # Stage 2: mixed-scale temporal convolution -> (B, num_feat, C, T).
        x = self.mix_conv(x)

        # Stage 3: spatial convolution block -> (B, num_feat*dilatability, 1, T).
        x = self.scb(x)

        # Stage 4: split time into stride_factor segments and take log-variance
        # within each segment, then flatten (paper Section III-D).
        T = x.shape[3]
        seg = T // self.stride_factor
        x = x[:, :, :, : seg * self.stride_factor]    # drop remainder for even split
        x = x.reshape(x.shape[0], x.shape[1], self.stride_factor, seg)
        x = self.temporal_layer(x)                    # (B, num_feat*dilatability, stride_factor, 1)
        x = self.dropout(x)
        return torch.flatten(x, start_dim=1)          # (B, out_features)
