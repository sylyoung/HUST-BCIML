# ===========================================================================
# EEGWaveNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/IoBT-VISTEC/EEGWaveNet
#
# Reference (IEEE BibTeX):
#   @Article{Thuwajit2022,
#     author  = {Thuwajit, Punnawish and Rangpong, Phurin and Sawangjai, Phattarapong and Autthasan, Phairot and Chaisaen, Rattanaphon and Banluesombatkul, Nannapas and Boonchit, Puttaranun and Tatsaringkansakul, Nattasate and Sudhawiyangkul, Thapanun and Wilaiprasitporn, Theerawit},
#     journal = {IEEE Transactions on Industrial Informatics},
#     title   = {{EEGWaveNet}: Multiscale {CNN}-Based Spatiotemporal Feature Extraction for {EEG} Seizure Detection},
#     year    = {2022},
#     number  = {8},
#     pages   = {5547-5557},
#     volume  = {18},
#     doi     = {10.1109/TII.2021.3133307},
#   }
# ===========================================================================
"""EEGWaveNet backbone (Punnawish Thuwajit et al., 2022) is a multiscale
CNN for spatiotemporal EEG feature extraction.

The network has two stages that mirror the paper's Figure 2 and Section III-B.

  * Multiscale temporal decomposition. A cascade of six depthwise (per-channel)
    Conv1d layers with kernel 2 and stride 2 repeatedly halves the sampling rate,
    so each successive layer looks at a coarser time scale. Following the paper,
    the first layer output is treated as a preliminary transform and the five
    coarser levels feed the next stage, giving a wavelet-like multiscale pyramid.
  * Per-scale spatial-temporal pooling. Each of the five retained scales is passed
    through its own two-layer pointwise-over-channels convolution block (the
    paper's spatial feature extractor) of the form Conv1d -> BatchNorm ->
    LeakyReLU -> Conv1d -> BatchNorm -> LeakyReLU, then averaged over time to a
    32-dimensional descriptor. The five 32-dimensional descriptors are
    concatenated into a 160-dimensional multiscale feature vector.

The paper closes with a classifier MLP Linear(160, 64) -> LeakyReLU ->
Linear(64, 32) -> Sigmoid -> Linear(32, n_classes). As in the DBConformer port,
the final Linear that maps to n_classes is removed and the rest of that MLP is
folded into forward_features, so out_features is 32 and the shared hustbciml
Linear head produces the logits.

Source: github.com/IoBT-VISTEC/EEGWaveNet (also carried in DBConformer's
models/EEGWaveNet.py). The only deviation is behaviour-preserving: out_features
is confirmed by a dummy forward in __init__ rather than hardcoded, so any
(C, T) shape works.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hustbciml.core.stages import Backbone


class EEGWaveNet(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float, **_):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times

        # --- Stage 1: multiscale temporal decomposition (paper Section III-B). ---
        # Six depthwise Conv1d(kernel=2, stride=2) layers, one filter per channel
        # (groups=n_chans). Each layer halves the time length, so the cascade
        # yields progressively coarser temporal scales, wavelet-like.
        self.temp_conv1 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv2 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv3 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv4 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv5 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv6 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)

        # --- Stage 2: per-scale spatial feature extractor (paper Section III-B). ---
        # One block per retained scale. Each mixes across channels with two
        # pointwise-over-time Conv1d(kernel=4) layers, each followed by BatchNorm
        # and LeakyReLU(0.01), producing 32 spatial feature maps per scale.
        self.chpool1 = self._chpool(n_chans)
        self.chpool2 = self._chpool(n_chans)
        self.chpool3 = self._chpool(n_chans)
        self.chpool4 = self._chpool(n_chans)
        self.chpool5 = self._chpool(n_chans)

        # --- Feature MLP (paper's classifier, minus the final class Linear). ---
        # Linear(160, 64) -> LeakyReLU -> Linear(64, 32) -> Sigmoid. The concat of
        # the five 32-d per-scale descriptors is 160-d; this compresses it to the
        # 32-d pre-logit feature vector. The paper's trailing Linear(32, n_classes)
        # is dropped so the shared Linear head maps features to logits.
        self.feat = nn.Sequential(
            nn.Linear(160, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.Sigmoid(),
        )

        # Infer the pre-logit feature width by a dummy forward (dataset-agnostic).
        with torch.no_grad():
            self.out_features = self.forward_features(
                torch.zeros(1, 1, n_chans, n_times)).shape[1]

    @staticmethod
    def _chpool(n_chans: int) -> nn.Sequential:
        """Per-scale spatial block: two Conv1d(kernel=4) mixing across channels,
        each with BatchNorm and LeakyReLU(0.01), yielding 32 feature maps."""
        return nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, C, T)
        x = x.squeeze(1)                       # (B, C, T)

        # Multiscale cascade. temp_conv1 is the preliminary transform; the five
        # coarser outputs temp_w1..temp_w5 are the retained scales fed to Stage 2.
        temp_x = self.temp_conv1(x)
        temp_w1 = self.temp_conv2(temp_x)
        temp_w2 = self.temp_conv3(temp_w1)
        temp_w3 = self.temp_conv4(temp_w2)
        temp_w4 = self.temp_conv5(temp_w3)
        temp_w5 = self.temp_conv6(temp_w4)

        # Per-scale spatial extraction, then average over time -> (B, 32) each.
        w1 = self.chpool1(temp_w1).mean(dim=-1)
        w2 = self.chpool2(temp_w2).mean(dim=-1)
        w3 = self.chpool3(temp_w3).mean(dim=-1)
        w4 = self.chpool4(temp_w4).mean(dim=-1)
        w5 = self.chpool5(temp_w5).mean(dim=-1)

        # Concatenate the five per-scale descriptors -> (B, 160), then compress.
        concat_vector = torch.cat([w1, w2, w3, w4, w5], dim=1)
        return self.feat(concat_vector)        # (B, out_features) == (B, 32)
