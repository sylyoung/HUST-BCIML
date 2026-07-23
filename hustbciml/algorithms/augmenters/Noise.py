# ===========================================================================
# Noise.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Freer2020,
#     author  = {Freer, Daniel and Yang, Guang-Zhong},
#     journal = {Journal of Neural Engineering},
#     title   = {Data Augmentation for Self-Paced Motor Imagery Classification with {C}-{LSTM}},
#     year    = {2020},
#     number  = {1},
#     pages   = {016041},
#     volume  = {17},
#     doi     = {10.1088/1741-2552/ab57c0},
#   }
# ===========================================================================
"""Additive noise, the simplest label-preserving EEG augmentation and a
comparison baseline in Ziwei Wang's augmentation studies (Channel Reflection,
Neural Networks 2024; CSDA, Knowledge-Based Systems 2025).

Each trial is copied once with zero-mean Gaussian noise added, scaled to the
trial's own amplitude so the perturbation is relative rather than absolute:

    X' = X + n,   n ~ Normal(0, (std(X) / C_noise) ** 2)          # C_noise = 2

where std(X) is taken per channel over time. The noisy copy keeps the label,
doubling the batch. Scaling the noise by std(X)/C_noise follows the run baseline
used by Wang et al.; the underlying idea traces to Gaussian-white-noise
augmentation (Wang et al., MultiMedia Modeling 2018). Being amplitude-only it is
montage-agnostic and composes with any spatial aligner such as EA.
"""
from __future__ import annotations

import torch

from hustbciml.core.batch import EEGBatch
from hustbciml.core.stages import Augmenter


class Noise(Augmenter):
    train_only = True

    c_noise: float = 2.0     # noise std = per-channel std(X) / c_noise (paper: 2)

    def __init__(self, ch_names=None, n_classes: int = 2, c_noise: float = None, **_):
        self.c_noise = float(c_noise) if c_noise is not None else Noise.c_noise

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        x = batch.x                                   # (B, 1, C, T)
        std = x.std(dim=-1, keepdim=True)             # (B, 1, C, 1) per-channel amplitude
        x_aug = x + torch.randn_like(x) * (std / self.c_noise)

        x_new = torch.cat([x, x_aug], dim=0)
        y_new = torch.cat([batch.y, batch.y], dim=0)
        d_new = torch.cat([batch.domain, batch.domain], dim=0)
        return EEGBatch(x_new, y_new, d_new)
