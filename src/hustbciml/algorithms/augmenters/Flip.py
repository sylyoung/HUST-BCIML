# ===========================================================================
# Flip.py  —  HUST-BCIML EEG-decoding benchmark
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
"""Amplitude flip, a label-preserving baseline in Ziwei Wang's augmentation
studies (Channel Reflection, Neural Networks 2024; CSDA, Knowledge-Based Systems
2025).

Each channel is mirrored vertically about its own maximum:

    X'_c = max_t(X_c) - X_c

This inverts peaks and troughs while preserving the temporal envelope and the
band-power magnitude, so the class evidence is retained. The flipped copy keeps
the label and is appended to the batch. Being amplitude-only it is
montage-agnostic and composes with any spatial aligner. This is distinct from
Channel Reflection, which permutes electrodes across hemispheres and, for
two-class motor imagery, swaps the label.
"""
from __future__ import annotations

import torch

from hustbciml.core.batch import EEGBatch
from hustbciml.core.stages import Augmenter


class Flip(Augmenter):
    train_only = True

    def __init__(self, ch_names=None, n_classes: int = 2, **_):
        self.n_classes = int(n_classes)

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        x = batch.x                                   # (B, 1, C, T)
        cmax = x.amax(dim=-1, keepdim=True)           # (B, 1, C, 1) per-channel max
        x_aug = cmax - x                              # X'_c = max(X_c) - X_c

        x_new = torch.cat([x, x_aug], dim=0)
        y_new = torch.cat([batch.y, batch.y], dim=0)
        d_new = torch.cat([batch.domain, batch.domain], dim=0)
        return EEGBatch(x_new, y_new, d_new)
