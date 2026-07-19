# ===========================================================================
# Scaling.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Chen2026,
#     author  = {Chen, Xiaoqing and Jia, Tianwang and Tu, Yunlu and Wu, Dongrui},
#     journal = {Fundamental Research},
#     title   = {{PAT}: Privacy-Preserving Adversarial Transfer for Accurate, Robust and Privacy-Preserving {EEG} Decoding},
#     year    = {2026},
#     doi     = {10.1016/j.fmre.2026.04.034},
#   }
# Original authors' code: https://github.com/xqchen914/PAT
# ===========================================================================
"""Amplitude scaling — a simple, label-preserving EEG data augmentation.

Each trial is copied once with its amplitude multiplied by a coefficient close
to one, ``X' = X * (1 +/- beta)`` (Chen et al., 2026, PAT, Eq. 4; the scaling
trick also appears in the ABAT / augmentation literature). The augmenter appends
the scaled copy to the batch with the label unchanged, doubling the effective
training set at no data cost — the augmentation half of the PAT pipeline
(``aligner: EA`` + ``augmenter: Scaling`` + ``strategy: PAT``).

``beta = 0.05`` matches the paper. The sign is drawn per trial so a batch mixes
slightly-amplified and slightly-attenuated copies, which is why the transform is
useful beyond a global gain (a constant scale would be undone by BatchNorm).
Being an amplitude-only transform it is montage-agnostic, so unlike Channel
Reflection it composes with a spatial aligner such as EA.
"""
from __future__ import annotations

import torch

from hustbciml.core.batch import EEGBatch
from hustbciml.core.stages import Augmenter


class Scaling(Augmenter):
    train_only = True

    beta: float = 0.05     # amplitude perturbation magnitude (paper: 0.05)

    def __init__(self, ch_names=None, n_classes: int = 2, beta: float = None, **_):
        self.beta = float(beta) if beta is not None else Scaling.beta

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        x = batch.x                                   # (B, 1, C, T)
        B = x.shape[0]
        # per-trial coefficient 1 +/- beta (random sign), broadcast over (1, C, T)
        signs = torch.where(torch.rand(B, device=x.device) < 0.5, -1.0, 1.0)
        coef = (1.0 + self.beta * signs).view(B, 1, 1, 1)
        x_aug = x * coef

        x_new = torch.cat([x, x_aug], dim=0)
        y_new = torch.cat([batch.y, batch.y], dim=0)
        d_new = torch.cat([batch.domain, batch.domain], dim=0)
        return EEGBatch(x_new, y_new, d_new)
