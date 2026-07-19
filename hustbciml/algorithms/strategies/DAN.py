# ===========================================================================
# DAN.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Long2015,
#     author    = {Long, Mingsheng and Cao, Yue and Wang, Jianmin and Jordan, Michael I.},
#     booktitle = {Proc. Int'l Conf. Machine Learning},
#     title     = {Learning Transferable Features with Deep Adaptation Networks},
#     year      = {2015},
#     pages     = {97-105},
#     address   = {Lille, France},
#     month     = {Jul.},
#   }
# ===========================================================================
"""DAN — Deep Adaptation Network (Long et al., ICML 2015), as used for
cross-subject EEG in DeepTransferEEG ``tl/dan.py``.

Transductive, non-adversarial: train the source classifier and add a
multi-kernel Maximum Mean Discrepancy (MK-MMD) between source and target
backbone features, pulling the two subjects' feature distributions together.
No auxiliary module.

mode='gradient', uses_target=True. Kernels and trade-off match DeepTransferEEG:
five Gaussian kernels (alpha = 2^-3 .. 2^1), linear MK-MMD, weight 1.0.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import forward_logits, transductive_train
from ._mmd import GaussianKernel, MultipleKernelMaximumMeanDiscrepancy


class DAN(Strategy):
    mode = "gradient"
    uses_target = True

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        criterion = nn.CrossEntropyLoss()
        mkmmd = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)], linear=True)

        def da_step(m, bs, bt, aux, it, max_iter, ctx):
            feat_s, out_s = m(bs.x)
            feat_t, _ = m(bt.x)
            return criterion(out_s, bs.y) + mkmmd(feat_s, feat_t)   # alignment_weight = 1.0

        return transductive_train(model, source, ctx, da_step)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
