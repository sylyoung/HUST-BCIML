# ===========================================================================
# DAN.py  —  HUST-BCIML EEG-decoding benchmark

# Original authors:    Mingsheng Long, Yue Cao, Jianmin Wang, Michael I. Jordan (2015) — "Learning Transferable Features with Deep Adaptation Networks", Proc. ICML
#                      Original code: https://github.com/thuml/Transfer-Learning-Library (reference implementation)
# Implementation:      Junguang Jiang et al. (thuml) — thuml/Transfer-Learning-Library (https://github.com/thuml/Transfer-Learning-Library) (canonical implementation)
# Current code:        Siyang Li — sylyoung/DeepTransferEEG (https://github.com/sylyoung/DeepTransferEEG) (ported from)
# Integrated by:       Siyang Li <lsyyoungll@gmail.com> — HUST-BCIML
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.

# References (IEEE BibTeX):
#   @InProceedings{Long2015,
#     author    = {Long, Mingsheng and Cao, Yue and Wang, Jianmin and Jordan, Michael I.},
#     booktitle = {Proceedings of the International Conference on Machine Learning},
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

        # Transfer-loss weight. 1.0 is the reference default and what the
        # published row used; ``--hp dan_align=<w>`` makes it sweepable, which it
        # was not before — the repository has an ``hp`` mechanism precisely for
        # method coefficients and this one was hard-coded past it.
        align_w = float(ctx.cfg.hp.get("dan_align", 1.0))

        def da_step(m, bs, bt, aux, it, max_iter, ctx):
            feat_s, out_s = m(bs.x)
            feat_t, _ = m(bt.x)
            return criterion(out_s, bs.y) + align_w * mkmmd(feat_s, feat_t)

        return transductive_train(model, source, ctx, da_step)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
