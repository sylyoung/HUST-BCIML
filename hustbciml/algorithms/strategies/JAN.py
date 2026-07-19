# ===========================================================================
# JAN.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Long2017,
#     author    = {Long, Mingsheng and Zhu, Han and Wang, Jianmin and Jordan, Michael I.},
#     booktitle = {Proc. Int'l Conf. Machine Learning},
#     title     = {Deep Transfer Learning with Joint Adaptation Networks},
#     year      = {2017},
#     pages     = {2208-2217},
#     address   = {Sydney, Australia},
#     month     = {Aug.},
#   }
# ===========================================================================
"""JAN — Joint Adaptation Network (Long et al., ICML 2017), as used for
cross-subject EEG in DeepTransferEEG ``tl/jan.py``.

Transductive, non-adversarial: like DAN, but the discrepancy is a *joint* MMD
over the pair (backbone features, softmax predictions) — aligning the joint
distribution rather than the marginal feature distribution. No auxiliary module.

mode='gradient', uses_target=True. Kernel set matches DeepTransferEEG: feature
layer uses five Gaussian kernels (alpha = 2^-3 .. 2^1), prediction layer a single
fixed-bandwidth Gaussian (sigma = 0.92); non-linear joint MMD, weight 1.0.
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
from ._mmd import GaussianKernel, JointMultipleKernelMaximumMeanDiscrepancy


class JAN(Strategy):
    mode = "gradient"
    uses_target = True

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        criterion = nn.CrossEntropyLoss()
        jmmd = JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=([GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                     (GaussianKernel(sigma=0.92, track_running_stats=False),)),
            linear=False)

        def da_step(m, bs, bt, aux, it, max_iter, ctx):
            feat_s, out_s = m(bs.x)
            feat_t, out_t = m(bt.x)
            align = jmmd((feat_s, torch.softmax(out_s, dim=1)),
                         (feat_t, torch.softmax(out_t, dim=1)))
            return criterion(out_s, bs.y) + align                  # alignment_weight = 1.0

        return transductive_train(model, source, ctx, da_step)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
