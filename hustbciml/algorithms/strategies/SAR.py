# ===========================================================================
# SAR.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/mr-eggplant/SAR
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Niu2023,
#     author    = {Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Wen, Zhiquan and Chen, Yaofo and Zhao, Peilin and Tan, Mingkui},
#     booktitle = {International Conference on Learning Representations},
#     title     = {Towards Stable Test-Time Adaptation in Dynamic Wild World},
#     year      = {2023},
#   }
# ===========================================================================
"""SAR — Sharpness-Aware Reliable test-time adaptation (Niu et al., ICLR 2023).

Online test-time adaptation like Tent/T-TIME, but the entropy-minimization step
is taken with a Sharpness-Aware Minimization (SAM) optimizer: a two-pass update
that first ascends to the worst-case parameter perturbation in the neighborhood,
then descends from there — seeking a *flat* entropy minimum that is more robust
to the noisy, single-batch test-time signal.

This is the DeepTransferEEG ``tl/sar.py`` variant: SAM + temperature-scaled
entropy over all model parameters, on the online sliding batch. It omits the
optional reliable-sample filter (entropy thresholding) and model recovery from
the full SAR paper — the lab implementation keeps the sharpness-aware core only.

Vendored from DeepTransferEEG ``tl/sar.py`` + ``tl/models/sam.py``, over the
shared ``online_tta_loop`` skeleton. Source training reuses the shared ERM loop.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import entropy, online_tta_loop, supervised_train
from ._sam import SAM


class SAR(Strategy):
    mode = "tta"

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return supervised_train(model, source, ctx)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        def make_opt(m, cfg):
            # SAM wraps a base optimizer; the sharpness-aware entropy update
            # perturbs *all* parameters (the lab's tl/sar.py choice).
            return SAM(m.parameters(), torch.optim.Adam, lr=cfg.lr)

        def update(m, xb, opt, cfg):
            m.train()
            for _ in range(cfg.steps):
                # first pass: gradient of entropy, then climb to w + e(w)
                _, logits = m(xb)
                loss = torch.mean(entropy(torch.softmax(logits / cfg.temperature, dim=1)))
                opt.zero_grad()
                loss.backward()
                opt.first_step(zero_grad=True)
                # second pass: re-forward at the perturbed weights, then descend
                _, logits2 = m(xb)
                loss2 = torch.mean(entropy(torch.softmax(logits2 / cfg.temperature, dim=1)))
                loss2.backward()
                opt.second_step(zero_grad=True)

        return online_tta_loop(model, target, ctx, update, make_opt)
