# ===========================================================================
# Tent.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/DequanWang/tent
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Wang2021,
#     author    = {Wang, Dequan and Shelhamer, Evan and Liu, Shaoteng and Olshausen, Bruno and Darrell, Trevor},
#     booktitle = {International Conference on Learning Representations},
#     title     = {{T}ent: Fully Test-Time Adaptation by Entropy Minimization},
#     year      = {2021},
#   }
# ===========================================================================
"""Tent — fully test-time adaptation by entropy minimization (Wang et al., ICLR 2021).

Online test-time adaptation restricted to the BatchNorm affine parameters:
stream the target trials, predict each with the current model (after an
incremental Euclidean-Alignment update when EA is the composed aligner), then
minimize the prediction entropy of a sliding batch by updating only the BN
scale/shift, with the BN layers in train mode so their running statistics track
the target batch. This is T-TIME's information-maximization loss reduced to its
conditional-entropy term with the parameter set narrowed to BN-affine only.

Vendored from DeepTransferEEG ``tl/tent.py`` + ``tl/models/tent.py``, over the
shared ``online_tta_loop`` skeleton.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import (collect_bn_params, entropy, online_tta_loop, set_bn_train,
                      supervised_train)


class Tent(Strategy):
    mode = "tta"

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return supervised_train(model, source, ctx)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        def make_opt(m, cfg):
            return torch.optim.Adam(collect_bn_params(m), lr=cfg.lr)

        def update(m, xb, opt, cfg):
            set_bn_train(m)                       # only BN trains; dropout stays deterministic
            for _ in range(cfg.steps):
                _, logits = m(xb)
                loss = torch.mean(entropy(torch.softmax(logits, dim=1)))
                opt.zero_grad()
                loss.backward()
                opt.step()

        return online_tta_loop(model, target, ctx, update, make_opt)
