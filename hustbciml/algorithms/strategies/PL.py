# ===========================================================================
# PL.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Lee2013,
#     author    = {Lee, Dong-Hyun},
#     booktitle = {Proc. {ICML} Workshop on Challenges in Representation Learning},
#     title     = {Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks},
#     year      = {2013},
#     address   = {Atlanta, GA},
#     month     = {Jun.},
#   }
# ===========================================================================
"""PL — online pseudo-labeling test-time adaptation.

Stream the target trials with incremental Euclidean Alignment; predict each
trial, then on a sliding batch take the model's own argmax predictions as
pseudo-labels and take a cross-entropy step on all parameters. The self-training
test-time baseline of DeepTransferEEG (``tl/pl.py``), over the shared
``online_tta_loop`` skeleton.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import online_tta_loop, supervised_train


class PL(Strategy):
    mode = "tta"

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return supervised_train(model, source, ctx)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        def make_opt(m, cfg):
            return torch.optim.Adam(m.parameters(), lr=cfg.lr)

        def update(m, xb, opt, cfg):
            m.train()
            criterion = nn.CrossEntropyLoss()
            for _ in range(cfg.steps):
                _, logits = m(xb)
                pseudo = logits.argmax(dim=1).detach()      # model's own hard labels
                loss = criterion(logits, pseudo)
                opt.zero_grad()
                loss.backward()
                opt.step()

        return online_tta_loop(model, target, ctx, update, make_opt)
