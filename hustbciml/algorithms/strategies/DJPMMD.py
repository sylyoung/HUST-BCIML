# ===========================================================================
# DJPMMD.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @InProceedings{Zhang2020,
#     author    = {Zhang, Wen and Wu, Dongrui},
#     booktitle = {Proceedings of the International Joint Conference on Neural Networks},
#     title     = {Discriminative Joint Probability Maximum Mean Discrepancy ({DJP}-{MMD}) for Domain Adaptation},
#     year      = {2020},
#     address   = {Glasgow, UK},
#     month     = {Jul.},
#     pages     = {1-8},
#     doi       = {10.1109/IJCNN48605.2020.9207365},
#   }
# ===========================================================================
"""DJP-MMD — Discriminative Joint Probability MMD (Zhang & Wu, IJCNN 2020).

A lab domain-adaptation method, ported as a transductive strategy on EA + EEGNet.
It adds the DJP-MMD discrepancy (``M_T - mu * M_D`` over the joint P(X, Y), using
source labels and target pseudo-labels) to the source cross-entropy, over the
shared ``transductive_train`` skeleton — the same family as DAN / JAN / MDD.

mode='gradient', uses_target=True. mu = 0.1 (source default); alignment weight 1.0.
The paper evaluates DJP-MMD on image datasets; this is a faithful adaptation of the
loss to EEG features (see ``_djp.py`` for the disclosed corrections to the source).
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
from ._djp import djp_mmd


class DJPMMD(Strategy):
    mode = "gradient"
    uses_target = True

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        criterion = nn.CrossEntropyLoss()
        mu = float(ctx.cfg.hp.get("djpmmd_mu", 0.1))           # M_T - mu * M_D discriminability weight
        align_w = float(ctx.cfg.hp.get("djpmmd_align", 1.0))  # DA-term trade-off vs source CE

        def da_step(m, bs, bt, aux, it, max_iter, ctx):
            feat_s, out_s = m(bs.x)
            feat_t, out_t = m(bt.x)
            n = min(feat_s.size(0), feat_t.size(0))            # equal-size batches
            y_t = out_t[:n].argmax(1).detach()                 # target pseudo-labels
            align = djp_mmd(feat_s[:n], feat_t[:n], bs.y[:n], y_t, out_s.size(1), mu=mu)
            return criterion(out_s, bs.y) + align_w * align

        return transductive_train(model, source, ctx, da_step)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
