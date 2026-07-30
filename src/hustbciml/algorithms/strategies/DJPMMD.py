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
"""DJP-MMD — Discriminative Joint Probability MMD (Zhang & Wu, 2020).

Unsupervised domain adaptation for C-class problems. DJP-MMD replaces the joint
MMD used by prior feature-based DA with a *discriminative* joint-probability
discrepancy that acts directly on the joint distribution P(X, Y) rather than on a
weighted sum of the marginal and conditional MMDs (Sec. III). The joint MMD splits
into a same-class-across-domains term M_T and a different-class-across-domains term
M_D (Eq. 7); DJP-MMD is defined as ``M_T - mu * M_D`` (Eq. 8), so it MINIMIZES the
cross-domain joint discrepancy of the same class (transferability) while MAXIMIZING
it between different classes (discriminability), with trade-off mu > 0. In the
paper the discrepancy is minimized over a linear projection A by embedding it in
the JPDA framework: a generalized eigen-decomposition (Eq. 27, Algorithm 1) that
iteratively refines target pseudo-labels, evaluated on six image datasets with a
1-NN classifier.

This benchmark reuses only the DJP-MMD discrepancy, not the JPDA eigen-solver: the
discrepancy is added to the source cross-entropy and minimized by gradient descent
over the EEGNet backbone features (source labels + target pseudo-labels), through
the shared ``transductive_train`` skeleton. mode='gradient', uses_target=True.
mu = 0.1 (the paper's default, Sec. IV-B); DA-term weight 1.0. See ``_djp.py`` for
the discrepancy and its corrections relative to the authors' released code.
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
        mu = float(ctx.cfg.hp.get("djpmmd_mu", 0.1))           # mu in M_T - mu * M_D (Eq. 8): discriminability trade-off
        align_w = float(ctx.cfg.hp.get("djpmmd_align", 1.0))  # weight of the DJP-MMD discrepancy vs source cross-entropy

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
