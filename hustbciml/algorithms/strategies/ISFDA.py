# ===========================================================================
# ISFDA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Li2021,
#     author    = {Li, Xinhao and Li, Jingjing and Zhu, Lei and Wang, Guoqing and Huang, Zi},
#     booktitle = {Proc. ACM Int'l Conf. Multimedia},
#     title     = {Imbalanced Source-Free Domain Adaptation},
#     year      = {2021},
#     pages     = {3330-3339},
#     doi       = {10.1145/3474085.3475487},
#   }
# ===========================================================================
"""ISFDA — Imbalanced Source-Free Domain Adaptation, online-TTA variant.

Online test-time adaptation like Tent/T-TIME: stream the target, predict each
trial with the frozen model (after incremental Euclidean Alignment), then update
on a sliding batch of recent trials. The adaptation loss combines two terms:

  * information maximization — the same entropy-minimization + marginal-diversity
    objective as T-TIME/SHOT, on temperature-scaled logits; and
  * an ISFDA class-structure term — tighten intra-class feature clusters and
    separate inter-class ones (cosine distance to pseudo-label class centers),
    with a secondary-label rule that also counts an ambiguous sample (winning
    probability in [0.5, 0.6)) toward the *other* class, to resist class
    imbalance in the streamed target.

Binary task only: the class-structure term is defined for two classes, matching
the lab source and the BNCI2014001 left/right-hand evaluation.

Vendored from DeepTransferEEG ``tl/isfda.py`` over the shared ``online_tta_loop``
skeleton. Source training reuses the shared ERM loop.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import entropy, online_tta_loop, supervised_train

_EPS = 1e-5


def _isfda_dist_loss(feats: torch.Tensor, prob: torch.Tensor, batch: int) -> torch.Tensor:
    """Intra-class tightening minus inter-class separation on pseudo-labeled
    features. Ambiguous samples (winning prob in [0.5, 0.6)) are additionally
    counted toward the other class (the ISFDA secondary-label rule). Returns 0 if
    either class is empty in this batch (distance undefined), matching the source.
    """
    pl = prob.argmax(dim=1)
    c0 = torch.where(pl == 0)[0].tolist()
    c1 = torch.where(pl == 1)[0].tolist()
    for l in range(prob.size(0)):
        if 0.5 <= prob[l, 0] < 0.6:
            c1.append(l)
        elif 0.5 <= prob[l, 1] < 0.6:
            c0.append(l)
    if len(c0) == 0 or len(c1) == 0:
        return feats.new_zeros(())
    f0 = feats[torch.tensor(c0, device=feats.device)]
    f1 = feats[torch.tensor(c1, device=feats.device)]
    ctr0 = f0.mean(dim=0, keepdim=True) if f0.size(0) > 1 else f0   # (1, D) either way
    ctr1 = f1.mean(dim=0, keepdim=True) if f1.size(0) > 1 else f1
    inter = (1 - F.cosine_similarity(f0, ctr1)).sum() + (1 - F.cosine_similarity(f1, ctr0)).sum()
    intra = (1 - F.cosine_similarity(f0, ctr0)).sum() + (1 - F.cosine_similarity(f1, ctr1)).sum()
    return (intra - inter) / batch


class ISFDA(Strategy):
    mode = "tta"

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return supervised_train(model, source, ctx)

    def predict(self, model: nn.Module, target: EEGEpochs,
                ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        if ctx.cfg.n_classes != 2:
            raise ValueError("ISFDA's class-structure term is binary-only "
                             "(got n_classes=%d)" % ctx.cfg.n_classes)

        def make_opt(m, cfg):
            return torch.optim.Adam(m.parameters(), lr=cfg.lr)

        def update(m, xb, opt, cfg):
            m.train()
            for _ in range(cfg.steps):
                feats, logits = m(xb)
                prob = torch.softmax(logits / cfg.temperature, dim=1)
                cem = torch.mean(entropy(prob))                     # entropy minimization
                msoftmax = prob.mean(dim=0)
                mdr = torch.sum(msoftmax * torch.log(msoftmax + _EPS))   # marginal diversity
                im_loss = cem + mdr
                dist_loss = _isfda_dist_loss(feats, prob, cfg.test_batch)
                loss = im_loss + dist_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

        return online_tta_loop(model, target, ctx, update, make_opt)
