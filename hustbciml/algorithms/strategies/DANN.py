# ===========================================================================
# DANN.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @Article{Ganin2016,
#     author  = {Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle, Hugo and Laviolette, Fran\c{c}ois and Marchand, Mario and Lempitsky, Victor},
#     journal = {Journal of Machine Learning Research},
#     title   = {Domain-Adversarial Training of Neural Networks},
#     year    = {2016},
#     pages   = {1-35},
#     volume  = {17},
#   }
# ===========================================================================
"""DANN — Domain-Adversarial Neural Network (Ganin et al., 2016), as used for
cross-subject EEG in DeepTransferEEG ``tl/dann.py``.

Transductive: trains on labeled (EA-aligned) source + unlabeled (EA-aligned)
target. A gradient-reversal layer feeds backbone features to a domain
discriminator, so the backbone learns subject-invariant features while the
classifier fits the source labels.

mode='gradient', uses_target=True → the Exp puts the aligned, label-masked
target in ``ctx.target_unlabeled``.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from hustbciml.core.batch import EEGBatch, EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from hustbciml.data_provider.collate import iterate_batches
from ._common import forward_logits


class _ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def _cycle(epochs: EEGEpochs, bs: int, seed: int) -> Iterator[EEGBatch]:
    e = 0
    while True:
        yielded = False
        for b in iterate_batches(epochs, bs, shuffle=True, drop_last=True, seed=seed + e):
            yielded = True
            yield b
        if not yielded:  # dataset smaller than one batch -> don't drop_last
            for b in iterate_batches(epochs, bs, shuffle=True, drop_last=False, seed=seed + e):
                yield b
        e += 1


class DANN(Strategy):
    mode = "gradient"
    uses_target = True

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        cfg, device = ctx.cfg, ctx.device
        target = ctx.target_unlabeled
        if target is None:
            raise RuntimeError("DANN requires ctx.target_unlabeled (transductive)")
        model.to(device)
        disc = weight_norm(nn.Linear(model.backbone.out_features, 2)).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(disc.parameters()), lr=cfg.lr)
        criterion = nn.CrossEntropyLoss()

        bps = max(1, len(source) // cfg.batch_size)
        max_iter = cfg.epochs * bps
        src = _cycle(source, cfg.batch_size, cfg.seed)
        tgt = _cycle(target, cfg.batch_size, cfg.seed + 9973)

        model.train()
        for it in range(max_iter):
            bs = next(src).to(device)
            bt = next(tgt).to(device)
            if bs.x.size(0) <= 1 or bt.x.size(0) <= 1:
                continue

            feat_s, out_s = model(bs.x)
            feat_t, _ = model(bt.x)

            p = it / max(1, max_iter)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            dom_s = disc(_ReverseLayerF.apply(feat_s, alpha))
            dom_t = disc(_ReverseLayerF.apply(feat_t, alpha))
            dl_s = torch.ones(bs.x.size(0), dtype=torch.long, device=device)
            dl_t = torch.zeros(bt.x.size(0), dtype=torch.long, device=device)

            cls_loss = criterion(out_s, bs.y)
            adv_loss = criterion(dom_s, dl_s) + criterion(dom_t, dl_t)
            loss = cls_loss + adv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
