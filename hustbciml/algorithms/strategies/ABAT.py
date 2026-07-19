# ===========================================================================
# ABAT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/xqchen914/ABAT
#
# Reference (IEEE BibTeX):
#   @Article{Chen2024,
#     author  = {Chen, Xiaoqing and Wang, Ziwei and Wu, Dongrui},
#     journal = {IEEE Trans. Neural Systems and Rehabilitation Engineering},
#     title   = {Alignment-Based Adversarial Training ({ABAT}) for Improving the Robustness and Accuracy of {EEG}-Based {BCI}s},
#     year    = {2024},
#     pages   = {1703-1714},
#     volume  = {32},
#     doi     = {10.1109/TNSRE.2024.3391936},
#   }
# ===========================================================================
"""ABAT — Alignment-Based Adversarial Training (Chen et al., 2024).

Two ingredients improve both robustness and clean accuracy of EEG classifiers:

  1. **Alignment** — Euclidean Alignment of the trials (supplied here by the
     pipeline's aligner stage; compose with ``aligner: EA``).
  2. **Adversarial training** — after a short warmup of clean training, each
     training batch is replaced by a PGD adversarial batch and the model is
     trained on those. ABAT's signature is the **channel-std-scaled** budget:
     the per-sample perturbation on each channel is scaled by that channel's
     temporal standard deviation, so the attack respects each channel's
     amplitude instead of using one global epsilon.

Defaults follow the source (``train.py`` ``ATchastd`` + ``attack_lib.PGD_batch_cha``):
eps ``AT_eps=0.01``, step size ``eps/5``, ``steps=10``, warmup at 20 of 100 epochs.
The warmup is expressed as a fraction (0.2) so it scales to any epoch budget and
is exact on the 100-epoch benchmark.

Faithful-adaptation notes. (1) The source applies EEGNet's ``MaxNormConstraint``
each step; the shared hustbciml EEGNet has no such constraint and every other
strategy trains that same backbone, so it is omitted here — ABAT differs from ERM
in exactly the adversarial training, keeping the strategy-axis comparison
controlled. (2) Optimizer / early-stopping match the shared ERM trainer (Adam +
val early stop) rather than the source's fixed LR schedule, for the same reason.

Source: github.com/xqchen914/ABAT (``train.py``, ``attack_lib.py``).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.core.batch import EEGBatch, EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import forward_logits, supervised_train


def _pgd_batch_cha(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                   eps: float, alpha: float, steps: int) -> torch.Tensor:
    """Channel-std-scaled PGD (L-inf). Perturbs each channel by up to
    ``eps * channel_temporal_std``; returns the adversarial batch (detached)."""
    was_training = model.training
    model.eval()
    x = x.detach()
    cha_std = x.std(dim=-1, keepdim=True).detach()             # (B, 1, C, 1)
    adv = (x + torch.empty_like(x).uniform_(-eps, eps) * cha_std).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        _, logits = model(adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * cha_std * grad.detach().sign()
        delta = torch.clamp(adv - x, min=-eps * cha_std, max=eps * cha_std)
        adv = (x + delta).detach()
    if was_training:
        model.train()
    return adv


class ABAT(Strategy):
    mode = "gradient"

    eps: float = 0.01          # AT_eps (channel-std-scaled)
    steps: int = 10
    warmup_frac: float = 0.2   # clean-training warmup as a fraction of epochs (20/100 in the paper)

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        hp = ctx.cfg.hp
        eps = float(hp.get("abat_eps", self.eps))              # AT budget (channel-std-scaled)
        steps = int(hp.get("abat_steps", self.steps))          # PGD iterations
        warmup_frac = float(hp.get("abat_warmup", self.warmup_frac))
        warmup = int(round(warmup_frac * ctx.cfg.epochs))
        alpha = eps / 5

        def at_batch(m, batch, epoch, _ctx):
            if epoch < warmup:
                return batch                                   # clean warmup
            adv = _pgd_batch_cha(m, batch.x, batch.y, eps, alpha, steps)
            m.train()
            return EEGBatch(adv, batch.y, batch.domain)

        return supervised_train(model, source, ctx, batch_fn=at_batch)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
