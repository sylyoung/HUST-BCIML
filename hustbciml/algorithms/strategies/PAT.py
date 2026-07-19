# ===========================================================================
# PAT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/xqchen914/PAT
#
# Reference (IEEE BibTeX):
#   @Article{Chen2026,
#     author  = {Chen, Xiaoqing and Jia, Tianwang and Tu, Yunlu and Wu, Dongrui},
#     journal = {Fundamental Research},
#     title   = {{PAT}: Privacy-Preserving Adversarial Transfer for Accurate, Robust and Privacy-Preserving {EEG} Decoding},
#     year    = {2026},
#     doi     = {10.1016/j.fmre.2026.04.034},
#   }
# ===========================================================================
"""PAT — Privacy-preserving Adversarial Transfer (Chen et al., 2026, Fund. Res.).

PAT is a single training pipeline: **Euclidean Alignment -> scaling augmentation
-> adversarial training**, jointly improving clean accuracy and adversarial
robustness. In the full paper it is instantiated in three privacy-preserving
transfer scenarios (centralized / federated source-free, and transfer with
privacy-preserved source data), which supply a source prior on top of this core.

This benchmark ports the **training procedure** as a Strategy plug-in, on par
with ABAT: composed with ``aligner: EA`` and ``augmenter: Scaling`` (Eq. 4), it
does adversarial training on the aligned + augmented source under the shared
cross-subject protocol, and reports clean (benign) accuracy. The privacy /
source-prior scenarios are out of scope for the current cross-subject-LOSO
protocol (they use target-calibration fine-tuning); this row isolates PAT's
alignment+augmentation+adversarial-training core.

Adversarial samples follow the paper's PGD (Eqs. 5-7), which differs from ABAT's
channel-std-scaled budget: a **global L-inf epsilon** ball, started from a noisy
benign sample (Eq. 6, ``X^{adv,0} = X + xi``, ``xi ~ U(-eps, eps)``) and updated
by projected sign-gradient steps (Eq. 7). Defaults are the paper's:
``eps = 0.03``, step size ``alpha = 0.005``, ``steps = 10``. A short clean warmup
(fraction of epochs, default 0.2) stabilizes the early gradient steps and mirrors
the source pre-training the paper relies on; it is exposed as a hyperparameter.

Faithful-adaptation notes. (1) Optimizer / early stopping match the shared ERM
trainer (Adam + held-out-source early stop) as for every other strategy, keeping
the strategy-axis comparison controlled, rather than the source's fixed schedule.
(2) The scaling augmentation is supplied by the composed ``Scaling`` augmenter
stage (Eq. 4), so PAT differs from ERM in exactly EA-augment-adversarial-train.
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


def _pgd_batch_global(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      eps: float, alpha: float, steps: int) -> torch.Tensor:
    """Global L-inf PGD (Chen et al. 2026, Eqs. 6-7). Starts from a noisy benign
    sample ``x + U(-eps, eps)`` and takes ``steps`` projected sign-gradient steps
    of size ``alpha``, keeping the perturbation within the eps-ball of ``x``.
    Returns the adversarial batch (detached)."""
    was_training = model.training
    model.eval()
    x = x.detach()
    adv = (x + torch.empty_like(x).uniform_(-eps, eps)).detach()   # Eq. 6: noisy init
    for _ in range(steps):
        adv.requires_grad_(True)
        _, logits = model(adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.detach().sign()          # Eq. 7: sign step
        adv = x + torch.clamp(adv - x, min=-eps, max=eps)          # project to eps-ball
        adv = adv.detach()
    if was_training:
        model.train()
    return adv


class PAT(Strategy):
    mode = "gradient"

    eps: float = 0.03          # global L-inf budget (paper)
    alpha: float = 0.005       # PGD step size (paper)
    steps: int = 10            # PGD iterations (paper)
    warmup_frac: float = 0.2   # clean-training warmup as a fraction of epochs

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        hp = ctx.cfg.hp
        eps = float(hp.get("pat_eps", self.eps))
        alpha = float(hp.get("pat_alpha", self.alpha))
        steps = int(hp.get("pat_steps", self.steps))
        warmup_frac = float(hp.get("pat_warmup", self.warmup_frac))
        warmup = int(round(warmup_frac * ctx.cfg.epochs))

        def at_batch(m, batch, epoch, _ctx):
            if epoch < warmup:
                return batch                                        # clean warmup
            adv = _pgd_batch_global(m, batch.x, batch.y, eps, alpha, steps)
            m.train()
            return EEGBatch(adv, batch.y, batch.domain)

        # the composed Scaling augmenter (Eq. 4) is applied per batch by
        # supervised_train before at_batch, so PGD runs on the aligned+augmented batch.
        return supervised_train(model, source, ctx, batch_fn=at_batch)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
