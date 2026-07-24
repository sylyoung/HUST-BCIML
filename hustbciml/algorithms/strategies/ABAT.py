# ===========================================================================
# ABAT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/xqchen914/ABAT
#
# Reference (IEEE BibTeX):
#   @Article{Chen2024,
#     author  = {Chen, Xiaoqing and Wang, Ziwei and Wu, Dongrui},
#     journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
#     title   = {Alignment-Based Adversarial Training ({ABAT}) for Improving the Robustness and Accuracy of {EEG}-Based {BCI}s},
#     year    = {2024},
#     pages   = {1703-1714},
#     volume  = {32},
#     doi     = {10.1109/TNSRE.2024.3391936},
#   }
# ===========================================================================
"""ABAT (Chen et al., 2024, IEEE TNSRE) — Alignment-Based Adversarial Training:
a training strategy that improves both the robustness and the benign-sample
accuracy of EEG classifiers by aligning the EEG data BEFORE adversarial training.

Adversarial training (AT) alone solves the min-max/saddle-point problem (Eq. 8):
minimize over the model the worst-case loss over adversarial examples X^adv drawn
from an l_inf ball of radius eps around each trial X. AT hardens the model against
attacks but often lowers accuracy on benign samples (Sec. III). ABAT's proposal is
to prepend Euclidean Alignment (EA) so alignment and AT act together. Per
Algorithm 1 (Sec. III) and Fig. 3-4, ABAT is two sequential stages, IN THIS ORDER:

  1. Data alignment — Euclidean Alignment (EA; Eqs. 1-2, Sec. II-A). EA whitens
     the per-domain average spatial covariance so trials from different
     subjects/sessions share a more consistent distribution. EA is unsupervised
     (uses no labels). Here it is supplied by the pipeline's aligner stage, so run
     ABAT with ``aligner: EA``.
  2. Adversarial training — on the aligned source data, generate strong
     adversarial examples and minimize the (supervised) classification loss on
     them (Eq. 8). The paper generates X^adv with FGSM (single step, Eq. 5),
     PGD (iterative, Eqs. 6-7), or AutoAttack. This file uses PGD: start from a
     uniformly perturbed benign trial X_0 = X + xi, xi in (-eps, eps) (Eq. 6),
     then take ``steps`` gradient-sign ascent steps of size alpha <= eps,
     projecting back into the eps-ball each step (Eq. 7). Defaults follow the
     paper: 10 PGD steps, step size alpha = eps/5 (Sec. IV-E).

Per-channel perturbation budget. The paper ties the attack magnitude to the
signal scale: "the perturbation magnitude [is] eps times the EEG signal standard
deviation" (Sec. IV-E). This file follows the authors' released implementation,
which applies that scaling per channel — each channel is perturbed by up to
``eps`` times that channel's temporal standard deviation (source
``attack_lib.PGD_batch_cha``, "cha" = channel), so higher-amplitude channels get
a proportionally larger budget rather than one absolute eps for all channels.

Faithful-adaptation notes.
  * Warmup: this file trains cleanly for the first ``warmup`` epochs, then swaps in
    the adversarial batch. This warmup is an implementation detail of the authors'
    ``train.py`` (``ATchastd``, 20 of 100 epochs), NOT part of Algorithm 1 in the
    paper; it is kept as a fraction (0.2) so it scales to any epoch budget and is
    exact on the 100-epoch benchmark.
  * The source applies EEGNet's ``MaxNormConstraint`` after each step; the shared
    hustbciml EEGNet has no such constraint and every other strategy trains that
    same backbone, so it is omitted — ABAT then differs from ERM in exactly the
    adversarial training, keeping the strategy-axis comparison controlled.
  * Optimizer / early stopping match the shared ERM trainer (Adam + validation
    early stop) rather than the source's fixed LR schedule, for the same reason.

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
    """PGD attack (l_inf) with a per-channel budget (Eqs. 6-7; Sec. IV-E).
    Init X_0 = X + xi with xi uniform in (-eps, eps) (Eq. 6), then ``steps``
    gradient-sign ascent steps of size ``alpha``, each projected back into the
    eps-ball (Eq. 7). The budget on every channel is ``eps`` times that channel's
    temporal std (source ``PGD_batch_cha``). Returns the adversarial batch,
    detached."""
    was_training = model.training
    model.eval()
    x = x.detach()
    cha_std = x.std(dim=-1, keepdim=True).detach()             # (B, 1, C, 1) per-channel temporal std
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

    eps: float = 0.01          # perturbation magnitude, in units of per-channel std (Sec. IV-E)
    steps: int = 10            # PGD iterations (Sec. IV-E)
    warmup_frac: float = 0.2   # clean-training warmup fraction (source train.py, 20/100; not in Algorithm 1)

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        hp = ctx.cfg.hp
        eps = float(hp.get("abat_eps", self.eps))              # perturbation magnitude (per-channel std units)
        steps = int(hp.get("abat_steps", self.steps))          # PGD iterations (Sec. IV-E)
        warmup_frac = float(hp.get("abat_warmup", self.warmup_frac))
        warmup = int(round(warmup_frac * ctx.cfg.epochs))
        alpha = eps / 5                                        # PGD step size alpha = eps/5 (Sec. IV-E)

        # EA (Eqs. 1-2) is applied upstream by the aligner stage; this hook is the
        # AT stage of Algorithm 1: replace each aligned batch with its adversarial
        # counterpart (Eq. 8) once past the source-code warmup.
        def at_batch(m, batch, epoch, _ctx):
            if epoch < warmup:
                return batch                                   # clean warmup (source train.py; not in Algorithm 1)
            adv = _pgd_batch_cha(m, batch.x, batch.y, eps, alpha, steps)
            m.train()
            return EEGBatch(adv, batch.y, batch.domain)

        return supervised_train(model, source, ctx, batch_fn=at_batch)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
