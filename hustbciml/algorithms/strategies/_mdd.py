# ===========================================================================
# _mdd.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Zhang2019,
#     author    = {Zhang, Yuchen and Liu, Tianle and Long, Mingsheng and Jordan, Michael I.},
#     booktitle = {Proc. Int'l Conf. Machine Learning},
#     title     = {Bridging Theory and Algorithm for Domain Adaptation},
#     year      = {2019},
#     pages     = {7404-7413},
#     address   = {Long Beach, CA},
#     month     = {Jun.},
#   }
# ===========================================================================
"""Margin Disparity Discrepancy machinery for MDD (Zhang et al., ICML 2019).

Vendored from DeepTransferEEG ``tl/utils/loss.py``, which in turn adapts the
authors' reference implementation. Underscore-prefixed so the registry
auto-scan skips it.

Note on the warm-start gradient-reversal layer: the authors' reference
implementation advances it
with ``classifier.step()`` every training iteration (the coefficient warms from
``lo`` to ``hi``); the DeepTransferEEG training script omitted that call, leaving
the reversal inert. The MDD strategy here restores the ``step()`` call so the
adversarial reversal is active, matching the reference.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class _GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
        ctx.coeff = coeff
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.coeff, None


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient reversal with a coefficient that warms from ``lo`` to ``hi`` over
    ``max_iters`` as ``step()`` is called. ``auto_step=False`` → the training
    loop must call ``step()`` each iteration (as the authors' reference does)."""

    def __init__(self, alpha: float = 1.0, lo: float = 0.0, hi: float = 1.0,
                 max_iters: int = 1000, auto_step: bool = False):
        super().__init__()
        self.alpha, self.lo, self.hi = alpha, lo, hi
        self.iter_num, self.max_iters, self.auto_step = 0, max_iters, auto_step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return _GradientReverseFunction.apply(x, coeff)

    def step(self):
        self.iter_num += 1


def _shift_log(x: torch.Tensor, offset: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x + offset, max=1.))


class ClassificationMarginDisparityDiscrepancy(nn.Module):
    """MDD for classification: gamma-weighted source disparity + target disparity
    between the main and adversarial classifier heads."""

    def __init__(self, margin: float = 4.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, y_s, y_s_adv, y_t, y_t_adv) -> torch.Tensor:
        _, pred_s = y_s.max(dim=1)
        _, pred_t = y_t.max(dim=1)
        source_loss = -self.margin * F.cross_entropy(y_s_adv, pred_s, reduction="none")
        target_loss = -F.nll_loss(_shift_log(1. - F.softmax(y_t_adv, dim=1)), pred_t, reduction="none")
        loss = source_loss + target_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MDDClassifier(nn.Module):
    """Auxiliary head for MDD over backbone features: a bottleneck, a main head,
    and an adversarial head fed through the warm-start gradient-reversal layer.
    ``forward`` returns (main_logits, adversarial_logits) in training mode."""

    def __init__(self, backbone_dim: int, num_classes: int, bottleneck_dim: int = 50,
                 grl: Optional[WarmStartGradientReverseLayer] = None):
        super().__init__()
        self.grl_layer = grl or WarmStartGradientReverseLayer(
            alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False)
        self.bottleneck = nn.Sequential(
            nn.Linear(backbone_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.bottleneck[1].weight.data.normal_(0, 0.005)
        self.bottleneck[1].bias.data.fill_(0.1)
        self.head = nn.Linear(bottleneck_dim, num_classes)
        self.adv_head = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x: torch.Tensor):
        feats = self.bottleneck(x)
        outputs = self.head(feats)
        outputs_adv = self.adv_head(self.grl_layer(feats))
        if self.training:
            return outputs, outputs_adv
        return outputs

    def step(self):
        self.grl_layer.step()
