# ===========================================================================
# CDAN.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Long2018,
#     author    = {Long, Mingsheng and Cao, Zhangjie and Wang, Jianmin and Jordan, Michael I.},
#     booktitle = {Advances in Neural Information Processing Systems},
#     title     = {Conditional Adversarial Domain Adaptation},
#     year      = {2018},
#     pages     = {1647-1657},
#     address   = {Montreal, Canada},
#     month     = {Dec.},
#   }
# ===========================================================================
"""CDAN — Conditional Domain-Adversarial Network (Long et al., NeurIPS 2018),
as used for cross-subject EEG in DeepTransferEEG ``tl/cdan.py`` (multilinear
conditioning variant).

Transductive adversarial DA. Like DANN, but the domain discriminator sees the
multilinear map of backbone features and the softmax class prediction
(feature ⊗ softmax), so alignment is conditioned on the predicted class. An
entropy weighting focuses the discriminator on confident samples, and a
gradient-reversal (with a warm-up coefficient) makes the backbone produce
domain-invariant conditional features.

mode='gradient', uses_target=True. Vendors the AdversarialNetwork + calc_coeff +
CDANE machinery from DeepTransferEEG (self-contained; no lab-repo import).
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


def _calc_coeff(iter_num: int, alpha: float = 10.0, max_iter: float = 10000.0) -> float:
    return float(2.0 / (1.0 + np.exp(-alpha * iter_num / max_iter)) - 1.0)


def _grl_hook(coeff: float):
    def hook(grad):
        return -coeff * grad.clone()
    return hook


def _init_weights(m: nn.Module):
    cn = m.__class__.__name__
    if cn.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif cn.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class _AdversarialNetwork(nn.Module):
    """Domain discriminator with a built-in gradient-reversal hook and warm-up
    coefficient (vendored from DeepTransferEEG ``tl/utils/network.py``)."""

    def __init__(self, in_feature: int, hidden1: int = 32, hidden2: int = 8):
        super().__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden1)
        self.ad_layer2 = nn.Linear(hidden1, hidden2)
        self.ad_layer3 = nn.Linear(hidden2, 1)
        self.relu1, self.relu2 = nn.ReLU(), nn.ReLU()
        self.dropout1, self.dropout2 = nn.Dropout(0.5), nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.iter_num, self.alpha, self.max_iter = 0, 10.0, 10000.0
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.iter_num += 1
        coeff = _calc_coeff(self.iter_num, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(_grl_hook(coeff))
        x = self.dropout1(self.relu1(self.ad_layer1(x)))
        x = self.dropout2(self.relu2(self.ad_layer2(x)))
        return self.sigmoid(self.ad_layer3(x))


def _entropy_vec(softmax_out: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return -torch.sum(softmax_out * torch.log(softmax_out + eps), dim=1)


def _cdane(feature: torch.Tensor, softmax_out: torch.Tensor, entropy: torch.Tensor,
           ad_net: nn.Module, coeff: float, device) -> torch.Tensor:
    """Entropy-conditioned adversarial loss over feature ⊗ softmax (multilinear
    conditioning). ``feature`` and ``entropy`` carry gradient; the softmax used
    for the outer product is detached (vendored ``CDANE``)."""
    softmax_output = softmax_out.detach()
    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
    ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    half = feature.size(0) // 2
    dc_target = torch.tensor([[1.0]] * half + [[0.0]] * half, device=device)

    entropy.register_hook(_grl_hook(coeff))
    entropy = 1.0 + torch.exp(-entropy)
    source_mask = torch.ones_like(entropy); source_mask[half:] = 0
    source_weight = entropy * source_mask
    target_mask = torch.ones_like(entropy); target_mask[:half] = 0
    target_weight = entropy * target_mask
    weight = (source_weight / torch.sum(source_weight).detach().item()
              + target_weight / torch.sum(target_weight).detach().item())
    bce = nn.BCELoss(reduction="none")(ad_out, dc_target)
    return torch.sum(weight.view(-1, 1) * bce) / torch.sum(weight).detach().item()


class CDAN(Strategy):
    mode = "gradient"
    uses_target = True

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        criterion = nn.CrossEntropyLoss()

        def setup(m, ctx):
            in_dim = m.backbone.out_features * ctx.cfg.n_classes    # multilinear conditioning
            ad_net = _AdversarialNetwork(in_dim).to(ctx.device)
            return ad_net, list(ad_net.parameters())

        def da_step(m, bs, bt, ad_net, it, max_iter, ctx):
            feat_s, out_s = m(bs.x)
            feat_t, out_t = m(bt.x)
            features = torch.cat((feat_s, feat_t), dim=0)
            outputs = torch.cat((out_s, out_t), dim=0)
            softmax_out = torch.softmax(outputs, dim=1)
            entropy = _entropy_vec(softmax_out)
            transfer = _cdane(features, softmax_out, entropy, ad_net, _calc_coeff(it), ctx.device)
            return criterion(out_s, bs.y) + transfer

        return transductive_train(model, source, ctx, da_step, setup=setup)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
