# ===========================================================================
# ASFA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Xia2022,
#     author  = {Xia, Kun and Deng, Lingfei and Duch, Wlodzislaw and Wu, Dongrui},
#     journal = {IEEE Trans. Biomedical Engineering},
#     title   = {Privacy-Preserving Domain Adaptation for Motor Imagery-Based Brain-Computer Interfaces},
#     year    = {2022},
#     number  = {11},
#     pages   = {3365-3376},
#     volume  = {69},
#     doi     = {10.1109/TBME.2022.3168570},
#   }
# ===========================================================================
"""ASFA — Augmentation-based Source-Free Adaptation (Xia, Deng, Duch & Wu, IEEE TBME 2022).

Source-free domain adaptation: train a source model (the shared ERM loop on the
EA-aligned source), then adapt it to the unlabeled target WITHOUT any source data
— freeze the classifier head (the source hypothesis) and update only the feature
extractor, exactly like SHOT. ASFA differs from SHOT-IM in the adaptation
objective: instead of information maximization it minimizes

  * an **uncertainty-reduction** loss ``L_UR`` (paper Eq. 9): a Tsallis-entropy
    sharpening of the predictions with three MCC-style refinements — temperature
    rescaling (T=2), per-sample entropy reweighting w_i, and per-class
    normalization gamma_k; and
  * a **consistency-regularization** loss ``L_CR`` (paper Eq. 10): the main
    (frozen-head) softmax is pulled toward a randomly-initialized auxiliary head
    fed the feature vector under a DropMin perturbation (drop the smallest
    floor(d * p_d) features, p_d ~ U(0,1)).

Total target loss ``L_t = L_UR + beta * L_CR`` (beta=0.1), SGD lr 0.01.

Faithful-adaptation notes (disclosed in the card): (1) the paper's own backbone
is a small MLP on Riemannian tangent-space features, NOT EEGNet; here ASFA runs
on the benchmark's EA-EEGNet so it is directly comparable to the other source-free
rows (SHOT, ISFDA, DELTA) — the ported contribution is the source-free adaptation
objective. (2) The paper's per-epoch channel-weakening source augmentation is
omitted so every transfer row shares one ERM source model (single-axis
comparison). (3) w_i and gamma_k are computed per mini-batch and detached
(constants w.r.t. the gradient), matching the MCC convention.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from hustbciml.data_provider.collate import iterate_batches
from ._common import forward_logits, supervised_train

ASFA_ADAPT_EPOCHS = 10
ASFA_LR = 0.01          # paper: SGD lr 0.01
ASFA_TEMP = 2.0         # T — probability rescaling (paper Eq. 6)
ASFA_A = 2.0            # a — Tsallis entropy order (paper Eq. 5)
ASFA_BETA = 0.1         # beta — consistency trade-off (paper Eq. 11)


def _dropmin(feats: torch.Tensor) -> torch.Tensor:
    """DropMin feature perturbation: zero the smallest floor(d * p_d) features per
    row, with p_d ~ U(0,1) resampled per call (paper §3.F, the default)."""
    d = feats.size(1)
    p_d = float(torch.rand(1).item())
    k = int(d * p_d)
    if k <= 0:
        return feats
    idx = feats.argsort(dim=1)[:, :k]              # smallest-k feature indices per row
    mask = torch.ones_like(feats)
    mask.scatter_(1, idx, 0.0)
    return feats * mask


class ASFA(Strategy):
    # non-tta source-free: adapts offline on the whole (offline-EA-aligned) target,
    # like SHOT; fit is plain source ERM, predict performs the adaptation.
    mode = "gradient"
    uses_target = False

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return supervised_train(model, source, ctx)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        cfg, device = ctx.cfg, ctx.device
        hp = cfg.hp
        adapt_lr = float(hp.get("asfa_lr", ASFA_LR))          # adaptation SGD LR (paper 0.01)
        beta = float(hp.get("asfa_beta", ASFA_BETA))          # consistency trade-off
        adapt_epochs = int(hp.get("asfa_epochs", ASFA_ADAPT_EPOCHS))
        temp = float(hp.get("asfa_temp", ASFA_TEMP))          # probability rescaling T
        a_ord = float(hp.get("asfa_a", ASFA_A))               # Tsallis entropy order a
        model.to(device)
        K = cfg.n_classes

        # freeze the classifier head (source hypothesis); adapt only the feature
        # extractor. A random auxiliary head drives the consistency term.
        for p in model.head.parameters():
            p.requires_grad_(False)
        aux = nn.Linear(model.backbone.out_features, K).to(device)
        opt = torch.optim.SGD(list(model.backbone.parameters()) + list(aux.parameters()),
                              lr=adapt_lr, momentum=0.9)

        for epoch in range(adapt_epochs):
            model.backbone.train()
            model.head.eval()
            for batch in iterate_batches(target, cfg.batch_size, shuffle=True,
                                         drop_last=True, seed=cfg.seed + epoch):
                if batch.x.size(0) <= 1:                       # BatchNorm needs >1
                    continue
                xb = batch.x.to(device)
                feats, logits = model(xb)

                # --- L_UR: Tsallis uncertainty with T-rescaling / reweight / class-norm ---
                pt = torch.softmax(logits / temp, dim=1)                   # rescaled probs (B, K)
                H = -(pt * torch.log(pt + 1e-5)).sum(dim=1)                # (B,)
                w = 1.0 + torch.exp(-H)
                w = (w.size(0) * w / w.sum()).detach()                     # sample weights (B,)
                gamma = pt.sum(dim=0).detach() + 1e-5                      # class norm (K,)
                l_ur = -(w.unsqueeze(1) * pt.pow(a_ord) / gamma.unsqueeze(0)).sum() \
                    / ((a_ord - 1.0) * K * xb.size(0))

                # --- L_CR: consistency of main head vs aux head on perturbed features ---
                main_p = torch.softmax(logits, dim=1)
                aux_p = torch.softmax(aux(_dropmin(feats)), dim=1)
                l_cr = ((main_p - aux_p) ** 2).mean()

                loss = l_ur + beta * l_cr
                opt.zero_grad()
                loss.backward()
                opt.step()

        for p in model.head.parameters():
            p.requires_grad_(True)

        logits = forward_logits(model, target, device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return y_score.argmax(1), y_score
