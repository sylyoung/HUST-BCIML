# ===========================================================================
# ASFA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Xia2022,
#     author  = {Xia, Kun and Deng, Lingfei and Duch, Wlodzislaw and Wu, Dongrui},
#     journal = {IEEE Transactions on Biomedical Engineering},
#     title   = {Privacy-Preserving Domain Adaptation for Motor Imagery-Based Brain-Computer Interfaces},
#     year    = {2022},
#     number  = {11},
#     pages   = {3365-3376},
#     volume  = {69},
#     doi     = {10.1109/TBME.2022.3168570},
#   }
# ===========================================================================
"""ASFA — Augmentation-based Source-Free Adaptation (Xia et al., 2022, IEEE TBME).

ASFA is a source-free unsupervised domain-adaptation (SFUDA) method for
motor-imagery BCIs: it improves cross-subject accuracy while protecting the
source user's privacy, because adaptation needs only a trained source model, not
the source EEG data (paper Abstract; Sec. III; Fig. 1). The source and target
models share a feature-extractor + classifier structure and operate on
Riemannian tangent-space features. ASFA has two parts (Sec. III, Fig. 2):

  1. Source model training (Sec. III-A): a novel channel data-augmentation —
     each epoch, randomly weaken some EEG channels (weakening probability p,
     magnitude scaled by lambda ~ U[lambda_0, 1]) so the model cannot rely on a
     few subject-specific channels — followed by cross-entropy with label
     smoothing (Eq. 4). This is the "augmentation-based" part of the name and it
     raises cross-subject generalization.
  2. Target model training (Sec. III-B): initialize the target model from the
     source model, freeze the target classifier h_t (the source hypothesis), and
     update only the feature extractor f_t by minimizing
       * an uncertainty-reduction loss ``L_UR`` (Eq. 9): the Tsallis entropy of
         the predictions (Eq. 5, order a>1) sharpened with three enhancements
         adapted from minimum class confusion (MCC) — probability rescaling by
         temperature T (Eq. 6, T=2), per-sample entropy reweighting w_i (Eq. 7),
         and per-class normalization gamma_k (Eq. 8) — to make confident,
         class-balanced predictions; and
       * a consistency-regularization loss ``L_CR`` (Eq. 10): the target
         classifier's softmax is pulled toward that of M auxiliary classifiers
         (same structure as h_t, randomly initialized) fed a perturbed copy of
         the features, so the model is robust to small perturbations. The paper's
         default perturbation is DropMin (drop the smallest floor(d * p_d)
         features, p_d ~ U(0,1)); the experiments use a single auxiliary
         classifier, M=1 (Sec. IV-B).
     Total target loss ``L_t = L_UR + beta * L_CR`` (Eq. 11, beta=0.1),
     optimized with SGD at lr 0.01. Target training is unsupervised (no target
     labels). The paper further extends ASFA to a black-box setting where even
     the source parameters are inaccessible (Sec. III-C, Eq. 12), which this file
     does not implement.

This file implements the target-model adaptation objective (L_UR + L_CR).
Benchmark-integration notes: (1) the paper's backbone is a 3-layer MLP on
tangent-space features, not EEGNet; here ASFA runs on the benchmark's EA-EEGNet
so it is directly comparable to the other source-free rows — the ported piece is
the source-free adaptation objective. (2) The source-training channel-weakening
augmentation (part 1 above) is omitted so every transfer row shares one common
source model, isolating the adaptation objective. (3) w_i and gamma_k are
computed per mini-batch and detached (constants w.r.t. the gradient).
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
ASFA_TEMP = 2.0         # T — probability rescaling temperature (Eq. 6, T=2)
ASFA_A = 2.0            # a — Tsallis entropy order, a>1 (Eq. 5)
ASFA_BETA = 0.1         # beta — L_CR consistency trade-off (Eq. 11)


def _dropmin(feats: torch.Tensor) -> torch.Tensor:
    """DropMin feature perturbation (Sec. III-B, "Consistency Regularization",
    item 3): zero the smallest floor(d * p_d) features per row, where p_d ~ U(0,1)
    is resampled per call. This is the paper's default perturbation for L_CR."""
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
    # Source-free UDA (not test-time): fit trains the source model with plain
    # source ERM; predict runs the offline target-model adaptation (Sec. III-B)
    # over the whole unlabeled, offline-EA-aligned target set.
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

        # freeze the target classifier h_t (source hypothesis); adapt only the
        # feature extractor f_t (Sec. III-B). A single (M=1) auxiliary classifier,
        # same shape as h_t and randomly initialized, drives L_CR (Sec. IV-B).
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

                # --- L_UR (Eq. 9): Tsallis-entropy uncertainty reduction with the
                # three MCC-inspired enhancements: probability rescaling (Eq. 6),
                # sample reweighting w_i (Eq. 7), category normalization gamma_k (Eq. 8) ---
                pt = torch.softmax(logits / temp, dim=1)                   # rescaled probs (Eq. 6), (B, K)
                H = -(pt * torch.log(pt + 1e-5)).sum(dim=1)                # entropy of rescaled probs (B,)
                w = 1.0 + torch.exp(-H)
                w = (w.size(0) * w / w.sum()).detach()                     # sample weights w_i (Eq. 7), (B,)
                gamma = pt.sum(dim=0).detach() + 1e-5                      # category norm gamma_k (Eq. 8), (K,)
                l_ur = -(w.unsqueeze(1) * pt.pow(a_ord) / gamma.unsqueeze(0)).sum() \
                    / ((a_ord - 1.0) * K * xb.size(0))

                # --- L_CR (Eq. 10): consistency between the target classifier and the
                # auxiliary classifier on DropMin-perturbed features ---
                main_p = torch.softmax(logits, dim=1)
                aux_p = torch.softmax(aux(_dropmin(feats)), dim=1)
                l_cr = ((main_p - aux_p) ** 2).mean()

                loss = l_ur + beta * l_cr                  # L_t = L_UR + beta * L_CR (Eq. 11)
                opt.zero_grad()
                loss.backward()
                opt.step()

        for p in model.head.parameters():
            p.requires_grad_(True)

        logits = forward_logits(model, target, device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return y_score.argmax(1), y_score
