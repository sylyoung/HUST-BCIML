# ===========================================================================
# MDMAML.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Li2022,
#     author  = {Li, Siyang and Wu, Huanyu and Ding, Lieyun and Wu, Dongrui},
#     journal = {IEEE Computational Intelligence Magazine},
#     title   = {Meta-Learning for Fast and Privacy-Preserving Source Knowledge Transfer of {EEG}-Based {BCI}s},
#     year    = {2022},
#     number  = {4},
#     pages   = {16-26},
#     volume  = {17},
#     doi     = {10.1109/MCI.2022.3199622},
#   }
# ===========================================================================
"""MDMAML — Multi-Domain Model-Agnostic Meta-Learning (Li, Wu, Ding & Wu, IEEE CIM 2022).

A privacy-preserving (source-free) transfer method: meta-learn a model over the
SOURCE subjects — each source subject is one domain — so that a one-step
adaptation on one source domain reduces loss on a *different* source domain. The
meta-learned weights are then applied to a new target subject FORWARD-ONLY, with
no target fine-tuning (the paper's optional online calibration and SHOT post-step
are excluded by design here — this is the paper's "0-shot" setting).

Per episode (paper Algorithm 1, Eqs. 1-6): form ``M/2`` random source domain
pairs ``(S_i, S_j)``; take one inner SGD step on the support domain ``S_i`` to get
fast weights ``theta' = theta - alpha * grad L_{S_i}(theta)`` (Eq. 1); evaluate the
cross-entropy query loss on the *different* domain ``S_j`` at ``theta'`` (Eq. 2);
average over pairs (Eq. 3) and meta-update the original weights
``theta <- theta - beta * grad L_MDMAML`` (Eq. 4). A **first-order** MAML
approximation is used (the paper's FOMAML): the query-loss gradient at ``theta'``
is accumulated as the meta-gradient, avoiding the second-order term.

Inference is a plain forward pass with BatchNorm in eval mode (fixed running
statistics) — the paper explicitly disables transductive/test-batch BN so
predictions stay causal.

Faithful-adaptation notes (disclosed in the card): (1) the optional data-driven
inner-loop layer freezing and the per-pair negative-transfer guard (paper §3.8,
called "ad hoc" there) are omitted — core MDMAML is domain-paired FOMAML; (2) the
backbone is the benchmark's EA-EEGNet, shared with the other transfer rows.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import cycle_batches, forward_logits

MDMAML_INNER_LR = 0.001     # alpha — inner-loop adaptation LR
MDMAML_META_LR = 0.001      # beta  — meta (outer-loop) LR
MDMAML_INNER_STEPS = 1      # paper: all inner-loop steps were 1


class MDMAML(Strategy):
    mode = "gradient"
    uses_target = False

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        cfg, device = ctx.cfg, ctx.device
        model.to(device)
        crit = nn.CrossEntropyLoss()
        inner_lr = float(cfg.hp.get("mdmaml_inner_lr", MDMAML_INNER_LR))   # alpha
        meta_lr = float(cfg.hp.get("mdmaml_meta_lr", MDMAML_META_LR))      # beta
        domains = [int(d) for d in source.domains()]
        M = len(domains)
        if M < 2:                                   # need >=2 domains to form a pair
            from ._common import supervised_train
            return supervised_train(model, source, ctx)
        n_pairs = max(1, M // 2)
        cyc = {k: cycle_batches(source.select(source.domain == k), cfg.batch_size,
                                cfg.seed + 101 * (k + 1)) for k in domains}
        params = list(model.parameters())
        # Adam meta-optimizer for the outer loop (learn2learn's default): plain SGD
        # at the paper's beta=0.001 badly undertrains EEGNet, so the 0-shot meta-model
        # collapses to near chance; Adam adapts the per-parameter step and recovers it.
        meta_opt = torch.optim.Adam(params, lr=meta_lr)
        rng = np.random.RandomState(cfg.seed)
        # many meta-updates per epoch: one episode (M/2 pairs -> one meta-update) per
        # minibatch step, covering the source once per epoch — so 200 epochs give
        # enough updates to meta-train from scratch (one episode/epoch does not).
        avg_len = int(np.mean([len(source.select(source.domain == k)) for k in domains]))
        steps_per_epoch = max(1, avg_len // cfg.batch_size)

        for epoch in range(cfg.epochs):
            model.train()
            for _step in range(steps_per_epoch):
                perm = list(rng.permutation(domains))
                pairs = []
                for p in range(n_pairs):
                    si = perm[(2 * p) % M]
                    sj = perm[(2 * p + 1) % M]
                    if si == sj:
                        sj = perm[(2 * p + 2) % M]
                    pairs.append((si, sj))

                saved = [p.detach().clone() for p in params]
                meta_grads = [torch.zeros_like(p) for p in params]
                n_used = 0
                for si, sj in pairs:
                    bi = next(cyc[si]).to(device)
                    bj = next(cyc[sj]).to(device)
                    if bi.x.size(0) <= 1 or bj.x.size(0) <= 1:
                        continue
                    # ---- inner step on support S_i: theta -> theta' (Eq. 1) ----
                    _, logits_i = model(bi.x)
                    loss_i = crit(logits_i, bi.y)
                    grads_i = torch.autograd.grad(loss_i, params, create_graph=False)
                    with torch.no_grad():
                        for pr, g in zip(params, grads_i):
                            pr.sub_(inner_lr * g)
                    # ---- query loss on the different domain S_j at theta' (Eq. 2) ----
                    _, logits_j = model(bj.x)
                    loss_j = crit(logits_j, bj.y)
                    grads_j = torch.autograd.grad(loss_j, params, create_graph=False)
                    for acc, g in zip(meta_grads, grads_j):    # FOMAML: accumulate query grad
                        acc.add_(g.detach())
                    with torch.no_grad():                       # restore theta for the next pair
                        for pr, s in zip(params, saved):
                            pr.copy_(s)
                    n_used += 1
                # ---- meta-update on the original weights (Eqs. 3-4), via Adam ----
                if n_used > 0:
                    meta_opt.zero_grad()
                    for pr, mg in zip(params, meta_grads):
                        pr.grad = mg / n_used
                    meta_opt.step()
        return model

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        # forward-only (Eq. 8), BN in eval mode with fixed running stats
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return y_score.argmax(1), y_score
