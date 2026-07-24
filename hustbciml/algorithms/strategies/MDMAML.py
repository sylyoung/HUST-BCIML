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
"""MDMAML — Multi-Domain Model-Agnostic Meta-Learning (Li et al., 2022, IEEE CIM).

An optimization-based meta-learning framework for source-free, cross-subject,
few-shot EEG classification (paper §III). Each source subject is treated as a
separate domain S_i (M domains, N labeled trials each, §III-A). MDMAML adapts
MAML's episodic inner/outer loops to META-LEARN THE DOMAIN-ADAPTATION PROCESS
ACROSS SOURCE DOMAINS: a good initialization theta is learned so that a one-step
inner adaptation on one source domain reduces the loss on a *different* source
domain, i.e. the meta-objective minimizes cross-domain shift. Once trained, the
initialization theta_MDMAML is saved and the source data is discarded, so a new
target subject is served without any access to the source EEG — the paper's
privacy-protection / source-free contribution (§III-A, Alg. 1).

Per training episode (paper Alg. 1, Eqs. 1-6): form ``M/2`` random source domain
pairs ``(S_i, S_j)`` (i != j); take one inner SGD step on the support domain
``S_i`` to get temporary weights ``theta'_i = theta - alpha * grad L_{S_i}(theta)``
(Eq. 1, alpha = inner-loop LR); evaluate the query loss of the *different* domain
``S_j`` at ``theta'_i``, ``L_{(S_i,S_j)}(theta) = L_{S_j}(theta'_i)`` (Eq. 2);
average over the M/2 pairs (Eq. 3) and meta-update the ORIGINAL weights
``theta <- theta - beta * grad L_MDMAML`` (Eq. 4, beta = meta LR). The task loss
L is standard cross-entropy (Eq. 6). Following §IV-E, a first-order approximation
replaces the second-order gradient: the query-loss gradient evaluated at
``theta'_i`` is accumulated directly as the meta-gradient.

At deployment the paper describes three regimes off the same theta_MDMAML: 0-shot
(a plain forward pass, Eq. 8), online few-shot (fine-tune on L labeled
calibration trials, ``theta <- theta - gamma * grad L_C``, Eq. 7), and offline
source-free domain adaptation on the unlabeled test set (e.g. SHOT). This file
implements the 0-SHOT setting: theta_MDMAML is applied to the target subject
forward-only, with no calibration fine-tune (Eq. 7) and no offline step.
Inference runs BatchNorm in eval mode with fixed running statistics — per §IV-F
the paper deactivates transductive/test-batch normalization so predictions stay
causal for real-time use.

Two optional components of the full method (paper §III-B, §III-C) are NOT ported
here: data-driven inner-loop LAYER FREEZING (§III-B, Fig. 4 — freeze layers whose
parameters change little after adapting to a held-out validation domain) and the
NEGATIVE-TRANSFER MITIGATION (§III-C, described there as an "ad hoc" strategy —
skip the Eq. 1 update of a pair when it worsens the loss on S_j). This file keeps
the core domain-paired episodic loop. The backbone is the benchmark's shared
EA-EEGNet, as used by the other transfer rows.
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
        # Adam meta-optimizer for the outer loop (a benchmark deviation from the
        # plain-SGD meta-update in Eq. 4): SGD at the paper's beta=0.001 badly
        # undertrains EEGNet here, so the 0-shot meta-model collapses to near
        # chance; Adam adapts the per-parameter step and recovers it.
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
                    for acc, g in zip(meta_grads, grads_j):    # first-order: accumulate query grad (Eq. 2)
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
