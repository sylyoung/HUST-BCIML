# ===========================================================================
# PAT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/xqchen914/PAT
#
# Reference (IEEE BibTeX). Journal pre-proof (accepted, in press); no final
# volume/pages yet, so only verifiable fields are set and ``note`` marks it as
# in press.
#   @Article{Chen2026,
#     author  = {Chen, Xiaoqing and Jia, Tianwang and Tu, Yunlu and Wu, Dongrui},
#     journal = {Fundamental Research},
#     title   = {{PAT}: Privacy-Preserving Adversarial Transfer for Accurate, Robust and Privacy-Preserving {EEG} Decoding},
#     year    = {2026},
#     doi     = {10.1016/j.fmre.2026.04.034},
#     note    = {In press},
#   }
# ===========================================================================
"""PAT (Chen et al., 2026, Fundamental Research, in press) — Privacy-Preserving
Adversarial Transfer: a unified transfer-learning framework that jointly
integrates data alignment, adversarial training, and privacy-preserving transfer
so that a target user's EEG decoder is at once accurate, adversarially robust, and
privacy-preserving (Sec. 3.2). Given a small labeled calibration set from the
target user plus a *source prior* (Sec. 3.2.1), PAT applies the SAME three-step
pipeline regardless of scenario: (1) align the target calibration trials with
Euclidean Alignment (EA; Eqs. 2-3, Sec. 3.2.5); (2) augment them by amplitude
scaling ``X' = X * (1 +/- beta)``, beta = 0.05 (Eq. 4, Sec. 3.2.5); (3) fit the
target classifier by adversarial training on the aligned+augmented target data,
optionally combined with a supervised loss on the source prior. Adversarial
training is the min-max / saddle-point problem of Eq. 5 with adversarial examples
X^adv drawn from the ``l_inf`` ball of radius eps around each trial. The single
pipeline is instantiated in three privacy-preserving scenarios that differ ONLY in
which source prior is available (Algorithm 1): centralized source-free transfer
(a model trained on aggregated source data, Sec. 3.2.2), federated source-free
transfer (a federated global model, Sec. 3.2.3), and transfer with
privacy-preserved source data (a user-wise perturbed source set that is safe to
share, Sec. 3.2.4; then Eq. 8 adds the source-benign loss). PAT extends ABAT
(single-domain EA -> AT; Chen et al., 2024) to the cross-domain, source-prior,
privacy-preserving setting (Table 1). The paper reports benign, adversarial, and
noisy classification accuracy on five EEG datasets (Sec. 4.2).

This file implements PAT's alignment + scaling-augmentation + adversarial-training
procedure as a Strategy plug-in for the shared cross-subject LOSO protocol,
composed with ``aligner: EA`` and ``augmenter: Scaling`` (Eq. 4). The three
privacy-preserving scenarios and the source-prior guidance are out of scope for
this single-axis protocol; here the procedure runs on the labeled source (train)
data alone, with no source prior and no target calibration, and the leaderboard
row measures benign accuracy. The privacy / robustness / multi-scenario story of
the full method is not exercised by this row.

Adversarial samples follow the paper's PGD (Eqs. 6-7), with a GLOBAL ``l_inf``
epsilon budget (one absolute eps for all channels, from the Eq. 5 ball
``B(X_T, eps)``) — unlike ABAT's per-channel std-scaled budget. PGD starts from a
noisy benign sample (Eq. 6: ``X^{adv,0} = X + xi``, ``xi ~ U(-eps, eps)``) and
takes projected gradient-sign steps of size alpha <= eps that keep X^adv within
the eps-ball (Eq. 7). Defaults follow the paper: ``eps = 0.03`` (0.01 on SEED,
Sec. 4.4), step size ``alpha = 0.005``, ``steps = 10`` (Sec. 4.4).

Faithful-adaptation notes. (1) A short clean warmup (fraction of epochs, default
0.2) trains benignly before the adversarial phase; the paper's own algorithm has
no warmup, so this follows the authors' released training code and is exposed as a
hyperparameter. (2) Optimizer / early stopping match the shared ERM trainer
(Adam + held-out-source early stop) as for every other strategy, keeping the
strategy-axis comparison controlled, rather than the source's fixed LR schedule.
(3) The scaling augmentation is supplied by the composed ``Scaling`` augmenter
stage (Eq. 4), so PAT differs from ERM in exactly EA + scaling-augment +
adversarial training.
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
    """PGD with a GLOBAL l_inf budget (Chen et al. 2026, Eqs. 5-7): one absolute
    eps for all channels, matching the Eq. 5 ball ``B(x, eps)``. Starts from a
    noisy benign sample ``x + U(-eps, eps)`` (Eq. 6) and takes ``steps`` projected
    sign-gradient steps of size ``alpha`` (Eq. 7), keeping the perturbation within
    the eps-ball of ``x``. Returns the adversarial batch (detached)."""
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

    eps: float = 0.03          # global l_inf budget (Eq. 5 ball; Sec. 4.4, 0.01 on SEED)
    alpha: float = 0.005       # PGD step size alpha <= eps (Eq. 7; Sec. 4.4)
    steps: int = 10            # PGD iterations (Sec. 4.4)
    warmup_frac: float = 0.2   # clean-training warmup fraction (authors' code; not in Algorithm 1)

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        hp = ctx.cfg.hp
        eps = float(hp.get("pat_eps", self.eps))
        alpha = float(hp.get("pat_alpha", self.alpha))
        steps = int(hp.get("pat_steps", self.steps))
        warmup_frac = float(hp.get("pat_warmup", self.warmup_frac))
        warmup = int(round(warmup_frac * ctx.cfg.epochs))

        # EA (Eqs. 2-3) is applied upstream by the aligner stage; this hook is the
        # adversarial-training step (Eq. 5): once past the warmup, replace each
        # aligned+augmented batch with its global-budget PGD adversarial counterpart.
        def at_batch(m, batch, epoch, _ctx):
            if epoch < warmup:
                return batch                                        # clean warmup (authors' code; not in Algorithm 1)
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
