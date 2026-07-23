# ===========================================================================
# DELTA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Zhao2023,
#     author    = {Zhao, Bowen and Chen, Chen and Xia, Shu-Tao},
#     booktitle = {International Conference on Learning Representations},
#     title     = {{DELTA}: Degradation-Free Fully Test-Time Adaptation},
#     year      = {2023},
#   }
# ===========================================================================
"""DELTA — degradation-free fully test-time adaptation (Zhao et al., ICLR 2023),
online-TTA variant.

Online TTA like Tent/T-TIME, but the marginal-diversity half of the information-
maximization objective is corrected for class imbalance by Dynamic Online
re-weighting (DOT). A running estimate ``z`` of the target class distribution
(exponential moving average, momentum ``lambda_z``) is used to inverse-frequency
weight each trial by its pseudo-label class, so the batch class marginal — and
thus the diversity term — is not dominated by whatever class the stream happens
to over-represent. The loss is entropy minimization (unweighted) plus this
re-weighted marginal-diversity term.

Runs a bespoke streaming loop (not the shared ``online_tta_loop``) because ``z``
is per-stream state updated on a cadence tied to the trial index; the
incremental-EA + sliding-batch scaffolding otherwise matches ``online_tta_loop``,
including the ``steps == 0`` reduction to frozen inference (the faithfulness
guard). Generalized from the source's binary ``z`` to any class count (uniform
initial estimate).

Vendored from DeepTransferEEG ``tl/delta.py``. Source training reuses the shared
ERM loop.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.algorithms.aligners.EA import EA
from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import entropy, supervised_train

_EPS = 1e-5
_LAMBDA_Z = 0.9   # DOT momentum on the running class-distribution estimate (lab default)


class DELTA(Strategy):
    mode = "tta"

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return supervised_train(model, source, ctx)

    def predict(self, model: nn.Module, target: EEGEpochs,
                ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        cfg, device = ctx.cfg, ctx.device
        model.to(device)
        K = cfg.n_classes
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        do_align = cfg.aligner != "Identity"
        C, T = target.n_channels, target.n_times
        raw = target.X.astype(np.float64)                       # (N, C, T), chronological
        n = len(raw)
        tb, stride, steps, temp = cfg.test_batch, cfg.stride, cfg.steps, cfg.temperature

        z = torch.full((K,), 1.0 / K, device=device)            # running class-distribution estimate
        y_score = []
        R, W = 0, None
        for i in range(n):
            # ---- Phase 1: incremental EA, frozen predict ----
            model.eval()
            if do_align:
                R = EA.online_update(raw[i], R, i)
                W = np.real(EA.inv_sqrt(R))
                sample = W @ raw[i]
            else:
                sample = raw[i]
            xb = torch.from_numpy(sample.reshape(1, 1, C, T)).float().to(device)
            with torch.no_grad():
                _, logits = model(xb)
            y_score.append(torch.softmax(logits, dim=1).cpu().numpy().reshape(-1))

            # ---- Phase 2: sliding-batch DELTA update ----
            if (i + 1) >= tb and (i + 1) % stride == 0:
                batch_raw = raw[i - tb + 1: i + 1]
                if do_align:
                    batch_raw = np.matmul(W[None, :, :], batch_raw)
                bx = torch.from_numpy(batch_raw.reshape(tb, 1, C, T)).float().to(device)
                model.train()
                for _ in range(steps):
                    _, logits = model(bx)
                    prob = torch.softmax(logits / temp, dim=1)          # (tb, K)
                    msoftmax = prob.mean(dim=0)                          # (K,)
                    cem = torch.mean(entropy(prob))                     # entropy minimization
                    # DOT: inverse-frequency weights from the running estimate z (constant)
                    pl = prob.argmax(dim=1)
                    w = 1.0 / (z[pl] + _EPS)                            # (tb,)
                    w_bar = tb * w / w.sum()                            # (tb,)
                    weighted_marginal = (prob * w_bar.unsqueeze(1)).sum(dim=0) / tb   # (K,)
                    if (i + 1) % tb == 0:                               # z EMA cadence
                        z = z * _LAMBDA_Z + msoftmax.detach() * (1 - _LAMBDA_Z)
                    gdiv = torch.sum(weighted_marginal * torch.log(weighted_marginal + _EPS))
                    loss = cem + gdiv
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                model.eval()

        y_score = np.asarray(y_score)
        return y_score.argmax(1), y_score
