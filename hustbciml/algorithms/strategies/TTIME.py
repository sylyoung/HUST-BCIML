# ===========================================================================
# TTIME.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @Article{Li2024,
#     author  = {Li, Siyang and Wang, Ziwei and Luo, Hanbin and Ding, Lieyun and Wu, Dongrui},
#     journal = {IEEE Trans. Biomedical Engineering},
#     title   = {{T}-{TIME}: Test-Time Information Maximization Ensemble for Plug-and-Play {BCI}s},
#     year    = {2024},
#     number  = {2},
#     pages   = {423-432},
#     volume  = {71},
#     doi     = {10.1109/TBME.2023.3303289},
#   }
# ===========================================================================
"""T-TIME — Test-Time Information Maximization Ensemble (Li et al., IEEE TBME).

Online test-time adaptation: stream the target trials one by one; for each,
optionally update an incremental Euclidean-Alignment reference, predict, then
(on a sliding batch) update the model by minimizing conditional entropy plus a
marginal-distribution regularizer (the information-maximization loss).

Vendored logic from DeepTransferEEG ``tl/ttime.py`` (``TTIME``), balanced case,
wrapped as a Strategy with ``mode='tta'`` over the shared ``online_tta_loop``
skeleton. Source training reuses the shared ERM loop. The online-EA coupling is
handled by the skeleton: when the composed aligner is EA the incremental
reference is used, otherwise the raw stream is fed to the model.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import entropy, online_tta_loop, supervised_train


class TTIME(Strategy):
    mode = "tta"

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        # T-TIME adapts a source model; build it with the standard supervised loop.
        return supervised_train(model, source, ctx)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        def make_opt(m, cfg):
            return torch.optim.Adam(m.parameters(), lr=cfg.lr)

        def update(m, xb, opt, cfg):
            m.train()
            for _ in range(cfg.steps):
                _, logits = m(xb)
                softmax_out = torch.softmax(logits / cfg.temperature, dim=1)
                cem = torch.mean(entropy(softmax_out))                 # conditional entropy
                msoftmax = softmax_out.mean(dim=0)
                mdr = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))  # marginal-diversity regularizer
                loss = cem + mdr
                opt.zero_grad()
                loss.backward()
                opt.step()

        return online_tta_loop(model, target, ctx, update, make_opt)
