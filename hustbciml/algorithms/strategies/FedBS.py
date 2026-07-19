# ===========================================================================
# FedBS.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/TianwangJia/FedBS
#
# Reference (IEEE BibTeX):
#   @Article{Jia2024,
#     author  = {Jia, Tianwang and Meng, Lubin and Li, Siyang and Liu, Jiajing and Wu, Dongrui},
#     journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
#     title   = {Federated Motor Imagery Classification for Privacy-Preserving Brain-Computer Interfaces},
#     year    = {2024},
#     pages   = {3442-3451},
#     volume  = {32},
#     doi     = {10.1109/TNSRE.2024.3457504},
#   }
# ===========================================================================
"""FedBS — Federated learning with local Batch-specific BN + SAM
(Jia, Meng, Li, Liu & Wu, "Federated Motor Imagery Classification for
Privacy-Preserving Brain-Computer Interfaces," IEEE Trans. Neural Syst. Rehabil.
Eng., 2024). A lab method.

Privacy-preserving cross-subject transfer: every source subject is a federated
client that trains only on its own EEG; the server aggregates model parameters
(never raw data) each communication round. On top of FedAvg, FedBS keeps each
client's BatchNorm layer local and batch-specific (the server aggregates but does
not distribute BN, and BN uses the current batch's statistics at train and test),
and trains each client with a Sharpness-Aware Minimization optimizer so the
aggregated global model lands in a flatter, better-generalizing minimum. The
federated loop, the batch-specific-BN handling, and SAM are in ``_fed`` / ``_sam``.

Hyperparameters follow the paper's federated schedule: 200 communication rounds,
half the clients selected per round (P=0.5), 2 local epochs each (E=2), SAM radius
rho=0.1. The base optimizer is Adam (SAM-wrapped), matching Centralized Training and
the single-source models so every EEGNet row in this scenario shares one optimizer
and lr — only the privacy mechanism differs. lr / weight_decay / batch_size /
test_batch come from the preset. See ``_fed`` for the faithful-adaptation notes
(benchmark EEGNet width, EA aligner).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._fed import federated_predict, federated_train


class FedBS(Strategy):
    mode = "gradient"

    rounds: int = 200          # communication rounds (N_t)
    local_epochs: int = 2      # client local epochs per round (E)
    client_frac: float = 0.5   # fraction of clients selected per round (P)
    rho: float = 0.1           # SAM ascent radius

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return federated_train(model, source, ctx, batch_bn=True, sam=True,
                               rho=self.rho, rounds=self.rounds,
                               local_epochs=self.local_epochs,
                               client_frac=self.client_frac)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        return federated_predict(model, target, ctx, batch_bn=True,
                                 test_batch=ctx.cfg.test_batch)
