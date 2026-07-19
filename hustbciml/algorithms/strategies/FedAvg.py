# ===========================================================================
# FedAvg.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @InProceedings{McMahan2017,
#     author    = {McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and Ag{\"u}era y Arcas, Blaise},
#     booktitle = {Proc. Int. Conf. Artif. Intell. Statist. ({AISTATS})},
#     title     = {Communication-Efficient Learning of Deep Networks from Decentralized Data},
#     year      = {2017},
#     pages     = {1273-1282},
#     volume    = {54},
#   }
# ===========================================================================
"""FedAvg — Federated Averaging (McMahan et al., AISTATS 2017), the standard
federated-learning baseline.

Every source subject is a client that trains locally with plain Adam; each
communication round the server averages the selected clients' model parameters,
weighted by their sample counts, and redistributes the full global model
(BatchNorm included). Privacy-preserving in the same sense as FedBS — raw EEG never
leaves the client — but without FedBS's local batch-specific BN or SAM optimizer,
so it is the reference that isolates what those two additions contribute. Shares the
communication loop in ``_fed`` (batch_bn=False, sam=False).

Hyperparameters match FedBS's federated schedule (200 rounds, P=0.5, 2 local
epochs); the base optimizer is Adam, matching Centralized Training so only the
privacy mechanism differs. lr / weight_decay / batch_size from the preset.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._fed import federated_predict, federated_train


class FedAvg(Strategy):
    mode = "gradient"

    rounds: int = 200
    local_epochs: int = 2
    client_frac: float = 0.5

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return federated_train(model, source, ctx, batch_bn=False, sam=False,
                               rounds=self.rounds, local_epochs=self.local_epochs,
                               client_frac=self.client_frac)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        return federated_predict(model, target, ctx, batch_bn=False,
                                 test_batch=ctx.cfg.test_batch)
