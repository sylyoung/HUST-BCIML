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
"""FedBS (Jia et al., 2024, IEEE TNSRE) — a federated-learning strategy for
privacy-preserving motor-imagery (MI) BCIs. "FedBS" is the paper's short name for
"Federated classification with local Batch-specific batch normalization and
Sharpness-aware minimization" (Abstract; Fig. 1). A lab method.

The task is cross-subject MI classification without sharing raw EEG. Federated
learning (FL) treats each subject as one client that trains a local model on its
own EEG only; a central server distributes a global model, receives the locally
updated parameters, and aggregates them — the server never sees raw EEG and the
clients never see each other's, so user data stays private (Sec. II-A/II-C,
Fig. 1). FedBS builds on FedAvg (McMahan et al., 2017) and adds two mechanisms
(Sec. III):

* Local batch-specific batch normalization (Sec. III-C). Following FedBN, the
  BN-layer parameters are localized: clients upload every parameter (BN included)
  and the server aggregates all of them, but the server distributes the global
  model WITHOUT the BN-layer parameters (Algorithm 1: "Server does not distribute
  BN layer parameters"), so each client keeps its own BN and per-client feature
  shift is not averaged away. Unlike FedBN, the server still holds a complete
  model and can therefore classify previously unseen subjects. On top of that,
  FedBS makes BN batch-specific: the BN mean/variance are recomputed from the
  current batch (Eqs. 1-2) during both training AND testing, rather than reusing
  the running statistics fixed after training, so the model adapts to feature
  shift on new subjects (Sec. III-C).
* Sharpness-Aware Minimization (SAM; Sec. III-D, Eqs. 3-8). Each client trains
  with SAM, which minimizes the loss plus its sharpness by a two-step update —
  first an inner gradient ascent to the worst-case perturbation eps* within an
  L2 radius rho (Eqs. 5-6), then a descent step on the original weights using the
  gradient at the perturbed point (Eqs. 7-8). This drives each local model toward
  a flatter minimum so the aggregated global model generalizes better.

The federated loop (Algorithm 1: m = max(P*K, 1) clients selected per round over
N_t rounds, aggregation weighted by n_k / sum_k n_k), the batch-specific-BN
handling, and the SAM update are shared with FedAvg in ``_fed`` / ``_sam``; this
file just fixes FedBS's flags (batch_bn=True, sam=True).

Hyperparameters follow the paper's FL setup (Sec. IV-C): N_t=200 communication
rounds, client-selection weight P=0.5 (half the clients per round), E=2 local
epochs, and SAM radius rho=0.1 (the one extra hyperparameter FedBS introduces).
Adaptation note: the paper trains all methods with SGD (momentum 0.9); this
benchmark instead wraps SAM around Adam so every EEGNet row in this scenario
(Centralized Training, the single-source models, and the FL methods) shares one
base optimizer and lr and the comparison isolates the privacy mechanism. lr /
weight_decay / batch_size / test_batch come from the preset. See ``_fed`` for the
other faithful-adaptation notes (benchmark EEGNet width, EA aligner).
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

    rounds: int = 200          # N_t: max communication rounds (Algorithm 1)
    local_epochs: int = 2      # E: local optimization epochs per client per round
    client_frac: float = 0.5   # P: client-selection weight; m = max(P*K, 1) chosen per round
    rho: float = 0.1           # rho: SAM gradient-ascent step size (Eqs. 6, 8)

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return federated_train(model, source, ctx, batch_bn=True, sam=True,
                               rho=self.rho, rounds=self.rounds,
                               local_epochs=self.local_epochs,
                               client_frac=self.client_frac)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        return federated_predict(model, target, ctx, batch_bn=True,
                                 test_batch=ctx.cfg.test_batch)
