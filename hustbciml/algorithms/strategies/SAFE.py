# ===========================================================================
# SAFE.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Jia2026,
#     author  = {Jia, Tianwang and Chen, Xiaoqing and Wu, Dongrui},
#     journal = {arXiv preprint arXiv:2601.05789},
#     title   = {{SAFE}: Secure and Accurate Federated Learning for Privacy-Preserving Brain-Computer Interfaces},
#     year    = {2026},
#     doi     = {10.48550/arXiv.2601.05789},
#   }
# ===========================================================================
"""SAFE — Secure and Accurate Federated Learning (Jia, Chen & Wu, arXiv 2026).

A lab method built directly on FedBS: the same privacy-preserving federated loop
(one client per source subject, n_k-weighted aggregation, raw EEG never shared)
and the same Local Batch-Specific Normalization (LBSN — batch statistics at train
and test, BN kept client-local, never transmitted). SAFE replaces FedBS's SAM
optimizer with a dual adversarial defense applied inside each client's local
training:

  * FAT (Federated Adversarial Training) — a single-step FGSM perturbation
    ``delta = alpha * sign(grad_X L)`` with ``alpha = 0.03 * signal_std`` (Eq. 4);
    the client trains on the adversarial batch instead of the clean one.
  * AWP (Adversarial Weight Perturbation) — a one-step weight-space perturbation
    ``nu = xi * ||theta|| * grad/||grad||`` (xi=0.01, Eqs. 6-7): ascend to
    theta+nu, take the gradient step there, then reset nu, landing a SAM-like flat-
    minimum update at the original weights.

"Secure" here means adversarial robustness (FAT + AWP) plus privacy by the FL
topology + LBSN (no cryptographic secure aggregation, no differential privacy — so
there is nothing extra to simulate at aggregation). The federated schedule follows
the paper: 100 communication rounds, half the clients per round (P=0.5), 2 local
epochs. The base optimizer stays Adam at the scenario lr (as in FedBS), so SAFE
differs from FedBS only in rounds and the FAT+AWP robustness mechanism — a clean
single-axis delta. See ``_fed`` for the shared federated / LBSN faithful-adaptation
notes (benchmark EEGNet width, EA aligner).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._fed import federated_predict, federated_train


class SAFE(Strategy):
    mode = "gradient"

    rounds: int = 100          # communication rounds (R)
    local_epochs: int = 2      # client local epochs per round (E)
    client_frac: float = 0.5   # fraction of clients selected per round (m/K, P=0.5)
    fat_alpha: float = 0.03    # FAT/FGSM magnitude (x signal std, Eq. 4)
    awp_xi: float = 0.01       # AWP perturbation scale (Eq. 6)

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return federated_train(model, source, ctx, batch_bn=True, sam=False,
                               rounds=self.rounds, local_epochs=self.local_epochs,
                               client_frac=self.client_frac, adv=True,
                               fat_alpha=self.fat_alpha, awp_xi=self.awp_xi)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        return federated_predict(model, target, ctx, batch_bn=True,
                                 test_batch=ctx.cfg.test_batch)
