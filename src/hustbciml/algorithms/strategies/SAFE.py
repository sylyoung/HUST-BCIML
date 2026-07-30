# ===========================================================================
# SAFE.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (arXiv preprint; journal version not yet out — IEEE BibTeX):
#   @Article{Jia2026,
#     author = {Jia, Tianwang and Chen, Xiaoqing and Wu, Dongrui},
#     title  = {{SAFE}: Secure and Accurate Federated Learning for Privacy-Preserving Brain-Computer Interfaces},
#     year   = {2026},
#     note   = {arXiv preprint arXiv:2601.05789},
#   }
# ===========================================================================
"""SAFE (Jia et al., 2026) — Secure and Accurate FEderated learning for
privacy-preserving BCIs. arXiv preprint arXiv:2601.05789 (v1, 9 Jan 2026); the
journal version is not yet out, so the preprint is authoritative here.

Task and setup. Cross-subject EEG decoding (MI and ERP) with NO calibration data
from the target subject. SAFE targets three challenges at once — cross-subject
generalization, adversarial robustness, and privacy leakage (Sec. I, Abstract) —
on a federated-learning architecture (Sec. III-B; Fig. 2, 3; ref [48], FedAvg):
each of the K source subjects is a client that trains a local copy on its own EEG
only; the server averages the uploaded model parameters (n_k-weighted, Algorithm 1)
and republishes them, so raw EEG never leaves a client. SAFE adds three components
on top of this loop:

  * LBSN — Local Batch-Specific Normalization (Sec. III-C, Eqs. 1-2). Following
    FedBS (Jia et al., IEEE TNSRE 2024, ref [59]), BatchNorm uses the current
    batch's mean/variance at train AND test (mu_B, sigma_B^2; Eqs. 1-2) rather
    than fixed running statistics, to absorb the non-stationary, cross-subject
    distribution shift. The BN affine parameters (gamma, beta) are kept local to
    each client during training (not shared), then aggregated at the server into a
    complete model for the unseen target subject. LBSN also mitigates privacy
    leakage through BN parameters (Sec. III-C, advantage 3).
  * FAT — Federated Adversarial Training in the INPUT space (Sec. III-D, Eqs. 3-4).
    Adversarial training solves the min-max problem of Eq. 3; FAT generates the
    per-batch perturbation with single-step FGSM, ``delta = alpha * sign(grad_X L)``
    under an l_inf constraint ``||delta||_inf <= eps`` (Eq. 4), and the client
    trains on the adversarial batch X_adv = X + delta. The paper sets alpha to 0.03
    times the signal's standard deviation (Sec. IV-E).
  * AWP — Adversarial Weight Perturbation in the PARAMETER space (Sec. III-E,
    Eqs. 5-7; introduced by Wu et al., NeurIPS 2020, ref [61]). AWP solves the
    weight-space min-max of Eq. 5 to flatten the weight loss landscape; a one-step
    approximation gives ``nu = xi * ||theta|| * grad_nu / ||grad_nu||`` (Eq. 6),
    the weights are updated by Eq. 7, and nu is reset after the step (Sec. III-E).
    The perturbation scale is xi = 0.01 (Sec. IV-E).

FAT (input space) and AWP (parameter space) form the paper's "dual-defense
mechanism" against adversarial attacks (Sec. III-A). In the paper, "Secure" refers
to both privacy protection (data stays local; LBSN blocks BN-based reverse
engineering) and adversarial robustness (FAT + AWP); "Accurate" refers to the
cross-subject decoding accuracy. The paper reports SAFE beating 14 baselines
(7 centralized-training, 7 federated) on five datasets (Sec. IV-V), including
centralized methods that use no privacy protection; those numbers are not
reproduced here.

Schedule (Sec. IV-E): 100 communication rounds, half the clients selected per
round (m = K/2), 2 local epochs per client. This file implements SAFE's full
train path — federated loop + LBSN + FAT + AWP; the aggregation/LBSN plumbing is
shared with the other federated strategies in ``_fed`` (see that file for the
faithful-adaptation notes: benchmark EEGNet width, EA aligner, and — a deviation
from the paper, which uses SGD — the benchmark's Adam base optimizer at the
scenario lr, matched across rows for a single-axis comparison).
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

    rounds: int = 100          # R, max communication rounds (Algorithm 1)
    local_epochs: int = 2      # E, local optimization epochs per client (Algorithm 1)
    client_frac: float = 0.5   # sets m, selected clients per round = K/2 (Sec. IV-E)
    fat_alpha: float = 0.03    # FAT/FGSM magnitude alpha, x signal std (Eq. 4; Sec. IV-E)
    awp_xi: float = 0.01       # AWP perturbation scale xi (Eq. 6; Sec. IV-E)

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return federated_train(model, source, ctx, batch_bn=True, sam=False,
                               rounds=self.rounds, local_epochs=self.local_epochs,
                               client_frac=self.client_frac, adv=True,
                               fat_alpha=self.fat_alpha, awp_xi=self.awp_xi)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        return federated_predict(model, target, ctx, batch_bn=True,
                                 test_batch=ctx.cfg.test_batch)
