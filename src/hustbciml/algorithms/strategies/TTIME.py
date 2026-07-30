# ===========================================================================
# TTIME.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# T-TIME (Li et al., 2024): online test-time information maximization for
# plug-and-play cross-subject EEG BCIs (lab-proposed).
# Original authors' code: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @Article{Li2024,
#     author  = {Li, Siyang and Wang, Ziwei and Luo, Hanbin and Ding, Lieyun and Wu, Dongrui},
#     journal = {IEEE Transactions on Biomedical Engineering},
#     title   = {{T}-{TIME}: Test-Time Information Maximization Ensemble for Plug-and-Play {BCI}s},
#     year    = {2024},
#     number  = {2},
#     pages   = {423-432},
#     volume  = {71},
#     doi     = {10.1109/TBME.2023.3303289},
#   }
# ===========================================================================
"""T-TIME (Li et al., 2024, IEEE TBME) — Test-Time Information Maximization
Ensemble for plug-and-play, calibration-free cross-subject EEG BCIs.

Task (Sec. III-A): the most challenging online test-time-adaptation (TTA)
scenario. Unlabeled target trials from a new subject arrive one-by-one in a
stream and each must be classified immediately, using no target labels. As set
up in the paper, T-TIME has three parts:

* Source model training (Sec. III-B): Euclidean Alignment (EA, Eq. 1-2) is
  applied per source subject, the aligned trials are pooled, and M EEGNets
  {f_m} are trained independently with different random seeds by ordinary
  cross-entropy.
* Online prediction (Sec. III-C, III-D): a fresh trial is aligned by
  incremental EA (Eq. 3-4), then the M models' probability outputs are fused —
  by the Spectral Meta-Learner (SML, Eq. 5-6) once enough trials have arrived
  (a > M), else by simple averaging. The trial is predicted BEFORE the model
  updates on it (Algorithm 1). The "Ensemble" in the name is this M-model SML
  fusion.
* Target model update (Sec. III-E, Eq. 7): each f_m is fine-tuned on a sliding
  batch by the information-maximization loss L = L_CEM + L_MDR, updating ALL
  parameters (Sec. III-E-1), not only the normalization layers as in Tent.
  L_CEM is temperature-scaled conditional-entropy minimization (Eq. 8), pushing
  each trial's prediction toward one class. L_MDR is Adaptive Marginal
  Distribution Regularization (Eq. 9-13): the batch marginal p_bar_k (Eq. 9) is
  recalibrated by a pseudo-labeled class-frequency estimate (Eq. 10-12) so that
  it does not falsely penalize genuine test-time class-imbalance.

This file implements the per-model online adaptation of ONE EEGNet, in the
class-balanced regime (paper Sec. IV-C, the DeepTransferEEG ``tl/ttime.py``
default). There L_MDR reduces to the marginal-entropy diversity term of SHOT /
information maximization (Liang et al., 2020) — the sum of the batch-mean
softmax's Shannon entropy, encouraging balanced class usage — because the
class-frequency recalibration of Eq. 10-13 is not applied. The M-model SML
ensemble is not part of this Strategy; it is produced separately by the
benchmark's multi-seed ensemble runner (``scripts/ensemble.py``).

Refactored from the paper's own code (DeepTransferEEG ``tl/ttime.py``) onto the
``mode='tta'`` Strategy, over the shared ``online_tta_loop`` skeleton; no
third-party code. Source training reuses the shared ERM loop. The online-EA
coupling lives in the skeleton: when the composed aligner is EA the incremental
reference (Eq. 3-4) is used, otherwise the raw stream is fed to the model.
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
        # Source model training (Sec. III-B): train one EEGNet on the pooled,
        # EA-aligned source subjects by ordinary cross-entropy. (Per-subject EA is
        # applied upstream by the composed aligner; the M-model ensemble is built
        # by the separate multi-seed runner.) This is the model adapted online.
        return supervised_train(model, source, ctx)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        def make_opt(m, cfg):
            return torch.optim.Adam(m.parameters(), lr=cfg.lr)

        def update(m, xb, opt, cfg):
            # Fine-tune ALL parameters (Sec. III-E-1) on the sliding batch by the
            # information-maximization loss L = L_CEM + L_MDR (Eq. 7), all
            # probabilities temperature-scaled by factor T (Eq. 8-9).
            m.train()
            for _ in range(cfg.steps):
                _, logits = m(xb)
                softmax_out = torch.softmax(logits / cfg.temperature, dim=1)
                # L_CEM: conditional-entropy minimization (Eq. 8) — per-trial
                # mean Shannon entropy, sharpening each prediction to one class.
                cem = torch.mean(entropy(softmax_out))
                # L_MDR: Adaptive Marginal Distribution Regularization (Eq. 9-13).
                # In this class-balanced path the class-frequency recalibration
                # (Eq. 10-13) is omitted, so it reduces to the SHOT / information-
                # maximization diversity term: the Shannon entropy of the batch-
                # mean softmax, encouraging balanced class usage. Written here as
                # its negative (sum p log p), so adding it to +cem MAXIMIZES the
                # marginal entropy while minimizing the conditional entropy.
                msoftmax = softmax_out.mean(dim=0)
                mdr = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
                loss = cem + mdr
                opt.zero_grad()
                loss.backward()
                opt.step()

        return online_tta_loop(model, target, ctx, update, make_opt)
