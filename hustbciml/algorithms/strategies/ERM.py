# ===========================================================================
# ERM.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Standard empirical risk minimization (supervised source training) — the
# no-transfer baseline. No single-method reference.
# ===========================================================================
"""Empirical Risk Minimization — the supervised source-only baseline.

Train on the (aligned) source subjects, then predict the target with the
frozen model. No adaptation; this is the reference every transfer method must
beat.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import forward_logits, supervised_train


class ERM(Strategy):
    mode = "gradient"

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return supervised_train(model, source, ctx)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        y_pred = logits.argmax(1)
        return y_pred, y_score
