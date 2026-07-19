# ===========================================================================
# BNAdapt.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Schneider2020,
#     author    = {Schneider, Steffen and Rusak, Evgenia and Eck, Luisa and Bringmann, Oliver and Brendel, Wieland and Bethge, Matthias},
#     booktitle = {Proc. Advances in Neural Information Processing Systems},
#     title     = {Improving Robustness against Common Corruptions by Covariate Shift Adaptation},
#     year      = {2020},
#     pages     = {11539-11551},
#     month     = {Dec.},
#   }
# ===========================================================================
"""BN-adapt — test-time BatchNorm statistics adaptation (Nado et al. 2020;
Schneider et al. 2020).

The lightest test-time adaptation: no gradient step at all. Stream the target
trials with incremental Euclidean Alignment; on each sliding batch, put the
BatchNorm layers in train mode and forward the batch so their running mean/var
track the target distribution. Subsequent predictions then use these adapted
statistics. Vendored from DeepTransferEEG ``tl/bn-adapt.py``, over the shared
``online_tta_loop`` skeleton.

Registered under the CLI key ``BNAdapt`` (class name); the ``BN-adapt`` preset
composes it into the full algorithm.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import online_tta_loop, set_bn_train, supervised_train


class BNAdapt(Strategy):
    mode = "tta"

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        return supervised_train(model, source, ctx)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        def update(m, xb, opt, cfg):
            set_bn_train(m)                        # only BN layers track batch statistics
            with torch.no_grad():
                for _ in range(cfg.steps):         # each forward nudges BN running stats
                    m(xb)

        return online_tta_loop(model, target, ctx, update, make_optimizer=None)
