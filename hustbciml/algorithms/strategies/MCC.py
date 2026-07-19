# ===========================================================================
# MCC.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Jin2020,
#     author    = {Jin, Ying and Wang, Ximei and Long, Mingsheng and Wang, Jianmin},
#     booktitle = {Proc. European Conf. Computer Vision},
#     title     = {Minimum Class Confusion for Versatile Domain Adaptation},
#     year      = {2020},
#     pages     = {464-480},
#     doi       = {10.1007/978-3-030-58589-1_28},
#   }
# ===========================================================================
"""MCC — Minimum Class Confusion (Jin et al., ECCV 2020), as used for
cross-subject EEG in DeepTransferEEG ``tl/mcc.py``.

Transductive but non-adversarial: train the source classifier and, on unlabeled
target predictions, minimize the off-diagonal mass of an entropy-reweighted
class-confusion matrix — pushing target predictions to be individually confident
and mutually un-confused between classes. No auxiliary module.

mode='gradient', uses_target=True → the Exp supplies the aligned, label-masked
target in ``ctx.target_unlabeled``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._common import forward_logits, transductive_train


def _class_confusion(logits: torch.Tensor, t: float, eps: float = 1e-5) -> torch.Tensor:
    """Entropy-reweighted minimum-class-confusion loss on target logits.
    Vendored from DeepTransferEEG ``tl/utils/loss.py`` (``ClassConfusionLoss``)."""
    n_sample, n_class = logits.shape
    softmax_out = torch.softmax(logits / t, dim=1)
    entropy_weight = (-torch.sum(softmax_out * torch.log(softmax_out + eps), dim=1)).detach()
    entropy_weight = 1.0 + torch.exp(-entropy_weight)
    entropy_weight = (n_sample * entropy_weight / torch.sum(entropy_weight)).unsqueeze(1)
    confusion = torch.mm((softmax_out * entropy_weight).transpose(1, 0), softmax_out)
    confusion = confusion / torch.sum(confusion, dim=1)
    return (torch.sum(confusion) - torch.trace(confusion)) / n_class


class MCC(Strategy):
    mode = "gradient"
    uses_target = True

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        criterion = nn.CrossEntropyLoss()
        t_mcc = 2.0            # temperature; loss_trade_off = 1.0 (DeepTransferEEG defaults)

        def da_step(m, bs, bt, aux, it, max_iter, ctx):
            _, out_s = m(bs.x)
            _, out_t = m(bt.x)
            return criterion(out_s, bs.y) + _class_confusion(out_t, t_mcc)

        return transductive_train(model, source, ctx, da_step)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
