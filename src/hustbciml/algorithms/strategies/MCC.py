# ===========================================================================
# MCC.py  —  HUST-BCIML EEG-decoding benchmark

# Original authors:    Ying Jin, Ximei Wang, Mingsheng Long, Jianmin Wang (2020) — "Minimum Class Confusion for Versatile Domain Adaptation", Proc. ECCV
#                      Original code: https://github.com/thuml/Transfer-Learning-Library (reference implementation)
# Implementation:      Junguang Jiang et al. (thuml) — thuml/Transfer-Learning-Library (https://github.com/thuml/Transfer-Learning-Library) (canonical implementation)
# Current code:        Siyang Li — sylyoung/DeepTransferEEG (https://github.com/sylyoung/DeepTransferEEG) (ported from)

# References (IEEE BibTeX):
#   @InProceedings{Jin2020,
#     author    = {Jin, Ying and Wang, Ximei and Long, Mingsheng and Wang, Jianmin},
#     booktitle = {Proceedings of the European Conference on Computer Vision},
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
    Vendored from DeepTransferEEG ``tl/utils/loss.py`` (``ClassConfusionLoss``).

    KNOWN DEVIATION FROM THE PAPER — DO NOT "FIX" WITHOUT RE-MEASURING.
    The normalisation below is written ``confusion / torch.sum(confusion, dim=1)``
    with no ``keepdim=True``. The divisor therefore has shape ``(C,)`` and
    broadcasts along the *last* axis, so entry ``[i][j]`` is divided by row-sum
    ``j``, not by row-sum ``i``. Because the entropy weighting makes the confusion
    matrix asymmetric, that is not the row normalisation Jin et al. (2020) define.

    It is kept because it is what the reference implementation does: the line is
    character-for-character identical in DeepTransferEEG ``tl/utils/loss.py``
    (``ClassConfusionLoss``) and in the widely circulated official MCC code, so
    every published MCC baseline — including this lab's own — was measured with
    it. Adding ``keepdim=True`` would change this benchmark's MCC row, currently
    the transfer-table leader, and make it non-comparable with all of them.

    So: the port is faithful to the reference implementation, and the reference
    implementation departs from the paper's equation. Both halves of that sentence
    matter; see the MCC card, which records the same thing for readers.
    """
    n_sample, n_class = logits.shape
    softmax_out = torch.softmax(logits / t, dim=1)
    entropy_weight = (-torch.sum(softmax_out * torch.log(softmax_out + eps), dim=1)).detach()
    entropy_weight = 1.0 + torch.exp(-entropy_weight)
    entropy_weight = (n_sample * entropy_weight / torch.sum(entropy_weight)).unsqueeze(1)
    confusion = torch.mm((softmax_out * entropy_weight).transpose(1, 0), softmax_out)
    confusion = confusion / torch.sum(confusion, dim=1)   # see note above (no keepdim, as upstream)
    return (torch.sum(confusion) - torch.trace(confusion)) / n_class


class MCC(Strategy):
    mode = "gradient"
    uses_target = True

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        criterion = nn.CrossEntropyLoss()
        # temperature; loss_trade_off = 1.0 (DeepTransferEEG defaults).
        # Sweepable with ``--hp mcc_temp=<t>``.
        t_mcc = float(ctx.cfg.hp.get("mcc_temp", 2.0))

        def da_step(m, bs, bt, aux, it, max_iter, ctx):
            _, out_s = m(bs.x)
            _, out_t = m(bt.x)
            return criterion(out_s, bs.y) + _class_confusion(out_t, t_mcc)

        return transductive_train(model, source, ctx, da_step)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
