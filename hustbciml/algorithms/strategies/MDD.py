# ===========================================================================
# MDD.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Zhang2019,
#     author    = {Zhang, Yuchen and Liu, Tianle and Long, Mingsheng and Jordan, Michael I.},
#     booktitle = {Proceedings of the International Conference on Machine Learning},
#     title     = {Bridging Theory and Algorithm for Domain Adaptation},
#     year      = {2019},
#     pages     = {7404-7413},
#     address   = {Long Beach, CA},
#     month     = {Jun.},
#   }
# ===========================================================================
"""MDD — Margin Disparity Discrepancy (Zhang et al., ICML 2019), as used for
cross-subject EEG in DeepTransferEEG ``tl/mdd.py``.

Transductive adversarial DA. An auxiliary classifier (bottleneck + main head +
adversarial head) is attached to the backbone features; the margin disparity
discrepancy between the two heads on source vs target is minimized adversarially
through a warm-start gradient-reversal layer, while the pipeline's own head fits
the source labels. Prediction uses the pipeline model (backbone + Linear head).

Follows the authors' reference implementation: the warm-start GRL is advanced every
iteration via ``classifier.step()`` (the DeepTransferEEG training script omitted
this, leaving the reversal inert; restored here). margin=4, bottleneck=50,
trade-off=1.0 as in DeepTransferEEG.

mode='gradient', uses_target=True.

Structural note, verified against the reference. Source cross-entropy is computed
on the *pipeline head's* logits, and ``MDDClassifier``'s own main head is trained
only through the disparity term — it receives no label supervision, and its
argmax is what defines the adversarial head's pseudo-targets. That is not an
adaptation made here: DeepTransferEEG's ``tl/mdd.py`` has exactly this structure
(``classifier_loss = criterion(outputs_source, labels_source)`` on the base
network, then ``outputs, outputs_adv = mdd_classifier(x)`` for the transfer term
alone). In the original tllib MDD the MDD classifier *is* the task classifier, so
the two differ. Recorded so the row is read as "reproduces DeepTransferEEG's MDD",
which is the claim this port can support.
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
from ._mdd import ClassificationMarginDisparityDiscrepancy, MDDClassifier


class MDD(Strategy):
    mode = "gradient"
    uses_target = True

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        criterion = nn.CrossEntropyLoss()
        # margin=4.0 and trade_off=1.0 are the reference defaults the published
        # row used; sweepable with ``--hp mdd_margin=`` / ``--hp mdd_trade_off=``.
        mdd_loss = ClassificationMarginDisparityDiscrepancy(
            margin=float(ctx.cfg.hp.get("mdd_margin", 4.0)))

        def setup(m, ctx):
            clf = MDDClassifier(backbone_dim=m.backbone.out_features,
                                num_classes=ctx.cfg.n_classes, bottleneck_dim=50).to(ctx.device)
            clf.train()
            return clf, list(clf.parameters())

        trade_off = float(ctx.cfg.hp.get("mdd_trade_off", 1.0))

        def da_step(m, bs, bt, clf, it, max_iter, ctx):
            feat_s, out_s = m(bs.x)
            feat_t, _ = m(bt.x)
            x = torch.cat((feat_s, feat_t), dim=0)
            outputs, outputs_adv = clf(x)
            # Split at the real source batch size, not with ``chunk(2)``. The two
            # agree whenever the batches are equal, which they are for the shipped
            # datasets — but ``cycle_batches`` falls back to a short final batch for
            # a domain smaller than ``batch_size``, and an even split of an odd
            # concatenation silently relabels source rows as target and vice versa.
            ns = feat_s.size(0)
            y_s, y_t = outputs[:ns], outputs[ns:]
            y_s_adv, y_t_adv = outputs_adv[:ns], outputs_adv[ns:]
            transfer = -mdd_loss(y_s, y_s_adv, y_t, y_t_adv)
            loss = criterion(out_s, bs.y) + trade_off * transfer
            clf.step()                                          # advance warm-start GRL (authors' reference)
            return loss

        return transductive_train(model, source, ctx, da_step, setup=setup)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
