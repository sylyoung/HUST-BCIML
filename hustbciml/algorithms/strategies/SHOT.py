# ===========================================================================
# SHOT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/tim-learn/SHOT
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# Reference (IEEE BibTeX):
#   @InProceedings{Liang2020,
#     author    = {Liang, Jian and Hu, Dapeng and Feng, Jiashi},
#     booktitle = {Proceedings of the International Conference on Machine Learning},
#     title     = {Do We Really Need to Access the Source Data? {S}ource Hypothesis Transfer for Unsupervised Domain Adaptation},
#     year      = {2020},
#     pages     = {6028-6039},
#     month     = {Jul.},
#   }
# ===========================================================================
"""SHOT — Source HypOthesis Transfer / source-free domain adaptation
(Liang et al., ICML 2020).

Source-free: train a source model (here the shared ERM loop), then adapt it to
the unlabeled target WITHOUT any source data — freeze the classifier head (the
fixed "hypothesis") and fine-tune only the feature extractor so target features
move to fit the frozen source decision boundary. The adaptation objective is
information maximization (IM): minimize each trial's prediction entropy (make
predictions confident) while keeping the batch-average prediction diverse (use
all classes). This is exactly the IM loss T-TIME uses, applied offline.

This is the DeepTransferEEG ``tl/shot.py`` variant, which runs SHOT-IM: the
clustering pseudo-label term is present in that file but overwritten by the IM
loss, so only information maximization is active.

Offline, not streaming: unlike Tent/T-TIME/SAR, SHOT adapts on the whole target
set for a few epochs before predicting, so it is a non-tta strategy and receives
the offline-EA-aligned target from the Exp (mode='gradient' here only routes that
offline target alignment; SHOT's ``fit`` is plain source ERM, and its ``predict``
performs the source-free adaptation). Vendored from DeepTransferEEG ``tl/shot.py``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from hustbciml.data_provider.collate import iterate_batches
from ._common import entropy, forward_logits, supervised_train

# Source-hypothesis-transfer hyperparameters (DeepTransferEEG tl/shot.py):
# adapt for a few epochs at a reduced learning rate (lr * lr_decay1).
SHOT_ADAPT_EPOCHS = 5
SHOT_LR_SCALE = 0.1


class SHOT(Strategy):
    # non-tta: adapts offline on the whole (offline-EA-aligned) target, so it
    # takes the aligned target from the Exp rather than an online raw stream.
    mode = "gradient"
    uses_target = False

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        # the source "hypothesis": a standard ERM source model.
        return supervised_train(model, source, ctx)

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        cfg, device = ctx.cfg, ctx.device
        model.to(device)

        # freeze the classifier head (the source hypothesis); adapt only the
        # feature extractor to the unlabeled target.
        for p in model.head.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(model.backbone.parameters(), lr=cfg.lr * SHOT_LR_SCALE)

        for epoch in range(SHOT_ADAPT_EPOCHS):
            model.backbone.train()
            model.head.eval()
            for batch in iterate_batches(target, cfg.batch_size, shuffle=True,
                                         drop_last=True, seed=cfg.seed + epoch):
                if batch.x.size(0) <= 1:                # BatchNorm needs >1
                    continue
                xb = batch.x.to(device)
                _, logits = model(xb)
                softmax_out = torch.softmax(logits, dim=1)
                cem = torch.mean(entropy(softmax_out))                  # confident predictions
                msoftmax = softmax_out.mean(dim=0)
                mdr = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))  # class-diversity (maximize marginal entropy)
                loss = cem + mdr                                        # information maximization
                opt.zero_grad()
                loss.backward()
                opt.step()

        # leave the model in a clean state (head grads re-enabled) then predict
        for p in model.head.parameters():
            p.requires_grad_(True)

        logits = forward_logits(model, target, device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return y_score.argmax(1), y_score
