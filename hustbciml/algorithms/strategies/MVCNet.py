# ===========================================================================
# MVCNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/wzwvv/MVCNet
#
# References (IEEE BibTeX):
#   @Article{Wang2025b,
#     author  = {Wang, Ziwei and Li, Siyang and Chen, Xiaoqing and Wu, Dongrui},
#     journal = {Knowledge-Based Systems},
#     title   = {{MVCN}et: Multi-View Contrastive Network for Motor Imagery Classification},
#     year    = {2025},
#     pages   = {114205},
#     volume  = {328},
#     doi     = {10.1016/j.knosys.2025.114205},
#   }
#   @Article{Wang2023,
#     author  = {Wang, Jiaheng and Yao, Lin and Wang, Yueming},
#     journal = {IEEE Trans. Neural Systems and Rehabilitation Engineering},
#     title   = {{IFN}et: An Interactive Frequency Convolutional Neural Network for Enhancing Motor Imagery Decoding from {EEG}},
#     year    = {2023},
#     pages   = {1900-1911},
#     volume  = {31},
#     doi     = {10.1109/TNSRE.2023.3257319},
#   }
# ===========================================================================
"""MVCNet — Multi-View Contrastive Network (Wang et al., 2025, Knowledge-Based
Systems). A supervised-contrastive training procedure that shapes a CNN backbone
(IFNet) using multi-view augmentation and two contrastive objectives.

Per source minibatch, three augmented views are formed — **flip** (amplitude
negation), **freq** (Hilbert frequency shift), and **cr** (channel reflection +
2-class label swap). All four signals (raw + 3 views) go through the backbone for
a multi-view classification loss, and through an auxiliary transformer-encoder +
projector for two NT-Xent contrastive losses:

  * **CVC** (cross-view): raw vs each augmented view, where each sample's
    representation is ``[backbone-feature ; projector-feature]`` stacked.
  * **CMC** (cross-modal): the stacked backbone-features vs the stacked
    projector-features across all four views.

Total loss = multi-view CE + ``lamda1``·CVC + ``lamda2``·CMC. At **inference only
the backbone + linear head are used** — the transformer/projector and views are
training-time machinery — so the hustbciml pipeline runs MVCNet as ``backbone: IFNet``
+ ``head: Linear`` + ``strategy: MVCNet``.

Faithful-adaptation notes. (1) ``lamda1``/``lamda2`` are passed via CLI in the
source with no hardcoded default; set to 1.0 here (flag for the real run). (2) The
auxiliary transformer feeds the full raw signal (the source drops the last sample
only to hit a hand-picked ``dim_e``); dims are derived from the data. (3) Optimizer
/ early stopping match the shared trainer. The SupConLoss/InfoNCE imported but
unused in the source script are omitted.

Source: github.com/wzwvv/MVCNet (``MVCNet_LOSO.py``).
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
from hustbciml.utils.metrics import accuracy
from hustbciml.utils.tools import EarlyStopping
from ._common import forward_logits, split_train_val
from ._mvcnet import (NTXentLoss, build_encoder, build_projector, flip_view,
                      freqshift_view, make_reflection_perm, reflect_view)


class MVCNet(Strategy):
    mode = "gradient"

    lamda1: float = 1.0       # CVC weight (source passes via CLI; no code default)
    lamda2: float = 1.0       # CMC weight
    temperature: float = 0.2
    f_shift: float = 0.1

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        cfg, device = ctx.cfg, ctx.device
        model.to(device)

        C, T = source.n_channels, source.n_times
        feat_dim = model.backbone.out_features
        netE = build_encoder(T).to(device)                 # transformer over channels
        netP = build_projector(C * T, feat_dim * 4, feat_dim).to(device)
        perm = make_reflection_perm(cfg.ch_names)

        params = list(model.parameters()) + list(netE.parameters()) + list(netP.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()

        tr_idx, va_idx = split_train_val(len(source), cfg.val_ratio, cfg.seed)
        train_epochs = source.select(tr_idx)
        has_val = len(va_idx) > 0
        val_epochs = source.select(va_idx) if has_val else None
        stopper = EarlyStopping(patience=cfg.early_stop_patience, mode="max")

        def project(sig):                                  # raw (B,1,C,T) -> z (B, feat_dim)
            h = netE(sig.squeeze(1))                        # (B, C, T)
            return netP(h.reshape(h.shape[0], -1))

        for epoch in range(cfg.epochs):
            model.train(); netE.train(); netP.train()
            for batch in iterate_batches(train_epochs, cfg.batch_size, shuffle=True,
                                         drop_last=True, seed=cfg.seed + epoch):
                if batch.x.size(0) <= 1:
                    continue
                batch = batch.to(device)
                x, y = batch.x, batch.y
                x1, y1 = flip_view(x), y
                x2, y2 = freqshift_view(x, source.sfreq, self.f_shift), y
                x3, y3 = reflect_view(x, y, perm, cfg.n_classes)

                f0, o0 = model(x)
                f1, o1 = model(x1)
                f2, o2 = model(x2)
                f3, o3 = model(x3)
                ce = criterion(o0, y) + criterion(o1, y1) + criterion(o2, y2) + criterion(o3, y3)

                z0, z1, z2, z3 = project(x), project(x1), project(x2), project(x3)
                bs = f0.shape[0]
                cvc = NTXentLoss(device, bs * 2, self.temperature)
                cmc = NTXentLoss(device, bs * 4, self.temperature)
                raw_rep = torch.cat([f0, z0])
                loss_cvc = (cvc(raw_rep, torch.cat([f1, z1]))
                            + cvc(raw_rep, torch.cat([f2, z2]))
                            + cvc(raw_rep, torch.cat([f3, z3]))) / 3
                loss_cmc = cmc(torch.cat([f0, f1, f2, f3]), torch.cat([z0, z1, z2, z3]))
                loss = ce + self.lamda1 * loss_cvc + self.lamda2 * loss_cmc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if has_val:
                logits = forward_logits(model, val_epochs, device)
                acc = accuracy(val_epochs.y, logits.argmax(1))
                is_best = stopper.step(acc, model)
                if (epoch + 1) % max(1, cfg.epochs // 5) == 0 or is_best:
                    ctx.log(f"  epoch {epoch + 1}/{cfg.epochs} val_acc={acc:.2f}{' *' if is_best else ''}")
                if stopper.should_stop:
                    ctx.log(f"  early stop at epoch {epoch + 1} (best val_acc={stopper.best:.2f})")
                    break

        if has_val:
            stopper.restore(model)
        return model

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        logits = forward_logits(model, target, ctx.device)
        y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return logits.argmax(1), y_score
