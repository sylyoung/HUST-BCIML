# ===========================================================================
# MVCNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/wzwvv/MVCNet
#
# Reference (IEEE BibTeX):
#   @Article{Wang2025b,
#     author  = {Wang, Ziwei and Li, Siyang and Chen, Xiaoqing and Wu, Dongrui},
#     journal = {Knowledge-Based Systems},
#     title   = {{MVCN}et: Multi-View Contrastive Network for Motor Imagery Classification},
#     year    = {2025},
#     pages   = {114205},
#     volume  = {328},
#     doi     = {10.1016/j.knosys.2025.114205},
#   }
# (MVCNet's IFNet backbone is documented and cited in models/IFNet.py.)
# ===========================================================================
"""MVCNet — Multi-View Contrastive Network for motor imagery classification
(Wang et al., 2025, Knowledge-Based Systems; Sec. 3).

MVCNet is a training STRATEGY layered on a dual-branch architecture (paper Sec.
3.2, Fig. 2): a CNN branch that captures local spatial-temporal features and a
Transformer branch that models global temporal dependencies. In this benchmark
the CNN branch is the configured backbone (IFNet by default — Wang et al., 2023,
a third-party multi-band CNN; its internals live in models/IFNet.py), and the
paper's Transformer branch is realized by the auxiliary transformer-encoder +
projector in _mvcnet.py. On top of the two branches MVCNet adds a multi-view data
augmentation pipeline and two contrastive regularizers, and is trained end-to-end
(paper Sec. 3.6, Eq. 6).

Multi-view augmentation (paper Sec. 3.3, Table 2). The paper defines seven
augmentations spanning three domains: time (Flip/Noise/Scale), frequency
(FShift/FSurr), and space (CR/HS). This port uses one representative view per
domain to keep the strategy dataset-agnostic:
  * flip  — time-domain amplitude negation (paper's Flip);
  * freq  — Hilbert-transform frequency shift (paper's FShift);
  * cr    — Channel Reflection: left/right hemisphere channel swap with the
            2-class label swap (paper's CR; Wang et al., 2024).
The raw trial plus these views feed both branches.

Losses (paper Sec. 3.4-3.6). Let f = CNN-branch (backbone) features and z =
Transformer-branch (projector) features per trial.
  * Classification L_CLS (Eq. 5): cross-entropy summed over the raw trial and the
    augmented views, on the backbone's linear head.
  * Cross-View Contrasting L_CVC (Sec. 3.4, Eq. 1-2): an NT-Xent objective with
    the raw trial as anchor and each augmented view as its positive (other trials
    are negatives), enforcing consistency across the time/frequency/space views.
  * Cross-Model Contrasting L_CMC (Sec. 3.5, Eq. 3-4): an NT-Xent objective that
    aligns the CNN-branch and Transformer-branch features of the same trial,
    contrasting them against other trials to align the two branches.
Total: L_all = L_CLS + lamda1 * L_CVC + lamda2 * L_CMC, where ``lamda1``/``lamda2``
are the paper's trade-off weights lambda and gamma (Eq. 6). At inference only the
CNN branch (backbone + linear head) is used — the Transformer branch and the
augmented views are training-time machinery — so the pipeline runs MVCNet as
``backbone: IFNet`` + ``head: Linear`` + ``strategy: MVCNet``.

Adaptation notes. (1) The paper fixes lambda = gamma = 0.1 in all experiments
(Sec. 4.3); the source script instead passes them via CLI with no hardcoded
default. They are set to 1.0 here as class attributes and can be overridden per
run (or via the tuner). (2) The auxiliary transformer takes the full raw signal;
its dimensions are derived from the data. (3) Optimizer / early stopping follow
the shared trainer. Loss classes imported but unused in the source script are
omitted.

Original authors' code: github.com/wzwvv/MVCNet (``MVCNet_LOSO.py``).
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

    lamda1: float = 1.0       # L_CVC weight = paper's lambda (Eq. 6; paper uses 0.1)
    lamda2: float = 1.0       # L_CMC weight = paper's gamma  (Eq. 6; paper uses 0.1)
    temperature: float = 0.2  # NT-Xent temperature tau (Eq. 1/3)
    f_shift: float = 0.1      # FShift amount in Hz (frequency-domain view)

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        cfg, device = ctx.cfg, ctx.device
        model.to(device)

        C, T = source.n_channels, source.n_times
        feat_dim = model.backbone.out_features
        netE = build_encoder(T).to(device)                 # Transformer-branch encoder (paper Sec. 3.2)
        netP = build_projector(C * T, feat_dim * 4, feat_dim).to(device)  # projector onto CNN feature dim
        perm = make_reflection_perm(cfg.ch_names)

        params = list(model.parameters()) + list(netE.parameters()) + list(netP.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()

        tr_idx, va_idx = split_train_val(len(source), cfg.val_ratio, cfg.seed)
        train_epochs = source.select(tr_idx)
        has_val = len(va_idx) > 0
        val_epochs = source.select(va_idx) if has_val else None
        stopper = EarlyStopping(patience=cfg.early_stop_patience, mode="max")

        def project(sig):                                  # Transformer-branch feature z (B, feat_dim)
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

                # CNN branch: features f* and logits o* for raw + 3 views
                f0, o0 = model(x)
                f1, o1 = model(x1)
                f2, o2 = model(x2)
                f3, o3 = model(x3)
                # L_CLS (Eq. 5): CE over the raw trial and the augmented views
                ce = criterion(o0, y) + criterion(o1, y1) + criterion(o2, y2) + criterion(o3, y3)

                # Transformer branch: features z* for the same signals
                z0, z1, z2, z3 = project(x), project(x1), project(x2), project(x3)
                bs = f0.shape[0]
                cvc = NTXentLoss(device, bs * 2, self.temperature)
                cmc = NTXentLoss(device, bs * 4, self.temperature)
                # L_CVC (Eq. 1-2): raw as anchor vs each view, averaged; per trial the
                # branch features [f;z] are stacked so both branches see the views.
                raw_rep = torch.cat([f0, z0])
                loss_cvc = (cvc(raw_rep, torch.cat([f1, z1]))
                            + cvc(raw_rep, torch.cat([f2, z2]))
                            + cvc(raw_rep, torch.cat([f3, z3]))) / 3
                # L_CMC (Eq. 3-4): align CNN-branch features vs Transformer-branch
                # features of the same trial (across all views).
                loss_cmc = cmc(torch.cat([f0, f1, f2, f3]), torch.cat([z0, z1, z2, z3]))
                # L_all (Eq. 6): classification + lambda*CVC + gamma*CMC
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
