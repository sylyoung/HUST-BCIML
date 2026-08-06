# ===========================================================================
# MVCNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.

# Credit chain († = co-first authors; every node except the integrator carries its GitHub link):
#   Original authors:    Ziwei Wang, Siyang Li, Xiaoqing Chen, Dongrui Wu (2025) — "MVCNet: Multi-View Contrastive Network for Motor Imagery Classification", Knowledge-Based Systems
#                        Original code: https://github.com/wzwvv/MVCNet
#   Implementation:      Ziwei Wang, Siyang Li, Xiaoqing Chen, Dongrui Wu — wzwvv/MVCNet (https://github.com/wzwvv/MVCNet) (official)
#   Current code:        Siyang Li — ported from wzwvv/MVCNet (https://github.com/wzwvv/MVCNet)
#   Integrated by:       Siyang Li <lsyyoungll@gmail.com> — HUST-BCIML

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

    # Defaults, overridable per run with ``--hp mvc_lamda1=0.1`` etc. The
    # benchmark's published MVCNet rows use the 1.0/1.0 weights below, which are
    # ten times the paper's 0.1/0.1 — a deviation that used to be visible only in
    # these two comments and reachable from nowhere, because the strategy never
    # read ``cfg.hp``. Both facts are now stated in the MVCNet card.
    lamda1: float = 1.0       # L_CVC weight = paper's lambda (Eq. 6; paper uses 0.1)
    lamda2: float = 1.0       # L_CMC weight = paper's gamma  (Eq. 6; paper uses 0.1)
    temperature: float = 0.2  # NT-Xent temperature tau (Eq. 1/3)
    f_shift: float = 0.1      # FShift amount in Hz (frequency-domain view)

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        cfg, device = ctx.cfg, ctx.device
        model.to(device)

        lamda1 = float(cfg.hp.get("mvc_lamda1", self.lamda1))
        lamda2 = float(cfg.hp.get("mvc_lamda2", self.lamda2))
        temperature = float(cfg.hp.get("mvc_temp", self.temperature))
        f_shift = float(cfg.hp.get("mvc_f_shift", self.f_shift))

        C, T = source.n_channels, source.n_times
        feat_dim = model.backbone.out_features
        netE = build_encoder(T).to(device)                 # Transformer-branch encoder (paper Sec. 3.2)
        netP = build_projector(C * T, feat_dim * 4, feat_dim).to(device)  # projector onto CNN feature dim
        # Channel Reflection is a *conditional* view: it needs a real 10-20
        # montage and a left/right class pair. On BNCI2014002 (generic EEG1..15
        # labels, right-hand vs feet) and BNCI2015001 (right-hand vs feet) it
        # applies to neither, so the view is dropped rather than fabricated. The
        # contrastive losses below are written over however many views survive.
        perm = make_reflection_perm(cfg.ch_names, cfg.classes)
        ctx.log(f"  [MVCNet] views: flip, freq-shift"
                f"{', channel-reflection' if perm is not None else ' (channel reflection N/A here)'}")

        params = list(model.parameters()) + list(netE.parameters()) + list(netP.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()

        tr_idx, va_idx = split_train_val(len(source), cfg.val_ratio, cfg.seed,
                                         domain=source.domain, mode=cfg.val_split)
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
                # The paper's three views: time-domain flip, frequency shift, and
                # (where valid) the space-domain channel reflection.
                views = [(flip_view(x), y),
                         (freqshift_view(x, source.sfreq, f_shift), y)]
                if perm is not None:
                    views.append(reflect_view(x, y, perm))

                # CNN branch: features f* and logits o* for raw + each view
                f0, o0 = model(x)
                fv, ov = zip(*(model(xv) for xv, _ in views))
                # L_CLS (Eq. 5): CE over the raw trial and the augmented views
                ce = criterion(o0, y) + sum(criterion(o, yv)
                                            for o, (_, yv) in zip(ov, views))

                # Transformer branch: features z* for the same signals
                z0 = project(x)
                zv = [project(xv) for xv, _ in views]
                bs = f0.shape[0]
                n_rep = 1 + len(views)
                cvc = NTXentLoss(device, bs * 2, temperature)
                cmc = NTXentLoss(device, bs * n_rep, temperature)
                # L_CVC (Eq. 1-2): raw as anchor vs each view, averaged; per trial the
                # branch features [f;z] are stacked so both branches see the views.
                raw_rep = torch.cat([f0, z0])
                loss_cvc = sum(cvc(raw_rep, torch.cat([f, z]))
                               for f, z in zip(fv, zv)) / len(views)
                # L_CMC (Eq. 3-4): align CNN-branch features vs Transformer-branch
                # features of the same trial (across all views).
                loss_cmc = cmc(torch.cat([f0, *fv]), torch.cat([z0, *zv]))
                # L_all (Eq. 6): classification + lambda*CVC + gamma*CMC
                loss = ce + lamda1 * loss_cvc + lamda2 * loss_cmc

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
