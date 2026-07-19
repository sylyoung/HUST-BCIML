# ===========================================================================
# _common.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Shared training / online-TTA loops used by the Strategy plug-ins.
# ===========================================================================
"""Shared training/inference helpers for gradient strategies.

Prefixed with ``_`` so the registry auto-scan skips it (it is not a plug-in).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from hustbciml.algorithms.aligners.EA import EA
from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.data_provider.collate import epochs_to_xyd, iterate_batches
from hustbciml.utils.metrics import accuracy
from hustbciml.utils.tools import EarlyStopping


def split_train_val(n: int, val_ratio: float, seed: int):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    if n_val == 0 or n_val >= n:
        return idx, np.array([], dtype=int)
    return idx[n_val:], idx[:n_val]


@torch.no_grad()
def forward_logits(model: nn.Module, epochs: EEGEpochs, device, chunk: int = 256) -> np.ndarray:
    model.eval()
    x, _, _ = epochs_to_xyd(epochs)
    outs = []
    for start in range(0, len(x), chunk):
        xb = x[start:start + chunk].to(device)
        _, logits = model(xb)
        outs.append(logits.detach().cpu())
    return torch.cat(outs, dim=0).numpy()


def supervised_train(model: nn.Module, source: EEGEpochs, ctx: RunContext, batch_fn=None) -> nn.Module:
    """ERM-style source training with early stopping on a held-out source split.
    This is the shared base used by ERM directly and by T-TIME to build the
    source model it then adapts.

    ``batch_fn(model, batch, epoch, ctx) -> batch`` optionally transforms each
    (already augmented, on-device) training batch before the forward pass — ABAT
    uses it to replace the batch with adversarial examples after a warmup. The
    default ``None`` leaves ERM / T-TIME behaviour byte-for-byte unchanged."""
    cfg = ctx.cfg
    device = ctx.device
    model.to(device)

    tr_idx, va_idx = split_train_val(len(source), cfg.val_ratio, cfg.seed)
    train_epochs = source.select(tr_idx)
    has_val = len(va_idx) > 0
    val_epochs = source.select(va_idx) if has_val else None

    # Data-dependent backbone init (e.g. CSP-Net's CSP-initialized spatial conv):
    # a backbone may need the source X, y, which the pipeline lacks at build time.
    # Fire the hook on the training split, before the optimizer, so any layer the
    # hook freezes is excluded below.
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "init_from_source"):
        backbone.init_from_source(train_epochs)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=cfg.early_stop_patience, mode="max")

    for epoch in range(cfg.epochs):
        model.train()
        for batch in iterate_batches(train_epochs, cfg.batch_size, shuffle=True,
                                     drop_last=True, seed=cfg.seed + epoch):
            if batch.x.size(0) <= 1:          # BatchNorm needs >1
                continue
            batch = ctx.augmenter(batch).to(device)
            if batch_fn is not None:              # e.g. ABAT swaps in adversarial examples
                batch = batch_fn(model, batch, epoch, ctx)
            _, logits = model(batch.x)
            loss = criterion(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if has_val:
            logits = forward_logits(model, val_epochs, device)
            acc = accuracy(val_epochs.y, logits.argmax(1))
            is_best = stopper.step(acc, model)
            if (epoch + 1) % max(1, cfg.epochs // 5) == 0 or is_best:
                ctx.log(f"  epoch {epoch + 1}/{cfg.epochs} val_acc={acc:.2f}"
                        f"{' *' if is_best else ''}")
            if stopper.should_stop:
                ctx.log(f"  early stop at epoch {epoch + 1} (best val_acc={stopper.best:.2f})")
                break

    if has_val:
        stopper.restore(model)
        # expose the best held-out-source validation accuracy for hyperparameter
        # selection (the exp aggregates it into summary["val_primary"]); this is a
        # source-only signal, so tuning by it never peeks at the target/test set.
        setattr(model, "_val_score", float(stopper.best))
    return model


# --------------------------------------------------------------------------
# Online test-time-adaptation skeleton (shared by the TTA strategies:
# T-TIME, Tent, PL, BN-adapt). They differ only in the Phase-2 update rule
# and which parameters the optimizer exposes; the streaming + incremental
# Euclidean-Alignment scaffolding below is identical for all of them.
# --------------------------------------------------------------------------
def entropy(p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Row-wise Shannon entropy of a probability tensor (B, K) -> (B,)."""
    return -torch.sum(p * torch.log(p + eps), dim=1)


def collect_bn_modules(model: nn.Module):
    """Every BatchNorm module in the model (targets for BN-affine adaptation
    and running-statistic re-estimation)."""
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    return [m for m in model.modules() if isinstance(m, bn_types)]


def collect_bn_params(model: nn.Module):
    """Affine parameters (weight, bias) of every BatchNorm module — the only
    parameters Tent adapts (Wang et al., ICLR 2021)."""
    params = []
    for m in collect_bn_modules(model):
        if m.affine:
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    if not params:
        raise ValueError("no BatchNorm affine parameters to adapt; the backbone "
                         "has no BatchNorm layers, so Tent/BN-adapt do not apply.")
    return params


def set_bn_train(model: nn.Module):
    """Put the whole model in eval mode, then switch only the BatchNorm modules
    to train mode — so BN uses/updates batch statistics while dropout and other
    stochastic layers stay deterministic during adaptation."""
    model.eval()
    for m in collect_bn_modules(model):
        m.train()


def online_tta_loop(model: nn.Module, target: EEGEpochs, ctx: RunContext,
                    update_fn, make_optimizer=None):
    """Stream target trials one-by-one in chronological order.

    For each trial: (Phase 1) predict it with the current model — after an
    incremental Euclidean-Alignment update when EA is the composed aligner —
    then (Phase 2) run ``update_fn`` on a sliding batch of the most recent
    ``cfg.test_batch`` trials. Predict-then-update means each trial is scored
    *before* the model adapts to it (the honest online protocol).

    ``update_fn(model, xb, optimizer, cfg)`` performs one strategy-specific
    adaptation on the aligned batch ``xb``; ``make_optimizer(model, cfg)``
    builds the optimizer once before the stream (or ``None`` for BN-adapt,
    which takes no gradient step). With ``cfg.steps == 0`` no adaptation fires
    and the loop reduces to frozen-model inference over the online-aligned
    stream — the faithfulness guard the T-TIME test relies on.
    """
    cfg, device = ctx.cfg, ctx.device
    do_align = cfg.aligner != "Identity"
    C, T = target.n_channels, target.n_times
    raw = target.X.astype(np.float64)          # (N, C, T), chronological order
    n = len(raw)

    optimizer = make_optimizer(model, cfg) if make_optimizer is not None else None
    y_score = []
    R = 0
    W = None
    for i in range(n):
        # ---- Phase 1: predict the current trial (frozen) ----
        model.eval()
        if do_align:
            R = EA.online_update(raw[i], R, i)
            W = np.real(EA.inv_sqrt(R))
            sample = W @ raw[i]
        else:
            sample = raw[i]
        xb = torch.from_numpy(sample.reshape(1, 1, C, T)).float().to(device)
        with torch.no_grad():
            _, logits = model(xb)
        y_score.append(torch.softmax(logits, dim=1).cpu().numpy().reshape(-1))

        # ---- Phase 2: sliding-batch adaptation ----
        if (i + 1) >= cfg.test_batch and (i + 1) % cfg.stride == 0:
            batch_raw = raw[i - cfg.test_batch + 1: i + 1]       # (test_batch, C, T)
            if do_align:
                batch_raw = np.matmul(W[None, :, :], batch_raw)
            xb = torch.from_numpy(batch_raw.reshape(cfg.test_batch, 1, C, T)).float().to(device)
            update_fn(model, xb, optimizer, cfg)
        model.eval()

    y_score = np.asarray(y_score)
    return y_score.argmax(1), y_score


# --------------------------------------------------------------------------
# Transductive domain-adaptation training skeleton (shared by CDAN, MCC, DAN,
# JAN ...). Cycles labeled source + unlabeled target minibatches; each method
# supplies its own per-iteration loss (source classification + a DA term) and,
# optionally, auxiliary trainable modules (a domain discriminator). DANN keeps
# its own loop; these newer methods share this one.
# --------------------------------------------------------------------------
def cycle_batches(epochs: EEGEpochs, bs: int, seed: int):
    """Endless shuffled minibatch stream over ``epochs`` (reshuffled each pass);
    falls back to keeping the last short batch if the set is smaller than ``bs``."""
    e = 0
    while True:
        yielded = False
        for b in iterate_batches(epochs, bs, shuffle=True, drop_last=True, seed=seed + e):
            yielded = True
            yield b
        if not yielded:
            for b in iterate_batches(epochs, bs, shuffle=True, drop_last=False, seed=seed + e):
                yield b
        e += 1


def transductive_train(model: nn.Module, source: EEGEpochs, ctx: RunContext,
                       da_step, setup=None) -> nn.Module:
    """Shared transductive DA loop.

    ``setup(model, ctx) -> (aux, extra_params)`` builds any auxiliary trainable
    modules once (e.g. a domain discriminator) and returns their parameters to
    add to the optimizer; return ``(None, [])`` for methods that need none.

    ``da_step(model, bs, bt, aux, it, max_iter, ctx) -> loss`` returns the full
    scalar loss (source classification + DA term) for iteration ``it`` of
    ``max_iter`` on source batch ``bs`` and unlabeled target batch ``bt`` —
    ``it``/``max_iter`` drive adversarial annealing schedules.
    """
    cfg, device = ctx.cfg, ctx.device
    target = ctx.target_unlabeled
    if target is None:
        raise RuntimeError("transductive DA requires ctx.target_unlabeled")
    model.to(device)
    aux, extra_params = (None, [])
    if setup is not None:
        aux, extra_params = setup(model, ctx)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(extra_params), lr=cfg.lr)

    bps = max(1, len(source) // cfg.batch_size)
    max_iter = cfg.epochs * bps
    src = cycle_batches(source, cfg.batch_size, cfg.seed)
    tgt = cycle_batches(target, cfg.batch_size, cfg.seed + 9973)

    model.train()
    for it in range(max_iter):
        bs = next(src).to(device)
        bt = next(tgt).to(device)
        if bs.x.size(0) <= 1 or bt.x.size(0) <= 1:
            continue
        loss = da_step(model, bs, bt, aux, it, max_iter, ctx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
