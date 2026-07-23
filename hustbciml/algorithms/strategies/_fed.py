# ===========================================================================
# _fed.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Shared federated / LBSN core (FedBS, SAFE, FedAvg).
#
# References (IEEE BibTeX):
#   @Article{Jia2024,
#     author  = {Jia, Tianwang and Meng, Lubin and Li, Siyang and Liu, Jiajing and Wu, Dongrui},
#     journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
#     title   = {Federated Motor Imagery Classification for Privacy-Preserving Brain-Computer Interfaces},
#     year    = {2024},
#     pages   = {3442-3451},
#     volume  = {32},
#     doi     = {10.1109/TNSRE.2024.3457504},
#   }
#   @InProceedings{McMahan2017,
#     author    = {McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and Ag{\"u}era y Arcas, Blaise},
#     booktitle = {Proceedings of the International Conference on Artificial Intelligence and Statistics},
#     title     = {Communication-Efficient Learning of Deep Networks from Decentralized Data},
#     year      = {2017},
#     pages     = {1273-1282},
#     volume    = {54},
#   }
# ===========================================================================
"""Federated-learning training + inference shared by FedAvg and FedBS.

Under leave-one-subject-out, ``fit`` treats every SOURCE subject as one federated
client that trains on its own local data only — the server never sees raw EEG, so
this is privacy-preserving cross-subject transfer. The loop is the standard FedAvg
communication protocol (McMahan et al., 2017); FedBS (Jia et al., IEEE TNSRE 2024)
adds two options on top, both switched here by flags:

* ``batch_bn`` — local batch-specific BatchNorm. Each client keeps its OWN BN layer
  (the server aggregates every parameter but does NOT distribute the BN parameters,
  so a client's BN is never overwritten by the global average), and BN uses the
  current batch's statistics at train AND test time (``track_running_stats=False``),
  never stored running statistics. This is FedBN improved: the server still holds a
  complete model, so it can classify unseen target subjects.
* ``sam`` — the client optimizes with Sharpness-Aware Minimization (Foret et al.,
  ICLR 2021; the vendored ``_sam.SAM`` wrapping the same Adam base optimizer)
  instead of plain Adam, seeking a flat local minimum so the aggregated global
  model generalizes better.

Faithful-adaptation notes (disclosed in the card): (1) the backbone is the
benchmark's EEGNet (F1=4, F2=8), not the paper's F1=8/F2=16, so every privacy-
preserving row shares one backbone with the centralized reference — a clean
single-axis comparison; the paper's own config/number is kept as the reference
range. (2) EA is the composed aligner (the paper also uses EA). (3) Clients keep a
persistent local model across rounds (so local BN carries over), initialised from
the initial global model; non-selected clients simply skip a round. (4) Aggregation
is n_k-weighted over the selected clients (Algorithm 1); integer buffers
(``num_batches_tracked``) are copied, not averaged.
"""
from __future__ import annotations

import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.data_provider.collate import epochs_to_xyd, iterate_batches
from ._common import collect_bn_modules, set_bn_train
from ._sam import SAM

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _bn_state_keys(model: nn.Module) -> set:
    """state_dict keys (params + buffers) belonging to every BatchNorm module —
    the ones FedBS keeps client-local (never distributes from the server)."""
    keys = set()
    for name, m in model.named_modules():
        if isinstance(m, _BN):
            for pn, _ in m.named_parameters(recurse=False):
                keys.add(f"{name}.{pn}" if name else pn)
            for bn, _ in m.named_buffers(recurse=False):
                keys.add(f"{name}.{bn}" if name else bn)
    return keys


def _make_batch_specific(model: nn.Module):
    """BN uses the current batch's mean/var at train AND test (no running stats) —
    FedBS's local batch-specific BN. Drops the running buffers entirely."""
    for m in collect_bn_modules(model):
        m.track_running_stats = False
        m.running_mean = None
        m.running_var = None
        m.num_batches_tracked = None


def _fat_awp_step(local, xb, yb, crit, opt, fat_alpha, awp_xi):
    """SAFE's client step (Jia et al. 2026): a single-step FGSM adversarial example
    (FAT, Eq. 4) followed by an Adversarial-Weight-Perturbation update (AWP, Eqs.
    6-7) — ascend to theta+nu in weight space, take the gradient step there, then
    subtract nu so the net update lands at the original weights (a SAM-like two-
    point step, nu scaled layer-wise by xi*||theta_l||)."""
    # FAT: FGSM perturbation delta = alpha * sign(grad_x L), alpha = 0.03 * signal std
    xb = xb.detach().requires_grad_(True)
    _, logits = local(xb)
    gx = torch.autograd.grad(crit(logits, yb), xb)[0]
    alpha = fat_alpha * xb.detach().std()
    x_adv = (xb.detach() + alpha * gx.sign()).detach()
    # AWP: ascend to theta+nu, step there, then reset nu
    params = [p for p in local.parameters() if p.requires_grad]
    _, logits = local(x_adv)
    grads = torch.autograd.grad(crit(logits, yb), params)
    nus = []
    with torch.no_grad():
        for p, g in zip(params, grads):
            nu = awp_xi * p.norm() * g / (g.norm() + 1e-12)
            p.add_(nu)
            nus.append(nu)
    opt.zero_grad()
    _, logits = local(x_adv)
    crit(logits, yb).backward()                       # gradient at theta+nu
    with torch.no_grad():
        for p, nu in zip(params, nus):
            p.sub_(nu)                                 # reset to original weights
    opt.step()                                         # net step at theta using grad at theta+nu


def _local_train(local: nn.Module, data: EEGEpochs, ctx: RunContext, *,
                 sam: bool, rho: float, local_epochs: int,
                 seed_offset: int, adv: bool = False,
                 fat_alpha: float = 0.03, awp_xi: float = 0.01):
    """One client's ``local_epochs`` of Adam (or SAM-Adam, or FAT+AWP) on its own
    data. The base optimizer is Adam to match the centralized reference and the
    single-source models (all Adam at the same scenario lr), so the federated rows
    differ from Centralized Training only in the privacy/robustness mechanism, not
    the optimizer. ``adv`` selects SAFE's FGSM+AWP client step over SAM."""
    cfg, device = ctx.cfg, ctx.device
    local.train()
    crit = nn.CrossEntropyLoss()
    if sam and not adv:
        opt = SAM(local.parameters(), torch.optim.Adam, rho=rho,
                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.Adam(local.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
    for e in range(local_epochs):
        for batch in iterate_batches(data, cfg.batch_size, shuffle=True,
                                     drop_last=False, seed=cfg.seed + seed_offset + e):
            xb, yb = batch.x.to(device), batch.y.to(device)
            if xb.size(0) <= 1:                      # batch-specific BN needs >1
                continue
            if adv:
                _fat_awp_step(local, xb, yb, crit, opt, fat_alpha, awp_xi)
            elif sam:
                _, logits = local(xb)
                crit(logits, yb).backward()
                opt.first_step(zero_grad=True)       # ascend to the sharp point
                _, logits = local(xb)
                crit(logits, yb).backward()
                opt.second_step(zero_grad=True)      # descend from w with that grad
            else:
                _, logits = local(xb)
                loss = crit(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()


def federated_train(model: nn.Module, source: EEGEpochs, ctx: RunContext, *,
                    batch_bn: bool, sam: bool, rho: float = 0.1, rounds: int = 200,
                    local_epochs: int = 2, client_frac: float = 0.5,
                    adv: bool = False, fat_alpha: float = 0.03, awp_xi: float = 0.01) -> nn.Module:
    """FedAvg communication loop over source subjects as clients. Returns the
    aggregated global ``model`` (complete, incl. aggregated BN affine)."""
    device = ctx.device
    model.to(device)
    if batch_bn:
        _make_batch_specific(model)

    clients = [int(d) for d in source.domains()]
    data = {k: source.select(source.domain == k) for k in clients}
    n_k = {k: len(data[k]) for k in clients}
    K = len(clients)
    m_sel = max(1, int(client_frac * K))             # floor(P*K), >=1

    bn_keys = _bn_state_keys(model) if batch_bn else set()
    global_state = copy.deepcopy(model.state_dict())
    # persistent per-client state so each client's local BN carries across rounds
    client_state = {k: copy.deepcopy(global_state) for k in clients}
    local = copy.deepcopy(model).to(device)
    # FedAvg-averaged keys = float tensors; ints (num_batches_tracked) are copied
    agg_keys = [k for k, v in global_state.items() if torch.is_floating_point(v)]

    rng = np.random.RandomState(ctx.cfg.seed)
    for t in range(rounds):
        selected = [int(s) for s in rng.choice(clients, size=m_sel, replace=False)]
        updates = {}
        for k in selected:
            state = client_state[k]                  # keeps this client's local BN
            for key in global_state:
                if batch_bn and key in bn_keys:
                    continue                         # BN not distributed (FedBS)
                state[key] = global_state[key].clone()
            local.load_state_dict(state)
            _local_train(local, data[k], ctx, sam=sam, rho=rho,
                         local_epochs=local_epochs, seed_offset=t * local_epochs,
                         adv=adv, fat_alpha=fat_alpha, awp_xi=awp_xi)
            new_state = copy.deepcopy(local.state_dict())
            client_state[k] = new_state              # persist (incl. updated local BN)
            updates[k] = new_state
        total = float(sum(n_k[k] for k in selected))
        for key in agg_keys:
            acc = sum((n_k[k] / total) * updates[k][key].to(torch.float64) for k in selected)
            global_state[key] = acc.to(global_state[key].dtype)
    model.load_state_dict(global_state)
    return model


@torch.no_grad()
def federated_predict(model: nn.Module, target: EEGEpochs, ctx: RunContext, *,
                      batch_bn: bool, test_batch: int) -> Tuple[np.ndarray, np.ndarray]:
    """Predict the aligned target with the server model. With ``batch_bn`` the BN
    layers run in batch-specific mode over test minibatches of ``test_batch``."""
    device = ctx.device
    model.to(device)
    x = epochs_to_xyd(target)[0]
    n = len(x)
    if batch_bn:
        set_bn_train(model)                          # eval all, BN in batch-stat mode
    else:
        model.eval()

    bounds = list(range(0, n, test_batch)) + [n]
    ranges = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    # batch-specific BN can't take a size-1 batch (no variance): fold a trailing
    # singleton into the previous minibatch.
    if batch_bn and len(ranges) >= 2 and (ranges[-1][1] - ranges[-1][0]) == 1:
        ranges = ranges[:-2] + [(ranges[-2][0], ranges[-1][1])]

    scores = []
    for a, b in ranges:
        xb = x[a:b].to(device)
        _, logits = model(xb)
        scores.append(torch.softmax(logits, dim=1).cpu().numpy())
    y_score = np.concatenate(scores, axis=0)
    return y_score.argmax(1), y_score
