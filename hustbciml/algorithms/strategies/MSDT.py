# ===========================================================================
# MSDT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Zhang2022,
#     author  = {Zhang, Wen and Wang, Ziwei and Wu, Dongrui},
#     journal = {IEEE Trans. Neural Systems and Rehabilitation Engineering},
#     title   = {Multi-Source Decentralized Transfer for Privacy-Preserving {BCI}s},
#     year    = {2022},
#     pages   = {2710-2720},
#     volume  = {30},
#     doi     = {10.1109/TNSRE.2022.3207494},
#   }
# ===========================================================================
"""MSDT — Multi-Source Decentralized Transfer (Zhang, Wang & Wu, IEEE TNSRE 2022).

A lab multi-source, source-free, privacy-preserving transfer method. Each source
subject trains its own small MLP on Riemannian tangent-space features
(decentralized — no source data is pooled). At transfer time only the source
models are shared: the target adapts each source feature extractor (classifiers
frozen) by information maximization — minimize per-instance entropy, maximize
batch diversity — plus a source-inconsistency term that pulls the sources into
agreement, with entropy-based per-source domain weighting. The final prediction
is the domain-weighted ensemble of the source models.

Per the LOSO protocol, each of the other subjects is one source domain (the
user's "each subject as source"). Ported from the authors'
``MSDT/{source_train_multi_mi, target_adapt_msdt_mi}.py`` (use_weight on).
It maps onto hustbciml's fit/predict as:

* ``fit``  — for every source subject: 7x signal augmentation, oas-covariance
  tangent-space features, and train ``SourceMLP`` (SGD, label-smoothed CE,
  best-val). The source raw data is then discarded (source-free / decentralized).
* ``predict`` — tangent-map the target, adapt the source extractors by the
  multi-source IM + inconsistency objective, predict the domain-weighted ensemble.

``mode='fit'`` (no neural backbone; the pipeline's EEGNet/Linear are unused).
Faithful-adaptation notes (disclosed in the card): (1) the per-subject tangent
map is the only alignment MSDT uses, matching the source (no EA/RA stage), so
the preset aligner is Identity; (2) made device-agnostic (runs on CPU or CUDA);
(3) the source models are (re)fit inside ``fit`` for each target rather than
loaded from disk checkpoints — same models, folded into the framework;
(4) the upstream ``use_mix`` mixup is a no-op in the authors' code — its
``mixup_loss.backward()`` runs but the next line ``optimizer.zero_grad()`` wipes
those gradients before ``loss_all.backward(); optimizer.step()``, so mixup never
affected the published numbers. It is therefore omitted here (reproducing the
effective optimization), not "fixed", which would change the numbers.
Requires pyriemann + scikit-learn (imported lazily via the shared helpers).
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from ._lsft import tangent_features
from ._msdt import (SourceMLP, augment_signals, batch_entropy_loss, domain_weights,
                    instance_entropy_loss, source_inconsistency_loss)


def _poly_lr(optimizer, lr0: float, it: int, max_it: int, gamma: float = 10.0, power: float = 0.75):
    decay = (1 + gamma * it / max(1, max_it)) ** (-power)
    for g in optimizer.param_groups:
        g["lr"] = lr0 * decay


def _label_smooth_ce(logits: torch.Tensor, y: torch.Tensor, n_classes: int, eps: float = 0.1):
    onehot = torch.zeros_like(logits).scatter_(1, y.view(-1, 1), 1.0)
    smooth = (1 - eps) * onehot + eps / n_classes
    return -(smooth * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


class MSDT(Strategy):
    mode = "fit"
    # Each source subject's model depends only on that subject's own data, not on the
    # target, so under leave-one-subject-out it is identical across the targets it is a
    # source for. Train it once per (dataset, seed, subject) and reuse a deep copy per
    # target (the target-adaptation phase mutates its own copy). This makes the source
    # training O(N) instead of the authors' O(N^2) per-target refit — the same models,
    # far fewer trainings. The cache lives for the process (one seed/dataset per run).
    _cache: dict = {}

    def __init__(self, n_epoch_src: int = 100, n_epoch_tgt: int = 10, batch_size: int = 4,
                 bottleneck: int = 50, **_):
        self.n_epoch_src = n_epoch_src
        self.n_epoch_tgt = n_epoch_tgt
        self.batch_size = batch_size
        self.bottleneck = bottleneck
        self._models: List[SourceMLP] = []

    # ---------------------------------------------------------------- fit ------
    def fit(self, model, source: EEGEpochs, ctx: RunContext):
        device = ctx.device
        hp = ctx.cfg.hp
        self.n_epoch_src = int(hp.get("msdt_src_epochs", self.n_epoch_src))
        self.n_epoch_tgt = int(hp.get("msdt_tgt_epochs", self.n_epoch_tgt))
        self.batch_size = int(hp.get("msdt_batch", self.batch_size))
        self.bottleneck = int(hp.get("msdt_bottleneck", self.bottleneck))
        self._src_lr = float(hp.get("msdt_src_lr", 0.01))     # source-MLP SGD LR
        self._tgt_lr = float(hp.get("msdt_tgt_lr", 0.001))    # target-adaptation SGD LR
        self._incons = float(hp.get("msdt_incons", 0.1))      # source-inconsistency weight
        self._n_classes = source.n_classes
        self._models = []
        # cache key includes the source-training-affecting knobs so distinct grid
        # configs never reuse each other's trained source models within a process.
        sig = (self.n_epoch_src, self.batch_size, self.bottleneck, self._src_lr)
        for d in source.domains():                       # each source subject = one domain
            key = (ctx.cfg.dataset, int(ctx.cfg.seed), int(d), sig)
            cached = MSDT._cache.get(key)
            if cached is not None:                        # trained on an earlier target — reuse
                self._models.append(copy.deepcopy(cached).to(device))
                continue
            Xd = source.X[source.domain == d]
            yd = source.y[source.domain == d]
            Xa, ya = augment_signals(Xd, yd, source.sfreq)
            feat = tangent_features(Xa.astype(np.float64), cov_type="oas")
            net = self._train_source(feat, ya, feat.shape[1], device, ctx)
            MSDT._cache[key] = copy.deepcopy(net).cpu()   # clean, target-independent copy
            self._models.append(net)
        return model                                     # neural pipeline model unused

    def _train_source(self, feat: np.ndarray, y: np.ndarray, input_dim: int,
                      device, ctx: RunContext) -> SourceMLP:
        net = SourceMLP(input_dim, self._n_classes, self.bottleneck).to(device)
        X = torch.from_numpy(feat.astype(np.float32)).to(device)
        Y = torch.from_numpy(y.astype(np.int64)).to(device)
        n = len(Y)
        n_tr = max(1, int(0.9 * n))
        opt = torch.optim.SGD(net.parameters(), lr=self._src_lr, momentum=0.9,
                              weight_decay=1e-3, nesterov=True)
        bs = self.batch_size
        iters_per = max(1, n_tr // bs)
        max_it = self.n_epoch_src * iters_per
        best_acc, best_state = -1.0, None
        rng = np.random.RandomState(ctx.cfg.seed)
        it = 0
        for epoch in range(self.n_epoch_src):
            net.train()
            perm = rng.permutation(n_tr)
            for s in range(0, n_tr - bs + 1, bs):
                idx = perm[s:s + bs]
                it += 1
                _poly_lr(opt, self._src_lr, it, max_it)
                logits = net(X[idx])
                loss = _label_smooth_ce(logits, Y[idx], self._n_classes)
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                acc = (net(X[n_tr:]).argmax(1) == Y[n_tr:]).float().mean().item() if n > n_tr else 0.0
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    # ------------------------------------------------------------- predict -----
    def predict(self, model, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        device = ctx.device
        Xt_np = tangent_features(target.X.astype(np.float64), cov_type="oas")
        Xt = torch.from_numpy(Xt_np.astype(np.float32)).to(device)

        # adapt feature extractors (netF); freeze classifiers (netC)
        params = []
        for m in self._models:
            m.train()
            for p in m.netF.parameters():
                params.append(p)
            for p in m.netC.parameters():
                p.requires_grad_(False)
        opt = torch.optim.SGD(params, lr=self._tgt_lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

        n = len(Xt)
        bs = self.batch_size
        iters_per = max(1, n // bs)
        for epoch in range(self.n_epoch_tgt):
            perm = torch.randperm(n, device=device)
            it = 0
            for s in range(0, n - bs + 1, bs):
                idx = perm[s:s + bs]
                if len(idx) <= 1:
                    continue
                it += 1
                _poly_lr(opt, self._tgt_lr, it, iters_per)
                xb = Xt[idx]
                logits = torch.stack([m(xb) for m in self._models], dim=1)   # (B, S, C)
                loss = (instance_entropy_loss(logits) + batch_entropy_loss(logits)
                        + self._incons * source_inconsistency_loss(logits))
                # entropy-based domain weights enter only at the weighted-ensemble
                # prediction below; the upstream mixup term is a no-op (module docstring).
                opt.zero_grad()
                loss.backward()
                opt.step()

        return self._predict_ensemble(Xt)

    @torch.no_grad()
    def _predict_ensemble(self, Xt: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        for m in self._models:
            m.eval()
        w = domain_weights(self._models, Xt).detach()
        logits = torch.stack([m(Xt) for m in self._models], dim=1)       # (N, S, C)
        y_score = (torch.softmax(logits, dim=2) * w).sum(dim=1)          # (N, C) probabilities
        return y_score.argmax(1).cpu().numpy().astype(np.int64), y_score.cpu().numpy()
