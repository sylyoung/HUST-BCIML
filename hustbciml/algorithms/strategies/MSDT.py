# ===========================================================================
# MSDT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Zhang2022,
#     author  = {Zhang, Wen and Wang, Ziwei and Wu, Dongrui},
#     journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
#     title   = {Multi-Source Decentralized Transfer for Privacy-Preserving {BCI}s},
#     year    = {2022},
#     pages   = {2710-2720},
#     volume  = {30},
#     doi     = {10.1109/TNSRE.2022.3207494},
#   }
# ===========================================================================
"""MSDT — Multi-Source Decentralized Transfer (Zhang et al., 2022, IEEE TNSRE).

Decentralized, privacy-preserving cross-subject transfer for EEG-based BCIs.
There are M source subjects, whose data and computations stay local; only the
pre-trained source models are shared, so no source EEG is ever centralized
(Sec. I, Sec. III, Fig. 1). MSDT runs in two stages. (1) Source model
pre-training (Sec. III-A): each subject trains its own deep model theta_m =
(g_m feature extractor . f_m classifier) locally on hand-crafted features --
Riemannian tangent-space vectors for motor imagery (Sec. II-A) -- after optional
signal augmentation (Table I), with a label-smoothed cross-entropy loss (Eq. 3).
(2) Decentralized transfer to an unlabeled target subject, in two settings: a
gray-box one (MSDT-G, Sec. III-B) that uses the source-model PARAMETERS, and a
black-box one (MSDT-B, Sec. III-C) that only queries the source models as APIs
and distills them into a single student (Eq. 11-13). The paper reports MSDT on
motor imagery (BNCI IV-2a) and affective BCI (SEED).

This file implements the GRAY-BOX variant (MSDT-G). Per source it builds a
target model theta'_m = (g'_m . f_m): the feature extractor g'_m is initialized
from g_m and adapted, while the classifier f_m is kept FROZEN (hypothesis
transfer). The extractors are adapted on the unlabeled target by three terms
(Sec. III-B): information maximization (IM, Eq. 5-6) -- minimize each source's
conditional entropy H(Y|X) to sharpen predictions and maximize the marginal
entropy so predictions stay class-balanced; source-consistency regularization
(Eq. 7) -- minimize the spread of the per-class probabilities across the M source
models so they agree on each target trial; and mixup regularization (Eq. 9). The
target prediction is a transferability-weighted ensemble of the adapted models,
theta_t(X_t) = sum_m alpha_m theta'_m(X_t) with sum_m alpha_m = 1 (Eq. 4); the
per-source weights alpha_m are the source-transferability estimates of Eq. (8),
so more transferable sources count more and negative transfer is suppressed.

Under leave-one-subject-out, each of the other subjects is one source domain, and
this file maps onto hustbciml's fit/predict as:

* ``fit``  -- source model pre-training (Sec. III-A). For every source subject:
  the 7x signal augmentation of Table I, oas-shrinkage tangent-space features,
  then train ``SourceMLP`` (SGD, label-smoothed CE Eq. 3, best-validation model).
  The source raw data is then dropped (decentralized / source-free).
* ``predict`` -- decentralized transfer (Sec. III-B). Tangent-map the target,
  adapt the source feature extractors by IM (Eq. 5-6) + source consistency
  (Eq. 7), then predict the transferability-weighted ensemble (Eq. 4, 8).

``mode='fit'`` (no neural backbone; the pipeline's EEGNet/Linear are unused).
Ported from the authors' ``MSDT/{source_train_multi_mi, target_adapt_msdt_mi}.py``
(gray-box, source-transferability weighting on). Faithful-adaptation notes,
disclosed in the model card: (1) the per-subject tangent map is the only
alignment MSDT uses, matching the paper (no EA/RA stage), so the preset aligner
is Identity; (2) made device-agnostic (runs on CPU or CUDA); (3) the source
models are (re)fit inside ``fit`` for each target rather than loaded from disk
checkpoints -- the same models, folded into the framework; (4) the paper's
gray-box objective is L_all = L_im + beta*L_sc + gamma*L_mix (Eq. 10, beta=0.1,
gamma=1); this port optimizes L_im + beta*L_sc (with beta the ``_incons`` weight,
0.1) and omits the mixup term L_mix, because in the authors' code the mixup
gradients are erased by an ``optimizer.zero_grad()`` before the backward/step of
``loss_all``, so mixup never affected the published numbers -- it is omitted to
reproduce the effective optimization, not "fixed", which would change them.
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
        self._incons = float(hp.get("msdt_incons", 0.1))      # beta: source-consistency weight (Eq. 7, 10)
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

        # adapt each source feature extractor g'_m (netF); freeze the classifier
        # f_m (netC) for hypothesis transfer (Sec. III-B)
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
                # information maximization (Eq. 5-6: conditional-entropy + marginal-
                # entropy terms) + source-consistency regularization L_sc (Eq. 7,
                # weighted by beta = _incons). Mixup L_mix (Eq. 9) is omitted (see
                # module docstring); the source-transferability weights alpha_m
                # (Eq. 8) enter only at the weighted-ensemble prediction below.
                loss = (instance_entropy_loss(logits) + batch_entropy_loss(logits)
                        + self._incons * source_inconsistency_loss(logits))
                opt.zero_grad()
                loss.backward()
                opt.step()

        return self._predict_ensemble(Xt)

    @torch.no_grad()
    def _predict_ensemble(self, Xt: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        for m in self._models:
            m.eval()
        # transferability-weighted ensemble theta_t = sum_m alpha_m theta'_m (Eq. 4),
        # with the per-source weights alpha_m from source-transferability estimation
        # (Eq. 8), normalized to sum to 1 over the S sources.
        w = domain_weights(self._models, Xt).detach()
        logits = torch.stack([m(Xt) for m in self._models], dim=1)       # (N, S, C)
        y_score = (torch.softmax(logits, dim=2) * w).sum(dim=1)          # (N, C) probabilities
        return y_score.argmax(1).cpu().numpy().astype(np.int64), y_score.cpu().numpy()
