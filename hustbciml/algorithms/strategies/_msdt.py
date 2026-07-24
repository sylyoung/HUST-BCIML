# ===========================================================================
# _msdt.py  —  HUST-BCIML EEG-decoding benchmark
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
"""MSDT helpers (Zhang et al., 2022, IEEE TNSRE) -- the per-source tangent-space
MLP, the multi-source information-maximization and source-consistency losses, the
source-transferability weighting (Eq. 8), and the source-signal augmentation
(Table I). Vendored and made device-agnostic (no hardcoded ``.cuda()``) from the
authors' ``MSDT/utils/{network,loss,data_augment}.py``.

MSDT operates on hand-crafted Riemannian tangent-space features, not raw EEG:
each source subject trains its own small MLP -- a two-layer feature extractor
(Table III) plus a weight-normalized classifier -- and the target adapts the
feature extractors by information maximization (Eq. 5-6) with a source-consistency
regularizer (Eq. 7). See ``MSDT.py`` for how these compose (gray-box MSDT-G).

Underscore-prefixed so the registry auto-scan skips it (a helper, not a plug-in).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert
from torch.nn.utils import weight_norm


# ------------------------------------------------------------- network ---------
def _init_linear(m: nn.Module) -> None:
    if m.__class__.__name__.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class SourceMLP(nn.Module):
    """One source model theta_m = (g_m . f_m), Table III: the feature extractor g_m
    (netF) is two fully-connected layers, each with layer normalization and ReLU;
    the classifier f_m (netC) is a weight-normalized linear layer. Exposes
    ``netF``/``netC`` so the gray-box adaptation can adapt g_m while freezing f_m."""

    def __init__(self, input_dim: int, n_classes: int, bottleneck: int = 50):
        super().__init__()
        n_hidden = 100 if bottleneck == 50 else 128
        self.netF = nn.Sequential(
            nn.Linear(input_dim, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, bottleneck), nn.LayerNorm(bottleneck), nn.ReLU(),
        )
        self.netF.apply(_init_linear)
        fc = weight_norm(nn.Linear(bottleneck, n_classes), name="weight")
        _init_linear(fc)
        self.netC = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.netC(self.netF(x))


# ------------------------------------------------------------- losses ----------
def _entropy(p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return -torch.sum(p * torch.log(p + eps), dim=1)


def instance_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Information-maximization conditional-entropy term (Eq. 5-6): mean
    per-instance prediction entropy H(Y|X) over the (B, S, C) stack. Minimizing it
    sharpens each source's target predictions toward one-hot."""
    ent = -(F.softmax(logits, dim=2) * F.log_softmax(logits, dim=2)).sum(dim=2)
    return ent.mean(dim=0).mean()


def batch_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Information-maximization marginal-entropy term (Eq. 6): the negative
    marginal entropy of the batch class distribution per source. Minimizing it
    maximizes H(marginal), keeping predictions class-balanced (prevents collapse)."""
    p = F.softmax(logits, dim=2).mean(dim=0)          # (S, C) class marginals per source
    neg_ent = -(-(p * p.log()).sum(dim=1))            # = -H(marginal) per source
    return neg_ent.mean()


def source_inconsistency_loss(logits: torch.Tensor) -> torch.Tensor:
    """Source-consistency regularization L_sc (Eq. 7): the spread (std) of the
    per-class predictions across the source models. Minimizing it pulls the M
    adapted models into agreement on each target trial (Zhang et al., 2022)."""
    return logits.std(dim=1).mean(dim=1).mean(dim=0)


def domain_weights(models: List[SourceMLP], Xt: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Source-transferability estimation (Eq. 8): the per-source weights alpha_m
    from an information/mutual-information score on the whole target (conditional
    minus marginal entropy per source, recomputed each epoch), normalized to sum
    to 1. Returns a (1, S, 1) tensor for broadcasting over (B, S, C)."""
    scores = []
    for m in models:
        soft = F.softmax(m(Xt), dim=1)
        im = _entropy(soft).mean()
        marg = soft.mean(dim=0)
        gentropy = torch.sum(-marg * torch.log(marg + eps))
        scores.append(im - gentropy)
    w = torch.stack(scores)
    w = w / w.sum()
    return w.reshape(1, len(models), 1)


# ---------------------------------------------------- source signal augment ----
def _freq_shift(x: np.ndarray, f_shift: float, dt: float) -> np.ndarray:
    """Analytic-signal frequency shift of one trial ``(T, C)`` by ``f_shift`` Hz."""
    T, C = x.shape
    pad_len = int(2 ** np.ceil(np.log2(abs(T))))
    padded = np.vstack((x, np.zeros((pad_len - T, C))))
    analytic = hilbert(padded, axis=0)
    t = np.arange(pad_len)
    shift = np.exp(2j * np.pi * f_shift * dt * t)
    out = np.zeros_like(x)
    for i in range(C):
        out[:, i] = (analytic[:, i] * shift)[:T].real
    return out


def augment_signals(X: np.ndarray, y: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """MSDT's 7x source-signal augmentation on ``(N, C, T)`` trials (Table I,
    Sec. III-A1): raw + data scaling (x(1 +/- C_mult), C_mult=0.05, x2) + noise
    injection (X + rand*std(X)/C_noise, C_noise=2) + data flipping (max(X) - X) +
    frequency shift (Hilbert-transform shift by +/- C_freq Hz, C_freq=0.2, x2).
    Faithful to ``data_aug([mult, noise, neg, freq])``; ``dt = 1/sfreq``."""
    d = np.transpose(X, (0, 2, 1)).astype(np.float64)     # (N, T, C), matching the source layout
    dt = 1.0 / float(sfreq)
    blocks = [d]                                          # raw
    labels = [y]
    blocks += [d * 1.05, d * 0.95]                        # data scaling, x(1 +/- 0.05)
    labels += [y, y]
    noise = np.stack([d[i] + (np.random.rand(*d[i].shape) - 0.5) * np.std(d[i]) / 2.0
                      for i in range(len(d))], axis=0)    # noise injection
    blocks.append(noise); labels.append(y)
    neg = np.stack([(-d[i]) - np.min(-d[i]) for i in range(len(d))], axis=0)   # data flipping, max(X) - X
    blocks.append(neg); labels.append(y)
    blocks += [np.stack([_freq_shift(d[i], -0.2, dt) for i in range(len(d))], axis=0),
               np.stack([_freq_shift(d[i], 0.2, dt) for i in range(len(d))], axis=0)]
    labels += [y, y]

    X_aug = np.concatenate(blocks, axis=0)
    X_aug = np.transpose(X_aug, (0, 2, 1))                # back to (N, C, T)
    return np.ascontiguousarray(X_aug, dtype=np.float32), np.concatenate(labels, axis=0)
