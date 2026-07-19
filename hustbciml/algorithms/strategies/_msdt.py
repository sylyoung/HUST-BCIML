# ===========================================================================
# _msdt.py  —  HUST-BCIML EEG-decoding benchmark
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
"""MSDT helpers — the per-source tangent-space MLP, the multi-source
information-maximization losses, entropy-based domain weighting, and the
source-signal augmentation. Vendored and made device-agnostic (no hardcoded
``.cuda()``) from the authors' ``MSDT/utils/{network,loss,data_augment}.py``.

MSDT operates on classical Riemannian tangent-space features, not raw EEG: each
source subject trains its own small MLP (``Net_ln2`` feature extractor + a
weight-normalized classifier), and the target adapts the feature extractors by
information maximization with a source-inconsistency term. See ``MSDT.py``.

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
    """One source model: ``Net_ln2`` feature extractor (netF) + weight-normalized
    linear classifier (netC). Exposes ``netF``/``netC`` so the target adaptation
    can adapt the extractor while freezing the classifier."""

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
    """Mean per-instance prediction entropy over the (B, S, C) stack — minimizing
    it sharpens each source's target predictions (the IM confidence term)."""
    ent = -(F.softmax(logits, dim=2) * F.log_softmax(logits, dim=2)).sum(dim=2)
    return ent.mean(dim=0).mean()


def batch_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Negative mean batch-marginal entropy — minimizing it maximizes class
    diversity across the batch (the IM diversity term); prevents collapse."""
    p = F.softmax(logits, dim=2).mean(dim=0)          # (S, C) class marginals per source
    neg_ent = -(-(p * p.log()).sum(dim=1))            # = -H(marginal) per source
    return neg_ent.mean()


def source_inconsistency_loss(logits: torch.Tensor) -> torch.Tensor:
    """Mean std across the source models — minimizing it pulls the source
    predictions into agreement on the target (Zhang & Wu 2022, weighted 0.1)."""
    return logits.std(dim=1).mean(dim=1).mean(dim=0)


def domain_weights(models: List[SourceMLP], Xt: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-source weights from an IM score on the whole target (recomputed each
    epoch); returns a (1, S, 1) tensor for broadcasting over (B, S, C)."""
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
    """MSDT's 7x source-signal augmentation on ``(N, C, T)`` trials: raw +
    multiplicative (x2) + additive-noise + negate-shift + frequency-shift (x2).
    Faithful to ``data_aug([mult, noise, neg, freq])``; ``dt = 1/sfreq``."""
    d = np.transpose(X, (0, 2, 1)).astype(np.float64)     # (N, T, C), matching the source layout
    dt = 1.0 / float(sfreq)
    blocks = [d]                                          # raw
    labels = [y]
    blocks += [d * 1.05, d * 0.95]                        # multiplicative
    labels += [y, y]
    noise = np.stack([d[i] + (np.random.rand(*d[i].shape) - 0.5) * np.std(d[i]) / 2.0
                      for i in range(len(d))], axis=0)    # additive noise
    blocks.append(noise); labels.append(y)
    neg = np.stack([(-d[i]) - np.min(-d[i]) for i in range(len(d))], axis=0)   # negate + shift
    blocks.append(neg); labels.append(y)
    blocks += [np.stack([_freq_shift(d[i], -0.2, dt) for i in range(len(d))], axis=0),
               np.stack([_freq_shift(d[i], 0.2, dt) for i in range(len(d))], axis=0)]
    labels += [y, y]

    X_aug = np.concatenate(blocks, axis=0)
    X_aug = np.transpose(X_aug, (0, 2, 1))                # back to (N, C, T)
    return np.ascontiguousarray(X_aug, dtype=np.float32), np.concatenate(labels, axis=0)
