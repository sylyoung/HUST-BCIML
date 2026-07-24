# ===========================================================================
# _mvcnet.py  —  HUST-BCIML EEG-decoding benchmark
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
# (The IFNet backbone is documented and cited in models/IFNet.py.)
# ===========================================================================
"""Building blocks for the MVCNet strategy (Wang et al., 2025, Knowledge-Based
Systems) — the SimCLR NT-Xent contrastive loss (paper Eq. 1/3), the three
representative augmented views used here (Flip / FShift / Channel Reflection, one
per time/frequency/space domain of Table 2), and the auxiliary transformer-encoder
+ projector that realize the paper's Transformer branch (Sec. 3.2). Prefixed ``_``
so the registry skips it.

Original authors' code: github.com/wzwvv/MVCNet (utils/contrastive_loss.py,
utils/data_augment.py, utils/network.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import UNLABELED
from hustbciml.utils.montage import reflection_permutation


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy (SimCLR). Contrasts two
    ``(N, d)`` batches where row i of one is the positive of row i of the other."""

    def __init__(self, device, batch_size: int, temperature: float, use_cosine: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.register_buffer("mask", self._mask(batch_size))
        self._cos = nn.CosineSimilarity(dim=-1)
        self.use_cosine = use_cosine
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    @staticmethod
    def _mask(bs):
        n = 2 * bs
        m = np.eye(n) + np.eye(n, k=-bs) + np.eye(n, k=bs)
        return torch.from_numpy((1 - m).astype(bool))

    def _sim(self, x, y):
        if self.use_cosine:
            return self._cos(x.unsqueeze(1), y.unsqueeze(0))
        return torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)

    def forward(self, zis, zjs):
        reps = torch.cat([zjs, zis], dim=0)
        sim = self._sim(reps, reps)
        l_pos = torch.diag(sim, self.batch_size)
        r_pos = torch.diag(sim, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = sim[self.mask.to(sim.device)].view(2 * self.batch_size, -1)
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(2 * self.batch_size, device=self.device).long()
        return self.criterion(logits, labels) / (2 * self.batch_size)


# ----------------------------- the three views -----------------------------
def flip_view(x: torch.Tensor) -> torch.Tensor:
    """Time-domain view: amplitude negation (paper's Flip augmentation, Table 2)."""
    return -x


def freqshift_view(x: torch.Tensor, sfreq: float, f_shift: float = 0.1) -> torch.Tensor:
    """Frequency-domain view: Hilbert-transform frequency shift by ``f_shift`` Hz
    (paper's FShift augmentation, Table 2)."""
    from scipy.signal import hilbert
    xb = x.squeeze(1).detach().cpu().numpy().astype(np.float64)     # (B, C, T)
    B, C, T = xb.shape
    dt = 1.0 / float(sfreq)
    padlen = 2 ** int(np.ceil(np.log2(T)))
    shift = np.exp(2j * np.pi * f_shift * dt * np.arange(padlen))   # (padlen,)
    out = np.zeros_like(xb)
    for b in range(B):
        padded = np.concatenate([xb[b], np.zeros((C, padlen - T))], axis=1)
        analytic = hilbert(padded, axis=1)
        out[b] = (analytic * shift[None, :]).real[:, :T]
    return torch.from_numpy(out.astype(np.float32)).unsqueeze(1).to(x.device)


def reflect_view(x: torch.Tensor, y: torch.Tensor, perm, n_classes: int):
    """Space-domain view: Channel Reflection — left/right hemisphere channel swap
    with the 2-class label swap (paper's CR augmentation, Table 2; Wang et al., 2024)."""
    C = x.shape[2]
    perm = torch.arange(C - 1, -1, -1, device=x.device) if perm is None else perm.to(x.device)
    xr = x[:, :, perm, :]
    yr = y.clone()
    if n_classes == 2:
        known = yr != UNLABELED
        yr[known] = 1 - yr[known]
    return xr, yr


def make_reflection_perm(ch_names):
    perm = reflection_permutation(list(ch_names) if ch_names else [])
    return None if len(perm) == 0 else torch.from_numpy(perm).long()


# --------------------------- auxiliary modules ------------------------------
def build_encoder(dim_e: int, nlayer: int = 1) -> nn.Module:
    """Transformer branch (paper Sec. 3.2): a self-attention encoder. Here channels
    are the tokens and ``dim_e`` (= n_times) is the model dimension. ``nhead``
    divides ``dim_e`` (the source notes head count barely matters; the paper's
    ablation, Fig. 9, finds 2 layers / 2 heads best)."""
    nhead = 2 if dim_e % 2 == 0 else 1
    layer = nn.TransformerEncoderLayer(
        d_model=dim_e, nhead=nhead, dim_feedforward=max(2 * dim_e, 128),
        dropout=0.1, batch_first=True)
    return nn.TransformerEncoder(layer, num_layers=nlayer)


def build_projector(dim_p: int, dim1: int, dim2: int) -> nn.Module:
    """Projector head (source netP): dim_p -> dim1 -> dim2."""
    return nn.Sequential(
        nn.Linear(dim_p, dim1), nn.LayerNorm(dim1), nn.ReLU(inplace=True),
        nn.Linear(dim1, dim2), nn.Tanh())
