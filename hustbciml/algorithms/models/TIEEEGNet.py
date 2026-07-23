# ===========================================================================
# TIEEEGNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Peng2022,
#     author  = {Peng, Ruimin and Zhao, Changming and Jiang, Jun and Kuang, Guangtao and Cui, Yuqi and Xu, Yifan and Du, Hao and Shao, Jianbo and Wu, Dongrui},
#     journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
#     title   = {{TIE}-{EEGN}et: Temporal Information Enhanced {EEGN}et for Seizure Subtype Classification},
#     year    = {2022},
#     pages   = {2567-2576},
#     volume  = {30},
#     doi     = {10.1109/TNSRE.2022.3204540},
#   }
# ===========================================================================
"""TIE-EEGNet — Temporal Information Enhanced EEGNet (Peng, ..., Wu, IEEE TNSRE 2022).

NOTE (domain fit): TIE-EEGNet was **developed for seizure (subtype) classification**,
not motor imagery — the paper is "TIE-EEGNet ... for Seizure Subtype Classification".
Its time-positional design targets seizure EEG and **may not be well-suited to MI**;
in this benchmark it only matches the EEGNet baseline on the MI datasets. It is kept
here for coverage of the lab's backbones, with this caveat flagged on the leaderboard
and in the algorithm card.

A lab backbone: standard EEGNet with exactly ONE layer changed — the first
temporal convolution becomes a "TIE-Conv2D" that adds a bounded, periodic
sinusoidal time-positional embedding to the ordinary convolution output. Every
other block is EEGNet verbatim, so it is a true drop-in Backbone whose output
shapes match EEGNet.

The TIE convolution computes (paper Eq. 6)

    Z[u, v] = (P * K)[u, v]  +  b · SE(t_v) · R[u, v]

where ``(P * K)`` is the ordinary temporal convolution, ``b`` is a single learned
scalar, ``R[u, v]`` is the *average* of the input over the same receptive field
as the convolution (paper Eq. 7 — average pooling with the temporal kernel's
window), and the sinusoidal positional term (paper Eq. 5) is

    SE(t_v) = sin(t_v / omega)  if t_v is even,   cos(t_v / omega)  if t_v is odd,

with the period regulator ``omega = sfreq / f_c``. It descends from EnK but swaps
EnK's linear encoding for a bounded sinusoid and EnK's sum for an average, fixing
the unboundedness and magnitude mismatch. It is a positional-encoding term, not
attention or recurrence.

Faithful-adaptation notes (disclosed in the card): (1) to keep a single-axis
network comparison, this uses the benchmark's EEGNet capacity (F1=4, D=2, F2=8,
dropout=0.25) shared with the other backbones; the paper's own config is
F1=8/F2=16/dropout=0.5, kept as the reference range. (2) The paper trains one
model per omega in a candidate set and keeps the lowest validation loss; here
``omega`` is a fixed hyperparameter (default 10.0, within the paper's candidate
set) so it drops into the shared per-backbone learning-rate tuner without a
nested omega sweep.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .EEGNet import EEGNet


class _TIEConv2d(nn.Module):
    """First temporal conv + additive sinusoidal time-positional embedding.

    Replaces EEGNet's ``ZeroPad2d`` + ``Conv2d(1, F1, (1, K))`` pair, doing its
    own same-padding so the output is ``(B, F1, C, T)`` — identical to the layer
    it stands in for."""

    def __init__(self, F1: int, kern_length: int, n_times: int, omega: float):
        super().__init__()
        K = int(kern_length)
        self.pad = nn.ZeroPad2d((K // 2 - 1, K - K // 2, 0, 0))
        self.conv = nn.Conv2d(1, F1, (1, K), stride=1, bias=False)
        self.avg = nn.AvgPool2d((1, K), stride=1)          # R: mean over the conv window
        self.b = nn.Parameter(torch.ones(1))               # learned scalar b (paper Eq. 6)
        # SE(t): sin(t/omega) at even t, cos(t/omega) at odd t (paper Eq. 5)
        t = torch.arange(n_times, dtype=torch.float32)
        se = torch.where(t.long() % 2 == 0, torch.sin(t / omega), torch.cos(t / omega))
        self.register_buffer("se", se.view(1, 1, 1, n_times))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = self.pad(x)                    # (B, 1, C, T+K-1)
        z = self.conv(xp)                   # (B, F1, C, T)
        r = self.avg(xp)                    # (B, 1,  C, T)
        return z + self.b * self.se * r     # broadcast over F1


class TIEEEGNet(EEGNet):
    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 F1: int = 4, D: int = 2, F2: int = 8,
                 kern_length: int = None, dropout: float = 0.25,
                 omega: float = 10.0, **_):
        super().__init__(n_chans=n_chans, n_times=n_times, n_classes=n_classes,
                         sfreq=sfreq, F1=F1, D=D, F2=F2,
                         kern_length=kern_length, dropout=dropout)
        self.omega = float(omega)
        # swap the first temporal conv (ZeroPad2d at [0] + Conv2d at [1]) for the
        # TIE convolution; keep the rest of block1 (BN, depthwise, ...) unchanged.
        self.block1[0] = _TIEConv2d(F1, self.kern_length, n_times, self.omega)
        self.block1[1] = nn.Identity()
