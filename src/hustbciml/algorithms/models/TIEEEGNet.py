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
"""TIE-EEGNet (Peng et al., 2022, IEEE TNSRE) — Temporal Information Enhanced
EEGNet for cross-patient seizure subtype classification.

The paper targets EEG-based seizure subtype classification (Sec. I): given a
short EEG segment, predict which subtype of epileptic seizure it belongs to,
across patients (train and test come from different patients). Standard EEGNet's
square temporal-convolution filters have small receptive fields and are weak at
capturing the temporal structure of seizure EEG, in particular the repeated
spike-and-wave discharges (SWDs, Fig. 1). To address this, the paper augments
EEGNet with a Temporal Information Enhancement (TIE) module that adds a
sinusoidal time-positional embedding to the feature maps of EEGNet's FIRST
convolution layer, turning it into a "TIE-Conv2D" layer (Sec. III-D, Fig. 4);
all other EEGNet blocks are unchanged. The method is supervised (trained with
cross-entropy). It was evaluated on the public TUSZ dataset and the authors' own
CHSZ (infant/child) dataset, and on a source-free transfer scenario from TUSZ to
CHSZ (Sec. IV).

This benchmark uses TIE-EEGNet as a backbone. Because the TIE module was designed
for seizure EEG rather than motor imagery, its periodic time-positional encoding
need not help on the MI datasets used here (the paper itself lists seizure
detection/classification and sleep staging — not MI — as its intended temporal
tasks, Sec. V); the algorithm card flags this domain caveat.

The TIE module descends from the Encoding Kernel (EnK) of Singh & Lin, 2020
(paper ref. [44]), which embeds a LINEAR time encoding into the convolution
(paper Eq. 3). TIE keeps EnK's idea of adding a time-dependent term to the
convolution output but replaces the linear encoding with a BOUNDED, PERIODIC
sinusoidal encoding, which the paper motivates by three requirements on the
positional term (Sec. III-C): it should be bounded, periodic (to simulate the
repeated SWDs), and preserve the intra-period time-sequence (wave-shape)
information. The TIE convolution is (paper Eq. 6)

    Z[u, v] = P[u, v] * K  +  b · SE(t_v) · R[u, v]

where ``P[u, v] * K`` is the ordinary (dot-product) temporal convolution at time
position ``t_v = v``, ``b`` is a single trainable scalar controlling the
embedding's intensity, and the sinusoidal encoding SE (paper Eq. 5) is

    SE(t_v) = sin(t_v / omega)  if t_v is even,   cos(t_v / omega)  if t_v is odd,

with ``omega`` the period regulator. ``R[u, v]`` is the "representation matrix"
(paper Eq. 7): the AVERAGE (via average pooling) of the input over the same
receptive field as the convolution, representing the local background of
``P[u, v]``. Using the average — rather than EnK's SUM — keeps the embedding at
the same magnitude as the convolved input (Sec. III-C). SE is a fixed positional
term, not attention or recurrence.

Notes on this benchmark's adaptation. (1) Capacity: this file uses the
benchmark's shared EEGNet configuration (F1=4, D=2, F2=8, dropout=0.25) so all
backbones differ only in architecture; these specific capacity hyperparameters
are a benchmark choice and are not fixed by the paper. (2) Period regulator: the
paper does not fix a single ``omega``. It derives candidates ``omega = f_s / f_c``
from the EEG-band frequency bounds f_c in {0.5, 4, 8, 12, 16, 32, 64} Hz
(paper Eqs. 8-9), trains one model per candidate in parallel, and keeps the one
with the lowest validation loss (Fig. 4, Fig. 8). Here ``omega`` is instead a
single fixed hyperparameter (default 10.0, near a paper candidate) so the model
plugs into the shared per-backbone tuner without a nested ``omega`` sweep.
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
        self.avg = nn.AvgPool2d((1, K), stride=1)          # R (Eq. 7): mean over the conv window
        self.b = nn.Parameter(torch.ones(1))               # trainable scalar b (paper Eq. 6)
        # SE(t): sin(t/omega) at even t, cos(t/omega) at odd t (paper Eq. 5)
        t = torch.arange(n_times, dtype=torch.float32)
        se = torch.where(t.long() % 2 == 0, torch.sin(t / omega), torch.cos(t / omega))
        self.register_buffer("se", se.view(1, 1, 1, n_times))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = self.pad(x)                    # (B, 1, C, T+K-1)
        z = self.conv(xp)                   # (B, F1, C, T)
        r = self.avg(xp)                    # (B, 1,  C, T)
        return z + self.b * self.se * r     # Eq. 6: P*K + b·SE(t)·R, broadcast over F1


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
