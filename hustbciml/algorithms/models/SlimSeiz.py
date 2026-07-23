# ===========================================================================
# SlimSeiz.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/guoruilu/SlimSeiz
#
# Reference (IEEE BibTeX):
#   @InProceedings{Lu2025,
#     author    = {Lu, Guorui and Peng, Jing and Huang, Bingyuan and Gao, Chang and Stefanov, Todor and Hao, Yong and Chen, Qinyu},
#     booktitle = {Proceedings of the IEEE International Symposium on Circuits and Systems},
#     title     = {{SlimSeiz}: Efficient Channel-Adaptive Seizure Prediction Using a Mamba-Enhanced Network},
#     year      = {2025},
#     pages     = {1-5},
#     doi       = {10.1109/ISCAS56072.2025.11043364},
#   }
# ===========================================================================
"""SlimSeiz (Lu et al., 2025) is an efficient channel-adaptive seizure-prediction
network that pairs a lightweight multi-branch 1D convolutional feature extractor
with a single Mamba selective-state-space mixer.

Architecture, mapped to the paper (Section 3, Efficient Network Architecture):

  * A first temporal conv block (kernel 21) with ReLU and MaxPool by 8 turns the
    multi-channel signal into 16 feature maps at a reduced time resolution. This
    is the paper's initial convolutional stage that captures short EEG patterns.
  * Two inception-style residual conv blocks. Each sums a 1x1 pointwise branch
    with a stacked multi-kernel branch. The first pair keeps 16 channels and is
    followed by MaxPool by 4. The second pair widens to 32 channels with stride 2.
    These are the paper's parallel multi-kernel convolution blocks that extract
    features at several temporal scales while staying parameter-cheap.
  * A single Mamba block (the paper's Mamba-enhanced temporal module, Section 3.3)
    mixes information along the time axis with a selective state-space scan,
    wrapped as ``x + Mamba(RMSNorm(x))`` in the style of the Mamba language-model
    layer. The Mamba block itself follows Figure 3 of the Mamba paper (Gu and Dao,
    2024): an input projection that gates two paths, a depthwise causal Conv1d,
    a SiLU nonlinearity, the input-dependent selective scan (Algorithm 2 of the
    Mamba paper), a SiLU gate from the second projected path, and an output
    projection.
  * Adaptive average pooling over time collapses the sequence to one 32-dim
    vector per trial, which the paper's final linear layer maps to the two
    seizure / non-seizure classes.

This ports the paper's default 1D configuration. The final classification Linear
(32 -> 2) is removed so the pre-logit 32-dim vector is exposed as the feature.
``out_features`` is 32 and is confirmed by a dummy forward in ``__init__`` so the
backbone works for any (C, T). The reference fixes the number of input channels
at 3 for its patient-specific setting. Here the input channel count is taken from
``n_chans`` so any montage works.

Deviations, all behaviour-preserving: the ``einops`` dependency is dropped
(``rearrange`` / ``repeat`` / ``einsum`` rewritten with plain torch ops), and the
Mamba selective scan uses the pure-PyTorch sequential loop from mamba-minimal
(no ``mamba_ssm`` package, no ``selective_scan_cuda`` kernel), so the forward runs
on CPU. Source: github.com/guoruilu/SlimSeiz (mamba-minimal block adapted from
github.com/johnma2006/mamba-minimal).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.core.stages import Backbone


class _MambaBlock(nn.Module):
    """A single Mamba block (Figure 3, Section 3.4 of the Mamba paper).

    Input and output are (B, L, D). D equals the number of feature channels fed
    in. This is the pure-PyTorch mamba-minimal variant: the selective scan is the
    sequential recurrence written in plain torch, not a fused CUDA kernel.
    """

    def __init__(self, input_channels: int):
        super().__init__()
        self.d_model = input_channels
        self.d_inner = self.d_model * 2
        self.dt_rank = math.ceil(self.d_model / 16)
        self.d_state = 16

        # Input projection: produces the two gated paths x and res, each d_inner.
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)

        # Depthwise causal Conv1d over time (padding 2, kernel 3, groups = d_inner).
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            groups=self.d_inner,
            padding=2,
        )

        # x_proj maps x to the input-dependent step size delta and the B, C matrices.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        # dt_proj lifts delta from dt_rank up to d_inner.
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is the input-independent state-transition matrix, stored in log space.
        # repeat(arange(1, d_state+1), 'n -> d n', d=d_inner) rewritten in plain torch.
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(self.d_inner, 1)          # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))     # skip connection term
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, L, D)
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)                          # (B, L, 2 * d_inner)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # Depthwise causal conv over the time axis, then crop to length L.
        x = x.transpose(1, 2)                                # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)                                # (B, L, d_inner)

        x = F.silu(x)

        y = self.ssm(x)                                      # selective state-space mixing

        y = y * F.silu(res)                                  # gate by the second path

        return self.out_proj(y)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective state-space model (Algorithm 2, Section 3.2 of the Mamba paper).

        A and D are input-independent. delta, B and C are produced from the input,
        which is what makes the state space selective.
        """
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())                   # (d_inner, d_state)
        D = self.D.float()

        x_dbl = self.x_proj(x)                               # (B, L, dt_rank + 2*d_state)
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))              # (B, L, d_inner)

        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        """Sequential selective scan (pure PyTorch, mamba-minimal).

        Runs the discrete state-space recurrence
            state(t) = deltaA(t) * state(t-1) + deltaB_u(t)
            y(t)     = C(t) . state(t)
        as an explicit loop over time. This mirrors the official selective scan
        without any fused CUDA kernel, so it runs on CPU.

        Shapes: u, delta (B, L, d_inner); A (d_inner, d_state); B, C (B, L, d_state);
        D (d_inner,). Returns (B, L, d_inner).
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize A (zero-order hold) and B (simplified Euler), einsum-free.
        #   deltaA  = exp(delta[...,None] * A)                 -> (B, L, d_inner, d_state)
        #   deltaB_u = (delta * u)[...,None] * B[:, :, None, :] -> (B, L, d_inner, d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = (delta * u).unsqueeze(-1) * B.unsqueeze(2)

        # Sequential scan over time. The official kernel does a parallel, hardware
        # aware scan; this plain loop is the pure-PyTorch equivalent.
        x = torch.zeros((b, d_in, n), device=deltaA.device, dtype=deltaA.dtype)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]            # (B, d_inner, d_state)
            # y(t) = sum_n state * C(t): einsum('b d n, b n -> b d') by broadcasting.
            y = (x * C[:, i, :].unsqueeze(1)).sum(dim=-1)    # (B, d_inner)
            ys.append(y)
        y = torch.stack(ys, dim=1)                           # (B, L, d_inner)

        return y + u * D                                     # add the skip term


class _RMSNorm(nn.Module):
    """Root-mean-square layer norm, as used around the Mamba mixer."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SlimSeiz(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float, **_):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times

        # Initial temporal conv stage: kernel 21, ReLU, MaxPool by 8.
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=n_chans, out_channels=16, kernel_size=21, stride=1, padding=10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
        )
        # First inception-style residual block (16 channels), 1x1 branch ...
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        # ... plus a stacked multi-kernel branch (11 then 3).
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        # Second inception-style residual block, widening to 32 channels, stride 2.
        self.conv4_1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=2),
            nn.ReLU(),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Mamba-enhanced temporal module: RMSNorm + Mamba mixer with a residual.
        self.mixer = _MambaBlock(32)
        self.norm = _RMSNorm(32)

        # Collapse the time axis to a single 32-dim descriptor per trial.
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # Infer the feature width via a dummy forward so the backbone is
        # dataset-agnostic. The paper's final Linear(32, 2) head is removed.
        with torch.no_grad():
            feat = self.forward_features(torch.zeros(1, 1, n_chans, n_times))
        self.out_features = feat.shape[1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, C, T)
        x = x.squeeze(1)                                     # (B, C, T)
        x = self.conv1(x)
        x = self.pool3(self.conv2_1(x) + self.conv2_2(x))    # first residual block + pool
        x = self.conv4_1(x) + self.conv4_2(x)                # second residual block
        x = x.permute(0, 2, 1)                               # (B, seq_len, 32)
        x = self.mixer(self.norm(x)) + x                     # Mamba mixer with residual
        x = x.permute(0, 2, 1)                               # (B, 32, seq_len)
        x = self.adaptive_avg_pool(x)                        # (B, 32, 1)
        return x.contiguous().view(x.size(0), -1)            # (B, 32)
