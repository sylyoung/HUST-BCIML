# ===========================================================================
# DBConformer.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/wzwvv/DBConformer
#
# Reference (IEEE BibTeX):
#   @Article{Wang2026,
#     author  = {Wang, Ziwei and Wang, Hongbin and Jia, Tianwang and He, Xingyi and Li, Siyang and Wu, Dongrui},
#     journal = {IEEE Journal of Biomedical and Health Informatics},
#     title   = {{DBC}onformer: Dual-Branch Convolutional Transformer for {EEG} Decoding},
#     year    = {2026},
#     number  = {5},
#     pages   = {4134-4147},
#     volume  = {30},
#     doi     = {10.1109/JBHI.2025.3622725},
#   }
# ===========================================================================
"""DBConformer (Wang et al., 2026, IEEE JBHI) — a dual-branch convolutional
Transformer for EEG decoding. It is a supervised end-to-end classifier
(cross-entropy loss, Eq. 15) validated on three paradigms: motor imagery,
epileptic seizure detection, and SSVEP (Sec. IV).

Unlike serial CNN-Transformer hybrids that push all spatial information through a
temporal bottleneck, DBConformer runs two parallel branches (Fig. 2, Sec. III-C)
and concatenates their features:

  * T-Conformer (temporal branch, Sec. III-C-2) -> F_t (Eq. 3). A "Temporal Patch
    Embedding" (a depthwise-separable 1D convolution: pointwise/spatial Conv1D ->
    BN -> depthwise temporal Conv1D -> BN&GELU -> average pooling) tokenizes the
    signal into P non-overlapping temporal patches Z_t; a "T-Transformer encoder"
    with learnable positional encoding (Eq. 6-7) models long-range temporal
    dependencies; mean pooling over patches (Eq. 8) yields F_t.
  * S-Conformer (spatial branch, Sec. III-C-3) -> F_s (Eq. 4). A "Spatial Patch
    Embedding" (depthwise Conv1D over time -> average pool -> flatten/reshape ->
    Linear) turns each EEG channel into one token Z_s; an "S-Transformer encoder"
    with learnable positional encoding (Eq. 9-10) models inter-channel
    dependencies; a lightweight "channel attention module" (Fig. 3, Eq. 11-13)
    re-weights the channel tokens by a data-driven, softmax-normalized channel
    importance score and sums them into F_s.

The two branch features are concatenated, F_fused = [F_t; F_s] (Eq. 14), and an
MLP classifier maps them to the class prediction (Eq. 14-15). The paper also
applies Euclidean Alignment (EA, Eq. 1-2, Sec. III-B) as a preprocessing step;
in this benchmark EA is a separate aligner stage, so it is not part of this file.

Faithful port of the paper's default architecture (Table II). As in the
EEGConformer port, the paper's MLP classifier (2D -> 64 -> 32 -> N_classes,
Table II; with D=40 the input is 2D=80) is split: the 80->64->32 stack is folded
into ``forward_features`` (``out_features`` is 32) and the paper's final linear
layer becomes the shared hustbciml ``Linear`` head.

Source: github.com/wzwvv/DBConformer (``models/DBConformer.py``). Deviations, all
behaviour-preserving: the ``einops`` and ``timm`` dependencies are dropped
(``rearrange`` rewritten with torch ops; ``timm.trunc_normal_`` ->
``torch.nn.init.trunc_normal_``), the global ``cudnn`` flags are not set, and the
number of temporal patches P is inferred by a dummy forward (the paper fixes it
via a hand-picked pooling window; here that window defaults to ``n_times//8`` so
P~=8 on any dataset).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.core.stages import Backbone


class _Conv(nn.Module):
    """conv (+ optional BN + optional activation); drops the conv bias under BN."""

    def __init__(self, conv, bn=None, activation=None):
        super().__init__()
        if bn is not None:
            conv.bias = None
        self.conv, self.bn, self.activation = conv, bn, activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class _InterFre(nn.Module):
    """sum the temporal-conv branches (one when radix=1, as used here), then GELU
    -- the "BN & GELU" non-linearity of the Temporal Patch Embedding (Table II)."""

    def forward(self, xs):
        return F.gelu(sum(xs))


class _Stem(nn.Module):
    """Temporal Patch Embedding (T-Conformer, Sec. III-C-2-a, Table II):
    depthwise-separable 1D conv (pointwise/spatial Conv1D -> depthwise temporal
    Conv1D) + GELU + average pooling into P temporal patches -> (B, D, P).
    radix defaults to 1, i.e. a single temporal scale as in the paper."""

    def __init__(self, in_planes, out_planes=40, kernel_size=63, patch_size=125,
                 radix=1, drop=0.5, drop_last_t=True):
        super().__init__()
        self.out_planes = out_planes
        self.drop_last_t = drop_last_t
        mid = out_planes * radix
        self.sconv = _Conv(nn.Conv1d(in_planes, mid, 1, bias=False, groups=radix),
                           bn=nn.BatchNorm1d(mid))
        self.tconv = nn.ModuleList()
        k = kernel_size
        for _ in range(radix):
            self.tconv.append(_Conv(
                nn.Conv1d(out_planes, out_planes, k, 1, groups=out_planes,
                          padding=k // 2, bias=False),
                bn=nn.BatchNorm1d(out_planes)))
            k //= 2
        self.inter = _InterFre()
        self.pool = nn.AvgPool1d(patch_size, patch_size)
        self.dp = nn.Dropout(drop)

    def forward(self, x):                                   # (B, in_planes, T)
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)      # radix chunks
        out = [m(o) for o, m in zip(out, self.tconv)]
        out = self.inter(out)
        if self.drop_last_t:
            out = out[:, :, :-1]                            # drop trailing sample
        out = self.pool(out)
        return self.dp(out)                                 # (B, out_planes, P)


class _TemporalPatch(nn.Module):
    def __init__(self, n_chans, emb, patch_size, kernel_size=63, drop=0.5, drop_last_t=True):
        super().__init__()
        self.stem = _Stem(n_chans, emb, kernel_size, patch_size, radix=1,
                          drop=drop, drop_last_t=drop_last_t)

    def forward(self, x):                                   # (B, C, T)
        return self.stem(x).permute(0, 2, 1)               # (B, P, emb)


class _SpatialPatch(nn.Module):
    """Spatial Patch Embedding (S-Conformer, Sec. III-C-3-a, Table II): a
    depthwise Conv1D over time is applied per channel, average-pooled over time,
    then a Linear projects each channel to one token of width emb."""

    def __init__(self, spa_dim, emb):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, spa_dim, kernel_size=25, stride=5, padding=12),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(spa_dim, emb),
        )

    def forward(self, x):                                   # (B, C, T)
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)
        x = self.encoder(x)
        return x.view(B, C, -1)                             # (B, C, emb)


class _MHA(nn.Module):
    """Multi-head self-attention inside the Transformer encoder (Fig. 2, the
    Q/K/V -> dot-product -> scaling -> softmax -> feed-forward path). einops-free;
    the softmax is scaled by sqrt(emb) rather than sqrt(head_dim), matching the
    original authors' code."""

    def __init__(self, emb, heads, drop):
        super().__init__()
        self.emb, self.heads = emb, heads
        self.q = nn.Linear(emb, emb)
        self.k = nn.Linear(emb, emb)
        self.v = nn.Linear(emb, emb)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(emb, emb)

    def forward(self, x):                                   # (B, N, emb)
        B, N, _ = x.shape
        h, d = self.heads, self.emb // self.heads
        q = self.q(x).view(B, N, h, d).permute(0, 2, 1, 3)
        k = self.k(x).view(B, N, h, d).permute(0, 2, 1, 3)
        v = self.v(x).view(B, N, h, d).permute(0, 2, 1, 3)
        energy = torch.einsum('bhqd,bhkd->bhqk', q, k)
        att = torch.softmax(energy / (self.emb ** 0.5), dim=-1)
        att = self.drop(att)
        out = torch.einsum('bhql,bhld->bhqd', att, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.emb)
        return self.proj(out)


class _FF(nn.Sequential):
    def __init__(self, emb, expansion=4, drop=0.5):
        super().__init__(
            nn.Linear(emb, expansion * emb), nn.GELU(),
            nn.Dropout(drop), nn.Linear(expansion * emb, emb))


class _Block(nn.Module):
    """One Transformer encoder layer (Eq. 7 / Eq. 10): pre-norm residual
    self-attention + pre-norm residual feed-forward."""

    def __init__(self, emb, heads=10, drop=0.5):
        super().__init__()
        self.n1, self.attn, self.d1 = nn.LayerNorm(emb), _MHA(emb, heads, drop), nn.Dropout(drop)
        self.n2, self.ff, self.d2 = nn.LayerNorm(emb), _FF(emb, 4, drop), nn.Dropout(drop)

    def forward(self, x):
        x = x + self.d1(self.attn(self.n1(x)))
        x = x + self.d2(self.ff(self.n2(x)))
        return x


class _Encoder(nn.Sequential):
    def __init__(self, depth, emb, heads=10, drop=0.5):
        super().__init__(*[_Block(emb, heads, drop) for _ in range(depth)])


class DBConformer(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 emb: int = 40, spa_dim: int = 16, tem_depth: int = 5,
                 chn_depth: int = 5, heads: int = 10, patch_size: int = None,
                 drop: float = 0.5, drop_last_t: bool = True, **_):
        super().__init__()
        if patch_size is None:
            patch_size = max(1, (n_times - (1 if drop_last_t else 0)) // 8)

        self.temporal = _TemporalPatch(n_chans, emb, patch_size, drop=drop, drop_last_t=drop_last_t)
        self.spatial = _SpatialPatch(spa_dim, emb)
        with torch.no_grad():                               # infer #temporal patches
            P = self.temporal(torch.zeros(1, n_chans, n_times)).shape[1]
        self.pos_t = nn.Parameter(torch.randn(1, P, emb))
        self.pos_s = nn.Parameter(torch.randn(1, n_chans, emb))

        self.temporal_tr = _Encoder(tem_depth, emb, heads, drop)   # T-Transformer encoder (Eq. 7)
        self.spatial_tr = _Encoder(chn_depth, emb, heads, drop)    # S-Transformer encoder (Eq. 10)
        # channel attention module (S-Conformer, Fig. 3, Eq. 11): Linear -> Tanh
        # -> Linear scores each channel token; softmax over channels (Eq. 12) then
        # gives the channel-importance weights used to pool the tokens (Eq. 13).
        self.attn_pool = nn.Sequential(nn.Linear(emb, emb), nn.Tanh(), nn.Linear(emb, 1))
        # first two layers of the paper's MLP classifier (2D -> 64 -> 32, Table II);
        # the final 32 -> N_classes layer is the shared hustbciml Linear head.
        self.feat = nn.Sequential(
            nn.Linear(emb * 2, 64), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ELU(), nn.Dropout(0.3),
        )
        self.out_features = 32
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, C, T)
        x = x.squeeze(1)                                    # (B, C, T)
        xt = self.temporal(x) + self.pos_t                  # T-Conformer patches + pos enc (Eq. 6)
        xs = self.spatial(x) + self.pos_s                   # S-Conformer channel tokens + pos enc (Eq. 9)
        xt = self.temporal_tr(xt)                           # T-Transformer encoder (Eq. 7)
        xs = self.spatial_tr(xs)                            # S-Transformer encoder (Eq. 10)
        x_t = xt.mean(dim=1)                                # F_t: mean pooling over patches (Eq. 8)
        attn = torch.softmax(self.attn_pool(xs), dim=1)     # channel-importance weights alpha (Eq. 12)
        x_s = torch.sum(attn * xs, dim=1)                   # F_s: attention-weighted channel sum (Eq. 13)
        return self.feat(torch.cat([x_t, x_s], dim=-1))     # concat F_fused=[F_t;F_s] (Eq. 14) -> MLP
