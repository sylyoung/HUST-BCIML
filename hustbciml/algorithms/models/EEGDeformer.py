# ===========================================================================
# EEGDeformer.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/yi-ding-cs/EEG-Deformer
#
# Reference (IEEE BibTeX):
#   @Article{Ding2025,
#     author  = {Ding, Yi and Li, Yong and Sun, Hao and Liu, Rui and Tong, Chengxuan and Liu, Chenyu and Zhou, Xinliang and Guan, Cuntai},
#     journal = {IEEE Journal of Biomedical and Health Informatics},
#     title   = {{EEG}-Deformer: A Dense Convolutional Transformer for Brain-Computer Interfaces},
#     year    = {2025},
#     number  = {3},
#     pages   = {1909-1918},
#     volume  = {29},
#     doi     = {10.1109/JBHI.2024.3504604},
#   }
# ===========================================================================
"""EEG-Deformer (Yi Ding et al., 2025) - a dense convolutional transformer for
brain-computer interfaces.

The network has three parts, mapped to the paper's Section III (Method):

  * Shallow CNN encoder (paper Section III-B, "Shallow feature encoder"). A
    temporal depthwise-style convolution followed by a spatial convolution over
    all channels, then BatchNorm, ELU and temporal max-pooling. It turns the raw
    (B, 1, C, T) signal into num_kernel temporal token sequences of length
    0.5*T. Both convolutions are weight-norm-constrained (max_norm=2), matching
    the paper's Conv2dWithConstraint.

  * Hierarchical Coarse-to-Fine Transformer (paper Section III-C). At each of the
    ``depth`` levels the token length is halved and processed by two parallel
    paths. The Coarse path max-pools the tokens then applies multi-head
    self-attention with a residual connection, capturing global temporal context
    at progressively coarser resolution. The Fine path is the Fine-grained
    Temporal Learning branch (paper Section III-C, Eq. for the CNN sub-branch): a
    small 1-D convolution block (Dropout, Conv1d, BatchNorm1d, ELU, MaxPool1d)
    that keeps local high-frequency detail. The two paths are recombined as
    ``x = FeedForward(coarse) + fine`` and passed to the next level.

  * Dense Information Purification (paper Section III-D). At every level a compact
    power descriptor ``log(mean(fine^2))`` is extracted per kernel and stacked
    across levels, then concatenated with the flattened final-level tokens. This
    dense multi-level feature is the pre-classifier representation.

The paper's ``mlp_head`` (a single Linear to ``num_classes``) is dropped here and
its input width becomes ``out_features``, so the shared hustbciml ``Linear`` head
produces the logits. ``out_features`` is inferred by a dummy forward in
``__init__`` (the reference derives it analytically via ``get_hidden_size``; the
dummy forward makes the backbone robust to any ``(C, T)`` where an intermediate
pooled length is odd).

Source: github.com/yi-ding-cs/EEG-Deformer (``models/EEGDeformer.py``, class
``Deformer``). The only deviations are behaviour-preserving: the ``einops``
dependency is dropped (``rearrange`` and ``Rearrange`` rewritten with plain torch
ops) and the pre-classifier width is exposed instead of the final Linear. Default
hyper-parameters follow the reference's own example configuration
(``temporal_kernel=11, num_kernel=64, depth=4, heads=16, mlp_dim=16, dim_head=16,
dropout=0.5``).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hustbciml.core.stages import Backbone


class _Conv2dWithConstraint(nn.Conv2d):
    """Conv2d whose weights are renormalised to a max L2 norm each forward.

    Faithful to the reference ``Conv2dWithConstraint``; used by the shallow CNN
    encoder (paper Section III-B) with ``max_norm=2``.
    """

    def __init__(self, *args, do_weight_norm: bool = True, max_norm: float = 1, **kwargs):
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class _FeedForward(nn.Module):
    """Pre-norm position-wise MLP (paper Section III-C, transformer sublayer)."""

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _Attention(nn.Module):
    """Multi-head self-attention over the coarse tokens (paper Section III-C).

    einops-free rewrite of the reference ``Attention``; the qkv projection and
    ``dim_head ** -0.5`` scaling are unchanged.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):                                   # (B, N, dim)
        b, n, _ = x.shape
        h, d = self.heads, self.dim_head
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 'b n (h d) -> b h n d' without einops
        q, k, v = (t.view(b, n, h, d).permute(0, 2, 1, 3) for t in qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)                        # (B, h, N, d)
        # 'b h n d -> b n (h d)' without einops
        out = out.permute(0, 2, 1, 3).reshape(b, n, h * d)
        return self.to_out(out)


class _Transformer(nn.Module):
    """Hierarchical Coarse-to-Fine transformer with Dense Information Purification.

    Implements paper Sections III-C and III-D. Each level halves the token length
    (``dim *= 0.5``), runs the coarse attention path and the Fine-grained Temporal
    Learning CNN path in parallel, harvests a per-kernel power descriptor for the
    dense feature, and fuses the paths for the next level.
    """

    def _cnn_block(self, in_chan, kernel_size, dp):
        # Fine-grained Temporal Learning branch (paper Section III-C).
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
                      kernel_size=kernel_size, padding=self._padding_1d(kernel_size)),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, in_chan,
                 fine_grained_kernel=11, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            dim = int(dim * 0.5)                            # coarse-to-fine halving
            self.layers.append(nn.ModuleList([
                _Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                _FeedForward(dim, mlp_dim, dropout=dropout),
                self._cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout),
            ]))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)   # coarse-path downsampler

    def forward(self, x):                                   # (B, in_chan, dim)
        dense_feature = []
        for attn, ff, cnn in self.layers:
            x_cg = self.pool(x)                             # coarse: downsample
            x_cg = attn(x_cg) + x_cg                        # coarse: attention + residual
            x_fg = cnn(x)                                   # fine: local CNN branch
            x_info = self._get_info(x_fg)                   # dense purification descriptor (B, in_chan)
            dense_feature.append(x_info)
            x = ff(x_cg) + x_fg                             # fuse coarse and fine
        x_dense = torch.cat(dense_feature, dim=-1)          # (B, in_chan*depth)
        x = x.view(x.size(0), -1)                           # (B, in_chan*len_last)
        emd = torch.cat((x, x_dense), dim=-1)               # (B, in_chan*(depth + len_last))
        return emd

    def _get_info(self, x):
        # Per-kernel log-power over time (paper Section III-D). x: (B, in_chan, L)
        return torch.log(torch.mean(x.pow(2), dim=-1))

    @staticmethod
    def _padding_1d(kernel):
        return int(0.5 * (kernel - 1))


class EEGDeformer(Backbone):
    task_name = "classification"

    def _cnn_block(self, out_chan, kernel_size, num_chan):
        # Shallow CNN encoder (paper Section III-B): temporal conv, spatial conv,
        # BN, ELU, temporal max-pool. Both convs are max-norm constrained.
        return nn.Sequential(
            _Conv2dWithConstraint(1, out_chan, kernel_size,
                                  padding=self._padding_2d(kernel_size[-1]), max_norm=2),
            _Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
        )

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 temporal_kernel: int = 11, num_kernel: int = 64, depth: int = 4,
                 heads: int = 16, mlp_dim: int = 16, dim_head: int = 16,
                 dropout: float = 0.5, **_):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.num_kernel = num_kernel

        # Shallow CNN encoder -> (B, num_kernel, 1, 0.5*n_times).
        self.cnn_encoder = self._cnn_block(out_chan=num_kernel,
                                           kernel_size=(1, temporal_kernel), num_chan=n_chans)

        dim = int(0.5 * n_times)                            # token length after the CNN encoder
        # Positional embedding over the num_kernel tokens (paper Section III-C).
        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        # Hierarchical Coarse-to-Fine transformer + Dense Information Purification.
        self.transformer = _Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout,
            in_chan=num_kernel, fine_grained_kernel=temporal_kernel,
        )

        # Infer the concatenated pre-classifier feature width via a dummy forward,
        # so the backbone is dataset-agnostic (paper derives it via get_hidden_size).
        with torch.no_grad():
            feat = self.forward_features(torch.zeros(1, 1, n_chans, n_times))
        self.out_features = int(feat.shape[1])

        # The reference nn.Linear(out_size, num_classes) classifier is dropped;
        # the shared hustbciml Linear head consumes out_features instead.

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, C, T)
        x = self.cnn_encoder(x)                             # (B, num_kernel, 1, 0.5*T)
        # 'b k c f -> b k (c f)' without einops (c == 1 after the spatial conv).
        b, k, c, f = x.shape
        x = x.reshape(b, k, c * f)                          # (B, num_kernel, dim)
        x = x + self.pos_embedding                          # additive positional encoding
        x = self.transformer(x)                             # dense multi-level feature
        return x                                            # (B, out_features)

    @staticmethod
    def _padding_2d(kernel):
        return (0, int(0.5 * (kernel - 1)))
