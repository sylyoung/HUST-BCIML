# ===========================================================================
# MSVTNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/SheldonLiu0412/MSVTNet
#
# Reference (IEEE BibTeX):
#   @Article{Liu2024,
#     author  = {Liu, Ke and Yang, Tao and Yu, Zhuliang and Yi, Weibo and Yu, Hong and Wang, Guoyin and Wu, Wei},
#     journal = {IEEE Journal of Biomedical and Health Informatics},
#     title   = {{MSVTNet}: Multi-Scale Vision Transformer Neural Network for {EEG}-Based Motor Imagery Decoding},
#     year    = {2024},
#     number  = {12},
#     pages   = {7126-7137},
#     volume  = {28},
#     doi     = {10.1109/JBHI.2024.3450753},
#   }
# ===========================================================================
"""MSVTNet backbone (Ke Liu et al., 2024) - a multi-scale convolutional plus
Transformer network for EEG motor-imagery decoding.

The architecture has two stages that map to Section III of the paper:

  * Multi-scale temporal-spatial convolution (MSTSConv, paper Section III-B).
    Several parallel EEGNet-style branches run in parallel. Each branch is a
    `TSConv` block: a temporal convolution with a branch-specific kernel length,
    a depthwise spatial convolution across all channels, ELU, average pooling,
    dropout, then a separable temporal convolution, ELU, average pooling and
    dropout. Every branch keeps the same feature width `F * D` but sees a
    different temporal scale (kernel lengths `C1 = [15, 31, 63, 125]`), which is
    the multi-scale idea. The per-branch outputs are laid out as tokens over the
    pooled time axis and concatenated along the feature axis, giving a token
    sequence of shape (B, seq_len, d_model) with d_model = sum of `F[b] * D`.

  * Transformer encoder over the token sequence (paper Section III-C). A learned
    class (CLS) token is prepended, learned positional embeddings are added, and
    a standard pre-norm `TransformerEncoder` (multi-head self-attention plus
    feed-forward) attends over the tokens. The final CLS token is read out as the
    global feature vector.

Head handling for this port. The reference MSVTNet has per-branch auxiliary
classification heads plus a main classification head, trained jointly with a
weighted cross-entropy loss. This port exposes the pre-main-head feature vector
(the Transformer CLS output, width d_model) as `out_features` and DROPS all
classification heads: both the auxiliary per-branch heads and the main head. The
shared hustbciml `Linear` head then produces the logits, so only the main
feature path (MSTSConv -> concat -> Transformer -> CLS) is kept here.

Source: github.com/SheldonLiu0412/MSVTNet. Deviations, all behaviour-preserving:
the `einops` `Rearrange('b d 1 t -> b t d')` is rewritten with plain torch ops
(`squeeze` plus `permute`), the in-place positional-encoding add is written
out-of-place, and `seq_len` and `d_model` are inferred by a dummy forward in
`__init__` (the reference derives them the same way from a random dummy input)
so any (C, T) dataset works. Default hyper-parameters match the reference
(F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7,
Pc = 0.3, nhead = 8, ff_ratio = 1, Pt = 0.5, transformer layers = 2).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hustbciml.core.stages import Backbone


class _TSConv(nn.Sequential):
    """One temporal-spatial convolution branch (paper Section III-B, `TSConv`).

    Temporal conv (branch-specific kernel `C1`) -> BN -> depthwise spatial conv
    over all `nCh` channels -> BN -> ELU -> avg-pool -> dropout -> separable
    temporal conv (kernel `C2`) -> BN -> ELU -> avg-pool -> dropout. Output width
    is `F * D`; the time axis is pooled by `P1 * P2`.
    """

    def __init__(self, nCh: int, F: int, C1: int, C2: int, D: int,
                 P1: int, P2: int, Pc: float) -> None:
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding="same", bias=False),
            nn.BatchNorm2d(F),
            nn.Conv2d(F, F * D, (nCh, 1), groups=F, bias=False),   # depthwise spatial
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(Pc),
            nn.Conv2d(F * D, F * D, (1, C2), padding="same", groups=F * D, bias=False),  # separable temporal
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(Pc),
        )


class _PositionalEncoding(nn.Module):
    """Learned additive positional embedding (paper Section III-C)."""

    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe                                          # out-of-place add


class _Transformer(nn.Module):
    """CLS token + positional embedding + pre-norm Transformer encoder.

    Paper Section III-C. Prepends a learned CLS token, adds positional
    embeddings, runs a standard `TransformerEncoder`, and returns the CLS token
    as the global feature (B, d_model).
    """

    def __init__(self, seq_len: int, d_model: int, nhead: int, ff_ratio: int,
                 Pt: float = 0.5, num_layers: int = 4) -> None:
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = _PositionalEncoding(seq_len + 1, d_model)   # +1 for CLS token
        dim_ff = d_model * ff_ratio
        self.dropout = nn.Dropout(Pt)
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_ff, Pt, batch_first=True, norm_first=True
            ),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # (B, seq_len, d_model)
        b = x.shape[0]
        x = torch.cat((self.cls_embedding.expand(b, -1, -1), x), dim=1)  # prepend CLS
        x = self.pos_embedding(x)
        x = self.dropout(x)
        return self.trans(x)[:, 0]                                 # read out CLS -> (B, d_model)


class MSVTNet(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 F=(9, 9, 9, 9), C1=(15, 31, 63, 125), C2: int = 15, D: int = 2,
                 P1: int = 8, P2: int = 7, Pc: float = 0.3, nhead: int = 8,
                 ff_ratio: int = 1, Pt: float = 0.5, layers: int = 2, **_):
        super().__init__()
        assert len(F) == len(C1), "The length of F and C1 should be equal."
        self.n_chans = n_chans
        self.n_times = n_times

        # Multi-scale temporal-spatial conv branches (paper Section III-B).
        # Each branch: TSConv then reshape (B, F*D, 1, T') -> (B, T', F*D) so the
        # pooled time steps become tokens. The einops Rearrange is a plain
        # squeeze+permute here.
        self.mstsconv = nn.ModuleList([
            _TSConv(n_chans, F[b], C1[b], C2, D, P1, P2, Pc)
            for b in range(len(F))
        ])

        # Infer token-sequence length and model width by a dummy forward, so the
        # backbone is dataset-agnostic (the reference derives these the same way).
        with torch.no_grad():
            tokens = self._forward_mstsconv(torch.zeros(1, 1, n_chans, n_times))
        seq_len, d_model = tokens.shape[1], tokens.shape[2]

        # Transformer encoder over the token sequence (paper Section III-C).
        self.transformer = _Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        # Expose the CLS feature vector; DROP the auxiliary and main heads.
        self.out_features = d_model

    def _forward_mstsconv(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, C, T)
        outs = []
        for tsconv in self.mstsconv:
            o = tsconv(x)                                          # (B, F*D, 1, T')
            o = o.squeeze(2).permute(0, 2, 1)                     # (B, T', F*D)
            outs.append(o)
        return torch.cat(outs, dim=2)                             # (B, T', d_model)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, C, T)
        x = self._forward_mstsconv(x)                             # (B, seq_len, d_model)
        return self.transformer(x)                                # (B, d_model) via CLS token
