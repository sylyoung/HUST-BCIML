# ===========================================================================
# CTNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/snailpt/CTNet
#
# Reference (IEEE BibTeX):
#   @Article{Zhao2024,
#     author  = {Zhao, Wei and Jiang, Xiaolu and Zhang, Baocan and Xiao, Shixiao and Weng, Sujun},
#     journal = {Scientific Reports},
#     title   = {{CTNet}: A Convolutional Transformer Network for {EEG}-Based Motor Imagery Classification},
#     year    = {2024},
#     pages   = {20237},
#     volume  = {14},
#     doi     = {10.1038/s41598-024-71118-7},
#   }
# ===========================================================================
"""CTNet (Zhao et al., 2024, Scientific Reports) - a convolutional transformer
network for EEG-based motor imagery classification.

The architecture has three parts, matching Section 2 (Methods) and Figure 1 of
the paper:

  * Convolutional module (paper Section 2.2, "EEG patch embedding"). An
    EEGNet-style front end. A temporal convolution (kernel 64, about 0.25 of the
    sampling rate) followed by a depthwise convolution across all channels, then
    ELU, average pooling and dropout, then one more pointwise-in-time
    convolution, ELU, average pooling and dropout. The time-pooled output is
    reshaped into a sequence of tokens, one token per remaining time step with
    the F2 = D * f1 = 40 feature maps as the embedding, exactly the patch
    tokens fed to a Vision-Transformer.

  * Transformer encoder (paper Section 2.3, "Transformer encoder"). The tokens
    are scaled by sqrt(emb_size), a learnable positional embedding is added,
    then `depth` = 6 standard encoder blocks run. Each block is a residual
    multi-head self-attention sublayer plus a residual position-wise
    feed-forward sublayer, each wrapped by dropout and a post-add LayerNorm. The
    self-attention is faithful to the authors' code, which scales the logits by
    sqrt(emb_size) rather than sqrt(head_dim).

  * Residual connection and classifier (paper Section 2.4). The encoder output
    is added back to its pre-encoder input (a residual skip over the whole
    transformer) and flattened into one feature vector.

This ports the paper's default and best BCI-IV-2a configuration: emb_size = 40,
depth = 6, heads = 4, f1 = 20, D = 2, temporal kernel 64, both average pooling
sizes 8, dropout 0.3 in the convolutional module and 0.5 inside the transformer.

Backbone only. The paper's final `ClassificationHead` (Dropout(0.5) then a
Linear to the class count) is removed. `forward_features` returns the flattened
residual feature vector and the shared hustbciml Linear head maps it to classes,
so the paper's final linear layer becomes that head. `out_features` is the
flattened width (emb_size times the number of tokens), inferred by a dummy
forward so any (C, T) input works instead of the source's hardcoded
`flatten_eeg1`.

Source: github.com/snailpt/CTNet. Deviations, all behaviour-preserving: the
`einops` dependency is dropped (the `Rearrange` reshape and the `rearrange`
calls inside attention are rewritten with plain torch ops), the learnable
positional encoding no longer forces `.cuda()` so it stays device-agnostic, and
the flattened feature width is inferred by a dummy forward rather than fixed by
the `flatten_eeg1` constant.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.core.stages import Backbone


class _PatchEmbeddingCNN(nn.Module):
    """EEGNet-style convolutional patch embedding (paper Section 2.2).

    Produces a sequence of temporal tokens (B, N, emb_size). The number of
    feature maps after the depthwise conv, f2 = D * f1, equals emb_size.
    """

    def __init__(self, f1=20, kernel_size=64, D=2, pooling_size1=8,
                 pooling_size2=8, dropout_rate=0.3, number_channel=22):
        super().__init__()
        f2 = D * f1
        self.cnn_module = nn.Sequential(
            # temporal conv, kernel 64 is about 0.25 * sampling rate
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            # channel depthwise conv over all electrodes
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1,
                      padding="valid", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1, slices the time axis into patches as in a ViT
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),
            # extra pointwise-in-time conv
            nn.Conv2d(f2, f2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 2, sets the token-sequence length for the encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, C, T)
        x = self.cnn_module(x)                           # (B, emb, 1, N)
        # einops Rearrange('b e h w -> b (h w) e'); here h == 1 after channel conv
        b, e, h, w = x.shape
        x = x.reshape(b, e, h * w).permute(0, 2, 1)      # (B, N, emb)
        return x


class _MultiHeadAttention(nn.Module):
    """Multi-head self-attention (paper Section 2.3), einops-free.

    Faithful to the authors' code, the attention logits are scaled by
    sqrt(emb_size) rather than by sqrt(head_dim).
    """

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, N, emb)
        b, n, _ = x.shape
        h, d = self.num_heads, self.emb_size // self.num_heads
        # rearrange('b n (h d) -> b h n d')
        queries = self.queries(x).view(b, n, h, d).permute(0, 2, 1, 3)
        keys = self.keys(x).view(b, n, h, d).permute(0, 2, 1, 3)
        values = self.values(x).view(b, n, h, d).permute(0, 2, 1, 3)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        # rearrange('b h n d -> b n (h d)')
        out = out.permute(0, 2, 1, 3).reshape(b, n, self.emb_size)
        out = self.projection(out)
        return out


class _FeedForwardBlock(nn.Sequential):
    """Position-wise feed-forward network (paper Section 2.3)."""

    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _ResidualAdd(nn.Module):
    """Residual sublayer with dropout then a post-add LayerNorm (paper Section 2.3)."""

    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x
        res = self.fn(x)
        out = self.layernorm(self.drop(res) + x_input)
        return out


class _TransformerEncoderBlock(nn.Sequential):
    """One encoder block: residual attention then residual feed-forward."""

    def __init__(self, emb_size, num_heads=4, drop_p=0.5,
                 forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            _ResidualAdd(
                _MultiHeadAttention(emb_size, num_heads, drop_p),
                emb_size, drop_p),
            _ResidualAdd(
                _FeedForwardBlock(emb_size, expansion=forward_expansion,
                                  drop_p=forward_drop_p),
                emb_size, drop_p),
        )


class _TransformerEncoder(nn.Sequential):
    """Stack of `depth` encoder blocks (paper Section 2.3)."""

    def __init__(self, heads, depth, emb_size):
        super().__init__(*[_TransformerEncoderBlock(emb_size, heads)
                           for _ in range(depth)])


class _PositionalEncoding(nn.Module):
    """Learnable positional embedding added to the token sequence (paper Section 2.3)."""

    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, emb)
        # source added `.cuda()` here; dropped so the module stays device-agnostic
        x = x + self.encoding[:, : x.shape[1], :]
        return self.dropout(x)


class CTNet(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 heads: int = 4, emb_size: int = 40, depth: int = 6,
                 f1: int = 20, kernel_size: int = 64, D: int = 2,
                 pooling_size1: int = 8, pooling_size2: int = 8,
                 dropout_rate: float = 0.3, **_):
        super().__init__()
        self.emb_size = emb_size

        # convolutional patch embedding (paper Section 2.2)
        self.cnn = _PatchEmbeddingCNN(
            f1=f1, kernel_size=kernel_size, D=D,
            pooling_size1=pooling_size1, pooling_size2=pooling_size2,
            dropout_rate=dropout_rate, number_channel=n_chans)
        # learnable positional embedding + transformer encoder (paper Section 2.3)
        self.position = _PositionalEncoding(emb_size, dropout=0.1)
        self.trans = _TransformerEncoder(heads, depth, emb_size)
        self.flatten = nn.Flatten()

        # infer the flattened residual-feature width (source hardcodes flatten_eeg1)
        with torch.no_grad():
            self.out_features = self.forward_features(
                torch.zeros(1, 1, n_chans, n_times)).shape[1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, C, T)
        cnn = self.cnn(x)                        # (B, N, emb) patch tokens
        # scale tokens then add positional embedding (paper Section 2.3)
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        trans = self.trans(cnn)
        # residual skip over the whole transformer, then flatten (paper Section 2.4)
        features = cnn + trans
        features = self.flatten(features)        # (B, N * emb)
        return features
