# ===========================================================================
# MSCFormer.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: not publicly released (paper: https://www.nature.com/articles/s41598-025-96611-5)
#
# Reference (IEEE BibTeX):
#   @Article{Zhao2025,
#     author  = {Zhao, Wei and Zhang, Baocan and Zhou, Haifeng and Wei, Dezhi and Huang, Chenxi and Lan, Quan},
#     journal = {Scientific Reports},
#     title   = {Multi-Scale Convolutional Transformer Network for Motor Imagery Brain-Computer Interface},
#     year    = {2025},
#     pages   = {12935},
#     volume  = {15},
#     doi     = {10.1038/s41598-025-96611-5},
#   }
# ===========================================================================
"""MSCFormer (Wei Zhao et al., 2025) is a multi-scale convolutional transformer
for motor imagery EEG decoding.

The network has three stages that mirror the paper's Figure 1 and Methods.

  * Multi-scale convolutional module (paper Section "Multi-scale convolution
    module"). Three parallel branches each run a temporal convolution at a
    different kernel length (85, 65, 45 samples) followed by a depthwise
    spatial convolution over all electrode channels, batch normalization, ELU,
    average pooling and dropout. Each branch outputs f1=16 feature maps. The
    three branches are concatenated along the feature dimension into an
    embedding of size emb_size = 16 * 3 = 48, then reshaped into a sequence of
    tokens of width 48.
  * Transformer encoder (paper Section "Transformer encoder"). A learnable
    class token is prepended to the token sequence in the manner of BERT, the
    sequence is scaled by sqrt(emb_size) and a learnable positional encoding is
    added. A stack of depth=5 encoder blocks, each an 8-head self-attention
    sublayer and a feed-forward sublayer, both wrapped as residual-add followed
    by layer normalization, attends over the tokens.
  * Classification (paper Section "Classification"). The paper reads out the
    class token as the final feature vector and passes it through a dropout plus
    a single linear layer to the class logits.

Port contract. This backbone exposes the pre-logit feature vector, so
out_features = emb_size = 48 (the class token). The paper's final linear layer
that maps 48 to n_classes is removed and becomes the shared hustbciml Linear
head. The dropout that the paper applies right before that linear layer is kept
as the last pre-logit operation.

Behaviour-preserving deviations from the reference. The einops Rearrange,
rearrange and Reduce calls are rewritten with plain torch reshape and permute
operations. The hard-coded .cuda() placements of the class token and the
positional encoding are dropped so the module runs on any device. The reference
fixes the positional-encoding length at 100 and the pooling size per dataset
(44 for BCI IV-2a, 52 for BCI IV-2b, default 52 here). Here the token-sequence
length is inferred by a dummy forward in __init__ so the positional encoding is
sized correctly for any (n_chans, n_times) input, and the global cudnn flags are
not set.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.core.stages import Backbone


class _PatchEmbeddingCNN(nn.Module):
    """Multi-scale convolutional module (paper Section "Multi-scale convolution
    module").

    Three parallel branches with temporal-conv kernel lengths 85, 65 and 45.
    Each branch does temporal conv -> depthwise spatial conv over all channels
    -> BatchNorm -> ELU -> average pool -> dropout, and emits f1 feature maps.
    The branches are concatenated on the feature dimension and flattened into a
    token sequence of width 3 * f1.
    """

    def __init__(self, f1=16, pooling_size=52, dropout_rate=0.5, number_channel=22):
        super().__init__()

        def branch(kernel_len):
            return nn.Sequential(
                # temporal convolution, same padding keeps the time length
                nn.Conv2d(1, f1, (1, kernel_len), (1, 1), padding="same"),
                # depthwise spatial convolution over all electrode channels
                nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
                nn.BatchNorm2d(f1),
                nn.ELU(),
                nn.AvgPool2d((1, pooling_size)),
                nn.Dropout(dropout_rate),
            )

        self.cnn1 = branch(85)
        self.cnn2 = branch(65)
        self.cnn3 = branch(45)

    def forward(self, x):                                   # (B, 1, C, T)
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        # concatenate the three scales along the feature-channel dimension
        x = torch.cat([x1, x2, x3], dim=1)                 # (B, 3*f1, 1, W)
        b, e, h, w = x.shape
        # Rearrange('b e h w -> b (h w) e'): flatten spatial dims into tokens
        x = x.reshape(b, e, h * w).permute(0, 2, 1)        # (B, h*w, 3*f1)
        return x


class _MultiHeadAttention(nn.Module):
    """Multi-head self-attention (paper Section "Transformer encoder"), einops
    free. Keeps the reference scaling of dividing by sqrt(emb_size)."""

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):                                  # (B, N, emb)
        b, n, _ = x.shape
        h, d = self.num_heads, self.emb_size // self.num_heads
        # "b n (h d) -> b h n d"
        queries = self.queries(x).view(b, n, h, d).permute(0, 2, 1, 3)
        keys = self.keys(x).view(b, n, h, d).permute(0, 2, 1, 3)
        values = self.values(x).view(b, n, h, d).permute(0, 2, 1, 3)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        # "b h n d -> b n (h d)"
        out = out.permute(0, 2, 1, 3).reshape(b, n, self.emb_size)
        out = self.projection(out)
        return out


class _FeedForwardBlock(nn.Sequential):
    """Position-wise feed-forward sublayer (paper Section "Transformer encoder")."""

    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _ResidualAdd(nn.Module):
    """Residual-add then layer normalization (the add-and-norm of each
    encoder sublayer, paper Section "Transformer encoder")."""

    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x):
        x_input = x
        res = self.fn(x)
        out = self.layernorm(self.drop(res) + x_input)
        return out


class _TransformerEncoderBlock(nn.Sequential):
    """One encoder block: add-and-norm(self-attention) + add-and-norm(feed
    forward), paper Section "Transformer encoder"."""

    def __init__(self, emb_size, num_heads=4, drop_p=0.5,
                 forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(_MultiHeadAttention(emb_size, num_heads, drop_p)),
                emb_size, drop_p),
            _ResidualAdd(
                nn.Sequential(
                    _FeedForwardBlock(emb_size, expansion=forward_expansion,
                                      drop_p=forward_drop_p)),
                emb_size, drop_p),
        )


class _TransformerEncoder(nn.Sequential):
    """Stack of depth encoder blocks (paper Section "Transformer encoder")."""

    def __init__(self, heads, depth, emb_size):
        super().__init__(
            *[_TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])


class _PositionalEncoding(nn.Module):
    """Learnable positional encoding added to the token sequence (paper Section
    "Transformer encoder"). Device-agnostic (no .cuda())."""

    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):                                  # (B, N, emb)
        x = x + self.encoding[:, : x.shape[1], :]
        return self.dropout(x)


class MSCFormer(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 f1: int = 16, heads: int = 8, depth: int = 5,
                 pooling_size: int = 52, dropout_rate: float = 0.5, **_):
        super().__init__()
        self.emb_size = f1 * 3

        # Multi-scale convolutional module -> token sequence.
        self.cnn = _PatchEmbeddingCNN(f1=f1, pooling_size=pooling_size,
                                      dropout_rate=dropout_rate,
                                      number_channel=n_chans)

        # Infer the token-sequence length (plus the class token) with a dummy
        # forward, so the learnable positional encoding is sized for any input.
        with torch.no_grad():
            n_tokens = self.cnn(torch.zeros(1, 1, n_chans, n_times)).shape[1]
        pos_length = n_tokens + 1                          # + class token

        self.position = _PositionalEncoding(self.emb_size, length=pos_length,
                                            dropout=0.1)
        self.trans = _TransformerEncoder(heads, depth, self.emb_size)

        # Dropout the paper applies right before its final linear classifier.
        # The linear layer itself is removed and becomes the shared Linear head.
        self.pre_logit_drop = nn.Dropout(0.25)
        self.out_features = self.emb_size                  # class-token width

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, C, T)
        # Multi-scale convolutional module -> tokens (B, N, emb).
        x = self.cnn(x)
        b, l, e = x.shape

        # Prepend a class token like BERT (device-agnostic zeros).
        cls = torch.zeros(b, 1, e, device=x.device, dtype=x.dtype)
        x = torch.cat((cls, x), dim=1)                     # (B, N+1, emb)
        x = x * math.sqrt(self.emb_size)
        x = self.position(x)
        trans = self.trans(x)

        # Read out the class token as the final feature vector.
        features = trans[:, 0, :]                          # (B, emb)
        features = self.pre_logit_drop(features)
        return features
