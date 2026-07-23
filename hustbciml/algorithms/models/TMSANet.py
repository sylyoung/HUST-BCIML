# ===========================================================================
# TMSANet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: not publicly released (paper: https://www.sciencedirect.com/science/article/pii/S1746809424012473)
#
# Reference (IEEE BibTeX):
#   @Article{Zhao2025a,
#     author  = {Zhao, Qian and Zhu, Weina},
#     journal = {Biomedical Signal Processing and Control},
#     title   = {{TMSA}-Net: A Novel Attention Mechanism for Improved Motor Imagery {EEG} Signal Processing},
#     year    = {2025},
#     pages   = {107189},
#     volume  = {102},
#     doi     = {10.1016/j.bspc.2024.107189},
#   }
# ===========================================================================
"""TMSA-Net (Zhao and Zhu, 2025) is a temporal multi-scale attention network
for motor imagery EEG decoding.

The model has three stages that mirror the paper's Section 2 (Methods):

  * **Feature extraction** (paper Section 2.2, the convolution module). Two
    parallel temporal convolutions with kernels 31 and 15 are summed, then a
    depthwise spatial convolution mixes across all channels, followed by GELU,
    batch normalization, and temporal average pooling. This turns the raw
    (C, T) signal into ``embed_dim`` feature maps over a shorter time axis.
  * **Temporal Multi-Scale Attention transformer** (paper Section 2.3, the TMSA
    encoder). A stack of pre-norm transformer blocks. Each block replaces the
    plain key projection with a two-branch attention: a *local* branch whose
    keys come from a multi-scale 1D convolution (kernels 3 and 5) of the input,
    and a *global* branch with ordinary linear keys. The two attention outputs
    are added, giving attention that is sensitive to both short local temporal
    patterns and long-range dependencies. A GELU feed-forward sublayer follows.
  * **Classification** (paper Section 2.4). The paper flattens the transformer
    output and applies one linear layer to the class logits. Following the
    benchmark contract that final linear layer is removed here. The flattened
    pre-logit vector of width ``embed_dim * temp_embedding_dim`` is exposed as
    the feature vector, and the shared hustbciml ``Linear`` head produces the
    logits instead.

Ported at the paper's default and best BCIC-IV-2a configuration: ``embed_dim``
= 19, temporal-conv kernels 31 and 15, pooling kernel 50 stride 15, one
transformer block (``depth`` = 1), 4 attention heads, feed-forward expansion
ratio 2, and dropout 0.5. ``temp_embedding_dim`` (the pooled time length) and
therefore ``out_features`` are inferred by a dummy forward so any (C, T) works.

Source: no standalone code repository was publicly released by the authors. The
architecture is reproduced from the paper. The only behaviour-preserving
deviation is that the ``einops.rearrange`` call is rewritten with a plain torch
transpose (identical result for a 3D tensor).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.core.stages import Backbone


# --- Multi-scale 1D convolution (paper Section 2.3, local-key generator) ----
class _MultiScaleConv1d(nn.Module):
    """Apply several 1D convolutions with different kernel sizes and concatenate
    their outputs along the channel axis. Used inside the attention module to
    produce local keys that see short temporal contexts of sizes 3 and 5."""

    def __init__(self, in_channels, out_channels, kernel_sizes, padding):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=p)
            for k, p in zip(kernel_sizes, padding)
        ])
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        conv_outs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)   # concatenate the multi-scale branches
        out = self.bn(out)
        out = self.dropout(out)
        return out


# --- Temporal multi-scale attention (paper Section 2.3, the TMSA mechanism) --
class _MultiHeadedAttention(nn.Module):
    """Multi-head attention that combines a local and a global branch. The local
    branch derives its keys from a multi-scale convolution of the sequence, the
    global branch derives its keys from an ordinary linear projection, and the
    two attention results are summed. This is the paper's core contribution."""

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head    # per-head width for keys
        self.d_v = d_model // n_head    # per-head width for values
        self.n_head = n_head

        # multi-scale conv over the sequence supplies the local keys
        kernel_sizes = [3, 5]
        padding = [1, 2]
        self.multi_scale_conv_k = _MultiScaleConv1d(d_model, d_model, kernel_sizes, padding)

        # linear projections: query, local key, global key, value, output
        self.w_q = nn.Linear(d_model, n_head * self.d_k)
        self.w_k_local = nn.Linear(d_model * len(kernel_sizes), n_head * self.d_k)
        self.w_k_global = nn.Linear(d_model, n_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_head * self.d_v)
        self.w_o = nn.Linear(n_head * self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):   # each (B, seq_len, d_model)
        bsz = query.size(0)

        # local keys: multi-scale conv wants (B, d_model, seq_len)
        key_local = key.transpose(1, 2)
        key_local = self.multi_scale_conv_k(key_local).transpose(1, 2)

        # project and split into heads
        q = self.w_q(query).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
        k_local = self.w_k_local(key_local).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
        k_global = self.w_k_global(key).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(bsz, -1, self.n_head, self.d_v).transpose(1, 2)

        # local attention (short-range, from multi-scale conv keys)
        scores_local = torch.matmul(q, k_local.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_local = F.softmax(scores_local, dim=-1)
        attn_local = self.dropout(attn_local)
        x_local = torch.matmul(attn_local, v)

        # global attention (long-range, from ordinary linear keys)
        scores_global = torch.matmul(q, k_global.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_global = F.softmax(scores_global, dim=-1)
        attn_global = self.dropout(attn_global)
        x_global = torch.matmul(attn_global, v)

        # combine local and global attention, then project out
        x = x_local + x_global
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.n_head * self.d_v)
        return self.w_o(x)


# --- Feed-forward sublayer (paper Section 2.3, transformer block) -----------
class _FeedForward(nn.Module):
    """Two-layer position-wise feed-forward network with GELU."""

    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


# --- One transformer encoder block (paper Section 2.3) ----------------------
class _TransformerEncoder(nn.Module):
    """Pre-norm transformer block: residual multi-scale attention followed by a
    residual feed-forward network."""

    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = _MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.feed_forward = _FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, data):
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)   # residual attention
        res = self.layernorm2(out)
        output = out + self.feed_forward(res)                  # residual feed-forward
        return output


# --- Convolutional feature extractor (paper Section 2.2) --------------------
class _ExtractFeature(nn.Module):
    """Temporal + spatial convolutional front end. Two temporal convolutions
    (kernels 31 and 15) are summed, a depthwise spatial convolution mixes over
    all channels, and average pooling shortens the time axis."""

    def __init__(self, num_channels, num_samples, embed_dim, pool_size, pool_stride):
        super().__init__()
        # two temporal convolutions with different kernel sizes
        self.temp_conv1 = nn.Conv2d(1, embed_dim, (1, 31), padding=(0, 15))
        self.temp_conv2 = nn.Conv2d(1, embed_dim, (1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(embed_dim)

        # spatial convolution across every channel at once
        self.spatial_conv1 = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1), padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.glu = nn.GELU()
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)   # temporal pooling

    def forward(self, x):                        # (B, num_channels, num_samples)
        x = x.unsqueeze(dim=1)                   # -> (B, 1, num_channels, num_samples)
        x1 = self.temp_conv1(x)                  # temporal conv, kernel 31
        x2 = self.temp_conv2(x)                  # temporal conv, kernel 15
        x = x1 + x2                              # fuse the two temporal scales
        x = self.bn1(x)
        x = self.spatial_conv1(x)                # spatial conv over channels
        x = self.glu(x)
        x = self.bn2(x)
        x = x.squeeze(dim=2)                     # -> (B, embed_dim, num_samples)
        x = self.avg_pool(x)                     # -> (B, embed_dim, temp_embedding_dim)
        return x


# --- Transformer stack (paper Section 2.3) ----------------------------------
class _TransformerModule(nn.Module):
    """Stack of ``depth`` TMSA transformer blocks over the pooled time axis."""

    def __init__(self, embed_dim, num_heads, fc_ratio, depth, attn_drop, fc_drop):
        super().__init__()
        self.transformer_encoders = nn.ModuleList([
            _TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop)
            for _ in range(depth)
        ])

    def forward(self, x):                        # (B, embed_dim, seq_len)
        x = x.transpose(1, 2)                    # -> (B, seq_len, embed_dim); was einops rearrange
        for encoder in self.transformer_encoders:
            x = encoder(x)
        x = x.transpose(1, 2)                    # -> (B, embed_dim, seq_len)
        x = x.unsqueeze(dim=2)                   # -> (B, embed_dim, 1, seq_len)
        return x


class TMSANet(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 embed_dim: int = 19, pool_size: int = 50, pool_stride: int = 15,
                 num_heads: int = 4, fc_ratio: int = 2, depth: int = 1,
                 attn_drop: float = 0.5, fc_drop: float = 0.5, **_):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times

        # feature extractor + TMSA transformer stack (paper Sections 2.2-2.3)
        self.extract_feature = _ExtractFeature(n_chans, n_times, embed_dim, pool_size, pool_stride)
        self.dropout = nn.Dropout()
        self.transformer_module = _TransformerModule(
            embed_dim, num_heads, fc_ratio, depth, attn_drop, fc_drop)

        # infer the pre-logit feature width via a dummy forward, so the backbone
        # is dataset-agnostic (the paper's final Linear to n_classes is removed)
        with torch.no_grad():
            feas = self._embed(torch.zeros(1, 1, n_chans, n_times))
        self.out_features = feas.reshape(1, -1).shape[1]

    def _embed(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, C, T) -> (B, embed_dim, 1, seq_len)
        x = x.squeeze(1)                          # -> (B, C, T)
        x = self.extract_feature(x)               # convolutional front end
        x = self.dropout(x)
        return self.transformer_module(x)         # TMSA transformer stack

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, C, T)
        feas = self._embed(x)                     # (B, embed_dim, 1, seq_len)
        return feas.reshape(feas.size(0), -1)     # flatten to (B, out_features)
