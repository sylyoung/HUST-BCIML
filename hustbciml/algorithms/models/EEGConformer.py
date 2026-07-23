# ===========================================================================
# EEGConformer.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/eeyhsong/EEG-Conformer
#
# Reference (IEEE BibTeX):
#   @Article{Song2023,
#     author  = {Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong},
#     journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
#     title   = {{EEG} {C}onformer: Convolutional Transformer for {EEG} Decoding and Visualization},
#     year    = {2023},
#     pages   = {710-719},
#     volume  = {31},
#     doi     = {10.1109/TNSRE.2022.3230250},
#   }
# ===========================================================================
"""EEG Conformer (Song et al., 2022, IEEE TNSRE) — convolutional tokenizer + a
transformer encoder for EEG decoding.

A ShallowConvNet-style front end (temporal conv -> spatial conv -> BN -> ELU ->
average pool) turns the signal into a sequence of temporal tokens; a standard
transformer encoder then models global temporal dependencies across those
tokens; a small MLP compresses the flattened sequence into features. Backbone
only: ``forward_features`` returns the pre-logit features and the shared Linear
head maps them to classes, so the final linear layer of the paper's classifier
is simply that head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hustbciml.core.stages import Backbone


class EEGConformer(Backbone):
    task_name = "classification"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 emb: int = 40, depth: int = 6, heads: int = 10,
                 mlp_hidden: int = 256, feat_dim: int = 32, drop: float = 0.5, **_):
        super().__init__()
        # Convolutional tokenizer of the paper: a ShallowConvNet-style front end
        # that turns the raw signal into a sequence of temporal tokens of width
        # `emb`. Input is (B, 1, C, T).
        self.patch = nn.Sequential(
            nn.Conv2d(1, emb, (1, 25)),               # temporal conv: band-pass-like filters
            nn.Conv2d(emb, emb, (n_chans, 1)),        # spatial conv over all electrodes
            nn.BatchNorm2d(emb),
            nn.ELU(),
            # Average-pool over a 75-sample window (stride 15) smooths activity
            # and sets the token rate, so each surviving time step is one token.
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            nn.Dropout(drop),
        )
        self.proj = nn.Conv2d(emb, emb, (1, 1))   # token embedding projection
        # Transformer encoder of the paper: standard multi-head self-attention
        # blocks that model global dependencies across the temporal tokens.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, dim_feedforward=emb * 4,
            dropout=drop, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Token count depends on T, so measure it with a dummy forward before
        # sizing the MLP that reads the flattened sequence.
        with torch.no_grad():
            n_tok = self._embed(torch.zeros(1, 1, n_chans, n_times)).shape[1]
        # Compression MLP of the paper's head, folded in here so `out_features`
        # is `feat_dim` and the shared Linear head produces the class logits.
        self.fc = nn.Sequential(
            nn.Linear(n_tok, mlp_hidden), nn.ELU(), nn.Dropout(drop),
            nn.Linear(mlp_hidden, feat_dim), nn.ELU(), nn.Dropout(drop),
        )
        self.out_features = feat_dim

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch(x)                 # (B, emb, 1, tokens): conv tokenizer
        z = self.proj(z)                  # 1x1 token embedding projection
        z = z.squeeze(2).transpose(1, 2)  # (B, tokens, emb): tokens as a sequence
        z = self.transformer(z)           # self-attention over the tokens
        return z.flatten(1)               # (B, tokens * emb): flatten the sequence

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self._embed(x))    # tokenize + encode, then compress to features
