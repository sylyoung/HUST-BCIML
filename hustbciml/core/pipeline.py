# pipeline.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Compose stage plug-ins into a runnable pipeline.

``build_pipeline`` resolves the aligner / augmenter / backbone / head /
strategy named in the Config, sizes the backbone from the data-derived dims
the Exp injected, and wires backbone+head into a ``PipelineModel`` whose
forward returns ``(features, logits)`` — the same contract the original
DeepTransferEEG ``FC_xy`` exposed, so DA/TTA strategies that need features
work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from . import registry
from .config import Config
from .stages import Aligner, Augmenter, Backbone, Head, Strategy


class PipelineModel(nn.Module):
    """backbone + head, returning ``(feats, logits)``."""

    def __init__(self, backbone: Backbone, head: Head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor):
        feats = self.backbone.forward_features(x)
        logits = self.head(feats)
        return feats, logits


@dataclass
class Pipeline:
    aligner: Aligner
    augmenter: Augmenter
    model: PipelineModel
    strategy: Strategy
    cfg: Config


def build_pipeline(cfg: Config) -> Pipeline:
    if not cfg.n_chans or not cfg.n_times or not cfg.n_classes:
        raise ValueError(
            "data-derived dims are unset; Exp._get_data must fill "
            "n_chans/n_times/n_classes/sfreq before build_pipeline"
        )

    aligner: Aligner = registry.build("aligners", cfg.aligner)
    augmenter: Augmenter = registry.build(
        "augmenters", cfg.augmenter,
        ch_names=cfg.ch_names, n_classes=cfg.n_classes,
    )

    backbone: Backbone = registry.build(
        "models", cfg.backbone,
        n_chans=cfg.n_chans, n_times=cfg.n_times,
        n_classes=cfg.n_classes, sfreq=cfg.sfreq,
        F1=cfg.F1, D=cfg.D, F2=cfg.F2, dropout=cfg.dropout,
    )
    head: Head = registry.build(
        "heads", cfg.head,
        in_features=backbone.out_features, n_classes=cfg.n_classes,
    )
    model = PipelineModel(backbone, head)

    strategy: Strategy = registry.build("strategies", cfg.strategy)

    # composition sanity checks
    if aligner.requires_labels and cfg.protocol == "cross_subject":
        # label alignment needs target labels; fine offline, flag online
        pass
    return Pipeline(aligner=aligner, augmenter=augmenter, model=model,
                    strategy=strategy, cfg=cfg)
