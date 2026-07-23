# pipeline.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Compose stage plug-ins into a runnable pipeline.

This module is the wiring step between a resolved Config and something that can
actually run. ``build_pipeline`` reads the five plug-in names off the Config,
asks the registry to instantiate each, sizes the backbone and head from the
data-derived dimensions the Exp measured, and bundles the result into a
``Pipeline``.

The one piece that needs care is the model. A backbone and a head are separate
plug-ins, but a strategy wants a single module to train and to run. So they are
joined into a ``PipelineModel`` whose forward returns ``(features, logits)``
rather than logits alone. Returning both is the same contract the original
DeepTransferEEG ``FC_xy`` exposed. Strategies that only classify read the
logits and ignore the features, while domain-adaptation and test-time-adaptation
strategies that operate on the embedding read the features, all without any
change to the forward signature.
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
    """Backbone followed by head, returning ``(feats, logits)``.

    Forward takes the ``(B, 1, C, T)`` batch tensor, runs the backbone to get
    ``feats`` of shape ``(B, out_features)``, and runs the head to get ``logits``
    of shape ``(B, n_classes)``. Both are returned so a strategy can use either.
    """

    def __init__(self, backbone: Backbone, head: Head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor):
        feats = self.backbone.forward_features(x)   # (B, 1, C, T) -> (B, out_features)
        logits = self.head(feats)                   # (B, out_features) -> (B, n_classes)
        return feats, logits


@dataclass
class Pipeline:
    """The fully assembled, ready-to-run composition for one Config.

    It bundles the four data-flow stages plus the driver: the numpy ``aligner``,
    the batch ``augmenter``, the ``model`` (backbone+head), the ``strategy`` that
    drives training and prediction, and the ``cfg`` they were built from. The Exp
    takes this bundle and runs it against its source and target splits.
    """
    aligner: Aligner
    augmenter: Augmenter
    model: PipelineModel
    strategy: Strategy
    cfg: Config


def build_pipeline(cfg: Config) -> Pipeline:
    """Instantiate and wire every stage named in ``cfg`` into a ``Pipeline``.

    Order matters here. The backbone is built first because its architecture
    depends on the data-derived dimensions, and it is the backbone that reports
    ``out_features``. The head is built second and sized from that number, so
    the two always fit together no matter which backbone was chosen. The aligner,
    augmenter, and strategy carry no cross-stage sizing and are built plainly.

    Requires the Exp to have already measured the dataset. The data-derived dims
    on ``cfg`` start at 0 and are meaningless until ``Exp._get_data`` fills them,
    so the guard below refuses to build a mis-sized model.
    """
    if not cfg.n_chans or not cfg.n_times or not cfg.n_classes:
        raise ValueError(
            "data-derived dims are unset; Exp._get_data must fill "
            "n_chans/n_times/n_classes/sfreq before build_pipeline"
        )

    # Aligner takes no data dims: it works on raw (C, T) trials per subject.
    aligner: Aligner = registry.build("aligners", cfg.aligner)
    # Augmenter gets montage and paradigm context so montage-aware augmentations
    # (e.g. left/right channel reflection) know the electrode layout.
    augmenter: Augmenter = registry.build(
        "augmenters", cfg.augmenter,
        ch_names=cfg.ch_names, n_classes=cfg.n_classes, sfreq=cfg.sfreq,
    )

    # Backbone is sized from the data (n_chans, n_times, n_classes, sfreq). The
    # F1/D/F2/dropout knobs are the EEGNet family's; backbones that do not use
    # them just ignore the extra kwargs.
    backbone: Backbone = registry.build(
        "models", cfg.backbone,
        n_chans=cfg.n_chans, n_times=cfg.n_times,
        n_classes=cfg.n_classes, sfreq=cfg.sfreq,
        F1=cfg.F1, D=cfg.D, F2=cfg.F2, dropout=cfg.dropout,
    )
    # Head input width is the backbone's output width, read back off the built
    # backbone. This is the coupling that lets any head follow any backbone.
    head: Head = registry.build(
        "heads", cfg.head,
        in_features=backbone.out_features, n_classes=cfg.n_classes,
    )
    model = PipelineModel(backbone, head)

    # Strategy is the driver; it is built last and left un-parameterized here
    # because it reads its hyperparameters from the config at run time.
    strategy: Strategy = registry.build("strategies", cfg.strategy)

    # composition sanity checks
    if aligner.requires_labels and cfg.protocol == "cross_subject":
        # A label-alignment aligner needs the target's labels to align it. Under
        # cross-subject that is available offline (all target labels known ahead
        # of scoring) but not in a live online stream. Placeholder for the online
        # guard; harmless offline, hence the bare pass.
        pass
    return Pipeline(aligner=aligner, augmenter=augmenter, model=model,
                    strategy=strategy, cfg=cfg)
