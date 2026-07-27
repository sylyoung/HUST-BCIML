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

# The strategy ``mode`` values the protocols know how to run. Anything else is a
# typo or an unimplemented procedure, and must fail before a run rather than fall
# through to the default path.
VALID_STRATEGY_MODES = frozenset({"gradient", "fit", "tta"})


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
    # Augmenter gets montage, class and paradigm context so montage-aware
    # augmentations (e.g. left/right channel reflection) know both the electrode
    # layout and what the classes actually are — a reflection may only swap
    # labels when the two classes are mirror images of each other.
    augmenter: Augmenter = registry.build(
        "augmenters", cfg.augmenter,
        ch_names=cfg.ch_names, n_classes=cfg.n_classes, sfreq=cfg.sfreq,
        classes=cfg.classes,
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

    _validate_composition(cfg, aligner, strategy)
    return Pipeline(aligner=aligner, augmenter=augmenter, model=model,
                    strategy=strategy, cfg=cfg)


def _validate_composition(cfg: Config, aligner: Aligner, strategy: Strategy) -> None:
    """Reject compositions whose result would carry a misleading label.

    Each check below exists because the alternative is not a crash but a number:
    the run completes, writes a metrics file, and lands on the leaderboard
    describing something other than what executed.
    """
    # 1. Strategy mode. The protocol branches on this string to decide whether the
    #    target is aligned offline or streamed to the strategy. An unrecognised
    #    value silently takes the offline branch, so a mistyped or unsupported mode
    #    would publish a number measured under the wrong procedure.
    if strategy.mode not in VALID_STRATEGY_MODES:
        raise ValueError(
            f"strategy {cfg.strategy!r} declares mode {strategy.mode!r}; expected one of "
            f"{sorted(VALID_STRATEGY_MODES)}"
        )

    # 2. Label-requiring aligners under a held-out-target protocol. Aligning the
    #    held-out subject with its own labels is leakage, and the LOSO score would
    #    be inflated with nothing to show for it in the results file. Every shipped
    #    aligner sets ``requires_labels = False``, so this guards what comes next.
    if aligner.requires_labels and cfg.protocol == "cross_subject":
        raise ValueError(
            f"aligner {cfg.aligner!r} declares requires_labels=True, but the "
            f"{cfg.protocol!r} protocol hides the target's labels: aligning the held-out "
            f"subject would need labels it must not see. Use a label-free aligner, or add "
            f"a calibrated protocol that grants a labelled target slice explicitly."
        )

    # 3. Online alignment under test-time adaptation. A TTA strategy walks the raw
    #    target stream and aligns it incrementally, which only some aligners can do.
    #    Without this check a composition like ``--aligner RA --strategy Tent`` runs
    #    and reports itself as RA while the online loop applies EA's update.
    if strategy.mode == "tta" and cfg.aligner != "Identity" and not aligner.supports_online:
        raise ValueError(
            f"strategy {cfg.strategy!r} is test-time (mode='tta') and needs to update the "
            f"alignment reference per trial, but aligner {cfg.aligner!r} sets "
            f"supports_online=False. Use EA (or Identity) for online strategies."
        )

    # 4. Transductive + test-time. ``uses_target`` is served by the Exp as
    #    ``ctx.target_unlabeled``, which is only filled on the offline path; a TTA
    #    strategy declaring it would receive None and quietly run without the target
    #    data its definition requires.
    if strategy.mode == "tta" and getattr(strategy, "uses_target", False):
        raise ValueError(
            f"strategy {cfg.strategy!r} sets both mode='tta' and uses_target=True. The "
            f"protocol supplies ctx.target_unlabeled only to offline strategies, so the "
            f"strategy would silently receive None. A TTA strategy reads the target from "
            f"the stream it is handed in predict()."
        )
