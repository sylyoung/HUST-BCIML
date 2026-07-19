# stages.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Stage abstract base classes — the five plug-in kinds.

An *algorithm* in hustbciml is a named composition of stage plug-ins:

    Aligner -> Augmenter -> Backbone -> Head , driven by a Strategy.

The registry resolves each stage from ``algorithms/<group>/<Name>.py`` where
the file name equals the class name equals the CLI key. Declarative class
attributes (``requires_labels``, ``supports_online``, ``mode`` ...) let the
pipeline/protocol validate a composition before running it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .batch import EEGEpochs, EEGBatch


class Aligner(ABC):
    """Numpy, per-domain signal alignment (e.g. Euclidean Alignment).

    Operates on ``EEGEpochs`` before tensors are formed. ``fit`` estimates
    per-domain reference statistics on the *training* domains only; a fresh
    aligner is fit on each target subject at test time (offline) or updated
    incrementally (online).
    """

    requires_labels: bool = False   # label-alignment style methods set True
    supports_online: bool = False   # can update per-sample in a stream

    @abstractmethod
    def fit(self, epochs: EEGEpochs) -> "Aligner":
        ...

    @abstractmethod
    def transform(self, epochs: EEGEpochs) -> EEGEpochs:
        ...

    def fit_transform(self, epochs: EEGEpochs) -> EEGEpochs:
        return self.fit(epochs).transform(epochs)


class Augmenter(ABC):
    """Train-only batch transform (e.g. Channel Reflection, time-freq aug)."""

    train_only: bool = True

    @abstractmethod
    def __call__(self, batch: EEGBatch) -> EEGBatch:
        ...


class Backbone(nn.Module, ABC):
    """Feature-extractor. Subclasses set ``self.out_features`` in ``__init__``
    and implement ``forward_features``. ``task_name`` selects the output mode
    so multi-head backbones can dispatch on it."""

    task_name: str = "classification"

    def __init__(self):
        super().__init__()
        self.out_features: int = 0

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, C, T) -> (B, out_features)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class Head(nn.Module, ABC):
    """Maps backbone features to logits. Gradient heads (Linear) train with
    the backbone; classical heads (LDA, MDM) set ``is_gradient=False`` and
    implement ``fit``/``predict`` on numpy features instead."""

    is_gradient: bool = True

    @abstractmethod
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        ...


class Strategy(ABC):
    """Owns the train / adapt / predict *procedure* (ERM, DANN, T-TIME ...).

    Separated from the Exp protocol: ``Exp_<Protocol>`` owns the *data axis*
    (which subjects/sessions are source vs target), the Strategy owns the
    *procedure axis*. One protocol × any strategy.

    mode:
      'gradient' — standard source training then frozen inference.
      'tta'      — test-time adaptation: model keeps updating on the target stream.
      'fit'      — classical fit/predict, no gradient loop.
    """

    mode: str = "gradient"
    # transductive strategies (DANN, MEKT ...) read ctx.target_unlabeled during fit
    uses_target: bool = False

    @abstractmethod
    def fit(self, model: nn.Module, source: EEGEpochs, ctx) -> nn.Module:
        """Train ``model`` on labeled source epochs. Returns the trained model."""

    @abstractmethod
    def predict(self, model: nn.Module, target: EEGEpochs, ctx) -> Tuple[np.ndarray, np.ndarray]:
        """Return (y_pred:(N,), y_score:(N, n_classes))."""
