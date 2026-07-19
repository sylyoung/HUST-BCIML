# batch.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Data contracts shared by every stage.

``EEGEpochs`` is the dataset-level numpy container (what aligners and data
splitters operate on). ``EEGBatch`` is the torch forward-contract fed to
backbones/heads/strategies. This split keeps the data provider (numpy)
separate from the ``forward`` signature (tensors), and is specialized for
EEG: a single spatial-temporal map ``(C, T)`` per trial,
plus a ``domain`` (subject) id so domain-adaptation methods (DANN, MEKT)
need no signature change.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

import numpy as np
import torch

UNLABELED = -1  # y value marking a trial whose label is hidden (target/online)


@dataclass
class EEGEpochs:
    """Dataset-level container of trials.

    Attributes
    ----------
    X : (N, C, T) float32 — trials × channels × time samples.
    y : (N,) int64 — class index in ``[0, n_classes)``; ``-1`` = unlabeled.
    domain : (N,) int64 — subject id per trial (the domain axis).
    sfreq : sampling rate in Hz.
    n_classes : number of classes.
    ch_names : channel names (len C) — needed by montage-aware methods.
    paradigm : 'MI' | 'P300' | 'SSVEP' | ...
    classes : human-readable class names (len n_classes).
    """

    X: np.ndarray
    y: np.ndarray
    domain: np.ndarray
    sfreq: float
    n_classes: int
    ch_names: List[str] = field(default_factory=list)
    paradigm: str = "MI"
    classes: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.X = np.ascontiguousarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64).reshape(-1)
        self.domain = np.asarray(self.domain, dtype=np.int64).reshape(-1)
        if not (len(self.X) == len(self.y) == len(self.domain)):
            raise ValueError(
                f"length mismatch: X={len(self.X)} y={len(self.y)} domain={len(self.domain)}"
            )
        if self.X.ndim != 3:
            raise ValueError(f"X must be (N, C, T), got {self.X.shape}")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_times(self) -> int:
        return self.X.shape[2]

    def domains(self) -> np.ndarray:
        """Unique domain (subject) ids, sorted."""
        return np.unique(self.domain)

    def select(self, idx) -> "EEGEpochs":
        """Return a new EEGEpochs with the selected trials (idx = mask or index array)."""
        idx = np.asarray(idx)
        return replace(self, X=self.X[idx], y=self.y[idx], domain=self.domain[idx])

    def with_X(self, X: np.ndarray) -> "EEGEpochs":
        """Return a copy with X replaced (metadata preserved). Used by aligners/augmenters."""
        return replace(self, X=np.ascontiguousarray(X, dtype=np.float32))

    def has_labels(self) -> bool:
        return bool(np.all(self.y != UNLABELED))


@dataclass
class EEGBatch:
    """torch forward-contract: what a Backbone/Head/Strategy consumes.

    x : (B, 1, C, T) float32 — leading singleton channel dim for 2-D convs.
    y : (B,) int64 — labels (``-1`` where unknown).
    domain : (B,) int64 — subject ids (for adversarial/DA strategies).
    """

    x: torch.Tensor
    y: torch.Tensor
    domain: torch.Tensor

    def to(self, device) -> "EEGBatch":
        return EEGBatch(self.x.to(device), self.y.to(device), self.domain.to(device))

    def __len__(self) -> int:
        return self.x.shape[0]


def epochs_to_tensor(epochs: EEGEpochs) -> torch.Tensor:
    """(N, C, T) -> (N, 1, C, T) float32 tensor."""
    x = torch.from_numpy(np.ascontiguousarray(epochs.X, dtype=np.float32))
    return x.unsqueeze(1)
