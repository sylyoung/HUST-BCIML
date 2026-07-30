# batch.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Data contracts shared by every stage.

Two containers carry EEG through the whole benchmark, and they draw the line
between the numpy world and the torch world.

``EEGEpochs`` is the dataset-level numpy container. It is what a dataset loader
produces, what data splitters slice, and what aligners (EA and friends) read
and rewrite. Its trials are stored as ``(N, C, T)``: N trials, each a spatial
map of C electrode channels by T time samples. Because alignment and splitting
are per-trial and per-subject bookkeeping, they never need gradients, so this
half of the data flow stays in numpy.

``EEGBatch`` is the torch forward-contract. It is the exact tuple a Backbone,
Head, or Strategy sees inside ``forward``. Its ``x`` gains a leading singleton
dimension, ``(B, 1, C, T)``, so the ``(C, T)`` map looks like a one-channel
image to the 2-D convolutions that EEGNet-style backbones use.

Splitting the two matters for one reason. The data provider stays numpy and the
``forward`` signature stays tensors, so neither side has to know about the
other. Both are specialized for EEG in the same two ways. First, a trial is a
single spatial-temporal map ``(C, T)`` rather than a generic feature vector.
Second, every trial carries a ``domain`` (subject) id alongside its label, so
domain-adaptation methods such as DANN and MEKT can read which subject a trial
came from without any change to the ``forward`` signature.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

import numpy as np
import torch

# Sentinel label. A trial with ``y == UNLABELED`` is one whose class is hidden
# from the method on purpose. This is how the target subject looks to a
# cross-subject or online run before its true labels are revealed for scoring.
UNLABELED = -1  # y value marking a trial whose label is hidden (target/online)


@dataclass
class EEGEpochs:
    """Dataset-level container of trials (the numpy half of the data flow).

    The three per-trial arrays ``X``, ``y``, and ``domain`` are row-aligned:
    row i of each describes the same trial i, and ``__post_init__`` enforces
    that they share length N. Everything else is dataset-wide metadata.

    Attributes
    ----------
    X : (N, C, T) float32 — trials × channels × time samples. C and T are the
        same for every trial in the container (a fixed montage and window).
    y : (N,) int64 — class index in ``[0, n_classes)``; ``-1`` (``UNLABELED``)
        marks a trial whose label is withheld.
    domain : (N,) int64 — subject id per trial. This is the domain axis that
        per-subject alignment and domain-adaptation strategies key off.
    sfreq : sampling rate in Hz (needed to interpret the T axis in seconds).
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
        # Normalize dtypes and shapes on construction so every downstream stage
        # can assume float32 (N, C, T) trials plus flat int64 label and domain
        # vectors, instead of each stage re-checking. The two invariants below
        # (equal length, X is 3-D) are the container's core contract.
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
        """Return a new EEGEpochs holding only the selected trials.

        ``idx`` is either a boolean mask of length N or an array of integer row
        indices. The three per-trial arrays are sliced together so the result
        stays row-aligned, and the metadata is carried over unchanged. This is
        the primitive that data splitters use to carve source from target and
        train from validation.
        """
        idx = np.asarray(idx)
        return replace(self, X=self.X[idx], y=self.y[idx], domain=self.domain[idx])

    def with_X(self, X: np.ndarray) -> "EEGEpochs":
        """Return a copy with the trial array replaced and all metadata kept.

        This is the workhorse for stages that rewrite the signal but leave the
        trial ordering, labels, and domains alone. An aligner whitens ``X`` and
        an augmenter perturbs it, then hands back a new container through here.
        """
        return replace(self, X=np.ascontiguousarray(X, dtype=np.float32))

    def has_labels(self) -> bool:
        """True only if every trial is labeled (no ``UNLABELED`` present).

        A supervised strategy calls this to confirm it may train on this split.
        A target split in a cross-subject run returns False here.
        """
        return bool(np.all(self.y != UNLABELED))


@dataclass
class EEGBatch:
    """torch forward-contract: what a Backbone/Head/Strategy consumes.

    This is the tensor mirror of one minibatch of ``EEGEpochs``. The three
    fields stay row-aligned exactly as in the numpy container, so ``x[i]``,
    ``y[i]``, and ``domain[i]`` still describe the same trial.

    x : (B, 1, C, T) float32 — the ``(C, T)`` map with a leading singleton
        channel dimension, which lets an EEGNet-style backbone treat each trial
        as a single-channel image and run 2-D convolutions over it.
    y : (B,) int64 — labels (``-1`` where unknown).
    domain : (B,) int64 — subject ids, read by adversarial and domain-adaptation
        strategies that condition on which subject a trial came from.
    """

    x: torch.Tensor
    y: torch.Tensor
    domain: torch.Tensor

    def to(self, device) -> "EEGBatch":
        """Move all three tensors to ``device`` and return a new batch."""
        return EEGBatch(self.x.to(device), self.y.to(device), self.domain.to(device))

    def __len__(self) -> int:
        return self.x.shape[0]


def epochs_to_tensor(epochs: EEGEpochs) -> torch.Tensor:
    """Turn a container's trials into the backbone-ready ``x`` tensor.

    Takes the numpy ``(N, C, T)`` trial array and returns an ``(N, 1, C, T)``
    float32 tensor. The inserted axis at position 1 is the singleton channel
    dimension ``EEGBatch.x`` expects, so 2-D convolutions see one input plane.
    """
    x = torch.from_numpy(np.ascontiguousarray(epochs.X, dtype=np.float32))
    return x.unsqueeze(1)
