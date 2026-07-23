# collate.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Turn ``EEGEpochs`` (numpy) into ``EEGBatch`` minibatches (torch).

This is the one crossing from the numpy data-provider world into the torch
forward-contract. It copies the three per-trial arrays into tensors, adds the
leading singleton channel dimension that the 2-D-convolution backbones expect,
and packs them into the ``EEGBatch`` that every backbone, head, and strategy
consumes. See ``hustbciml/core/batch.py`` for the two containers.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from hustbciml.core.batch import EEGBatch, EEGEpochs


def epochs_to_xyd(epochs: EEGEpochs):
    """Convert the whole container to three tensors: ``x`` (N, 1, C, T) float32,
    ``y`` (N,) int64, and ``domain`` (N,) int64.

    ``unsqueeze(1)`` inserts the singleton dimension so each ``(C, T)`` trial
    reads as a one-channel image to the 2-D convolutions. ``ascontiguousarray``
    guarantees a contiguous C-order buffer before ``from_numpy``, which shares
    memory with the array rather than copying, and the dtypes match the
    ``EEGBatch`` contract (float32 signal, int64 labels and subject ids).
    """
    x = torch.from_numpy(np.ascontiguousarray(epochs.X, dtype=np.float32)).unsqueeze(1)
    y = torch.from_numpy(np.ascontiguousarray(epochs.y, dtype=np.int64))
    d = torch.from_numpy(np.ascontiguousarray(epochs.domain, dtype=np.int64))
    return x, y, d


def epochs_to_batch(epochs: EEGEpochs) -> EEGBatch:
    """Pack an entire ``EEGEpochs`` into a single ``EEGBatch`` (no minibatching).
    Used when a stage wants every trial at once, for example to featurise a
    whole subject."""
    x, y, d = epochs_to_xyd(epochs)
    return EEGBatch(x, y, d)


def iterate_batches(epochs: EEGEpochs, batch_size: int, shuffle: bool = True,
                    drop_last: bool = False, seed: int = 0) -> Iterator[EEGBatch]:
    """Yield ``EEGBatch`` minibatches for a training or inference loop.

    Shuffling is seeded through a dedicated ``torch.Generator`` so the trial
    order is reproducible from the run seed and does not disturb the global RNG.
    ``drop_last`` discards the final short batch, which matters when a layer
    such as batch norm needs a full-size batch. Indexing the pre-built ``x`` /
    ``y`` / ``d`` tensors keeps the per-trial rows aligned inside every batch.
    """
    x, y, d = epochs_to_xyd(epochs)
    n = len(epochs)
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(n, generator=g)
    else:
        order = torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        if drop_last and len(idx) < batch_size:
            break
        yield EEGBatch(x[idx], y[idx], d[idx])
