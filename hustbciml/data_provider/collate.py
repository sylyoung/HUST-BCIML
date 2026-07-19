# collate.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Turn ``EEGEpochs`` (numpy) into ``EEGBatch`` minibatches (torch)."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from hustbciml.core.batch import EEGBatch, EEGEpochs


def epochs_to_xyd(epochs: EEGEpochs):
    """(N,C,T)->(N,1,C,T) tensor, plus y and domain tensors."""
    x = torch.from_numpy(np.ascontiguousarray(epochs.X, dtype=np.float32)).unsqueeze(1)
    y = torch.from_numpy(np.ascontiguousarray(epochs.y, dtype=np.int64))
    d = torch.from_numpy(np.ascontiguousarray(epochs.domain, dtype=np.int64))
    return x, y, d


def epochs_to_batch(epochs: EEGEpochs) -> EEGBatch:
    x, y, d = epochs_to_xyd(epochs)
    return EEGBatch(x, y, d)


def iterate_batches(epochs: EEGEpochs, batch_size: int, shuffle: bool = True,
                    drop_last: bool = False, seed: int = 0) -> Iterator[EEGBatch]:
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
