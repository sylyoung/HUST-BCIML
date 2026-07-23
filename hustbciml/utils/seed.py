# seed.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Reproducibility and device selection. Every run seeds all RNGs once through
``fix_random_seed`` and picks its compute device through ``resolve_device``, so
a benchmark result is reproducible from its seed and portable across machines.
"""
import os
import random

import numpy as np
import torch


def fix_random_seed(seed: int) -> None:
    """Seed every random-number source the pipeline touches, from one seed.

    Python's ``random``, NumPy, and both the CPU and CUDA torch generators are
    all seeded, because different stages draw from different ones (data shuffling
    from torch, augmentation noise from NumPy, and so on). ``PYTHONHASHSEED``
    fixes hash-based ordering in fresh subprocesses. The last two lines force
    cuDNN to pick deterministic convolution algorithms instead of
    auto-tuning the fastest one, which trades a little GPU speed for
    run-to-run reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a device string into a ``torch.device``.

    With ``"auto"`` it prefers a CUDA GPU, then Apple Silicon MPS if that build
    of torch exposes it, and falls back to CPU, so the same config runs on the
    GPU server and on a laptop. Any other value (for example ``"cuda:1"`` or
    ``"cpu"``) is passed through as an explicit choice.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)
