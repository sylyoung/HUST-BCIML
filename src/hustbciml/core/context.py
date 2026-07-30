# context.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Run context handed to a Strategy — everything it needs beyond the model
and the data.

A Strategy's ``fit`` and ``predict`` take the model and one split of epochs as
positional arguments. Everything else they might need is bundled here and passed
as ``ctx``: the resolved config, the torch device, the already-composed
augmenter and aligner from the pipeline, a log callback, and, for transductive
methods, the unlabeled target. Passing one context object keeps the Strategy
method signatures stable no matter how much side information a given method uses.

This lives in ``core`` on purpose. The Exp constructs a ``RunContext`` and the
Strategy consumes it, but if the type were defined in the exp package a Strategy
would have to import exp to be typed against it, and exp already imports
strategies. Defining it here breaks that would-be import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .batch import EEGEpochs
from .config import Config
from .stages import Aligner, Augmenter


@dataclass
class RunContext:
    """Side information a Strategy reads during ``fit`` and ``predict``."""

    cfg: Config                       # the resolved run configuration (all knobs)
    device: torch.device             # where tensors and the model live
    augmenter: Augmenter             # the composed train-time batch augmenter
    aligner: Aligner                 # the composed per-domain signal aligner
    log: Callable[[str], None] = print   # where progress messages go (defaults to stdout)
    # Aligned, label-masked target epochs. The Exp fills this only for
    # transductive strategies (those with ``uses_target = True``, such as DANN
    # or MEKT) that must see the target distribution while training. It stays
    # None for ordinary strategies, which never look at the target during fit.
    target_unlabeled: Optional[EEGEpochs] = None
