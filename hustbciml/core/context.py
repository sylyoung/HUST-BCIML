# context.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Run context handed to a Strategy — everything it needs beyond the model
and the data (config, device, the composed augmenter and aligner, a logger).
Keeping this in ``core`` avoids a strategy->exp import cycle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .batch import EEGEpochs
from .config import Config
from .stages import Aligner, Augmenter


@dataclass
class RunContext:
    cfg: Config
    device: torch.device
    augmenter: Augmenter
    aligner: Aligner
    log: Callable[[str], None] = print
    # aligned, label-masked target epochs — set by the Exp for transductive
    # (uses_target) strategies such as DANN; None otherwise.
    target_unlabeled: Optional[EEGEpochs] = None
