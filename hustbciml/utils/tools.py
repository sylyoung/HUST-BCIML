# tools.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Small training helpers (kept close to the DeepTransferEEG originals)."""
from __future__ import annotations

import copy

import numpy as np


class EarlyStopping:
    """Track a validation score; keep the best model state; stop after
    ``patience`` non-improving checks. ``mode='max'`` for accuracy/AUC."""

    def __init__(self, patience: int = 20, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        return value > self.best if self.mode == "max" else value < self.best

    def step(self, value: float, model) -> bool:
        """Return True if this was a new best."""
        if self._improved(value):
            self.best = value
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

    def restore(self, model) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
