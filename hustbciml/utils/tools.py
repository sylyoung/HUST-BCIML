# tools.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Small training helpers (kept close to the DeepTransferEEG originals)."""
from __future__ import annotations

import copy

import numpy as np


class EarlyStopping:
    """Track a validation score, keep the best model state, and stop after
    ``patience`` non-improving checks. ``mode='max'`` for accuracy/AUC.

    The training loop calls ``step`` once per validation check. On every new
    best the current weights are snapshotted, and ``restore`` at the end loads
    that snapshot back, so training returns the best-validation model rather
    than the last one. ``best`` doubles as the model's validation score that the
    experiment records for hyperparameter selection. Use ``mode='min'`` for a
    loss, where lower is better.
    """

    def __init__(self, patience: int = 20, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None          # best validation score seen so far
        self.best_state = None    # deep-copied weights at that best score
        self.counter = 0          # checks since the last improvement
        self.should_stop = False  # set once counter reaches patience

    def _improved(self, value: float) -> bool:
        # The very first score always counts as an improvement. After that the
        # comparison direction depends on whether higher or lower is better.
        if self.best is None:
            return True
        return value > self.best if self.mode == "max" else value < self.best

    def step(self, value: float, model) -> bool:
        """Return True if this was a new best."""
        if self._improved(value):
            # New best: record the score, snapshot the weights (a deep copy so
            # later in-place updates do not mutate it), and reset the patience
            # counter.
            self.best = value
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True
        # No improvement: count it, and once ``patience`` such checks pass in a
        # row raise the stop flag for the training loop to read.
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

    def restore(self, model) -> None:
        """Load the best-scoring weights back into ``model`` (no-op if ``step``
        was never called with an improvement)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
