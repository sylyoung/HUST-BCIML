# splitters.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Protocol split rules. Hard invariant: never leak the target across the
split. Under the cross-subject protocol the source and the validation set are
both drawn from source domains only, and the aligner is fit on the relevant
domain(s) only.

Why validation comes from source subjects and not the target. The whole point
of the cross-subject protocol is to estimate how well a model trained on some
subjects generalises to a brand-new subject it has never seen. If any target
trial or target label were used to pick the model (through a validation split,
early stopping, or hyperparameter tuning), the reported target score would no
longer measure that. So this module only ever splits the source subjects, and
the target subject is returned untouched to be scored once at the very end. The
target's own trials may still be whitened by Euclidean Alignment, because that
step is label-free and per-subject and therefore not leakage. See ``EA.py``.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from hustbciml.core.batch import EEGEpochs


def list_targets(epochs: EEGEpochs) -> List[int]:
    """Every subject id, in sorted order. The cross-subject experiment loops
    over this list and holds each one out in turn as the target fold."""
    return [int(d) for d in epochs.domains()]


def cross_subject(epochs: EEGEpochs, target_id: int) -> Tuple[EEGEpochs, EEGEpochs]:
    """Leave-one-subject-out split: source = all other subjects, target =
    ``target_id``.

    The split is purely by the per-trial ``domain`` (subject) array. Trials of
    ``target_id`` become the target, and every other trial becomes the source.
    Both halves are fresh ``EEGEpochs`` views selected by mask, so downstream
    alignment or label masking on one does not touch the other.
    """
    tgt_mask = epochs.domain == target_id
    if not tgt_mask.any():
        raise ValueError(f"no trials for target subject {target_id}")
    source = epochs.select(~tgt_mask)
    target = epochs.select(tgt_mask)
    return source, target
