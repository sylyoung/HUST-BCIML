# splitters.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Protocol split rules. Hard invariant: never leak the target across the
split — cross-subject source/val are drawn from source domains only; the
aligner is fit on the relevant domain(s) only.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from hustbciml.core.batch import EEGEpochs


def list_targets(epochs: EEGEpochs) -> List[int]:
    return [int(d) for d in epochs.domains()]


def cross_subject(epochs: EEGEpochs, target_id: int) -> Tuple[EEGEpochs, EEGEpochs]:
    """Leave-one-subject-out: source = all other subjects, target = ``target_id``."""
    tgt_mask = epochs.domain == target_id
    if not tgt_mask.any():
        raise ValueError(f"no trials for target subject {target_id}")
    source = epochs.select(~tgt_mask)
    target = epochs.select(tgt_mask)
    return source, target
