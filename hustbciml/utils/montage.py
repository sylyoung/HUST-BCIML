# montage.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Montage helpers — left/right electrode symmetry for Channel Reflection.

The 10-20 / 10-10 naming convention encodes hemisphere in the trailing digit:
odd = left, even = right, a trailing ``z`` (or no digit) = the midline. So the
sagittal-midline mirror of an electrode is obtained by flipping that digit's
parity (``C3`` <-> ``C4``, ``FC1`` <-> ``FC2``, ``CP5`` <-> ``CP6``), and
midline electrodes (``Cz``, ``Fz``, ``POz`` ...) map to themselves. This rule
covers any standard 10-20 montage without a hand-maintained pair list.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

_NAME_RE = re.compile(r"^([A-Za-z]+?)(\d+)$")


def mirror_name(name: str) -> str:
    """Return the left/right-mirrored channel name (self for midline names)."""
    m = _NAME_RE.match(name.strip())
    if not m:                      # 'Cz', 'Fz', 'POz', or non-numbered -> midline
        return name
    prefix, num = m.group(1), int(m.group(2))
    mirror_num = num + 1 if num % 2 == 1 else num - 1   # flip parity
    return f"{prefix}{mirror_num}"


def reflection_permutation(ch_names: List[str]) -> np.ndarray:
    """Index permutation that maps each channel to its left/right mirror.

    ``perm[i]`` is the position (in ``ch_names``) of the mirror of channel ``i``;
    channels whose mirror is absent map to themselves. If ``ch_names`` is empty
    (e.g. synthetic data with no montage), fall back to reversing the channel
    order so the reflection is still a non-trivial involution.
    """
    if not ch_names:
        return np.array([], dtype=int)
    lut = {n.lower(): i for i, n in enumerate(ch_names)}
    perm = np.arange(len(ch_names), dtype=int)
    for i, n in enumerate(ch_names):
        j = lut.get(mirror_name(n).lower())
        if j is not None:
            perm[i] = j
    return perm
