# montage.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Montage helpers — left/right electrode symmetry for Channel Reflection.

The 10-20 / 10-10 naming convention encodes hemisphere in the trailing digit:
odd = left, even = right, a trailing ``z`` (or no digit) = the midline. So the
sagittal-midline mirror of an electrode is obtained by flipping that digit's
parity (``C3`` <-> ``C4``, ``FC1`` <-> ``FC2``, ``CP5`` <-> ``CP6``), and
midline electrodes (``Cz``, ``Fz``, ``POz`` ...) map to themselves. This rule
covers any standard 10-20 montage without a hand-maintained pair list.

Channel Reflection is a data augmentation for motor imagery. Left- and
right-hand imagery produce mirror-image scalp patterns, so reflecting the
electrodes across the midline turns a left-hand trial into a plausible
right-hand one. The augmenter uses the permutation built here to reorder the
channel axis, and for the two-class left/right task it also swaps the label.
Because the mirror of the mirror is the original, the permutation is an
involution, which the reversal fallback below deliberately preserves.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

_NAME_RE = re.compile(r"^([A-Za-z]+?)(\d+)$")


def mirror_name(name: str) -> str:
    """Return the left/right-mirrored channel name (self for midline names).

    ``_NAME_RE`` splits a name into a letter prefix and a trailing number. A
    name with no trailing number (``Cz``, ``Fz``, ``POz``) sits on the midline
    and mirrors to itself. Otherwise the odd/even parity of the number encodes
    the hemisphere, so adding one to an odd number and subtracting one from an
    even number gives the electrode symmetric across the midline.
    """
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
    # Look names up case-insensitively so a montage that writes, say, "cz" still
    # matches. ``perm`` starts as the identity, so any channel whose mirror is
    # not present in this montage keeps its own position.
    lut = {n.lower(): i for i, n in enumerate(ch_names)}
    perm = np.arange(len(ch_names), dtype=int)
    for i, n in enumerate(ch_names):
        j = lut.get(mirror_name(n).lower())
        if j is not None:
            perm[i] = j
    return perm
