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
involution.

**The parity rule is only meaningful for real electrode names.** Applied blindly
it "works" on any label ending in a digit, which is how a montage exposed as
``EEG1 ... EEG15`` — BNCI2014002, one of this benchmark's three datasets — turns
into a confident-looking mirror (``EEG1<->EEG2``, ``EEG3<->EEG4`` ...) that
permutes sensors arbitrarily. Combined with the two-class label swap that is
mislabeled training data produced silently, under a lab-proposed method's name.
So the prefix is validated against the 10-20/10-10/10-5 position vocabulary
below, and a montage that is not recognisably anatomical yields *no* permutation
rather than a fabricated one. Callers must treat an empty permutation as "this
montage cannot be reflected" and fail closed.
"""
from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np

_NAME_RE = re.compile(r"^([A-Za-z]+?)(\d+)$")

# Electrode-position prefixes of the 10-20 system and its 10-10 / 10-5
# refinements, lower-cased. A trailing digit only encodes a hemisphere when it
# follows one of these; ``EEG7`` or ``Ch3`` is an index, not a position.
_POSITION_PREFIXES = frozenset({
    # 10-20 / 10-10
    "nz", "fp", "af", "f", "ft", "fc", "t", "tp", "c", "cp", "p", "po", "o", "i",
    # ear / mastoid references
    "a", "m",
    # 10-5 intermediate rows
    "afp", "aff", "fft", "fft", "ftt", "fcc", "ccp", "cpp", "ppo", "poo", "ttp",
    "tpp", "oi",
})

# Midline electrodes end in ``z`` (Cz, FCz, POz) or carry no digit at all.
_MIDLINE_RE = re.compile(r"^[A-Za-z]+z$", re.IGNORECASE)


def is_position_name(name: str) -> bool:
    """True if ``name`` looks like a 10-20/10-10/10-5 electrode position."""
    n = name.strip()
    if _MIDLINE_RE.match(n):
        return n[:-1].lower() in _POSITION_PREFIXES
    m = _NAME_RE.match(n)
    return bool(m) and m.group(1).lower() in _POSITION_PREFIXES


def is_midline(name: str) -> bool:
    """True for a midline electrode (``Cz``, ``FCz``, ``POz``, ``Nz``)."""
    return bool(_MIDLINE_RE.match(name.strip())) or not _NAME_RE.match(name.strip())


def mirror_name(name: str) -> str:
    """Return the left/right-mirrored channel name (self for midline names).

    ``_NAME_RE`` splits a name into a letter prefix and a trailing number. A
    name with no trailing number (``Cz``, ``Fz``, ``POz``) sits on the midline
    and mirrors to itself. Otherwise the odd/even parity of the number encodes
    the hemisphere, so adding one to an odd number and subtracting one from an
    even number gives the electrode symmetric across the midline.

    A name whose prefix is not an electrode position mirrors to itself: the
    parity rule says nothing about it, and inventing a partner is worse than
    admitting there is none.
    """
    m = _NAME_RE.match(name.strip())
    if not m:                      # 'Cz', 'Fz', 'POz', or non-numbered -> midline
        return name
    prefix, num = m.group(1), int(m.group(2))
    if prefix.lower() not in _POSITION_PREFIXES:
        return name                # not an electrode position; no mirror exists
    mirror_num = num + 1 if num % 2 == 1 else num - 1   # flip parity
    return f"{prefix}{mirror_num}"


def check_montage(ch_names: List[str]) -> Tuple[bool, str]:
    """Can this montage be reflected across the sagittal midline?

    Returns ``(ok, reason)``. Two ways to fail, both of which used to pass
    silently:

    * the labels are not electrode positions at all (``EEG1 ... EEG15``), so the
      odd/even parity rule is meaningless;
    * a lateral electrode has no partner in this montage, so reflecting it would
      leave it in place while every other channel moved — a partial reflection
      that is still shipped with a swapped label.
    """
    if not ch_names:
        return False, "no channel names available"
    bad = [n for n in ch_names if not is_position_name(n)]
    if bad:
        return False, (f"channel names are not 10-20/10-10 electrode positions "
                       f"(e.g. {bad[:3]}); the odd/even hemisphere rule does not apply")
    lut = {n.strip().lower() for n in ch_names}
    unpaired = [n for n in ch_names
                if not is_midline(n) and mirror_name(n).lower() not in lut]
    if unpaired:
        return False, (f"lateral channels with no mirror partner in this montage: "
                       f"{unpaired}; reflection would be partial")
    return True, "ok"


def left_right_class_swap(classes: List[str]) -> Tuple[bool, str]:
    """Do these class names form a left/right pair whose labels a reflection swaps?

    Reflecting the montage only turns a trial into a valid example of *another*
    class when the two classes are mirror images of each other — left hand vs
    right hand. On right-hand-vs-feet data (BNCI2014002, BNCI2015001) the
    reflected trial is still a right-hand trial recorded upside down, so swapping
    its label fabricates training data. The augmenter must therefore ask about
    the class *names*, not merely count that there are two of them.
    """
    if len(classes) != 2:
        return False, f"label swap is defined for 2 classes, got {len(classes)}: {classes}"
    a, b = (c.strip().lower() for c in classes)
    if ("left" in a and "right" in b) or ("right" in a and "left" in b):
        return True, "ok"
    return False, (f"classes {classes} are not a left/right pair; a midline reflection "
                   f"does not map one onto the other")


def reflection_permutation(ch_names: List[str]) -> np.ndarray:
    """Index permutation mapping each channel to its left/right mirror.

    ``perm[i]`` is the position (in ``ch_names``) of the mirror of channel ``i``;
    midline channels map to themselves. Returns an **empty array** when the
    montage cannot be reflected (see ``check_montage``) — the caller is expected
    to treat that as a hard "do not reflect", not as a licence to substitute some
    other channel permutation.
    """
    ok, _ = check_montage(ch_names)
    if not ok:
        return np.array([], dtype=int)
    # Look names up case-insensitively so a montage that writes, say, "cz" still
    # matches. Every lateral channel is known to have a partner at this point.
    lut = {n.strip().lower(): i for i, n in enumerate(ch_names)}
    perm = np.arange(len(ch_names), dtype=int)
    for i, n in enumerate(ch_names):
        j = lut.get(mirror_name(n).lower())
        if j is not None:
            perm[i] = j
    return perm
