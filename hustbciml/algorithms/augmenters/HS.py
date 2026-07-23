# ===========================================================================
# HS.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Pei2021,
#     author  = {Pei, Yu and Luo, Zhiguo and Yan, Ye and Yan, Huijiong and Jiang, Jing and Li, Wei and Xie, Liang and Yin, Erwei},
#     journal = {Frontiers in Human Neuroscience},
#     title   = {Data Augmentation: Using Channel-Level Recombination to Improve Classification Performance for Motor Imagery {EEG}},
#     year    = {2021},
#     pages   = {645952},
#     volume  = {15},
#     doi     = {10.3389/fnhum.2021.645952},
#   }
# ===========================================================================
"""Half-sample recombination (Pei et al., 2021), a comparison baseline in CSDA
(Wang et al., Knowledge-Based Systems 2025).

Motor-imagery activity is lateralized, so left- and right-hemisphere channels
carry semi-independent evidence for the same class. Splicing the two hemispheres
from two same-class trials therefore yields a new, plausible trial of that class:

    X'[:, L] = X_a[:, L]      # left and midline channels from trial a
    X'[:, R] = X_b[:, R]      # right channels from a same-class trial b

Hemisphere membership is read from the montage (even trailing digit = right,
odd or midline = kept on the left side). With no montage the channel axis is
split in half. A random same-class partner is drawn per trial; trials whose
class has no second member in the batch are left unaugmented. The spliced copy
keeps the class label.
"""
from __future__ import annotations

import re

import numpy as np
import torch

from hustbciml.core.batch import UNLABELED, EEGBatch
from hustbciml.core.stages import Augmenter

_DIGIT_RE = re.compile(r"^[A-Za-z]+?(\d+)$")


class HS(Augmenter):
    train_only = True

    def __init__(self, ch_names=None, n_classes: int = 2, **_):
        self.n_classes = int(n_classes)
        self._ch_names = list(ch_names) if ch_names else []

    def _right_mask(self, C: int) -> np.ndarray:
        """Boolean mask (len C): True where the channel is right-hemisphere."""
        names = self._ch_names
        if len(names) != C:                       # no / mismatched montage: split in half
            m = np.zeros(C, dtype=bool)
            m[C - C // 2:] = True
            return m
        mask = np.zeros(C, dtype=bool)
        for i, n in enumerate(names):
            mm = _DIGIT_RE.match(n.strip())
            if mm and int(mm.group(1)) % 2 == 0:  # even trailing digit -> right hemisphere
                mask[i] = True
        return mask

    def _pair_same_class(self, y: np.ndarray) -> np.ndarray:
        """A random same-class partner index for each trial (-1 if none)."""
        partner = -np.ones(len(y), dtype=int)
        for c in np.unique(y):
            if c == UNLABELED:
                continue
            idx = np.where(y == c)[0]
            if len(idx) < 2:
                continue
            perm = np.random.permutation(idx)
            fixed = perm == idx                   # avoid pairing a trial with itself
            if fixed.any():
                perm[fixed] = np.roll(idx, 1)[fixed]
            partner[idx] = perm
        return partner

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        x = batch.x                               # (B, 1, C, T)
        C = x.shape[2]
        partner = self._pair_same_class(batch.y.cpu().numpy())
        valid = partner >= 0
        if not valid.any():
            return batch

        right = torch.from_numpy(self._right_mask(C)).to(x.device).view(1, 1, C, 1)
        keep_t = torch.from_numpy(np.where(valid)[0]).long().to(x.device)
        part_t = torch.from_numpy(partner[valid.nonzero()[0]]).long().to(x.device)

        x_aug = torch.where(right, x[part_t], x[keep_t])   # right from partner, left+mid from self
        x_new = torch.cat([x, x_aug], dim=0)
        y_new = torch.cat([batch.y, batch.y[keep_t]], dim=0)
        d_new = torch.cat([batch.domain, batch.domain[keep_t]], dim=0)
        return EEGBatch(x_new, y_new, d_new)
