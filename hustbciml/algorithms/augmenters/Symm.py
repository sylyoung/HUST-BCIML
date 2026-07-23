# ===========================================================================
# Symm.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Misc{Deiss2018,
#     author = {Deiss, Olivier and Biswal, Siddharth and Jin, Jing and Sun, Haoqi and Westover, M. Brandon and Sun, Jimeng},
#     title  = {{HAMLET}: Interpretable Human And Machine Co-Learning Technique},
#     year   = {2018},
#     note   = {arXiv preprint arXiv:1803.09702},
#   }
# ===========================================================================
"""Channel symmetry, the closest prior-art contrast to Channel Reflection and a
comparison baseline in Wang et al. (Channel Reflection, Neural Networks 2024).

The montage is reflected across the sagittal midline (C3 <-> C4, FC1 <-> FC2,
...) so the hemispheres swap, but the label is kept unchanged. For a left/right
motor-imagery task the reflected copy therefore actually depicts the opposite
class, which is exactly why Channel Reflection argues the label must be swapped.
Keeping the label here reproduces the weaker Symm baseline the paper compares
against, so a benchmark can quantify the gain from the label swap. The reflected
copy is appended to the batch.

The left/right pairing comes from ``utils.montage.reflection_permutation``. With
no montage (synthetic data) it falls back to reversing the channel order so the
stage stays a non-trivial involution.
"""
from __future__ import annotations

import torch

from hustbciml.core.batch import EEGBatch
from hustbciml.core.stages import Augmenter
from hustbciml.utils.montage import reflection_permutation


class Symm(Augmenter):
    train_only = True

    def __init__(self, ch_names=None, n_classes: int = 2, **_):
        self.n_classes = int(n_classes)
        perm = reflection_permutation(list(ch_names) if ch_names else [])
        self._perm = None if len(perm) == 0 else torch.from_numpy(perm).long()

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        x = batch.x                               # (B, 1, C, T)
        C = x.shape[2]
        perm = self._perm
        perm = torch.arange(C - 1, -1, -1, device=x.device) if perm is None else perm.to(x.device)

        x_ref = x[:, :, perm, :]                  # reflect electrodes, label kept
        x_new = torch.cat([x, x_ref], dim=0)
        y_new = torch.cat([batch.y, batch.y], dim=0)
        d_new = torch.cat([batch.domain, batch.domain], dim=0)
        return EEGBatch(x_new, y_new, d_new)
