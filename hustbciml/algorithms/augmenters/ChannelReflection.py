# ===========================================================================
# ChannelReflection.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Wang2024,
#     author  = {Wang, Ziwei and Li, Siyang and Luo, Jingwei and Liu, Jiajing and Wu, Dongrui},
#     journal = {Neural Networks},
#     title   = {Channel Reflection: Knowledge-Driven Data Augmentation for {EEG}-Based Brain-Computer Interfaces},
#     year    = {2024},
#     pages   = {106351},
#     volume  = {176},
#     doi     = {10.1016/j.neunet.2024.106351},
#   }
# ===========================================================================
"""Channel Reflection (Wang et al., 2024, Neural Networks) — knowledge-driven
data augmentation for EEG-based BCIs.

For a left/right-symmetric motor-imagery task, an electrode montage reflected
across the sagittal midline turns a left-hand-imagery trial into one that looks
like right-hand imagery (motor cortex activity swaps hemispheres), so the trial
is a valid *label-swapped* example of the opposite class. This augmenter appends
to each batch the hemisphere-reflected copy of every trial with its label
flipped, doubling the effective training set at no data cost.

The benchmark pairs this augmenter with ``aligner: Identity`` (see the CR preset)
so the leaderboard measures its own contribution as a pure electrode-space
transform, isolated from any aligner: on BNCI2014001 it lifts cross-subject
EEGNet from ~69% (no augmentation) to ~73%. That pairing is a measurement choice,
not a requirement of the method — the original paper (Wang et al., 2024, Fig. 3)
applies EA before CR, and that EA+CR pipeline composes cleanly (~74%, on par with
the raw-space regime). The montage's left/right pairing comes from
``utils.montage.reflection_permutation``.

Label swap is defined only for the 2-class (left/right) case; with any other
class count the channels are still reflected but labels are kept unchanged.
"""
# ---------------------------------------------------------------------------
# Prior-art contrast: "channel symmetry" (reflect the montage but KEEP the label,
# e.g. Deiss et al., HAMLET, arXiv:1803.09702, 2018). On a left/right task the
# reflected trial actually depicts the OPPOSITE class, so keeping its label
# mislabels the synthetic copy and drives cross-subject accuracy toward chance
# (measured ~53% on BNCI2014001, well below the ~69% no-augmentation baseline).
# That failure is exactly what the label swap below fixes, which is the whole
# point of Channel Reflection. Channel symmetry is therefore not carried as a
# separate benchmarked augmenter; the contrast is documented here instead. To
# study it, drop the `1 - y_ref[known]` label swap in __call__ (keep y_ref = y).
# ---------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch

from hustbciml.core.batch import UNLABELED, EEGBatch
from hustbciml.core.stages import Augmenter
from hustbciml.utils.montage import reflection_permutation


class ChannelReflection(Augmenter):
    train_only = True

    def __init__(self, ch_names=None, n_classes: int = 2, **_):
        self.n_classes = int(n_classes)
        perm = reflection_permutation(list(ch_names) if ch_names else [])
        # empty (no montage, e.g. synthetic data) -> reverse channel order so the
        # reflection is still a non-trivial involution and the stage stays runnable.
        self._perm = None if len(perm) == 0 else torch.from_numpy(perm).long()

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        x = batch.x                               # (B, 1, C, T)
        C = x.shape[2]
        perm = self._perm
        if perm is None:
            perm = torch.arange(C - 1, -1, -1, device=x.device)   # reverse fallback
        else:
            perm = perm.to(x.device)

        x_ref = x[:, :, perm, :]
        y_ref = batch.y.clone()
        if self.n_classes == 2:                   # left <-> right label swap
            known = y_ref != UNLABELED
            y_ref[known] = 1 - y_ref[known]

        x_new = torch.cat([x, x_ref], dim=0)
        y_new = torch.cat([batch.y, y_ref], dim=0)
        d_new = torch.cat([batch.domain, batch.domain], dim=0)
        return EEGBatch(x_new, y_new, d_new)
