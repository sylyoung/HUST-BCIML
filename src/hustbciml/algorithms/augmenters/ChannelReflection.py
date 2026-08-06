# ===========================================================================
# ChannelReflection.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.

# Credit chain († = co-first authors; every node except the integrator carries its GitHub link):
#   Original authors:    Ziwei Wang† & Siyang Li†, Jingwei Luo, Jiajing Liu, Dongrui Wu (2024) — "Channel Reflection: Knowledge-Driven Data Augmentation for EEG-Based Brain-Computer Interfaces", Neural Networks
#                        Original code: https://github.com/sylyoung/DeepTransferEEG (paper's official code) + https://github.com/wzwvv/EEGAug
#   Implementation:      Ziwei Wang, Siyang Li, Jingwei Luo, Jiajing Liu, Dongrui Wu — sylyoung/DeepTransferEEG (https://github.com/sylyoung/DeepTransferEEG), wzwvv/EEGAug (https://github.com/wzwvv/EEGAug) (official)
#   Current code:        Siyang Li — sylyoung/DeepTransferEEG (https://github.com/sylyoung/DeepTransferEEG) (eegdec port)
#   Integrated by:       Siyang Li <lsyyoungll@gmail.com> — HUST-BCIML

# References (IEEE BibTeX):
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

Two preconditions are checked at construction, and a failure is an error rather
than a fallback — a silent fallback here produces mislabeled training data that
still yields a plausible leaderboard number.

* **The montage must be anatomical.** The mirror is derived from the 10-20
  odd/even hemisphere rule, which says nothing about labels like ``EEG1 ...
  EEG15`` (BNCI2014002 exposes exactly those). An earlier reverse-the-channel-
  order fallback turned a missing montage into an arbitrary permutation.
* **The two classes must be a left/right pair.** Only then is the reflected
  trial an example of the *other* class. On right-hand-vs-feet data
  (BNCI2014002, BNCI2015001) the reflected trial is still a right-hand trial, so
  swapping its label would fabricate data. This — not the montage — is why the
  CR row is reported on BNCI2014001 only.
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
from hustbciml.utils.montage import (check_montage, left_right_class_swap,
                                     reflection_permutation)


class ChannelReflection(Augmenter):
    train_only = True

    def __init__(self, ch_names=None, n_classes: int = 2, classes=None, **_):
        self.n_classes = int(n_classes)
        names = list(ch_names) if ch_names else []
        ok, why = check_montage(names)
        if not ok:
            raise ValueError(
                f"ChannelReflection needs a left/right-symmetric electrode montage: {why}. "
                f"Reflecting without one permutes sensors arbitrarily while still swapping "
                f"the label, which fabricates training data."
            )
        ok, why = left_right_class_swap(list(classes or []))
        if not ok:
            raise ValueError(
                f"ChannelReflection needs a left/right two-class task: {why}. "
                f"A midline reflection of a non-lateral class (feet, tongue) is still the "
                f"same class, so the label swap would mislabel every augmented trial."
            )
        self._perm = torch.from_numpy(reflection_permutation(names)).long()

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        x = batch.x                               # (B, 1, C, T)
        perm = self._perm.to(x.device)

        x_ref = x[:, :, perm, :]
        y_ref = batch.y.clone()
        known = y_ref != UNLABELED                # left <-> right label swap
        y_ref[known] = 1 - y_ref[known]

        x_new = torch.cat([x, x_ref], dim=0)
        y_new = torch.cat([batch.y, y_ref], dim=0)
        d_new = torch.cat([batch.domain, batch.domain], dim=0)
        return EEGBatch(x_new, y_new, d_new)
