# ===========================================================================
# EA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/hehe03/EA
#
# Reference (IEEE BibTeX):
#   @Article{He2020,
#     author  = {He, He and Wu, Dongrui},
#     journal = {IEEE Trans. Biomedical Engineering},
#     title   = {Transfer Learning for Brain-Computer Interfaces: A {E}uclidean Space Data Alignment Approach},
#     year    = {2020},
#     number  = {2},
#     pages   = {399-410},
#     volume  = {67},
#     doi     = {10.1109/TBME.2019.2913914},
#   }
# ===========================================================================
"""Euclidean Alignment (EA) — label-free cross-subject covariance whitening.

Reference
---------
H. He and D. Wu, "Transfer learning for Brain-Computer Interfaces: A Euclidean
space data alignment approach," IEEE Trans. Biomedical Engineering, 67(2):
399-410, 2020.

What problem it solves
----------------------
In cross-subject EEG decoding, every subject's trials live in a slightly
different distribution: electrode impedance, head geometry, and the exact cap
placement change the scale and spatial covariance of the raw signal from person
to person. A model trained on some subjects therefore sees a shifted input when
tested on a new subject. EA removes the first-order (second-moment) part of that
shift — cheaply and without labels — *before* any backbone or classifier is fit.

The idea
--------
For one subject with n trials, each trial X_i of shape (C channels, T samples),
form the mean spatial covariance over that subject's trials:

    R = (1/n) * sum_i cov(X_i)                    # (C, C), symmetric PSD

then whiten every trial of that subject by the inverse square root of R:

    X_i_aligned = R^{-1/2} @ X_i

After this transform the subject's *new* mean covariance is the identity:

    (1/n) * sum_i cov(R^{-1/2} X_i) = R^{-1/2} R R^{-1/2} = I

So every subject, whatever their raw covariance, is mapped to the same reference
point (the identity). The between-subject covariance mismatch is gone, while the
discriminative within-trial structure (e.g. what distinguishes left- from
right-hand motor imagery) is preserved, because the *same* whitening matrix is
applied to every trial of that subject.

Why R^{-1/2} specifically
-------------------------
R^{-1/2} is the unique symmetric positive-definite inverse square root, so it
decorrelates and rescales the channels in place without imposing an arbitrary
rotation. That keeps the transform deterministic and consistent across subjects,
which is what makes the aligned trials comparable from one subject to the next.

Per-domain, label-free, and leakage-safe
----------------------------------------
- Per-domain: R is estimated *separately* for each subject (domain), because the
  shift is per-subject. A single shared R would not equalise anyone.
- Label-free: R is built only from the trials, never the labels, so EA applies
  identically to source and target subjects and never touches test labels.
- Leakage-safe under leave-one-subject-out: the held-out target subject's own R
  is estimated from that subject's (unlabeled) trials. This is a legitimate,
  unsupervised, per-subject normalisation — not information transfer from the
  held-out labels.

Online variant
--------------
For test-time / streaming use the reference can be accumulated incrementally as
trials arrive (`online_update`) rather than from a full batch, so the identical
alignment applies to a live BCI where trials come one at a time. This is why the
stage advertises `supports_online = True`.

Implementation notes
--------------------
The core numpy math (`EA` / `EA_online`) is vendored from DeepTransferEEG
(`tl/utils/alg_utils.py`) and wrapped here in the Aligner stage interface, made
per-domain. `scipy.linalg.fractional_matrix_power(R, -0.5)` computes R^{-1/2};
its output is real up to numerical round-off, so `transform` takes the real part
after the matrix multiply.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import fractional_matrix_power

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.stages import Aligner


def _reference_inv_sqrt(X: np.ndarray) -> np.ndarray:
    """Return R^{-1/2} for one subject's trials ``X`` of shape (n, C, T).

    R is the mean spatial covariance across the subject's n trials; whitening by
    R^{-1/2} maps that mean covariance to the identity (see module docstring).
    """
    # per-trial spatial covariance: cov(X[i]) is (C, C) over the T time samples
    cov = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
    for i in range(X.shape[0]):
        cov[i] = np.cov(X[i])
    ref = np.mean(cov, axis=0)                 # R: subject's mean covariance (C, C)
    return fractional_matrix_power(ref, -0.5)  # R^{-1/2}


class EA(Aligner):
    # declarative contract read by the pipeline resolver:
    requires_labels = False   # EA uses trials only, never labels
    supports_online = True    # a running reference can be built trial-by-trial

    def __init__(self, **_):
        # domain id (subject) -> that subject's whitening matrix R^{-1/2}
        self._inv_sqrt = {}

    def fit(self, epochs: EEGEpochs) -> "EA":
        """Estimate and cache one R^{-1/2} per domain (subject) from its trials."""
        self._inv_sqrt = {}
        for d in epochs.domains():
            Xd = epochs.X[epochs.domain == d]           # this subject's trials only
            self._inv_sqrt[int(d)] = _reference_inv_sqrt(Xd)
        return self

    def transform(self, epochs: EEGEpochs) -> EEGEpochs:
        """Whiten every trial by its own subject's R^{-1/2} (left-multiply).

        A domain not seen during ``fit`` (e.g. an unlabeled target that arrives
        later) has its reference estimated on the fly from its own trials, so the
        transform is always defined and never leaks across subjects.
        """
        X = epochs.X.copy()
        for d in epochs.domains():
            mask = epochs.domain == d
            W = self._inv_sqrt.get(int(d))
            if W is None:                               # unseen domain -> fit on the fly
                W = _reference_inv_sqrt(epochs.X[mask])
            Xd = epochs.X[mask]
            # broadcast the (C, C) whitening matrix over every trial of subject d;
            # take the real part to drop fractional_matrix_power round-off
            X[mask] = np.real(np.matmul(W[None, :, :], Xd))
        return epochs.with_X(X)

    # --- incremental reference, for online / test-time strategies ---
    @staticmethod
    def online_update(x: np.ndarray, R, n: int):
        """Fold one new trial ``x`` (C, T) into a running mean covariance ``R``.

        ``R`` is the running estimate after ``n`` trials; pass the sentinel int 0
        for the first sample. Returns the updated (C, C) reference, which
        ``inv_sqrt`` then turns into the whitening matrix for streaming EA.
        """
        cov = np.cov(x)
        if isinstance(R, int):     # first sample: no prior reference yet
            return cov
        return (R * n + cov) / (n + 1)     # running mean of per-trial covariances

    @staticmethod
    def inv_sqrt(R: np.ndarray) -> np.ndarray:
        """R^{-1/2} for a reference covariance ``R`` (used by the online path)."""
        return fractional_matrix_power(R, -0.5)
