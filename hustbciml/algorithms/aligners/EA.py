# ===========================================================================
# EA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/sylyoung/DeepTransferEEG
#
# References (IEEE BibTeX):
#   @Article{He2020,
#     author  = {He, He and Wu, Dongrui},
#     journal = {IEEE Transactions on Biomedical Engineering},
#     title   = {Transfer Learning for Brain-Computer Interfaces: A {E}uclidean Space Data Alignment Approach},
#     year    = {2020},
#     number  = {2},
#     pages   = {399-410},
#     volume  = {67},
#     doi     = {10.1109/TBME.2019.2913914},
#   }
#   @Article{Wu2025,
#     author  = {Wu, Dongrui},
#     journal = {Journal of Neural Engineering},
#     title   = {Revisiting {E}uclidean Alignment for Transfer Learning in {EEG}-Based Brain-Computer Interfaces},
#     year    = {2025},
#     number  = {3},
#     pages   = {031005},
#     volume  = {22},
#     doi     = {10.1088/1741-2552/addd49},
#   }
#   @Article{Li2024,
#     author  = {Li, Siyang and Wang, Ziwei and Luo, Hanbin and Ding, Lieyun and Wu, Dongrui},
#     journal = {IEEE Transactions on Biomedical Engineering},
#     title   = {{T}-{TIME}: Test-Time Information Maximization Ensemble for Plug-and-Play {BCI}s},
#     year    = {2024},
#     number  = {2},
#     pages   = {423-432},
#     volume  = {71},
#     doi     = {10.1109/TBME.2023.3303289},
#   }
# ===========================================================================
"""Euclidean Alignment (EA) — unsupervised, per-subject alignment of EEG trials
in Euclidean space (He & Wu, 2020, IEEE TBME).

References
----------
Original method: H. He and D. Wu, "Transfer Learning for Brain-Computer
Interfaces: A Euclidean Space Data Alignment Approach," IEEE Transactions on
Biomedical Engineering, 67(2): 399-410, 2020. Equation numbers below refer to
this paper.

Later re-examination: D. Wu, "Revisiting Euclidean Alignment for Transfer
Learning in EEG-Based Brain-Computer Interfaces," Journal of Neural Engineering,
22(3): 031005, 2025.

Online / test-time use in this benchmark: S. Li, Z. Wang, H. Luo, L. Ding and
D. Wu, "T-TIME: Test-Time Information Maximization Ensemble for Plug-and-Play
BCIs," IEEE Transactions on Biomedical Engineering, 71(2): 423-432, 2024.

What problem it solves
----------------------
Across subjects (and sessions), EEG trials follow different distributions:
electrode configuration, head geometry, and cap placement shift the spatial
covariance of the raw signal from person to person, so a model trained on some
subjects sees a distribution-shifted input on a new subject. EA reduces this
cross-subject mismatch in the *second-order statistics* (the covariance) — the
paper motivates it via maximum mean discrepancy and its relationship to CORAL
(Section III) — cheaply and without any labels from the new subject, *before*
any signal processing, feature extraction, or classifier is applied.

The method (He & Wu, 2020)
--------------------------
A subject has n trials, each trial X_i of shape (C channels, T samples). The
reference matrix R-bar is the arithmetic (Euclidean) mean of that subject's
per-trial spatial covariance matrices (Eq. 10; the paper's trial covariance is
X_i X_i^T, Eq. 7):

    R-bar = (1/n) * sum_i X_i X_i^T               # (C, C), symmetric PSD

Every trial of that subject is then aligned by the inverse square root of R-bar
(Eq. 11):

    X-tilde_i = R-bar^{-1/2} @ X_i

After alignment the subject's mean covariance is the identity (Eq. 12):

    (1/n) * sum_i X-tilde_i X-tilde_i^T = R-bar^{-1/2} R-bar R-bar^{-1/2} = I

So every subject's aligned trials share the same reference point (mean
covariance = I), which makes their distributions more similar and improves
transfer to a new subject. Because the *same* matrix R-bar^{-1/2} is applied to
every trial of a subject, discriminative within-trial structure (e.g. what
distinguishes left- from right-hand motor imagery) is preserved. Unlike
Riemannian Alignment (RA, Zanini et al., 2018), which recenters the covariance
matrices on the Riemannian manifold, EA aligns the time-domain EEG trials
directly in Euclidean space and uses the arithmetic mean, so it is much cheaper
to compute and any downstream algorithm applies unchanged (Section III.A).

Unsupervised and per-domain
---------------------------
- Per-domain: R-bar is estimated *separately* for each subject (domain), because
  the shift is per-subject; a single shared reference would not align anyone.
- Unsupervised: R-bar is built from the trials only, never the labels, so EA
  needs no label information from the new subject (Section III.A) and applies
  identically to source and target subjects.
- Under leave-one-subject-out this is a legitimate per-subject normalization: the
  held-out target subject's own R-bar comes from that subject's (unlabeled)
  trials, not from any held-out label.

Online variant
--------------
He & Wu (2020) also study a simulated-online setting (Section VI) in which the
reference is updated incrementally as target trials arrive rather than from a
full batch. `online_update` implements that running-mean reference so the same
alignment applies to a live BCI where trials come one at a time, which is why the
stage advertises `supports_online = True`.

Implementation notes
--------------------
The core numpy math is taken from DeepTransferEEG (`tl/utils/alg_utils.py`,
functions `EA` / `EA_online`) and wrapped here in the Aligner stage interface,
made per-domain. Per-trial covariance uses `numpy.cov` (which mean-centers each
channel and divides by T-1); on the band-pass-filtered EEG used in the benchmark
this is a benign numerical variant of the paper's X_i X_i^T (Eq. 7), since the
per-channel mean is ~0 and a common scale factor cancels under R-bar^{-1/2}.
`scipy.linalg.fractional_matrix_power(R, -0.5)` computes R-bar^{-1/2}; its output
is real up to round-off, so `transform` takes the real part after the matrix
multiply.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import fractional_matrix_power

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.stages import Aligner


def _reference_inv_sqrt(X: np.ndarray) -> np.ndarray:
    """Return R-bar^{-1/2} for one subject's trials ``X`` of shape (n, C, T).

    R-bar is the arithmetic mean of the subject's per-trial spatial covariances
    (He & Wu, 2020, Eq. 10); aligning each trial by R-bar^{-1/2} (Eq. 11) makes
    the subject's mean covariance the identity (Eq. 12; see module docstring).
    """
    # per-trial (C, C) spatial covariance over the T time samples
    cov = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
    for i in range(X.shape[0]):
        cov[i] = np.cov(X[i])
    ref = np.mean(cov, axis=0)                 # R-bar: arithmetic mean covariance (Eq. 10)
    return fractional_matrix_power(ref, -0.5)  # R-bar^{-1/2} (Eq. 11)


class EA(Aligner):
    # declarative contract read by the pipeline resolver:
    requires_labels = False   # EA uses trials only, never labels (unsupervised)
    supports_online = True    # a running reference can be built trial-by-trial

    def __init__(self, **_):
        # domain id (subject) -> that subject's alignment matrix R-bar^{-1/2}
        self._inv_sqrt = {}

    def fit(self, epochs: EEGEpochs) -> "EA":
        """Estimate and cache one R-bar^{-1/2} per domain (subject) from its trials."""
        self._inv_sqrt = {}
        for d in epochs.domains():
            Xd = epochs.X[epochs.domain == d]           # this subject's trials only
            self._inv_sqrt[int(d)] = _reference_inv_sqrt(Xd)
        return self

    def transform(self, epochs: EEGEpochs) -> EEGEpochs:
        """Align every trial by its own subject's R-bar^{-1/2} (Eq. 11, left-multiply).

        A domain not seen during ``fit`` (e.g. an unlabeled target that arrives
        later) has its reference estimated on the fly from its own trials, so the
        transform is always defined and never mixes references across subjects.
        """
        X = epochs.X.copy()
        for d in epochs.domains():
            mask = epochs.domain == d
            W = self._inv_sqrt.get(int(d))
            if W is None:                               # unseen domain -> fit on the fly
                W = _reference_inv_sqrt(epochs.X[mask])
            Xd = epochs.X[mask]
            # apply the (C, C) reference R-bar^{-1/2} to every trial of subject d;
            # take the real part to drop fractional_matrix_power round-off
            X[mask] = np.real(np.matmul(W[None, :, :], Xd))
        return epochs.with_X(X)

    # --- incremental reference for the online / test-time setting ---
    #     (He & Wu, 2020, Section VI: simulated online, reference updated as
    #      trials arrive; code from DeepTransferEEG `EA_online`) ---
    @staticmethod
    def online_update(x: np.ndarray, R, n: int):
        """Fold one new trial ``x`` (C, T) into a running mean covariance ``R``.

        ``R`` is the running R-bar estimate after ``n`` trials; pass the sentinel
        int 0 for the first sample. Returns the updated (C, C) reference, which
        ``inv_sqrt`` then turns into R-bar^{-1/2} for streaming EA.
        """
        cov = np.cov(x)
        if isinstance(R, int):     # first sample: no prior reference yet
            return cov
        return (R * n + cov) / (n + 1)     # running arithmetic mean of trial covariances

    @staticmethod
    def inv_sqrt(R: np.ndarray) -> np.ndarray:
        """R-bar^{-1/2} for a reference covariance ``R`` (Eq. 11; used by the online path)."""
        return fractional_matrix_power(R, -0.5)
