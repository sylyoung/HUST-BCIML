# ===========================================================================
# CSDA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/wzwvv/CSDA
#
# Reference (IEEE BibTeX):
#   @Article{Wang2025a,
#     author  = {Wang, Ziwei and Li, Siyang and Chen, Xiaoqing and Wu, Dongrui},
#     journal = {Knowledge-Based Systems},
#     title   = {Time--Frequency Transform Based {EEG} Data Augmentation for Brain--Computer Interfaces},
#     year    = {2025},
#     volume  = {311},
#     pages   = {113074},
#     doi     = {10.1016/j.knosys.2025.113074},
#   }
# ===========================================================================
"""CSDA (Wang et al., 2025, Knowledge-Based Systems) — time-frequency transform
cross-subject EEG data augmentation.

The paper proposes two parameter-free cross-subject augmenters that share three
steps (Sec. 3, Fig. 2): time-frequency domain signal decomposition, cross-subject
sub-signal reassembling, and time domain reconstruction. The two variants differ
only in the transform used to decompose a trial: DWTaug uses the discrete wavelet
transform (DWT), HHTaug uses the Hilbert-Huang transform (empirical mode
decomposition into intrinsic mode functions). Both keep the class label unchanged,
combine sub-signals only across trials OF THE SAME CLASS, and expand the training
set to about three times the original size (Sec. 3.1).

This file implements DWTaug (Sec. 3.2, Fig. 2a); the HHTaug variant (Sec. 3.3,
Eq. 7-10) is a separate, heavier method not ported here.

DWTaug decomposes each trial X with the order-4 Daubechies wavelet (db4) into
approximation coefficients cA (low-frequency) and detail coefficients cD
(high-frequency); at level j=1 this is Eq. 1, and inverse DWT (iDWT) reconstructs
the trial (Eq. 3-4). "Cross-Subject Coefficient Reassembling" (Sec. 3.2, step 2)
then forms an augmented trial from one trial's approximation and a same-class
partner's detail coefficients, and vice versa, reconstructing in the time domain:
X_tilde = iDWT(cA_self, cD_partner) (Eq. 5) and iDWT(cA_partner, cD_self) (Eq. 6).
Single-level DWTaug (j=1) is the paper's default (Sec. 3.2). Each pair therefore
yields two augmented trials, appended to the originals.

Adaptation to this benchmark. The paper reassembles source with target-train
trials of the same class (Eq. 5-6). Under the cross-subject leave-one-subject-out
protocol the target is unlabeled and the Augmenter contract sees only source
minibatches, but a source batch spans many source subjects, so pairing each trial
with a random same-class partner in the batch preserves the cross-subject
coefficient-reassembling mechanism within the contract. The paper applies
Euclidean Alignment before the transform for the MI paradigm (Sec. 4.1, 4.10);
here that is supplied by the pipeline's aligner stage (compose with ``aligner: EA``).

Requires PyWavelets (``pip install PyWavelets``), imported lazily.
Source: github.com/wzwvv/CSDA (``DWTAug.py``).
"""
from __future__ import annotations

import numpy as np
import torch

from hustbciml.core.batch import UNLABELED, EEGBatch
from hustbciml.core.stages import Augmenter


class CSDA(Augmenter):
    train_only = True

    def __init__(self, ch_names=None, n_classes: int = 2,
                 wavelet: str = "db4", mode: str = "smooth", **_):
        self.n_classes = int(n_classes)
        self.wavelet = wavelet
        self.mode = mode

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
            fixed = perm == idx                     # avoid pairing a trial with itself
            if fixed.any():
                perm[fixed] = np.roll(idx, 1)[fixed]
            partner[idx] = perm
        return partner

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        try:
            import pywt
        except ImportError as e:                    # pragma: no cover
            raise ImportError("CSDA needs PyWavelets: pip install PyWavelets") from e

        x = batch.x                                 # (B, 1, C, T), cpu
        B, _, C, T = x.shape
        xn = x.squeeze(1).cpu().numpy().astype(np.float64)      # (B, C, T)
        yn = batch.y.cpu().numpy()

        partner = self._pair_same_class(yn)
        valid = partner >= 0
        if not valid.any():
            return batch
        safe = np.where(valid, partner, 0)          # dummy index for invalid rows (discarded)

        # DWT into approximation (cA) and detail (cD) coefficients (Eq. 1, db4).
        cA, cD = pywt.dwt(xn, self.wavelet, axis=-1)            # (B, C, T') each
        # Cross-subject coefficient reassembling + iDWT (Sec. 3.2, step 2):
        aug1 = pywt.idwt(cA, cD[safe], self.wavelet, self.mode, axis=-1)[..., :T]  # Eq. 5: self cA + partner cD
        aug2 = pywt.idwt(cA[safe], cD, self.wavelet, self.mode, axis=-1)[..., :T]  # Eq. 6: partner cA + self cD

        keep = np.where(valid)[0]
        extra = np.concatenate([aug1[keep], aug2[keep]], axis=0).astype(np.float32)   # (2K, C, T)
        extra_t = torch.from_numpy(extra).unsqueeze(1)         # (2K, 1, C, T)

        keep_t = torch.from_numpy(keep).long()
        y_keep = batch.y[keep_t]
        d_keep = batch.domain[keep_t]
        x_new = torch.cat([x, extra_t], dim=0)
        # both augmented copies keep the class (same-class reassembling is label-preserving)
        y_new = torch.cat([batch.y, y_keep, y_keep], dim=0)
        d_new = torch.cat([batch.domain, d_keep, d_keep], dim=0)
        return EEGBatch(x_new, y_new, d_new)
