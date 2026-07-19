# ===========================================================================
# CSDA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/wzwvv/CSDA
#
# Reference (IEEE BibTeX):
#   @Article{Wang2025a,
#     author  = {Wang, Ziwei and Li, Siyang and Chen, Xiaoqing and Wu, Dongrui},
#     journal = {Knowledge-Based Systems},
#     title   = {Time-Frequency Transform Based {EEG} Data Augmentation for Brain-Computer Interfaces},
#     year    = {2025},
#     pages   = {113074},
#     volume  = {311},
#     doi     = {10.1016/j.knosys.2025.113074},
#   }
# ===========================================================================
"""CSDA / DWTaug (Ziwei Wang et al., 2025, Knowledge-Based Systems) — time-frequency
(wavelet) cross-subject EEG data augmentation.

Decompose each trial with a ``db4`` discrete wavelet transform into approximation
(low-frequency) and detail (high-frequency) coefficients, then *cross-reassemble*:
pair two SAME-CLASS trials and swap their detail bands while keeping each one's
approximation, and reconstruct in the time domain. This transplants one trial's
high-frequency characteristics onto another of the same class. Each pair yields
two augmented trials, so the training set is expanded ~threefold:
``[original, self-approx + partner-detail, partner-approx + self-detail]``.

Faithful-adaptation note. The paper reassembles SOURCE with TARGET-train trials of
the same class. Under the hustbciml cross-subject LOSO protocol the target is
unlabeled and the Augmenter contract sees only source minibatches — but a source
batch spans many source subjects, so pairing each trial with a random same-class
partner *in the batch* preserves the cross-subject detail-swap mechanism within
the contract. The paper's optional Euclidean Alignment before the transform is
supplied by the pipeline's aligner stage (compose with ``aligner: EA``).

Only DWTaug is ported (the repo's key code); the paper's HHTaug variant (empirical
mode decomposition via pyhht) is a separate, heavier method not included here.

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

        cA, cD = pywt.dwt(xn, self.wavelet, axis=-1)            # (B, C, T') each
        aug1 = pywt.idwt(cA, cD[safe], self.wavelet, self.mode, axis=-1)[..., :T]  # self approx + partner detail
        aug2 = pywt.idwt(cA[safe], cD, self.wavelet, self.mode, axis=-1)[..., :T]  # partner approx + self detail

        keep = np.where(valid)[0]
        extra = np.concatenate([aug1[keep], aug2[keep]], axis=0).astype(np.float32)   # (2K, C, T)
        extra_t = torch.from_numpy(extra).unsqueeze(1)         # (2K, 1, C, T)

        keep_t = torch.from_numpy(keep).long()
        y_keep = batch.y[keep_t]
        d_keep = batch.domain[keep_t]
        x_new = torch.cat([x, extra_t], dim=0)
        y_new = torch.cat([batch.y, y_keep, y_keep], dim=0)    # both augmented copies keep the class
        d_new = torch.cat([batch.domain, d_keep, d_keep], dim=0)
        return EEGBatch(x_new, y_new, d_new)
