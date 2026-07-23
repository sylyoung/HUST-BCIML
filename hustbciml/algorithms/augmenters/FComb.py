# ===========================================================================
# FComb.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Zhao2022,
#     author  = {Zhao, Xuyang and Sol\'e-Casals, Jordi and Sugano, Hidenori and Tanaka, Toshihisa},
#     journal = {Journal of Neural Engineering},
#     title   = {Seizure Onset Zone Classification Based on Imbalanced {iEEG} with Data Augmentation},
#     year    = {2022},
#     number  = {6},
#     pages   = {065001},
#     volume  = {19},
#     doi     = {10.1088/1741-2552/aca04f},
#   }
# ===========================================================================
"""Frequency recombination (FComb), a comparison baseline in CSDA (Wang et al.,
Knowledge-Based Systems 2025).

Each trial is moved to the frequency domain with the discrete cosine transform,
its spectrum is split into several contiguous bands, and a new trial is built by
taking each band from a (possibly different) same-class trial, then transforming
back:

    D  = DCT(X)                       # per-channel cosine spectrum
    D'[:, band_k] = D_{donor_k}[:, band_k]   for each band k, donor_k same class
    X' = IDCT(D')

Because every band is copied whole from a real same-class trial, the recombined
signal keeps class-consistent spectral content in each band while mixing which
trial contributes which frequencies. A fresh donor is drawn per band per output
trial. Trials whose class has no second member in the batch are left
unaugmented; the recombined copy keeps the class label.

Uses the discrete cosine transform from SciPy (``scipy.fft``), imported lazily so
it is only required when this augmenter is selected.
"""
from __future__ import annotations

import numpy as np
import torch

from hustbciml.core.batch import UNLABELED, EEGBatch
from hustbciml.core.stages import Augmenter


class FComb(Augmenter):
    train_only = True

    n_bands: int = 4         # number of contiguous DCT bands recombined independently

    def __init__(self, ch_names=None, n_classes: int = 2, n_bands: int = None, **_):
        self.n_classes = int(n_classes)
        self.n_bands = int(n_bands) if n_bands is not None else FComb.n_bands

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        try:
            from scipy.fft import dct, idct
        except ImportError as e:                    # pragma: no cover
            raise ImportError("FComb needs SciPy: pip install scipy") from e

        x = batch.x                                 # (B, 1, C, T)
        T = x.shape[-1]
        xn = x.squeeze(1).cpu().numpy().astype(np.float64)     # (B, C, T)
        yn = batch.y.cpu().numpy()

        D = dct(xn, type=2, norm="ortho", axis=-1)             # (B, C, T) cosine spectrum
        edges = np.linspace(0, T, self.n_bands + 1).astype(int)

        out, keep = [], []
        for c in np.unique(yn):
            if c == UNLABELED:
                continue
            idx = np.where(yn == c)[0]
            if len(idx) < 2:
                continue
            for i in idx:
                newD = D[i].copy()
                for b in range(self.n_bands):       # each band from a random same-class donor
                    donor = idx[np.random.randint(len(idx))]
                    newD[:, edges[b]:edges[b + 1]] = D[donor][:, edges[b]:edges[b + 1]]
                out.append(idct(newD, type=2, norm="ortho", axis=-1))
                keep.append(i)
        if not out:
            return batch

        extra = torch.from_numpy(np.stack(out).astype(np.float32)).unsqueeze(1)  # (K, 1, C, T)
        keep_t = torch.from_numpy(np.asarray(keep)).long()
        x_new = torch.cat([x, extra.to(x.device)], dim=0)
        y_new = torch.cat([batch.y, batch.y[keep_t]], dim=0)
        d_new = torch.cat([batch.domain, batch.domain[keep_t]], dim=0)
        return EEGBatch(x_new, y_new, d_new)
