# ===========================================================================
# FSurr.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Misc{Schwabedal2018,
#     author = {Schwabedal, Justus T. C. and Snyder, John C. and Cakmak, Ayse and Nemati, Shamim and Clifford, Gari D.},
#     title  = {Addressing Class Imbalance in Classification Problems of Noisy Signals by Using {F}ourier Transform Surrogates},
#     year   = {2018},
#     note   = {arXiv preprint arXiv:1806.08675},
#   }
# ===========================================================================
"""Fourier-transform surrogate (FSurr), a comparison baseline in CSDA (Wang et
al., Knowledge-Based Systems 2025).

A surrogate trial is drawn that has exactly the same power spectrum as the
original but a randomized phase. For each trial the discrete Fourier transform is
taken, every frequency bin's magnitude is kept, and a fresh random phase is
assigned, then the signal is transformed back:

    X'(f) = |X(f)| * exp(j * phi_f),   phi_f ~ Uniform(0, 2*pi)

The same random phase is used across channels at each frequency, which preserves
the cross-channel spectral relationships while decorrelating the waveform from
the original. The direct-current bin (and the Nyquist bin for an even length) is
left with zero phase so the surrogate stays real and its mean is preserved. The
surrogate keeps the label and doubles the batch.

Implemented with a real-input Fourier transform so the output is exactly real and
no SciPy dependency is needed.
"""
from __future__ import annotations

import math

import torch

from hustbciml.core.batch import EEGBatch
from hustbciml.core.stages import Augmenter


class FSurr(Augmenter):
    train_only = True

    def __init__(self, ch_names=None, n_classes: int = 2, **_):
        self.n_classes = int(n_classes)

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        x = batch.x                                   # (B, 1, C, T) real
        B, _, _, T = x.shape
        R = torch.fft.rfft(x, dim=-1)                 # (B, 1, C, F) complex
        F = R.shape[-1]
        mag = R.abs()

        phi = (2 * math.pi) * torch.rand(B, 1, 1, F, device=x.device)   # shared across channels
        phi[..., 0] = 0.0                             # keep DC real (preserve mean)
        if T % 2 == 0:
            phi[..., -1] = 0.0                        # keep Nyquist real
        R_surr = torch.complex(mag * torch.cos(phi), mag * torch.sin(phi))
        x_aug = torch.fft.irfft(R_surr, n=T, dim=-1).to(x.dtype)

        x_new = torch.cat([x, x_aug], dim=0)
        y_new = torch.cat([batch.y, batch.y], dim=0)
        d_new = torch.cat([batch.domain, batch.domain], dim=0)
        return EEGBatch(x_new, y_new, d_new)
