# ===========================================================================
# FSurr.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.

# Credit chain († = co-first authors; every node except the integrator carries its GitHub link):
#   Original authors:    Justus T. C. Schwabedal, John C. Snyder, Ayse Cakmak, Shamim Nemati, Gari D. Clifford (2018) — "Addressing Class Imbalance in Classification Problems of Noisy Signals by Using Fourier Transform Surrogates", arXiv:1806.08675
#                        Original code: not publicly released
#   Implementation:      Ziwei Wang — wzwvv/CSDA (https://github.com/wzwvv/CSDA) (comparison baseline)
#   Current code:        Siyang Li — no intermediate repo; implemented in HUST-BCIML following those definitions
#   Integrated by:       Siyang Li <lsyyoungll@gmail.com> — HUST-BCIML

# References (IEEE BibTeX):
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

    X'(f) = X(f) * exp(j * phi_f),   phi_f ~ Uniform(0, 2*pi)

The phase *increment* phi_f is shared across channels at each frequency. That is
what preserves the cross-channel spectral relationships: rotating every channel's
bin by the same angle leaves all pairwise phase differences — the inter-channel
coupling a spatial filter reads — exactly as they were, while decorrelating the
waveform from the original.

Note this is a rotation of the original spectrum, not a rebuild from magnitude
alone. Writing ``|X(f)| * exp(j*phi_f)`` instead, as a straightforward reading of
the surrogate formula suggests, discards each channel's own phase and therefore
forces every channel to the *same* phase at each frequency: perfect zero-lag
coherence across the whole montage, which is not physiologically plausible EEG
and is the opposite of preserving cross-channel structure.

The direct-current bin (and the Nyquist bin for an even length) is copied
verbatim rather than rotated, so the surrogate stays real and the channel means
are preserved *including their sign* — rebuilding those bins from magnitude turns
a negative mean into a positive one, a baseline shift that has nothing to do with
Fourier-surrogate augmentation. The surrogate keeps the label and doubles the
batch.

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

        # One random phase per frequency, shared across channels: a common
        # rotation leaves every pairwise inter-channel phase difference intact.
        phi = (2 * math.pi) * torch.rand(B, 1, 1, F, device=x.device)
        R_surr = R * torch.complex(torch.cos(phi), torch.sin(phi))
        # DC (and Nyquist, for even T) must stay real for the inverse transform to
        # be real; copy them across unchanged so their sign survives too.
        R_surr[..., 0] = R[..., 0]
        if T % 2 == 0:
            R_surr[..., -1] = R[..., -1]
        x_aug = torch.fft.irfft(R_surr, n=T, dim=-1).to(x.dtype)

        x_new = torch.cat([x, x_aug], dim=0)
        y_new = torch.cat([batch.y, batch.y], dim=0)
        d_new = torch.cat([batch.domain, batch.domain], dim=0)
        return EEGBatch(x_new, y_new, d_new)
