# ===========================================================================
# FShift.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# References (IEEE BibTeX):
#   @Article{Freer2020,
#     author  = {Freer, Daniel and Yang, Guang-Zhong},
#     journal = {Journal of Neural Engineering},
#     title   = {Data Augmentation for Self-Paced Motor Imagery Classification with {C}-{LSTM}},
#     year    = {2020},
#     number  = {1},
#     pages   = {016041},
#     volume  = {17},
#     doi     = {10.1088/1741-2552/ab57c0},
#   }
#   @Article{Freeman2007,
#     author  = {Freeman, Walter J.},
#     journal = {Scholarpedia},
#     title   = {Hilbert Transform for Brain Waves},
#     year    = {2007},
#     number  = {1},
#     pages   = {1338},
#     volume  = {2},
#     doi     = {10.4249/scholarpedia.1338},
#   }
# ===========================================================================
"""Frequency shift (FShift / Freq), a comparison baseline in Ziwei Wang's
augmentation studies (Channel Reflection, Neural Networks 2024; CSDA,
Knowledge-Based Systems 2025).

The whole spectrum of a trial is translated by a small offset delta_f using the
analytic signal. For each channel the analytic signal x_a(t) is formed with the
Hilbert transform (Freeman, 2007), then modulated:

    x'(t) = Re{ x_a(t) * exp(j * 2*pi * delta_f * t) }        # delta_f = +/- 0.2 Hz

A positive delta_f slides every rhythm up by delta_f hertz, a negative one slides
it down; the sign is drawn per trial so the batch mixes both. The shift is tiny
relative to the motor-imagery mu/beta bands, so the class evidence survives while
the exact spectral fingerprint changes. The shifted copy keeps the label.

The analytic signal is computed with a discrete Fourier transform (single-sided
spectrum doubled) so no SciPy dependency is needed. The sampling rate is supplied
by the pipeline; it defaults to 250 Hz if unavailable.
"""
from __future__ import annotations

import math

import torch

from hustbciml.core.batch import EEGBatch
from hustbciml.core.stages import Augmenter


class FShift(Augmenter):
    train_only = True

    c_freq: float = 0.2      # shift magnitude in Hz (paper: 0.2)

    def __init__(self, ch_names=None, n_classes: int = 2,
                 sfreq: float = 250.0, c_freq: float = None, **_):
        self.sfreq = float(sfreq) if sfreq else 250.0
        self.c_freq = float(c_freq) if c_freq is not None else FShift.c_freq

    def _analytic(self, x: torch.Tensor) -> torch.Tensor:
        """Analytic signal along the last axis (complex tensor, same shape)."""
        T = x.shape[-1]
        X = torch.fft.fft(x, dim=-1)
        h = torch.zeros(T, device=x.device, dtype=X.dtype)   # single-sided doubling window
        if T % 2 == 0:
            h[0] = 1; h[T // 2] = 1; h[1:T // 2] = 2
        else:
            h[0] = 1; h[1:(T + 1) // 2] = 2
        return torch.fft.ifft(X * h, dim=-1)

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        x = batch.x                                          # (B, 1, C, T) real
        B, _, _, T = x.shape
        analytic = self._analytic(x)                         # complex (B, 1, C, T)

        n = torch.arange(T, device=x.device, dtype=x.dtype)  # sample index
        signs = torch.where(torch.rand(B, device=x.device) < 0.5, -1.0, 1.0)
        delta_f = (self.c_freq * signs).view(B, 1, 1, 1)
        angle = 2 * math.pi * delta_f * n / self.sfreq       # (B, 1, 1, T) broadcast over C
        phase = torch.complex(torch.cos(angle), torch.sin(angle))
        x_aug = (analytic * phase).real.to(x.dtype)          # Re{ x_a * e^{j w t} }

        x_new = torch.cat([x, x_aug], dim=0)
        y_new = torch.cat([batch.y, batch.y], dim=0)
        d_new = torch.cat([batch.domain, batch.domain], dim=0)
        return EEGBatch(x_new, y_new, d_new)
