# ===========================================================================
# KDFNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Jiang2026,
#     author  = {Jiang, Xue and Meng, Lubin and Chen, Xinru and Li, Wei and Wu, Dongrui},
#     journal = {Information Sciences},
#     title   = {{KDFN}et: Knowledge-Data Fusion Network for Motor Imagery Based Brain-Computer Interfaces},
#     year    = {2026},
#     pages   = {123001},
#     volume  = {734},
#     doi     = {10.1016/j.ins.2025.123001},
#   }
# ===========================================================================
"""KDFNet — Knowledge-Data Fusion Network (Jiang, ..., Wu, Information Sciences 2026).

A lab backbone that mirrors the classical FBCSP pipeline as a compact CNN whose
layers are *initialized* with the classical solution and then fine-tuned
end-to-end (no concatenation, no attention — knowledge is injected purely by
structured initialization, per the paper's Algorithm 1):

    Temporal filtering  (m band-pass FIR kernels, 1xl)   -> BN
    Spatial filtering    (n CSP kernels Cx1, one bank per band, grouped)
    Log-variance feature (variance over time -> log)      -> BN
    Classification       (one FC m*n -> k)                [the Linear head]

``forward_features`` returns the ``m*n`` log-band-power features; the composed
Linear head is the classifier. Two priors are injected here at fit time:

* Temporal conv <- a windowed-sinc **FIR filter bank** (data-free: needs only
  ``sfreq`` and the kernel length; set in ``__init__``). Bands are ``m`` equal
  4-Hz sub-bands spanning ``f_low..f_high`` (default 8-32 Hz -> 8-12, ..., 28-32).
* Spatial conv <- per-band **CSP filters** fit on the (aligned) source in
  ``init_from_source`` (the shared ``supervised_train`` fires this hook on the
  training split before the optimizer is built), estimated the same way as
  CSP-Net (mne ``transform_into='average_power', log=False, cov_est='epoch'``).

All weights are left trainable and fine-tuned — the paper's ablation shows
freezing any block hurts.

Faithful-adaptation notes (disclosed in the card): (1) the FIR kernels are
symmetric (linear-phase), so PyTorch's cross-correlation equals convolution — no
kernel flip is needed. (2) The paper also initializes the final FC from a
logistic regression fit on classical log-variance features; here the classifier
is the framework's separate Linear head, so that FC-init is omitted and the head
trains from scratch — a minor deviation the end-to-end fine-tuning absorbs (the
paper reports the LR-init is refined by training anyway). (3) A ``log(var+eps)``
stabilizer is used (the paper's Eq. 7 omits the eps).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.stages import Backbone


class KDFNet(Backbone):
    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 m: int = 6, l: int = 61, n_csp: int = 8,
                 f_low: float = 8.0, f_high: float = 32.0, eps: float = 1e-6, **_):
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.sfreq = float(sfreq)
        self.m = int(m)
        self.l = int(l)
        self.n_csp = int(n_csp)
        self.eps = float(eps)
        bw = (f_high - f_low) / self.m
        self.bands = [(f_low + i * bw, f_low + (i + 1) * bw) for i in range(self.m)]

        # temporal filter bank: 1 input channel -> m band maps, no padding
        self.temporal = nn.Conv2d(1, self.m, (1, self.l), bias=False)
        self.bn_t = nn.BatchNorm2d(self.m)
        # per-band spatial filters (grouped so band i -> its own n_csp filters)
        self.spatial = nn.Conv2d(self.m, self.m * self.n_csp, (self.n_chans, 1),
                                 groups=self.m, bias=False)
        self.bn_f = nn.BatchNorm1d(self.m * self.n_csp)
        self.out_features = self.m * self.n_csp

        self._init_fir()

    @torch.no_grad()
    def _init_fir(self) -> None:
        """Windowed-sinc band-pass FIR bank -> temporal conv weights (data-free)."""
        from scipy.signal import firwin
        nyq = self.sfreq / 2.0
        w = np.zeros((self.m, 1, 1, self.l), dtype=np.float32)
        for i, (lo, hi) in enumerate(self.bands):
            hi = min(hi, nyq - 1e-3)
            taps = firwin(self.l, [lo, hi], pass_zero=False, fs=self.sfreq)
            w[i, 0, 0, :] = taps.astype(np.float32)
        self.temporal.weight.copy_(torch.from_numpy(w))

    @torch.no_grad()
    def init_from_source(self, epochs: EEGEpochs) -> None:
        """Fit per-band CSP on the source epochs and write the spatial filters.

        Called once by ``supervised_train`` on the training split, before the
        optimizer is built; all weights stay trainable afterwards."""
        from mne.decoding import CSP
        from scipy.signal import lfilter
        X = epochs.X.astype(np.float64)                 # (N, C, T)
        y = epochs.y
        n_comp = min(self.n_csp, self.n_chans)
        W = np.zeros((self.m * self.n_csp, 1, self.n_chans, 1), dtype=np.float32)
        taps = self.temporal.weight.detach().cpu().numpy()   # (m,1,1,l)
        for i in range(self.m):
            Xb = lfilter(taps[i, 0, 0], [1.0], X, axis=-1)    # band-i FIR (symmetric kernel)
            csp = CSP(n_components=n_comp, transform_into="average_power",
                      log=False, cov_est="epoch")
            csp.fit(Xb, y)
            filt = np.asarray(csp.filters_[:n_comp], dtype=np.float32)   # (n_comp, C)
            if n_comp < self.n_csp:                            # tile if fewer comps than filters
                reps = math.ceil(self.n_csp / n_comp)
                filt = np.tile(filt, (reps, 1))[: self.n_csp]
            W[i * self.n_csp:(i + 1) * self.n_csp, 0, :, 0] = filt
        self.spatial.weight.copy_(torch.from_numpy(W).to(self.spatial.weight.device))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bn_t(self.temporal(x))          # (B, m, C, T-l+1)
        z = self.spatial(z)                       # (B, m*n_csp, 1, T-l+1)
        v = z.var(dim=3, unbiased=False).squeeze(2)   # (B, m*n_csp)  band power
        f = torch.log(v + self.eps)               # log-variance
        return self.bn_f(f)
