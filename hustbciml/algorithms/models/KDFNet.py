# ===========================================================================
# KDFNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Original authors' code: https://github.com/jxgogo/KDFNet
#
# Reference (IEEE BibTeX):
#   @Article{Jiang2026,
#     author  = {Jiang, Xue and Meng, Lubin and Chen, Xinru and Li, Wei and Wu, Dongrui},
#     journal = {Information Sciences},
#     title   = {{KDFN}et: Knowledge-data fusion network for motor imagery based brain-computer interfaces},
#     year    = {2026},
#     pages   = {123001},
#     volume  = {734},
#     doi     = {10.1016/j.ins.2025.123001},
#   }
# ===========================================================================
"""KDFNet (Jiang et al., 2026, Information Sciences) — Knowledge-Data Fusion
Network for motor imagery (MI) based brain-computer interfaces.

Supervised MI EEG classification. "Fusion" here means fusing knowledge with data
(Sec. 3.1, Fig. 3-4): KDFNet takes the classical Filter Bank Common Spatial
Pattern (FBCSP) pipeline as its knowledge source and mirrors its four stages as
one end-to-end CNN (Sec. 3.1.2, Fig. 5, Table 1), whose parameters are set by
STRUCTURED INITIALIZATION from the FBCSP solution and then refined by
gradient-descent fine-tuning on the labeled EEG (contribution 2; Algorithm 1).
Fusion is thus temporal/spatial/feature/classifier by initialization, not a
two-branch concatenation or an attention mechanism. The four modules:

    Temporal filtering  (m band-pass FIR kernels, 1xl)   -> BN      (Sec. 3.2)
    Spatial filtering    (n CSP kernels Cx1, per band, grouped)     (Sec. 3.3)
    Feature engineering  (log-variance over time)         -> BN     (Sec. 3.4)
    Classification       (FC m*n -> k, softmax)                     (Sec. 3.5)

``forward_features`` returns the ``m*n`` log-variance (band-power) features,
concatenated/flattened as in Sec. 3.4 (Fig. 5); the composed Linear head is the
FC classifier of Sec. 3.5 (Eq. 8). Two knowledge priors are injected at fit time:

* Temporal conv <- a windowed-sinc **band-pass FIR filter bank** (Sec. 3.2,
  Eq. 3-6): each kernel is a length-``l`` rectangular-windowed sinc for one
  4-Hz sub-band, giving a strong band-pass inductive bias. Data-free (needs only
  ``sfreq`` and ``l``; set in ``__init__``). Bands are ``m`` equal 4-Hz sub-bands
  spanning ``f_low..f_high`` (default 8-32 Hz -> 8-12, ..., 28-32; the paper's
  m=6, l=61 default).
* Spatial conv <- per-band **CSP filters** (Sec. 3.3; CSP objective in Sec. 2.1,
  Eq. 1-2), fit on the (aligned) source in ``init_from_source`` (the shared
  supervised training loop fires this hook on the training split before the
  optimizer is built; Algorithm 1, lines 2-3, 7). Estimated the same way as the
  sibling CSP-Net (mne ``transform_into='average_power', log=False,
  cov_est='epoch'``).

All initialized weights stay trainable and are fine-tuned. The paper's knowledge-
updating ablation (Sec. 4.10, Table 9) supports this: FIXING the initialized
temporal, spatial, or FC weights instead of training them lowered average
within-subject accuracy (by 2.84%, 1.28%, and 6.21% respectively), so fine-tuning
each initialized block helped, most of all the classification layer.

Adaptation notes (disclosed in the card): (1) the FIR kernels are symmetric
(linear-phase, Sec. 3.2), so PyTorch's cross-correlation equals convolution — no
kernel flip is needed. (2) The paper also initializes the FC layer from a logistic
-regression classifier fit on the classical log-variance features (Sec. 3.5,
Eq. 9-10; Algorithm 1, line 8); here the classifier is the framework's separate
Linear head, so that FC initialization is omitted and the head trains from
scratch. Table 9 shows the FC init+fine-tuning matters most among the modules, so
this is a genuine simplification (the paper's FC is still fine-tuned after init,
which the from-scratch head partially recovers). (3) A ``log(var+eps)`` stabilizer
is used; the paper's log-variance (Eq. 7) omits the eps.
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
        bw = (f_high - f_low) / self.m         # equal sub-band width (4 Hz for m=6 over 8-32)
        self.bands = [(f_low + i * bw, f_low + (i + 1) * bw) for i in range(self.m)]

        # temporal filtering (Sec. 3.2): 1 input map -> m band maps, no padding
        self.temporal = nn.Conv2d(1, self.m, (1, self.l), bias=False)
        self.bn_t = nn.BatchNorm2d(self.m)
        # spatial filtering (Sec. 3.3): grouped conv so each band i gets its own
        # bank of n_csp CSP-initialized spatial filters (Fig. 5)
        self.spatial = nn.Conv2d(self.m, self.m * self.n_csp, (self.n_chans, 1),
                                 groups=self.m, bias=False)
        self.bn_f = nn.BatchNorm1d(self.m * self.n_csp)
        self.out_features = self.m * self.n_csp

        self._init_fir()

    @torch.no_grad()
    def _init_fir(self) -> None:
        """Band-pass FIR bank -> temporal conv weights (Sec. 3.2, Eq. 3-6; data-free)."""
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

        The spatial-filtering initialization of Sec. 3.3 (CSP per frequency band;
        Algorithm 1, lines 2-3, 7). Called once by the shared supervised training
        loop on the training split, before the optimizer is built; all weights
        stay trainable afterwards."""
        from mne.decoding import CSP
        from scipy.signal import lfilter
        X = epochs.X.astype(np.float64)                 # (N, C, T)
        y = epochs.y
        n_comp = min(self.n_csp, self.n_chans)
        W = np.zeros((self.m * self.n_csp, 1, self.n_chans, 1), dtype=np.float32)
        taps = self.temporal.weight.detach().cpu().numpy()   # (m,1,1,l)
        for i in range(self.m):
            Xb = lfilter(taps[i, 0, 0], [1.0], X, axis=-1)    # band-i FIR (symmetric kernel)
            # per-band CSP (Sec. 3.3): mne returns the n_comp filters paired from
            # the top and bottom of the eigenspectrum (Sec. 2.1, Eq. 1-2)
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
        z = self.bn_t(self.temporal(x))          # temporal filtering + BN (Sec. 3.2)
        z = self.spatial(z)                       # spatial filtering (B, m*n_csp, 1, T-l+1)
        v = z.var(dim=3, unbiased=False).squeeze(2)   # variance over time -> (B, m*n_csp)
        f = torch.log(v + self.eps)               # log-variance feature (Sec. 3.4, Eq. 7)
        return self.bn_f(f)                       # BN over the concatenated m*n features
