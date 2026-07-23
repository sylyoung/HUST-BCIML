# ===========================================================================
# CSPNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# References (IEEE BibTeX):
#   @Article{Jiang2024,
#     author  = {Jiang, Xue and Meng, Lubin and Chen, Xinru and Xu, Yifan and Wu, Dongrui},
#     journal = {Knowledge-Based Systems},
#     title   = {{CSP}-Net: Common Spatial Pattern Empowered Neural Networks for {EEG}-Based Motor Imagery Classification},
#     year    = {2024},
#     pages   = {112668},
#     volume  = {305},
#     doi     = {10.1016/j.knosys.2024.112668},
#   }
#   @Article{Blankertz2008,
#     author  = {Blankertz, Benjamin and Tomioka, Ryota and Lemm, Steven and Kawanabe, Motoaki and M\"uller, Klaus-Robert},
#     journal = {IEEE Signal Processing Magazine},
#     title   = {Optimizing Spatial Filters for Robust {EEG} Single-Trial Analysis},
#     year    = {2008},
#     number  = {1},
#     pages   = {41-56},
#     volume  = {25},
#     doi     = {10.1109/MSP.2008.4408441},
#   }
# ===========================================================================
"""CSP-Net — CSP-initialized EEGNet backbone (Jiang, Meng, Chen, Xu & Wu, Knowledge-Based Systems 2024).

A lab backbone: a standard EEGNet whose depthwise spatial convolution is
initialized with Common Spatial Pattern filters estimated from the (aligned)
source data and then — in the default CSP-Net-2 configuration — frozen, so the
spatial layer *is* the CSP solution while the temporal and separable
convolutions train normally. It injects classical, well-understood CSP spatial
filtering into the network as a structural prior.

This ports CSP-Net-2 (the "spatial-layer replacement" variant) from the authors'
``CSP-Net/main_CSP_Net_2.py``: their ``block1.3.weight`` is our ``block1[3]``
depthwise conv, of shape (F1*D, 1, n_chans, 1). ``n_csp`` CSP filters (default 8)
are tiled to fill the F1*D spatial-conv output channels, matching the source's
``filters.repeat(ceil(F1*D / n_csp), 1, 1, 1)[:F1*D]``. CSP-Net-1 — an
alternative that *prepends* a CSP channel-projection layer rather than
overwriting the spatial conv — is described in the algorithm card.

Data-dependent init: the CSP filters need the source X, y, which the pipeline
does not have when the backbone is built. So the backbone exposes
``init_from_source(epochs)``, and the shared ``supervised_train`` loop calls it
once, on the training split, before building the optimizer (frozen filters are
then left out of the optimizer).

Faithful-adaptation notes (disclosed in the card): (1) like the source, the CSP
filters are estimated on the raw epochs but applied *after* EEGNet's temporal
conv, since they overwrite the depthwise layer — an approximation the paper
adopts; (2) CSP is fit with mne ``transform_into='average_power', log=False,
cov_est='epoch'`` to match the authors' trainer; (3) ``freeze_spatial=True``
reproduces their default ``baseline=2`` (replace-and-freeze); set it False for
``baseline=0`` (replace-and-fine-tune).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from hustbciml.core.batch import EEGEpochs
from .EEGNet import EEGNet


class CSPNet(EEGNet):
    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float,
                 F1: int = 4, D: int = 2, F2: int = 8,
                 kern_length: int = None, dropout: float = 0.25,
                 n_csp: int = 8, freeze_spatial: bool = True, **_):
        super().__init__(n_chans=n_chans, n_times=n_times, n_classes=n_classes,
                         sfreq=sfreq, F1=F1, D=D, F2=F2,
                         kern_length=kern_length, dropout=dropout)
        self.n_csp = int(n_csp)
        self.freeze_spatial = bool(freeze_spatial)
        self._n_spatial = F1 * D              # depthwise-conv output channels (spatial filters)
        self._spatial_conv = self.block1[3]   # nn.Conv2d weight: (F1*D, 1, n_chans, 1)

    @torch.no_grad()
    def init_from_source(self, epochs: EEGEpochs) -> None:
        """Estimate CSP on the source epochs and write the filters into the
        depthwise spatial conv, tiling to F1*D channels; freeze if configured.

        Called once by ``supervised_train`` before the optimizer is built."""
        from mne.decoding import CSP
        n_comp = min(self.n_csp, self.n_chans)
        csp = CSP(n_components=n_comp, transform_into="average_power",
                  log=False, cov_est="epoch")
        csp.fit(epochs.X.astype(np.float64), epochs.y)
        filt = np.asarray(csp.filters_[:n_comp], dtype=np.float32)      # (n_comp, n_chans)
        w = torch.from_numpy(filt).reshape(n_comp, 1, self.n_chans, 1)
        reps = math.ceil(self._n_spatial / n_comp)
        w = w.repeat(reps, 1, 1, 1)[: self._n_spatial]                  # (F1*D, 1, n_chans, 1)
        self._spatial_conv.weight.copy_(w.to(self._spatial_conv.weight.device))
        if self.freeze_spatial:
            self._spatial_conv.weight.requires_grad_(False)
