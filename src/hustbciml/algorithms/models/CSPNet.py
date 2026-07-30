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
"""CSP-Net (Jiang et al., 2024, Knowledge-Based Systems) — Common Spatial
Pattern empowered neural networks for EEG-based motor imagery classification.

Supervised binary MI classification. The paper embeds classical Common Spatial
Pattern (CSP) spatial filters into a CNN as knowledge-driven prior. CSP (Sec.
2.1) designs spatial filters W that maximize the class-variance ratio J(W) =
(W^T C1 W) / (W^T C2 W) (Eq. 1), solved as the generalized eigendecomposition
of C2^-1 C1 (Eq. 2). The paper proposes TWO architectures (Sec. 2.3-2.4, Fig. 2):
  * CSP-Net-1 (Sec. 2.3, Fig. 2b, Alg. 1): PREPENDS a CSP layer before a CNN
    backbone, spatially filtering the raw EEG to improve input discriminability.
  * CSP-Net-2 (Sec. 2.4, Fig. 2c, Alg. 2): REPLACES the CNN's spatial-filter
    convolution with a CSP layer (for the EEGNet backbone, the DepthwiseConv2D
    of the depthwise spatial filter block; Table 1). The CSP layer is
    INITIALIZED with CSP filters designed on the training data.
In both variants the CSP-initialized layer is then either kept FIXED (the "-fix"
setting) or optimized by gradient descent together with the CNN (the "-upd"
setting); the paper reports the fixed setting generally does better (Sec. 3.3).
When a backbone's spatial block needs more kernels than f CSP filters, the CSP
filters are REPLICATED to match (Sec. 3.2); the default is f = 8 filters
(Sec. 3.6). Experiments on four public MI datasets show both CSP-Nets
consistently improve over their CNN backbones, especially with few training
samples (Sec. 3.3, Tables 7-11).

This file implements CSP-Net-2 with the EEGNet backbone, in the fixed setting
(CSP-Net-2-fix) by default. It subclasses EEGNet and overwrites ``block1[3]``,
the (n_chans, 1) depthwise spatial conv of shape (F1*D, 1, n_chans, 1), with the
CSP filter matrix. ``n_csp`` CSP filters (default 8) are replicated to fill the
F1*D output channels; with EEGNet's default F1*D = 8 this equals n_csp, so no
replication happens. CSP-Net-1 (the prepend variant) is not implemented here.

Data-dependent init: the CSP filters need the source X, y, which the pipeline
does not have when the backbone is built. So the backbone exposes
``init_from_source(epochs)``, and the shared supervised training loop calls it
once, on the training split, before building the optimizer (a frozen spatial
layer is then left out of the optimizer).

Adaptation notes: (1) CSP is fit with mne ``transform_into='average_power',
log=False, cov_est='epoch'`` to obtain the spatial filters ``csp.filters_``;
(2) ``freeze_spatial=True`` gives CSP-Net-2-fix (initialize the spatial layer
with CSP and keep it fixed), matching the paper's default and better-performing
setting; set it False for CSP-Net-2-upd (fine-tune the CSP layer with the CNN).
The replaced layer is EEGNet's depthwise spatial conv, which follows the temporal
conv — exactly the block CSP-Net-2 targets in Fig. 2c, not a raw-input filter.
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
        self.n_csp = int(n_csp)               # f: number of CSP spatial filters (Sec. 3.6, default 8)
        self.freeze_spatial = bool(freeze_spatial)   # True -> CSP-Net-2-fix; False -> CSP-Net-2-upd
        self._n_spatial = F1 * D              # depthwise-conv output channels (the spatial filters to replace)
        self._spatial_conv = self.block1[3]   # EEGNet's depthwise spatial conv, weight (F1*D, 1, n_chans, 1)

    @torch.no_grad()
    def init_from_source(self, epochs: EEGEpochs) -> None:
        """Design CSP filters on the source epochs (Eqs. 1-2) and write them into
        EEGNet's depthwise spatial conv, replicating to F1*D channels (Sec. 3.2);
        freeze the layer for the fixed setting (CSP-Net-2-fix).

        Called once by the shared supervised training loop, on the training split,
        before the optimizer is built."""
        from mne.decoding import CSP
        n_comp = min(self.n_csp, self.n_chans)
        # mne CSP solves the same class-variance-ratio problem as Eqs. 1-2 via a
        # generalized eigendecomposition; csp.filters_ are the spatial filters W.
        csp = CSP(n_components=n_comp, transform_into="average_power",
                  log=False, cov_est="epoch")
        csp.fit(epochs.X.astype(np.float64), epochs.y)
        filt = np.asarray(csp.filters_[:n_comp], dtype=np.float32)      # (n_comp, n_chans): W^T rows
        w = torch.from_numpy(filt).reshape(n_comp, 1, self.n_chans, 1)
        # Replicate the CSP filters to fill the F1*D spatial-conv channels when
        # the block needs more kernels than filters (Sec. 3.2). For EEGNet's
        # default F1*D == n_csp == 8 this is a no-op (reps == 1).
        reps = math.ceil(self._n_spatial / n_comp)
        w = w.repeat(reps, 1, 1, 1)[: self._n_spatial]                  # (F1*D, 1, n_chans, 1)
        self._spatial_conv.weight.copy_(w.to(self._spatial_conv.weight.device))
        if self.freeze_spatial:                # CSP-Net-2-fix: keep the CSP layer fixed
            self._spatial_conv.weight.requires_grad_(False)
