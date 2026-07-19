# ===========================================================================
# _mmd.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
# Adapted from: https://github.com/sylyoung/DeepTransferEEG
#
# References (IEEE BibTeX):
#   @InProceedings{Long2015,
#     author    = {Long, Mingsheng and Cao, Yue and Wang, Jianmin and Jordan, Michael I.},
#     booktitle = {Proc. Int'l Conf. Machine Learning},
#     title     = {Learning Transferable Features with Deep Adaptation Networks},
#     year      = {2015},
#     pages     = {97-105},
#     address   = {Lille, France},
#     month     = {Jul.},
#   }
#   @InProceedings{Long2017,
#     author    = {Long, Mingsheng and Zhu, Han and Wang, Jianmin and Jordan, Michael I.},
#     booktitle = {Proc. Int'l Conf. Machine Learning},
#     title     = {Deep Transfer Learning with Joint Adaptation Networks},
#     year      = {2017},
#     pages     = {2208-2217},
#     address   = {Sydney, Australia},
#     month     = {Aug.},
#   }
# ===========================================================================
"""Maximum Mean Discrepancy machinery shared by DAN and JAN.

Vendored from DeepTransferEEG ``tl/utils/loss.py`` (which credits the Transfer-
Learning-Library implementation). Underscore-prefixed so the registry auto-scan
skips it — it is a helper, not a plug-in. Device-agnostic (index matrix follows
the input tensor's device), so it runs on CUDA or CPU unchanged.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class GaussianKernel(nn.Module):
    """Gaussian kernel matrix over a stacked (source+target) feature group.
    With ``track_running_stats`` the bandwidth is the mean pairwise squared
    distance scaled by ``alpha``; otherwise a fixed ``sigma`` is used."""

    def __init__(self, sigma: Optional[float] = None, track_running_stats: bool = True,
                 alpha: float = 1.0):
        super().__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2 = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)
        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2.detach())
        return torch.exp(-l2 / (2 * self.sigma_square))


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor],
                         linear: bool) -> torch.Tensor:
    """Sign matrix that turns a (2B, 2B) kernel matrix into the MMD estimate."""
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """MK-MMD between source activations ``z_s`` and target activations ``z_t``
    (the DAN objective). Sums the given Gaussian kernels."""

    def __init__(self, kernels: Sequence[nn.Module], linear: bool = False):
        super().__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)
        kernel_matrix = sum(kernel(features) for kernel in self.kernels)
        return (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)


class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """Joint MK-MMD over multiple layers' activations (the JAN objective):
    the per-layer kernel matrices are multiplied elementwise, so the discrepancy
    is over the *joint* distribution of (features, predictions)."""

    def __init__(self, kernels: Sequence[Sequence[nn.Module]], linear: bool = True,
                 thetas: Sequence[nn.Module] = None):
        super().__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        self.thetas = thetas if thetas else [nn.Identity() for _ in kernels]

    def forward(self, z_s, z_t) -> torch.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s[0].device)
        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            kernel_matrix *= sum(kernel(layer_features) for kernel in layer_kernels)
        return (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
