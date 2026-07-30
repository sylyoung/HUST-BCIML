# ===========================================================================
# _djp.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @InProceedings{Zhang2020,
#     author    = {Zhang, Wen and Wu, Dongrui},
#     booktitle = {Proceedings of the International Joint Conference on Neural Networks},
#     title     = {Discriminative Joint Probability Maximum Mean Discrepancy ({DJP}-{MMD}) for Domain Adaptation},
#     year      = {2020},
#     address   = {Glasgow, UK},
#     month     = {Jul.},
#     pages     = {1-8},
#     doi       = {10.1109/IJCNN48605.2020.9207365},
#   }
# ===========================================================================
"""Discriminative Joint Probability MMD (DJP-MMD; Zhang & Wu, 2020).

The DJP-MMD discrepancy over the joint distribution P(X, Y), which the paper writes
as ``M_T - mu * M_D`` (Eq. 8): the same-class-across-domains term M_T measures
transferability and is minimized, while the different-class-across-domains term M_D
measures discriminability and is maximized (weighted by mu > 0). Each term is a
squared distance between mapped class means (Eq. 17-18, 20-21): the c-th column of
``X @ N`` is the mean feature of class c, where the one-hot label matrix N is scaled
by 1/n per domain (source labels for N_s, target pseudo-labels for the target
matrix). In the paper's linear form the discrepancy equals
``trace(A^T X (R_min - mu * R_max) X^T A)`` (Eq. 24), with the domain-block matrices
R_min (Eq. 25) and R_max (Eq. 26) assembled from those indicator matrices.

This helper computes the kernelized value used by the gradient-based EEG strategy:
``trace(K M K^T)`` with M = ``R_min - mu * R_max`` and an RBF kernel K over the
stacked (source, target) batch features (the paper's RKHS form, Eq. 28, with the
kernel in place of the projected features). It returns a scalar loss for
autograd, in place of the paper's generalized-eigen solution for A (Eq. 27).

Adapted from the authors' released ``DaNN_DJP/djp_mmd.py`` (the deep DJP-MMD
variant), with three corrections: (1) the class count is the actual number of
classes rather than a hardcoded 10; (2) the target indicator is always built from
the target pseudo-labels -- the source built it only when the batch collapsed to a
single pseudo-label, which zeroed the cross-domain terms; (3) device-agnostic (no
.cpu()/.cuda() round-trips), so it runs on CPU or CUDA unchanged.

Underscore-prefixed so the registry auto-scan skips it (a helper, not a plug-in).
"""
from __future__ import annotations

import torch


def _rbf_kernel(Z: torch.Tensor) -> torch.Tensor:
    """RBF kernel matrix over rows of Z, bandwidth = mean pairwise sq. distance."""
    zzt = Z @ Z.t()
    d = torch.diag(zzt).unsqueeze(1)
    dist = d - 2 * zzt + d.t()
    sigma2 = torch.clamp(dist.detach().mean(), min=1e-6)
    return torch.exp(-dist / (2 * sigma2))


def _indicator(y: torch.Tensor, C: int, m: int) -> torch.Tensor:
    """(m, C) one-hot label matrix scaled by 1/m (N_s, N_t in Eq. 18)."""
    N = torch.zeros(m, C, device=y.device, dtype=torch.float32)
    N.scatter_(1, y.view(-1, 1), 1.0)
    return N / m


def djp_mmd(feat_s: torch.Tensor, feat_t: torch.Tensor, y_s: torch.Tensor,
            y_t: torch.Tensor, num_class: int, mu: float = 0.1) -> torch.Tensor:
    """DJP-MMD discrepancy between source and target batch features (Eq. 8).

    feat_s/feat_t : (m, k) backbone features (equal m). y_s : source labels;
    y_t : target pseudo-labels. mu : discriminability trade-off (paper default 0.1).
    """
    m, C = feat_s.size(0), num_class
    K = _rbf_kernel(torch.cat([feat_s, feat_t], dim=0))
    Ns = _indicator(y_s, C, m)
    Nt = _indicator(y_t, C, m)

    # R_min (Eq. 25): the transferability term M_T -- same class across domains, minimized
    Rmin = torch.cat([torch.cat([Ns @ Ns.t(), -Ns @ Nt.t()], dim=1),
                      torch.cat([-Nt @ Ns.t(), Nt @ Nt.t()], dim=1)], dim=0)

    # R_max (Eq. 26): the discriminability term M_D -- each class vs the other C-1
    # classes across domains, maximized (M_s, M_t of Eq. 21 built column-wise here)
    ms = torch.cat([Ns[:, i:i + 1].repeat(1, C - 1) for i in range(C)], dim=1)
    mt = torch.cat([Nt[:, [j for j in range(C) if j != i]] for i in range(C)], dim=1)
    Rmax = torch.cat([torch.cat([ms @ ms.t(), -ms @ mt.t()], dim=1),
                      torch.cat([-mt @ ms.t(), mt @ mt.t()], dim=1)], dim=0)

    M = Rmin - mu * Rmax                       # M_T - mu * M_D in block form (Eq. 24)
    return torch.trace(K @ M @ K.t())
