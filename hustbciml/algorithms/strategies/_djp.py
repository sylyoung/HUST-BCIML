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
"""Discriminative Joint Probability MMD (DJP-MMD; Zhang & Wu, IJCNN 2020).

Vendored and corrected from the authors' ``DaNN_DJP/djp_mmd.py``. The DJP-MMD
discrepancy is ``M_T - mu * M_D`` over the joint probability distribution:
``M_T`` (transferability) pulls the same-class source/target class-means together,
``M_D`` (discriminability) pushes different-class source/target means apart. It is
computed in an RKHS as ``trace(K M K^T)`` with an RBF kernel over the stacked
(source, target) batch features.

Fixes vs the source, disclosed in the card: (1) the class count is the real number
of classes, not a hardcoded 10; (2) the target indicator ``Nt`` is always built
from the target pseudo-labels — the source only built it when the batch collapsed
to a single pseudo-label, which zeroed the cross-domain terms; (3) device-agnostic
(no .cpu()/.cuda() round-trips), so it runs on CPU or CUDA unchanged.

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
    """(m, C) one-hot class indicator scaled by 1/m."""
    N = torch.zeros(m, C, device=y.device, dtype=torch.float32)
    N.scatter_(1, y.view(-1, 1), 1.0)
    return N / m


def djp_mmd(feat_s: torch.Tensor, feat_t: torch.Tensor, y_s: torch.Tensor,
            y_t: torch.Tensor, num_class: int, mu: float = 0.1) -> torch.Tensor:
    """DJP-MMD discrepancy between source and target batch features.

    feat_s/feat_t : (m, k) backbone features (equal m). y_s : source labels;
    y_t : target pseudo-labels. mu : discriminability trade-off (source default 0.1).
    """
    m, C = feat_s.size(0), num_class
    K = _rbf_kernel(torch.cat([feat_s, feat_t], dim=0))
    Ns = _indicator(y_s, C, m)
    Nt = _indicator(y_t, C, m)

    # transferability: same class across domains -> minimize
    Rmin = torch.cat([torch.cat([Ns @ Ns.t(), -Ns @ Nt.t()], dim=1),
                      torch.cat([-Nt @ Ns.t(), Nt @ Nt.t()], dim=1)], dim=0)

    # discriminability: each class vs the other C-1 classes across domains -> maximize
    ms = torch.cat([Ns[:, i:i + 1].repeat(1, C - 1) for i in range(C)], dim=1)
    mt = torch.cat([Nt[:, [j for j in range(C) if j != i]] for i in range(C)], dim=1)
    Rmax = torch.cat([torch.cat([ms @ ms.t(), -ms @ mt.t()], dim=1),
                      torch.cat([-mt @ ms.t(), mt @ mt.t()], dim=1)], dim=0)

    M = Rmin - mu * Rmax
    return torch.trace(K @ M @ K.t())
