# ===========================================================================
# EBCC.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# EBCC enhanced Bayesian classifier combination, vendored (numpy/scipy) from
# TestEnsemble/algs/EBCC.py (https://github.com/sylyoung/TestEnsemble). Defaults
# follow the lab's ensemble.py (num_groups=10, empirical_prior=True).
#
# References (IEEE BibTeX):
#   @InProceedings{Li2019EBCC,
#     author    = {Li, Yuan and Rubinstein, Benjamin I. P. and Cohn, Trevor},
#     booktitle = {Proc. 36th Int. Conf. Machine Learning (ICML)},
#     title     = {Exploiting Worker Correlation for Label Aggregation in Crowdsourcing},
#     year      = {2019},
#   }
# ===========================================================================
"""EBCC (Li et al., 2019): enhanced Bayesian classifier combination.

The most expressive of the crowd-aggregation baselines. Variational Bayes over
worker sub-type groups and per-group confusion matrices: each base model is a
mixture over ``num_groups`` behavior types, and models within a type share a
confusion matrix, which captures correlated errors that Dawid-Skene (independent
workers) cannot. The variational updates alternate estimating the group
memberships ``zg_ikm``, the consensus responsibilities ``z_ik``, and the per-group
Dirichlet parameters, until ``z_ik`` converges. Aggregates the hard votes only.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import fixed_seed


class EBCC(VoteCombiner):
    """Enhanced Bayesian classifier combination via variational inference over
    low-rank worker-correlation groups (vendored numpy/scipy)."""

    name = "EBCC"

    def __init__(self, num_groups: int = 10, a_pi: float = 0.1, alpha: float = 1.0,
                 a_v: float = 4.0, b_v: float = 1.0, max_iter: int = 500,
                 empirical_prior: bool = True):
        self.num_groups = num_groups                     # worker sub-type groups
        self.a_pi = a_pi                                 # Dirichlet prior on group mixing
        self.alpha = alpha                               # class-prior concentration (overridden if empirical)
        self.a_v = a_v                                   # confusion-matrix prior: diagonal strength
        self.b_v = b_v                                   # confusion-matrix prior: off-diagonal strength
        self.max_iter = max_iter
        self.empirical_prior = empirical_prior

    def aggregate(self, votes: np.ndarray) -> np.ndarray:
        import scipy.sparse as ssp
        from scipy.special import digamma

        preds = votes                                    # (K, N) integer hard votes
        # bind hyperparameters locally (alpha is reassigned below under empirical_prior)
        num_groups, a_pi, alpha = self.num_groups, self.a_pi, self.alpha
        a_v, b_v, max_iter, empirical_prior = self.a_v, self.b_v, self.max_iter, self.empirical_prior

        K, N = preds.shape
        num_classes = int(preds.max()) + 1
        with fixed_seed(0):
            first = np.repeat(np.arange(N), K)
            second = np.tile(np.arange(K), N)
            third = preds.T.flatten()
            tuples = np.vstack((first, second, third)).T

            # per-class sparse vote incidence matrices (trial x worker), and transposes
            y_lij, y_lji = [], []
            for k in range(num_classes):
                sel = tuples[:, 2] == k
                coo = ssp.coo_matrix((np.ones(sel.sum()), tuples[sel, :2].T),
                                     shape=(N, K), dtype=bool)
                y_lij.append(coo.tocsr())
                y_lji.append(coo.T.tocsr())

            beta_kl = np.eye(num_classes) * (a_v - b_v) + b_v   # confusion-matrix prior
            z_ik = np.zeros((N, num_classes))
            for l in range(num_classes):
                z_ik[:, [l]] += y_lij[l].sum(axis=-1)
            z_ik /= z_ik.sum(axis=-1, keepdims=True)            # init responsibilities = vote fractions
            if empirical_prior:
                alpha = z_ik.sum(axis=0)

            zg_ikm = np.random.dirichlet(np.ones(num_groups), z_ik.shape) * z_ik[:, :, None]
            for _ in range(max_iter):
                eta_km = a_pi / num_groups + zg_ikm.sum(axis=0)          # group-mixing posterior
                nu_k = alpha + z_ik.sum(axis=0)                          # class-prior posterior
                mu_jkml = np.zeros((K, num_classes, num_groups, num_classes)) + beta_kl[None, :, None, :]
                for l in range(num_classes):
                    for k in range(num_classes):
                        mu_jkml[:, k, :, l] += y_lji[l].dot(zg_ikm[:, k, :])
                Eq_log_pi = digamma(eta_km) - digamma(eta_km.sum(axis=-1, keepdims=True))
                Eq_log_tau = digamma(nu_k) - digamma(nu_k.sum())
                Eq_log_v = digamma(mu_jkml) - digamma(mu_jkml.sum(axis=-1, keepdims=True))
                zg_ikm[:] = Eq_log_pi[None, :, :] + Eq_log_tau[None, :, None]
                for l in range(num_classes):
                    for k in range(num_classes):
                        zg_ikm[:, k, :] += y_lij[l].dot(Eq_log_v[:, k, :, l])
                zg_ikm = np.exp(zg_ikm)
                zg_ikm /= zg_ikm.reshape(N, -1).sum(axis=-1)[:, None, None]
                last = z_ik
                z_ik = zg_ikm.sum(axis=-1)                               # marginalize groups -> class responsibilities
                if np.allclose(last, z_ik, atol=1e-3):                  # converged
                    break
        return np.argmax(z_ik, axis=1).astype(int)
