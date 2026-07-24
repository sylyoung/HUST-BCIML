# ===========================================================================
# StackingNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# StackingNet: black-box meta-ensemble for collective inference (lab-proposed).
# Original authors' code: https://github.com/sylyoung/TestEnsemble
#
# References (IEEE BibTeX):
#   @Article{Li2026a,
#     author  = {Li, Siyang and Liu, Chenhao and Wu, Dongrui and Zeng, Zhigang and Ding, Lieyun},
#     journal = {Advanced Science},
#     title   = {{S}tacking{N}et: Collective Inference Across Independent {AI} Foundation Models},
#     year    = {2026},
#     pages   = {e76488},
#     doi     = {10.1002/advs.76488},
#   }
# ===========================================================================
"""StackingNet (Li et al., 2026, Advanced Science) — a black-box meta-ensemble
for "collective inference": it aggregates only the OUTPUT predictions of
independent, pre-trained models, with no access to their weights or training
data. The paper unifies two output types — regression (supervised; Eq. 16-17: a
weighted sum of base outputs plus a scalar bias, fit by MSE) and classification
(Eq. 25: a weighted sum of the base one-hot predictions, no bias) — and, for
classification, two label regimes: supervised S-StackingNet (cross-entropy,
Eq. 26) and unsupervised U-StackingNet (Eq. 28). The benchmark uses it to combine
its base EEG classifiers, so this file implements the CLASSIFICATION path.

``combine`` runs the UNSUPERVISED variant (U-StackingNet): it learns one
non-negative reliability weight w_j per base model from the test predictions
alone, with no labels. Faithful to the authors' ``StackingNet_classification``
(the ``golden_num == 0`` path with ``use_weight_init``):

* weights are INITIALIZED from the normalized balanced accuracy of each model vs
  the majority-vote consensus (the paper's unsupervised init, not uniform 1/M),
  which anchors the optimization;
* they are refined by the unsupervised objective L_unsup (Eq. 28) — a
  consensus-disagreement indicator loss: each model pays its weight w_j on every
  test trial whose prediction disagrees with the current weighted-ensemble
  consensus pseudo-label yhat (Eq. 27). This is StackingNet's own objective; the
  code's disagreement term is NOT the "PM" (Participant-Mine voting) baseline of
  Table 2. It enters the combined objective (Eq. 31) scaled by
  lambda_1 = ``unsupervised_weight`` (0.001);
* a strong sum-to-one regularizer L_reg = (1 - sum_j w_j)^2 (Eq. 30), scaled by
  lambda_2 = ``reg_weight`` (100), keeps the weights a convex combination; each
  step clamps w_j >= 0 (the non-negativity constraint, Eq. 29).

The small L_unsup weight + strong regularizer + informed init keep the weights a
sensible convex combination instead of collapsing to zero (an earlier uniform-
init / weak-regularizer port collapsed to the constant class, i.e. chance).
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import Combiner

from ._common import balanced_accuracy, majority_vote, onehot


class StackingNet(Combiner):
    """U-StackingNet: the unsupervised classification variant (lab-proposed).

    The hyperparameters are the authors' unsupervised defaults; they are exposed
    as constructor arguments so a sweep can vary them, while the registry builds
    the combiner with the published defaults.
    """

    name = "StackingNet"
    lab_proposed = True

    def __init__(self, epochs: int = 200, lr: float = 0.001,
                 unsupervised_weight: float = 0.001, reg_weight: float = 100.0,
                 seed: int = 0):
        self.epochs = epochs
        self.lr = lr
        self.unsupervised_weight = unsupervised_weight   # lambda_1 (Eq. 31): weight on L_unsup (Eq. 28)
        self.reg_weight = reg_weight                     # lambda_2 (Eq. 31): weight on the sum-to-one L_reg (Eq. 30)
        self.seed = seed

    def combine(self, scores: np.ndarray) -> np.ndarray:
        import torch

        K, N, C = scores.shape
        preds = scores.argmax(axis=2)                    # (K, N)
        oh = onehot(preds, C).astype(np.float32)         # (K, N, C)

        # weight init = each model's balanced accuracy vs the majority-vote consensus
        consensus = majority_vote(scores)                # (N,) label-free consensus
        w0 = np.array([balanced_accuracy(consensus, preds[k], C) for k in range(K)])
        w0 = w0 / w0.sum() if w0.sum() > 0 else np.ones(K) / K

        X = torch.from_numpy(oh.transpose(1, 0, 2))      # (N, K, C)
        torch.manual_seed(self.seed)
        w = torch.nn.Parameter(torch.tensor(w0.reshape(K, 1), dtype=torch.float32))
        opt = torch.optim.Adam([w], lr=self.lr)
        for _ in range(self.epochs):
            out = (X * w.unsqueeze(0)).sum(dim=1)        # (N, C) weighted-ensemble scores
            ens = torch.zeros_like(out)
            ens.scatter_(1, out.argmax(1, keepdim=True), 1)  # hard consensus one-hot (detached by argmax)
            # unsupervised objective L_unsup (Eq. 28): each model pays its weight
            # on every trial whose prediction disagrees with the ensemble
            # consensus pseudo-label `ens` (yhat, Eq. 27). StackingNet's own term,
            # NOT the "PM" (Participant-Mine voting) baseline of Table 2.
            l_unsup = sum(((X[:, i, :] != ens).any(dim=1).float().sum()) * w[i] for i in range(K))
            loss = self.unsupervised_weight * l_unsup + self.reg_weight * (1 - w.sum()) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                w.clamp_(min=0)                          # non-negative weights
        with torch.no_grad():
            out = (X * w.unsqueeze(0)).sum(dim=1)
        return out.argmax(1).numpy()
