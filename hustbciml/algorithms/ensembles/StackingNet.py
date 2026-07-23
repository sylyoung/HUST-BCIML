# ===========================================================================
# StackingNet.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# StackingNet: unsupervised transductive meta-combiner (lab-proposed).
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
"""StackingNet — unsupervised transductive meta-combiner (Li, Liu & Wu, 2026).

Learns one non-negative weight per base model with NO labels, transductively over
the test predictions. Faithful to the authors' ``StackingNet_classification``
(the unsupervised ``golden_num == 0`` path with ``use_weight_init``):

* weights are INITIALIZED from each model's balanced-accuracy agreement with the
  majority-vote consensus (not uniform 1/K), which anchors the optimization;
* they are then refined by the PM indicator loss (each model pulled toward the
  current weighted-ensemble consensus), scaled DOWN by ``unsupervised_weight``
  (0.001), plus a STRONG ``reg_weight * (1 - sum w)^2`` (100) term that keeps the
  weights near a convex combination; weights are clamped >= 0 after each step.

The small PM weight + strong regularizer + informed init are what keep the weights
near a sensible convex combination instead of collapsing to zero (an earlier
uniform-init / unit-PM / weak-reg port collapsed to the constant class, i.e.
chance). Published in Advanced Science (Wiley), 2026.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import Combiner

from ._common import balanced_accuracy, majority_vote, onehot


class StackingNet(Combiner):
    """Unsupervised transductive StackingNet meta-combiner (lab-proposed).

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
        self.unsupervised_weight = unsupervised_weight   # down-weights the PM refinement term
        self.reg_weight = reg_weight                     # pulls sum(w) toward 1 (convex combination)
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
            # PM indicator loss: each model pays its weight for every trial it
            # disagrees with the current weighted-ensemble consensus.
            pm = sum(((X[:, i, :] != ens).any(dim=1).float().sum()) * w[i] for i in range(K))
            loss = self.unsupervised_weight * pm + self.reg_weight * (1 - w.sum()) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                w.clamp_(min=0)                          # non-negative weights
        with torch.no_grad():
            out = (X * w.unsqueeze(0)).sum(dim=1)
        return out.argmax(1).numpy()
