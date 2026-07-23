# ===========================================================================
# LAA.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# LAA label-aware autoencoder aggregator, vendored (torch) from
# TestEnsemble/algs/LAA.py (https://github.com/sylyoung/TestEnsemble).
#
# References (IEEE BibTeX):
#   @InProceedings{Yin2017LAA,
#     author    = {Yin, L. and others},
#     booktitle = {Proc. 26th Int. Joint Conf. Artificial Intelligence (IJCAI)},
#     title     = {Aggregating Crowd Wisdoms with Label-Aware Autoencoders},
#     year      = {2017},
#     doi       = {10.24963/ijcai.2017/184},
#   }
# ===========================================================================
"""LAA (Yin et al., 2017): label-aware autoencoder.

A neural aggregator. A linear classifier maps the concatenated one-hot votes to a
consensus label, and a source-wise-softmax decoder reconstructs each base model's
vote from that label. The classifier is first warmed up to the majority vote, then
classifier and decoder are trained jointly (reconstruction + a small KL-to-majority
term + an L1 penalty). The trained classifier's argmax is the consensus. Everything
runs under a local fixed seed, targeting the label-free majority vote.
"""
from __future__ import annotations

import numpy as np

from hustbciml.core.stages import VoteCombiner

from ._common import fixed_seed, onehot


class LAA(VoteCombiner):
    """Label-aware autoencoder: classifier + source-wise-softmax decoder, trained to
    the majority vote (vendored torch)."""

    name = "LAA"

    def aggregate(self, votes: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        preds = votes                                    # (K, N) integer hard votes
        K, N = preds.shape
        C = int(preds.max()) + 1
        oh = onehot(preds, C)                            # (K, N, C)
        user_labels = np.concatenate([oh[k] for k in range(K)], axis=1)  # (N, K*C)
        # majority vote target (label-free), local tie-break
        counts = np.zeros((C, N))
        for k in range(K):
            for j in range(N):
                counts[preds[k, j], j] += 1
        rng = np.random.RandomState(0)
        majority = np.array([rng.choice(np.flatnonzero(counts[:, j] == counts[:, j].max()))
                             for j in range(N)])

        in_features = K * C
        # Block-diagonal template: the decoder's per-source softmax normalizes each
        # model's C-slice of the reconstruction independently.
        template = torch.zeros((in_features, in_features))
        for i in range(K):
            template[i * C:(i + 1) * C, i * C:(i + 1) * C] = 1

        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = nn.Parameter(torch.empty(in_features, C))
                self.biases = nn.Parameter(torch.empty(C))
                nn.init.trunc_normal_(self.weights, std=0.01)
                nn.init.zeros_(self.biases)

            def forward(self, x):
                return F.softmax(x @ self.weights + self.biases, dim=1)

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = nn.Parameter(torch.empty(C, in_features))
                self.biases = nn.Parameter(torch.empty(in_features))
                nn.init.trunc_normal_(self.weights, std=0.01)
                nn.init.zeros_(self.biases)

            def forward(self, y):
                e = torch.exp(y @ self.weights + self.biases)
                return e / (e @ template)                # per-source softmax via the block template

        with fixed_seed(0):
            torch.manual_seed(0)
            clf, dec = Classifier(), Decoder()
            opt = torch.optim.Adam(list(clf.parameters()) + list(dec.parameters()), lr=0.005)
            x = torch.FloatTensor(user_labels)
            y = F.one_hot(torch.LongTensor(majority), num_classes=C).float()
            ce = nn.CrossEntropyLoss()
            for _ in range(50):                          # warm up classifier to MV
                opt.zero_grad()
                ce(clf(x), y).backward()
                opt.step()
            for _ in range(100):                         # autoencoder refinement
                opt.zero_grad()
                yc = clf(x)
                xr = dec(yc)
                loss = (torch.mean(torch.sum(-x * torch.log(xr + 1e-10), dim=1))
                        + 0.0001 * nn.KLDivLoss(reduction="batchmean")(torch.log(yc + 1e-10), y)
                        + 0.005 / (K * C * C) * torch.sum(torch.abs(clf.weights)))
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred = clf(x).argmax(1).numpy()
        return pred.astype(int)
