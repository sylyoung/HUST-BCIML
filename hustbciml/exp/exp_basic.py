# exp_basic.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Base experiment: owns the data axis (loading + dim injection + results IO).
Subclasses (one per ``--protocol``) implement ``run``.

This class holds everything a protocol does not need to reimplement: load the
dataset once and tell the config how big the data is, average per-subject
metrics into a summary, and write the results to disk. A protocol subclass such
as ``Exp_CrossSubject`` supplies only the loop that decides how subjects are
split and scored.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.config import Config
from hustbciml.data_provider.data_factory import get_epochs
from hustbciml.utils.seed import resolve_device


class Exp_Basic:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)

    def _get_data(self) -> EEGEpochs:
        """Load the dataset and inject data-derived dims into the config, so
        the pipeline can size backbone/head generically."""
        epochs = get_epochs(self.cfg)
        self.cfg.n_chans = epochs.n_channels
        self.cfg.n_times = epochs.n_times
        self.cfg.n_classes = epochs.n_classes
        self.cfg.sfreq = epochs.sfreq
        self.cfg.ch_names = list(epochs.ch_names)
        return epochs

    @staticmethod
    def aggregate(per_subject: List[Dict]) -> Dict:
        """Reduce the list of per-subject metric dicts to mean and std per
        metric, which is the leave-one-subject-out headline (average over the
        held-out subjects).

        ``primary`` is moved to the end only for tidy ordering. ``nanmean`` and
        ``nanstd`` are used so a fold whose AUC came back NaN (a class missing in
        that fold) is skipped for that metric instead of poisoning the average.
        """
        keys = [k for k in per_subject[0] if k != "primary"] + ["primary"]
        out = {}
        for k in keys:
            vals = np.array([m[k] for m in per_subject], dtype=float)
            out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
        return out

    def save_results(self, per_subject: List[Dict], summary: Dict, predictions=None) -> str:
        """Write the run's results under ``results_dir/<setting>/`` and return
        that directory.

        Two files are produced. ``metrics.json`` records the full run identity
        (which dataset, protocol, and pipeline stages were used) alongside the
        per-subject numbers and the summary, so a result is self-describing.
        ``predictions.npz`` is optional and holds the raw per-subject scores that
        the offline ensemble tool combines. The folder name comes from
        ``cfg.setting()``, so re-running the same configuration overwrites in
        place instead of piling up copies.
        """
        out_dir = os.path.join(self.cfg.results_dir, self.cfg.setting())
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "setting": self.cfg.setting(),
            "dataset": self.cfg.dataset,
            "protocol": self.cfg.protocol,
            "algorithm": self.cfg.algorithm,
            "stages": {
                "aligner": self.cfg.aligner, "augmenter": self.cfg.augmenter,
                "backbone": self.cfg.backbone, "head": self.cfg.head,
                "strategy": self.cfg.strategy,
            },
            "per_subject": per_subject,
            "summary": summary,
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
        if predictions is not None:                 # per-subject scores, for the ensemble tool
            def _obj(arrs):                          # 1-D object array of (unequal-length) arrays,
                a = np.empty(len(arrs), dtype=object)  # not a 2-D array when lengths happen to match
                for i, x in enumerate(arrs):
                    a[i] = np.asarray(x)
                return a
            np.savez(os.path.join(out_dir, "predictions.npz"),
                     subjects=np.array([p["subject"] for p in predictions]),
                     y_true=_obj([p["y_true"] for p in predictions]),
                     y_score=_obj([p["y_score"] for p in predictions]))
        return out_dir

    def run(self):
        """Run the protocol. Implemented by each protocol subclass."""
        raise NotImplementedError
