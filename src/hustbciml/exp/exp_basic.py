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

import dataclasses
import json
import math
import os
from typing import Dict, List

import numpy as np

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.config import Config
from hustbciml.data_provider.data_factory import get_epochs
from hustbciml.utils.seed import resolve_device


def _json_safe(obj):
    """Recursively replace non-finite floats with ``None``.

    ``json.dump(..., allow_nan=False)`` raises on NaN/Inf, which is the point —
    but a NaN metric is a legitimate outcome (an AUC for a fold in which one
    class never appears), so it is mapped to JSON ``null`` rather than aborting
    the write. Everything else that is genuinely non-finite would be a bug and
    still surfaces, as ``null``, in the file.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


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
        self.cfg.classes = list(epochs.classes)
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

    # Config fields that describe *where* a run wrote and how chatty it was, not
    # *what* it measured. Two runs that differ only in these are the same
    # experiment, so they are excluded from the overwrite comparison below.
    _NON_MEASUREMENT_FIELDS = frozenset({
        "data_dir", "results_dir", "device", "itr", "run_tag", "overwrite", "verbose",
        "n_chans", "n_times", "n_classes", "sfreq", "ch_names",
    })

    def _config_payload(self) -> Dict:
        """The resolved config as plain JSON-able data, for ``metrics.json``."""
        cfg = dataclasses.asdict(self.cfg)
        cfg["ch_names"] = list(cfg.get("ch_names") or [])
        return cfg

    def _measurement_config(self, cfg: Dict) -> Dict:
        return {k: v for k, v in cfg.items() if k not in self._NON_MEASUREMENT_FIELDS}

    def save_results(self, per_subject: List[Dict], summary: Dict, predictions=None) -> str:
        """Write the run's results under ``results_dir/<setting>/`` and return
        that directory.

        Two files are produced. ``metrics.json`` records the full run identity —
        dataset, protocol, pipeline stages, *and the entire resolved config*
        (seed, learning rate, epochs, batch size, architecture knobs, every
        ``hp`` entry) — alongside the per-subject numbers and the summary. That
        completeness is the point: a leaderboard cell has to be auditable back to
        the exact settings that produced it from the artifact alone, and the
        folder name cannot carry them all.

        ``predictions.npz`` is optional and holds the raw per-subject hard
        predictions and scores that the offline ensemble tool combines. ``y_pred``
        is stored explicitly rather than recomputed as ``argmax(y_score)``,
        because that identity does not hold for every strategy and the metrics
        were computed from ``y_pred``.

        The folder name comes from ``cfg.setting()``. Re-running the *same*
        configuration overwrites in place, which is what makes runs resumable;
        re-running a *different* configuration into the same folder is refused,
        because that silently replaces one measurement with another under one
        label. ``--run_tag`` (or ``--overwrite``) is the way past it.
        """
        out_dir = os.path.join(self.cfg.results_dir, self.cfg.setting())
        cfg_payload = self._config_payload()
        self._check_overwrite(out_dir, cfg_payload)
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
            "config": cfg_payload,
            "per_subject": per_subject,
            "summary": summary,
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
            # allow_nan=False: the default emits the bare token ``NaN``, which is
            # not JSON and breaks every non-Python reader of the results tree.
            # A metric that could not be computed is serialised as null instead.
            json.dump(_json_safe(payload), fh, indent=2, allow_nan=False)
        if predictions is not None:                 # per-subject scores, for the ensemble tool
            def _obj(arrs):                          # 1-D object array of (unequal-length) arrays,
                a = np.empty(len(arrs), dtype=object)  # not a 2-D array when lengths happen to match
                for i, x in enumerate(arrs):
                    a[i] = np.asarray(x)
                return a
            np.savez(os.path.join(out_dir, "predictions.npz"),
                     subjects=np.array([p["subject"] for p in predictions]),
                     y_true=_obj([p["y_true"] for p in predictions]),
                     y_pred=_obj([p["y_pred"] for p in predictions]),
                     y_score=_obj([p["y_score"] for p in predictions]))
        return out_dir

    def _check_overwrite(self, out_dir: str, cfg_payload: Dict) -> None:
        """Refuse to replace a result that a *different* config produced.

        Legacy results written before the config was recorded have no ``config``
        key; those are left alone (there is nothing to compare against).
        """
        path = os.path.join(out_dir, "metrics.json")
        if self.cfg.overwrite or not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                old = json.load(fh)
        except Exception:
            return                                   # unreadable: treat as absent
        old_cfg = old.get("config")
        if old_cfg is None:
            return
        new_m, old_m = self._measurement_config(cfg_payload), self._measurement_config(old_cfg)
        if new_m == old_m:
            return
        differing = sorted(k for k in set(new_m) | set(old_m) if new_m.get(k) != old_m.get(k))
        raise FileExistsError(
            f"{path} already holds a result from a different configuration "
            f"(differs in: {differing}). Writing here would replace one measurement with "
            f"another under the same label. Pass --run_tag <name> to keep both, a distinct "
            f"--results_dir, or --overwrite to replace it deliberately."
        )

    def run(self):
        """Run the protocol. Implemented by each protocol subclass."""
        raise NotImplementedError
