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
import uuid
from typing import Dict, List

import numpy as np

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.config import Config
from hustbciml.data_provider.data_factory import get_epochs
from hustbciml.utils.io import atomic_json_dump, atomic_savez
from hustbciml.utils.provenance import runtime_provenance
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
        self.cfg.resolved_device = str(self.device)

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
        self.cfg.data_provenance = dict(epochs.provenance or {})
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

    # Output location and presentation do not change a measurement. Device,
    # source/data identity, dimensions and preprocessing do, so they remain in the
    # comparison even though they are not part of the human-readable folder name.
    _NON_MEASUREMENT_FIELDS = frozenset({
        "data_dir", "results_dir", "itr", "run_tag", "overwrite", "verbose",
    })

    def _config_payload(self) -> Dict:
        """The resolved config as plain JSON-able data, for ``metrics.json``."""
        cfg = dataclasses.asdict(self.cfg)
        cfg["ch_names"] = list(cfg.get("ch_names") or [])
        return cfg

    def _measurement_config(self, cfg: Dict) -> Dict:
        return {k: v for k, v in cfg.items() if k not in self._NON_MEASUREMENT_FIELDS}

    def _measurement_identity(self, payload: Dict) -> Dict:
        """The complete, stable identity used to approve result reuse."""
        runtime = (payload.get("provenance") or {}).get("runtime") or {}
        return {
            "config": self._measurement_config(payload.get("config") or {}),
            "data": (payload.get("provenance") or {}).get("data") or {},
            "source_sha256": runtime.get("source_sha256"),
            "hustbciml_version": runtime.get("hustbciml_version"),
            "python": runtime.get("python"),
            "platform": runtime.get("platform"),
            "machine": runtime.get("machine") or {},
            "dependencies": runtime.get("dependencies") or {},
            "numpy_build": runtime.get("numpy_build") or {},
            "numerical_libraries": runtime.get("numerical_libraries") or [],
            "torch_runtime": runtime.get("torch_runtime") or {},
        }

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
        predictions and scores that the offline ensemble tool combines. Both files
        carry the same per-write ``artifact_id`` so an interrupted rewrite cannot
        pair new metrics with stale predictions. ``y_pred`` is stored explicitly
        rather than recomputed as ``argmax(y_score)``, because that identity does not
        hold for every strategy and the metrics were computed from ``y_pred``.

        The folder name comes from ``cfg.setting()``. Re-running the *same*
        configuration overwrites in place, which is what makes runs resumable;
        re-running a *different* configuration into the same folder is refused,
        because that silently replaces one measurement with another under one
        label. ``--run_tag`` (or ``--overwrite``) is the way past it.
        """
        out_dir = os.path.join(self.cfg.results_dir, self.cfg.setting())
        cfg_payload = self._config_payload()
        data_provenance = dict(self.cfg.data_provenance or {})
        runtime = runtime_provenance()
        is_measurement = bool(data_provenance.get("is_measurement", False))
        payload = {
            "artifact_id": uuid.uuid4().hex,
            "setting": self.cfg.setting(),
            "dataset": self.cfg.dataset,
            "protocol": self.cfg.protocol,
            "algorithm": self.cfg.algorithm,
            "is_measurement": is_measurement,
            "non_measurement_reason": None if is_measurement else
                data_provenance.get("reason", "dataset provenance is missing"),
            "stages": {
                "aligner": self.cfg.aligner, "augmenter": self.cfg.augmenter,
                "backbone": self.cfg.backbone, "head": self.cfg.head,
                "strategy": self.cfg.strategy,
            },
            "config": cfg_payload,
            "provenance": {"runtime": runtime, "data": data_provenance},
            "per_subject": per_subject,
            "summary": summary,
        }
        self._check_overwrite(out_dir, payload)
        os.makedirs(out_dir, exist_ok=True)
        atomic_json_dump(_json_safe(payload), os.path.join(out_dir, "metrics.json"))
        if predictions is not None:                 # per-subject scores, for the ensemble tool
            def _obj(arrs):                          # 1-D object array of (unequal-length) arrays,
                a = np.empty(len(arrs), dtype=object)  # not a 2-D array when lengths happen to match
                for i, x in enumerate(arrs):
                    a[i] = np.asarray(x)
                return a
            atomic_savez(
                os.path.join(out_dir, "predictions.npz"),
                artifact_id=np.asarray(payload["artifact_id"], dtype="U"),
                subjects=np.array([p["subject"] for p in predictions]),
                y_true=_obj([p["y_true"] for p in predictions]),
                y_pred=_obj([p["y_pred"] for p in predictions]),
                y_score=_obj([p["y_score"] for p in predictions]),
            )
        return out_dir

    def _check_overwrite(self, out_dir: str, payload: Dict) -> None:
        """Refuse reuse unless the complete measurement identity matches."""
        path = os.path.join(out_dir, "metrics.json")
        if self.cfg.overwrite or not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                old = json.load(fh)
        except Exception as exc:
            raise FileExistsError(
                f"{path} exists but is not readable strict JSON. Preserve it for forensic "
                "comparison and use a new --run_tag/results_dir, or pass --overwrite "
                "only if deliberate replacement is intended."
            ) from exc

        if old.get("config") is None or old.get("provenance") is None:
            raise FileExistsError(
                f"{path} is a legacy artifact without a complete config/provenance identity. "
                "It cannot be assumed equivalent to this run. Preserve it and use a new "
                "--run_tag or --results_dir."
            )
        new_identity = self._measurement_identity(payload)
        old_identity = self._measurement_identity(old)
        if new_identity == old_identity:
            return
        differing = sorted(
            key for key in set(new_identity) | set(old_identity)
            if new_identity.get(key) != old_identity.get(key)
        )
        raise FileExistsError(
            f"{path} already holds a different measurement identity "
            f"(differs in: {differing}). Use --run_tag/a distinct --results_dir, "
            "or --overwrite only for deliberate replacement."
        )

    def run(self):
        """Run the protocol. Implemented by each protocol subclass."""
        raise NotImplementedError
