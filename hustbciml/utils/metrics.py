# metrics.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Classification metrics. The paradigm picks the primary one (MI -> accuracy/
kappa, P300/ERP -> AUC); the leaderboard reports the primary plus the rest.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             roc_auc_score)

# paradigm -> primary metric name
PRIMARY = {
    "MI": "accuracy",
    "P300": "auc",
    "ERP": "auc",
    "SSVEP": "accuracy",
}


def accuracy(y_true, y_pred, **_) -> float:
    return float(accuracy_score(y_true, y_pred) * 100)


def cohen_kappa(y_true, y_pred, **_) -> float:
    return float(cohen_kappa_score(y_true, y_pred))


def macro_f1(y_true, y_pred, **_) -> float:
    return float(f1_score(y_true, y_pred, average="macro") * 100)


def roc_auc(y_true, y_score, n_classes=2, **_) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if n_classes == 2:
        pos = y_score[:, 1] if y_score.ndim == 2 else y_score
        return float(roc_auc_score(y_true, pos) * 100)
    return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro") * 100)


def score(y_true, y_pred, y_score, paradigm="MI", n_classes=2) -> dict:
    """All metrics as a dict; ``primary`` marks the paradigm's headline number."""
    out = {
        "accuracy": accuracy(y_true, y_pred),
        "kappa": cohen_kappa(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }
    try:
        out["auc"] = roc_auc(y_true, y_score, n_classes=n_classes)
    except Exception:
        out["auc"] = float("nan")
    out["primary"] = out[PRIMARY.get(paradigm, "accuracy")]
    return out
