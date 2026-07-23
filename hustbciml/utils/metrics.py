# metrics.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Classification metrics. The paradigm picks the primary one (MI -> accuracy/
kappa, P300/ERP -> AUC); the leaderboard reports the primary plus the rest.

Why the primary metric is paradigm-dependent. Motor imagery (MI) and SSVEP are
balanced, multi-class problems where getting the label right is the goal, so
accuracy is the headline number. P300 and other ERP paradigms are strongly
imbalanced (few target events among many non-targets), where accuracy is
misleading and ranking targets above non-targets is what matters, so AUC is the
headline instead. ``score`` always computes every metric and then copies the
paradigm's choice into ``primary``, so a run can be summarised by one number
without discarding the others.
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
    """Fraction of trials whose predicted class matches the true class, scaled
    to a percentage. The plain, chance-normalised headline metric for MI."""
    return float(accuracy_score(y_true, y_pred) * 100)


def cohen_kappa(y_true, y_pred, **_) -> float:
    """Cohen's kappa: agreement between prediction and truth after subtracting
    the agreement expected by chance. Reported as a ratio (0 = chance, 1 =
    perfect), which is why, unlike the others, it is not multiplied by 100. It
    is more informative than accuracy when the classes are imbalanced."""
    return float(cohen_kappa_score(y_true, y_pred))


def macro_f1(y_true, y_pred, **_) -> float:
    """Macro-averaged F1: the per-class F1 (harmonic mean of precision and
    recall) computed separately for each class and then averaged with equal
    weight. Equal weighting means every class counts the same regardless of how
    many trials it has, so a rare class cannot be ignored. Scaled to a
    percentage."""
    return float(f1_score(y_true, y_pred, average="macro") * 100)


def roc_auc(y_true, y_score, n_classes=2, **_) -> float:
    """Area under the ROC curve from the model's class scores, as a percentage.

    AUC needs a continuous score per trial, not a hard label, and it measures
    how well those scores rank the classes. For two classes it takes the
    positive-class column (or the raw 1-D score) and computes the standard
    binary AUC. For more than two it uses one-vs-rest per class and averages
    them with equal class weight, matching ``macro_f1``'s equal weighting.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if n_classes == 2:
        pos = y_score[:, 1] if y_score.ndim == 2 else y_score
        return float(roc_auc_score(y_true, pos) * 100)
    return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro") * 100)


def score(y_true, y_pred, y_score, paradigm="MI", n_classes=2) -> dict:
    """All metrics as a dict; ``primary`` marks the paradigm's headline number.

    ``y_pred`` are hard labels (used by accuracy, kappa, F1) and ``y_score`` are
    the continuous class scores (used by AUC). AUC is wrapped in ``try`` because
    it is undefined when a class is missing from ``y_true`` in this fold, and in
    that case it is recorded as NaN so the aggregation can skip it rather than
    crash. ``primary`` copies whichever metric ``PRIMARY`` names for the
    paradigm, so the leaderboard has one agreed headline number to sort on while
    still keeping every metric.
    """
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
