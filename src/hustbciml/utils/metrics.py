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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, roc_auc_score)

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


def balanced_accuracy(y_true, y_pred, **_) -> float:
    """Mean per-class recall, as a percentage.

    Unlike plain accuracy this cannot be inflated by predicting the majority
    class: a detector that answers "non-target" for every P300 trial scores 50 on
    two classes, not 90-something. AUC ranks well but says nothing about where a
    method actually puts its decision threshold, so the two are reported together
    for the imbalanced paradigms.
    """
    return float(balanced_accuracy_score(y_true, y_pred) * 100)


def roc_auc(y_true, y_score, n_classes=2, positive_class=1, **_) -> float:
    """Area under the ROC curve from the model's class scores, as a percentage.

    AUC needs a continuous score per trial, not a hard label, and it measures
    how well those scores rank the classes. For two classes it takes the
    positive-class column (or the raw 1-D score) and computes the standard
    binary AUC. For more than two it uses one-vs-rest per class and averages
    them with equal class weight, matching ``macro_f1``'s equal weighting.

    ``positive_class`` is the column index treated as positive in the binary
    case. It defaults to 1, which is the convention every dataset adapter here
    follows (labels are encoded 0..C-1 in the dataset's own class order). It is a
    parameter rather than a hard-coded ``1`` so a paradigm whose positive class
    is encoded at 0 can say so instead of silently reporting the AUC of the
    opposite direction.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_score.ndim == 2 and y_score.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_score has {y_score.shape[0]} rows for {y_true.shape[0]} trials")
    if n_classes == 2:
        pos = y_score[:, positive_class] if y_score.ndim == 2 else y_score
        return float(roc_auc_score(y_true, pos) * 100)
    return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro") * 100)


def score(y_true, y_pred, y_score, paradigm="MI", n_classes=2) -> dict:
    """All metrics as a dict; ``primary`` marks the paradigm's headline number.

    ``y_pred`` are hard labels (used by accuracy, kappa, F1, balanced accuracy)
    and ``y_score`` are the continuous class scores (used by AUC). ``primary``
    copies whichever metric ``PRIMARY`` names for the paradigm, so the
    leaderboard has one agreed headline number to sort on while still keeping
    every metric.

    AUC is the one metric that can legitimately be undefined: it needs both
    classes present in ``y_true``, and a fold in which one never appears has no
    ROC curve. That single case is detected up front and recorded as NaN so the
    aggregation skips the fold for AUC instead of crashing. Every *other* failure
    — a malformed score array, a wrong class dimension, a None — is a bug, and it
    propagates. Catching those too, as a blanket ``except Exception`` did, turns a
    broken prediction path into a complete-looking results file, and for P300/ERP
    it degrades the *headline* number to NaN with nothing to show for it.
    """
    out = {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "kappa": cohen_kappa(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }
    if len(np.unique(np.asarray(y_true))) < 2:
        out["auc"] = float("nan")        # genuinely undefined for this fold
    else:
        out["auc"] = roc_auc(y_true, y_score, n_classes=n_classes)
    out["primary"] = out[PRIMARY.get(paradigm, "accuracy")]
    return out
