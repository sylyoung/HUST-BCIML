# exp_cross_subject.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Cross-subject leave-one-subject-out protocol.

A fold can run normally or in ``selection_only`` mode. Selection-only mode is
the firewall used by nested tuning: it trains on source subjects and returns the
source-validation score without aligning, predicting, scoring, printing, or
otherwise consulting the outer target subject.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hustbciml.core.batch import UNLABELED, EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.pipeline import build_pipeline
from hustbciml.data_provider.splitters import cross_subject, list_targets
from hustbciml.utils.metrics import score
from hustbciml.utils.seed import fix_random_seed
from .exp_basic import Exp_Basic


def _mask_labels(epochs):
    """A copy of ``epochs`` with every label replaced by ``UNLABELED``."""
    return replace(epochs, y=np.full(len(epochs), UNLABELED, dtype=np.int64))


@dataclass
class FoldResult:
    target: int
    val_primary: float | None
    metrics: dict | None = None
    prediction: dict | None = None


class Exp_CrossSubject(Exp_Basic):
    def run_fold(self, epochs: EEGEpochs, target_id: int, *,
                 selection_only: bool = False) -> FoldResult:
        """Run one LOSO fold without saving an artifact.

        ``selection_only=True`` is intentionally narrower than an ordinary fold:
        only the source partition is constructed. The target is not passed to an
        aligner or strategy and its labels are never scored. This is what lets a
        nested tuner compare candidates without evaluating the outer test fold.
        """
        cfg = self.cfg
        target_id = int(target_id)
        pipe = build_pipeline(cfg)
        model = pipe.model.to(self.device)

        # Construct source directly. In selection-only mode there is deliberately
        # no target object below this line, so future preprocessing cannot start
        # reading target data merely because cross_subject returned it.
        source = epochs.select(epochs.domain != target_id)
        if not len(source):
            raise ValueError(f"target {target_id} leaves no source trials")
        pipe.aligner.fit(source)
        source_a = pipe.aligner.transform(source)

        if selection_only and (
            pipe.strategy.mode != "gradient" or getattr(pipe.strategy, "uses_target", False)
        ):
            raise ValueError(
                f"selection-only folds require a source-only gradient strategy; "
                f"{cfg.strategy!r} has mode={pipe.strategy.mode!r}, "
                f"uses_target={getattr(pipe.strategy, 'uses_target', False)}"
            )

        log = (lambda message: print(message, flush=True)) if cfg.verbose else (lambda message: None)
        if selection_only:
            ctx = RunContext(
                cfg=cfg, device=self.device, augmenter=pipe.augmenter,
                aligner=pipe.aligner, log=log, target_unlabeled=None,
            )
            pipe.strategy.fit(model, source_a, ctx)
            val_primary = getattr(model, "_val_score", None)
            if val_primary is None:
                raise RuntimeError(
                    f"strategy {cfg.strategy!r} produced no source-validation score for "
                    f"selection-only target {target_id}"
                )
            return FoldResult(target=target_id, val_primary=float(val_primary))

        # The ordinary evaluation path starts consulting the target only after the
        # selection-only return above.
        _, target = cross_subject(epochs, target_id)
        is_tta = pipe.strategy.mode == "tta"
        target_masked = _mask_labels(target)
        target_a = None if is_tta else replace(
            pipe.aligner.transform(target_masked), y=target.y
        )

        target_unlabeled = None
        if not is_tta and getattr(pipe.strategy, "uses_target", False):
            target_unlabeled = _mask_labels(target_a)
        ctx = RunContext(
            cfg=cfg, device=self.device, augmenter=pipe.augmenter,
            aligner=pipe.aligner, log=log, target_unlabeled=target_unlabeled,
        )
        pipe.strategy.fit(model, source_a, ctx)
        val_primary = getattr(model, "_val_score", None)

        if is_tta:
            y_pred, y_score = pipe.strategy.predict(model, target, ctx)
        else:
            y_pred, y_score = pipe.strategy.predict(model, target_a, ctx)

        metrics = score(
            target.y, y_pred, y_score, paradigm=epochs.paradigm,
            n_classes=epochs.n_classes,
        )
        prediction = {
            "subject": target_id, "y_true": target.y,
            "y_pred": y_pred, "y_score": y_score,
        }
        return FoldResult(
            target=target_id,
            val_primary=None if val_primary is None else float(val_primary),
            metrics=metrics,
            prediction=prediction,
        )

    def run(self):
        """Run one full leave-one-subject-out sweep and save its artifacts."""
        cfg = self.cfg
        fix_random_seed(cfg.seed)
        epochs = self._get_data()
        targets = list_targets(epochs)

        # Development-subset runs use held-out subject labels as a selection signal
        # and are therefore tagged so they cannot be mistaken for reportable LOSO.
        dev = cfg.hp.get("dev_targets")
        if dev is not None:
            requested = {
                int(value) for value in
                (dev if isinstance(dev, (list, tuple)) else str(dev).split(","))
            }
            targets = [target for target in targets if int(target) in requested]
            dev_tag = "dev" + "-".join(str(value) for value in sorted(requested))
            cfg.run_tag = f"{cfg.run_tag}_{dev_tag}" if cfg.run_tag else dev_tag

        print(
            f"[data] {cfg.dataset}: {len(epochs)} trials, {len(targets)} subjects, "
            f"C={cfg.n_chans} T={cfg.n_times} classes={cfg.n_classes} sfreq={cfg.sfreq}"
            f"{' [DEV tuning subset — selection signal, not a reportable LOSO result]' if dev is not None else ''}"
        )

        per_subject = []
        predictions = []
        val_scores = []
        for target_id in targets:
            if cfg.fold_seed:
                fix_random_seed(cfg.seed * 1000 + int(target_id))
            result = self.run_fold(epochs, int(target_id))
            per_subject.append(result.metrics)
            predictions.append(result.prediction)
            if result.val_primary is not None:
                val_scores.append(result.val_primary)
            metrics = result.metrics
            print(
                f"[S{target_id}] primary={metrics['primary']:.2f}  "
                f"acc={metrics['accuracy']:.2f}  kappa={metrics['kappa']:.3f}  "
                f"auc={metrics['auc']:.2f}"
            )

        summary = self.aggregate(per_subject)
        if val_scores:
            summary["val_primary"] = {
                "mean": float(np.mean(val_scores)), "std": float(np.std(val_scores)),
            }
        out_dir = self.save_results(per_subject, summary, predictions=predictions)
        primary = summary["primary"]
        print(f"\n== {cfg.setting()} ==")
        print(
            f"primary {primary['mean']:.2f} +/- {primary['std']:.2f}  "
            f"(acc {summary['accuracy']['mean']:.2f}, "
            f"kappa {summary['kappa']['mean']:.3f})"
        )
        print(f"saved -> {out_dir}/metrics.json")
        return summary


PROTOCOLS = {"cross_subject": Exp_CrossSubject}
