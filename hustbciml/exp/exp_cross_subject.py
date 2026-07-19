# exp_cross_subject.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Cross-subject leave-one-subject-out protocol.

For each target subject: source = every other subject. Fit the aligner on the
source domains, align source; a *gradient* strategy also aligns the target
offline (by the target's own reference — label-free), while a *tta* strategy
receives the raw target stream and aligns it online itself. Train, predict,
score; aggregate across subjects.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from hustbciml.core.batch import UNLABELED
from hustbciml.core.context import RunContext
from hustbciml.core.pipeline import build_pipeline
from hustbciml.data_provider.splitters import cross_subject, list_targets
from hustbciml.utils.metrics import score
from hustbciml.utils.seed import fix_random_seed
from .exp_basic import Exp_Basic


class Exp_CrossSubject(Exp_Basic):
    def run(self):
        cfg = self.cfg
        fix_random_seed(cfg.seed)
        epochs = self._get_data()
        targets = list_targets(epochs)
        # Hyperparameter-tuning hook (honest, opt-in): when ``hp.dev_targets`` is
        # set, score only those subjects as held-out pseudo-targets (sources = all
        # others, exactly as in the real protocol). The tuner selects one global HP
        # on this source-only signal, never peeking at the reported target results.
        dev = cfg.hp.get("dev_targets")
        if dev is not None:
            want = {int(s) for s in (dev if isinstance(dev, (list, tuple)) else str(dev).split(","))}
            targets = [t for t in targets if int(t) in want]
        print(f"[data] {cfg.dataset}: {len(epochs)} trials, {len(targets)} subjects, "
              f"C={cfg.n_chans} T={cfg.n_times} classes={cfg.n_classes} sfreq={cfg.sfreq}"
              f"{' [DEV tuning subset]' if dev is not None else ''}")

        per_subject = []
        predictions = []
        val_scores = []
        for tid in targets:
            pipe = build_pipeline(cfg)
            model = pipe.model.to(self.device)

            source, target = cross_subject(epochs, tid)
            pipe.aligner.fit(source)
            source_a = pipe.aligner.transform(source)

            # target alignment: offline (own reference) for gradient/fit strategies;
            # raw for tta (the strategy aligns online).
            is_tta = pipe.strategy.mode == "tta"
            target_a = None if is_tta else pipe.aligner.transform(target)

            # transductive strategies get the aligned target with labels masked out
            target_unlabeled = None
            if not is_tta and getattr(pipe.strategy, "uses_target", False):
                target_unlabeled = replace(
                    target_a, y=np.full(len(target_a), UNLABELED, dtype=np.int64))

            ctx = RunContext(cfg=cfg, device=self.device, augmenter=pipe.augmenter,
                             aligner=pipe.aligner, log=lambda m: None,
                             target_unlabeled=target_unlabeled)

            pipe.strategy.fit(model, source_a, ctx)
            val_scores.append(getattr(model, "_val_score", None))

            if is_tta:
                y_pred, y_score = pipe.strategy.predict(model, target, ctx)   # raw; online align
            else:
                y_pred, y_score = pipe.strategy.predict(model, target_a, ctx)

            m = score(target.y, y_pred, y_score, paradigm=epochs.paradigm,
                      n_classes=epochs.n_classes)
            per_subject.append(m)
            predictions.append({"subject": tid, "y_true": target.y, "y_score": y_score})
            print(f"[S{tid}] primary={m['primary']:.2f}  acc={m['accuracy']:.2f}  "
                  f"kappa={m['kappa']:.3f}  auc={m['auc']:.2f}")

        summary = self.aggregate(per_subject)
        # mean held-out-source validation accuracy across folds — used only for
        # hyperparameter selection (never for reporting), so it stays out of the
        # test metrics but is recorded alongside them for the tuner to read.
        _vv = [v for v in val_scores if v is not None]
        if _vv:
            summary["val_primary"] = {"mean": float(np.mean(_vv)), "std": float(np.std(_vv))}
        out_dir = self.save_results(per_subject, summary, predictions=predictions)
        p = summary["primary"]
        print(f"\n== {cfg.setting()} ==")
        print(f"primary {p['mean']:.2f} +/- {p['std']:.2f}  "
              f"(acc {summary['accuracy']['mean']:.2f}, kappa {summary['kappa']['mean']:.3f})")
        print(f"saved -> {out_dir}/metrics.json")
        return summary


PROTOCOLS = {"cross_subject": Exp_CrossSubject}
