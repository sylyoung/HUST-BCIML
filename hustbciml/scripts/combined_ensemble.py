# combined_ensemble.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Source-combined heterogeneous ensemble (the NON-decentralized paradigm).

The counterpart to ``decentralized.py --base hetero``, which trains five models per
source subject and aggregates (N-1)*5 hard votes per target. This paradigm instead
trains FIVE architectures — EEGNet, ShallowConvNet, DeepConvNet, EEGConformer,
CSPNet — each ONCE on the POOLED source (all subjects except the target, Euclidean-
aligned), giving five per-trial predictions on the target. The diversity the
combiners exploit here comes from the heterogeneous *architectures* (different
inductive biases), not from the subjects.

Under leave-one-subject-out, for each target subject t the five nets are trained on
the union of the other subjects (5 * N trainings across a run) and each predicts t.
Aggregation is over HARD labels only — each net contributes a single predicted class
per trial, never a soft score — so hard majority voting is the baseline, alongside
the same crowd/lab combiners as the decentralized ensemble (Dawid-Skene / Wawa /
M-MSR / MACE / GLAD / ZenCrowd / PM / LA / LAA / EBCC and the lab's SML / SML-OVR /
StackingNet). Restricting to hard labels keeps the two paradigms comparable
column-for-column: the decentralized paradigm exposes only hard votes, so no
soft-averaging baseline is reported here either.

    python -m hustbciml.scripts.combined_ensemble --dataset BNCI2014001 \
        --seeds 1,2,3 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score

from hustbciml.core.config import resolve_config
from hustbciml.core.context import RunContext
from hustbciml.core.pipeline import build_pipeline
from hustbciml.algorithms.strategies._common import forward_logits, supervised_train
from hustbciml.algorithms.ensembles import build_combiners
from hustbciml.scripts.decentralized import _load_aligned, _onehot
from hustbciml.utils.seed import fix_random_seed, resolve_device

# name -> combiner instance, auto-discovered from algorithms/ensembles/ (one file per method).
COMBINERS = build_combiners()


# Five heterogeneous backbones (distinct inductive biases): EEGNet (compact
# depthwise-separable conv), ShallowConvNet (shallow FBCSP-style), DeepConvNet
# (deep 4-block conv), EEGConformer (conv stem + transformer), CSPNet (CSP-
# initialized spatial conv). Each trains on the same pooled EA-aligned source.
BACKBONES = ["EEGNet", "ShallowConvNet", "DeepConvNet", "EEGConformer", "CSPNet"]


def _combined_scores(cfg, dev, epochs_a, subjects, C, backbones):
    """LOSO source-combined training: for each target t, train each backbone on the
    POOLED other subjects (EA-aligned) and predict t. Returns per-target ground truth
    and hard one-hot votes — one per backbone (hard labels only, no soft scores)."""
    ytrue = {}
    hard = {t: {} for t in subjects}
    for t in subjects:
        src = epochs_a.select(epochs_a.domain != t)          # pooled source (all but t)
        tgt = epochs_a.select(epochs_a.domain == t)
        ytrue[t] = tgt.y
        for bb in backbones:
            cfg_bb = copy.deepcopy(cfg)
            cfg_bb.backbone = bb                              # same pooled data, different net
            pipe = build_pipeline(cfg_bb)
            model = pipe.model.to(dev)
            ctx = RunContext(cfg=cfg_bb, device=dev, augmenter=pipe.augmenter,
                             aligner=pipe.aligner, log=lambda m: None)
            supervised_train(model, src, ctx)
            logits = forward_logits(model, tgt, dev)
            hard[t][bb] = _onehot(logits.argmax(1), C)       # single predicted class per trial
    return ytrue, hard


def _seed_run(dataset, seed, device, data_dir, results_dir, algorithm, combiners, backbones):
    cfg, _ = resolve_config(["--algorithm", algorithm, "--dataset", dataset,
                             "--seed", str(seed), "--device", device,
                             "--data_dir", data_dir, "--results_dir", results_dir])
    fix_random_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    epochs_a, C = _load_aligned(cfg)
    subjects = [int(s) for s in np.unique(epochs_a.domain)]
    ytrue, hard = _combined_scores(cfg, dev, epochs_a, subjects, C, backbones)

    # single-model context: mean accuracy of one architecture alone (over models x
    # targets). Each one-hot vote's argmax is that model's predicted class.
    single = [accuracy_score(ytrue[t], v.argmax(1))
              for t in subjects for v in hard[t].values()]

    out = {}
    # hard-vote combiners — identical suite to decentralized.py (voting is the
    # baseline), so the two paradigms line up column-for-column. Hard labels only:
    # no soft-averaging, matching what the decentralized paradigm can express.
    for c in combiners:
        if c == "SML" and C != 2:                # binary SML undefined for >2 classes
            continue
        if c == "SML-OVR" and C == 2:            # multi-class one-vs-rest variant
            continue
        try:
            accs = []
            for t in subjects:
                stack = np.stack(list(hard[t].values()))     # (n_backbones, N, C) one-hot
                accs.append(accuracy_score(ytrue[t], COMBINERS[c](stack)))
            out[c] = float(np.mean(accs) * 100)
        except Exception as e:                   # a degenerate combiner must not kill the run
            print(f"[warn] combiner {c!r} failed, skipping it — {type(e).__name__}: {e}")
    return float(np.mean(single) * 100), out


def main(argv=None):
    p = argparse.ArgumentParser(prog="hustbciml.scripts.combined_ensemble",
                                description="source-combined heterogeneous ensemble")
    p.add_argument("--dataset", default="Toy")
    p.add_argument("--algorithm", default="EA-EEGNet",
                   help="base preset supplying aligner/training config (backbone is overridden per learner)")
    p.add_argument("--backbones", default=",".join(BACKBONES),
                   help="comma-separated heterogeneous backbones (default: the 5 above)")
    p.add_argument("--seeds", default="1,2,3", help="comma-separated seeds")
    p.add_argument("--device", default="auto")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--combiners",
                   default="voting,Dawid-Skene,Wawa,M-MSR,MACE,GLAD,ZenCrowd,PM,"
                           "LA,LAA,EBCC,SML,SML-OVR,StackingNet")
    a = p.parse_args(argv)

    seeds = [int(s) for s in a.seeds.split(",")]
    backbones = [b for b in a.backbones.split(",") if b]
    combiners = [c for c in a.combiners.split(",") if c]
    reported = combiners                            # hard-label combiners only (voting is the baseline)
    single_all, comb_all = [], {c: [] for c in reported}
    os.makedirs(a.results_dir, exist_ok=True)
    out_path = os.path.join(a.results_dir, f"combined_{a.dataset}_hetero_{a.algorithm}.json")

    for seed in seeds:
        single, out = _seed_run(a.dataset, seed, a.device, a.data_dir,
                                a.results_dir, a.algorithm, combiners, backbones)
        single_all.append(single)
        for c in out:
            comb_all.setdefault(c, []).append(out[c])
        json.dump({"dataset": a.dataset, "algorithm": a.algorithm, "backbones": backbones,
                   "seeds_done": seeds[:len(single_all)],
                   "single_model": single_all, "combiners": comb_all},
                  open(out_path, "w"), indent=2)     # incremental save (flaky-link safe)
        print(f"[seed {seed}] single-model {single:.2f} | "
              + " ".join(f"{c} {out[c]:.2f}" for c in out))

    def ms(v):
        arr = np.array(v)
        return arr.mean(), arr.std()

    print(f"\n=== source-combined heterogeneous ensemble ({'/'.join(backbones)}) "
          f"on {a.dataset} — {len(seeds)} seeds {seeds} ===")
    m, s = ms(single_all)
    print(f"single model (mean over backbone x target): {m:.2f} +/- {s:.2f}")
    print(f"{'combiner':14s} {'acc':>8s} {'std':>7s}   delta-vs-single")
    for c in reported:
        if not comb_all.get(c):
            reason = "binary only" if c == "SML" else "failed — see [warn] above"
            print(f"{c:14s}   (skipped: {reason})")
            continue
        m2, s2 = ms(comb_all[c])
        print(f"{c:14s} {m2:8.2f} {s2:7.2f}   {m2 - m:+.2f}")


if __name__ == "__main__":
    main()
