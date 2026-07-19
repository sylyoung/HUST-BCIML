# ensemble.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Black-box test-time ensemble over K random seeds of one base algorithm.

Runs ``--algorithm`` for each ``--seed`` (reusing the normal Exp, so the ensemble
combines exactly the predictions the benchmark produces), then for every target
subject stacks the K seeds' per-trial predictions and fuses them with each post-hoc
black-box combiner — hard majority ``voting`` (the baseline), the crowd-label
aggregators (Dawid-Skene / Wawa / M-MSR / MACE / GLAD / ZenCrowd / PM / LA / LAA /
EBCC), and the lab's SML / SML-OVR / StackingNet (see ``_ensembles.py``). Every
combiner sees only hard votes — there is deliberately no soft-score averaging
combiner, so none has an information advantage over the label-only aggregators.
Reports per-combiner accuracy mean ± std across subjects, against the single-seed
base for reference.

    python -m hustbciml.scripts.ensemble --algorithm T-TIME --dataset BNCI2014001 \
        --seeds 1,2,3,4,5 --device cuda

Needs >= a few seeds to be meaningful (the lab uses 5-11). Each seed is a full
run, so this is a server job on real data; on Toy it runs locally in seconds.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score

from hustbciml import run as run_module
from hustbciml.scripts._ensembles import COMBINERS


def _setting_dir(results_dir, dataset, algorithm, seed, protocol="cross_subject"):
    return os.path.join(results_dir, f"{dataset}_{protocol}_{algorithm}_seed{seed}")


def _ensure_run(algorithm, dataset, seed, device, results_dir, data_dir):
    d = _setting_dir(results_dir, dataset, algorithm, seed)
    if os.path.exists(os.path.join(d, "predictions.npz")):
        print(f"[skip] seed {seed} already has predictions ({d})")
        return d
    print(f"[run ] {algorithm} on {dataset}, seed {seed}")
    run_module.main(["--algorithm", algorithm, "--dataset", dataset, "--seed", str(seed),
                     "--itr", "1", "--device", device,
                     "--results_dir", results_dir, "--data_dir", data_dir])
    return d


def _load(d):
    z = np.load(os.path.join(d, "predictions.npz"), allow_pickle=True)
    return z["subjects"], z["y_true"], z["y_score"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="hustbciml.scripts.ensemble",
                                description="black-box multi-seed ensemble")
    p.add_argument("--algorithm", required=True, help="base preset to ensemble (e.g. T-TIME)")
    p.add_argument("--dataset", default="Toy")
    p.add_argument("--seeds", default="1,2,3,4,5", help="comma-separated seeds")
    p.add_argument("--device", default="auto")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--combiners",
                   default="voting,Dawid-Skene,Wawa,M-MSR,MACE,GLAD,ZenCrowd,PM,"
                           "LA,LAA,EBCC,SML,SML-OVR,StackingNet")
    a = p.parse_args(argv)

    np.random.seed(0)                                   # voting tie-breaks
    seeds = [int(s) for s in a.seeds.split(",")]
    combiners = [c for c in a.combiners.split(",") if c]
    K = len(seeds)

    dirs = [_ensure_run(a.algorithm, a.dataset, s, a.device, a.results_dir, a.data_dir) for s in seeds]
    loaded = [_load(d) for d in dirs]
    subjects = loaded[0][0]

    # single-seed base accuracy (mean over seeds), for reference
    base = [np.mean([accuracy_score(yt[j], ys[j].argmax(1)) for j in range(len(subs))])
            for subs, yt, ys in loaded]
    print(f"\n=== {a.algorithm} on {a.dataset} — {K} seeds {seeds} ===")
    print(f"base single-seed acc: {np.mean(base) * 100:.2f} +/- {np.std(base) * 100:.2f}")

    results = {c: [] for c in combiners}
    failed = {}                                         # combiner -> error string, if it raised
    for j in range(len(subjects)):
        yt = loaded[0][1][j]
        C = loaded[0][2][j].shape[1]
        scores = np.stack([loaded[si][2][j] for si in range(K)])     # (K, N, C)
        for c in combiners:
            # binary SML needs exactly 2 classes; SML-OVR is the multi-class (K>2)
            # one-vs-rest extension, so it does not apply when there are 2 classes.
            if c in failed or (c == "SML" and C != 2) or (c == "SML-OVR" and C == 2):
                continue
            try:
                results[c].append(accuracy_score(yt, COMBINERS[c](scores)))
            except Exception as e:                      # a degenerate combiner (e.g. crowdkit
                failed[c] = f"{type(e).__name__}: {e}"  # M-MSR on too few / near-identical seeds)
                results[c] = []                         # must not kill the whole run — drop it,
                print(f"[warn] combiner {c!r} failed, skipping it — {failed[c]}")  # keep the rest

    print(f"{'combiner':14s} {'acc':>8s} {'std':>7s}   delta-vs-base")
    for c in combiners:
        if c in failed:
            print(f"{c:14s}   (failed: {failed[c][:48]})")
            continue
        if not results[c]:
            why = "multi-class only" if c == "SML-OVR" else "binary only"
            print(f"{c:14s}   (skipped: {why})")
            continue
        arr = np.array(results[c]) * 100
        print(f"{c:14s} {arr.mean():8.2f} {arr.std():7.2f}   {arr.mean() - np.mean(base) * 100:+.2f}")


if __name__ == "__main__":
    main()
