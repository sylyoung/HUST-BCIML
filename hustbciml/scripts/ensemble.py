# ensemble.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Black-box test-time ensemble over K random seeds of one base algorithm.

Runs ``--algorithm`` for each ``--seed`` (reusing the normal Exp, so the ensemble
combines exactly the predictions the benchmark produces), then for every target
subject stacks the K seeds' per-trial predictions and fuses them with each post-hoc
black-box combiner — hard majority ``voting`` (the baseline), the crowd-label
aggregators (Dawid-Skene / Wawa / M-MSR / MACE / GLAD / ZenCrowd / PM / LA / LAA /
EBCC), and the lab's SML / SML-OVR / StackingNet (see ``algorithms/ensembles/``). Every
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
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score

from hustbciml import run as run_module
from hustbciml.algorithms.ensembles import build_combiners

# name -> combiner instance, auto-discovered from algorithms/ensembles/ (one file
# per method). Built once at import; the run selects combiners by name below.
COMBINERS = build_combiners()


def _setting_dir(results_dir, dataset, algorithm, seed, protocol="cross_subject"):
    return os.path.join(results_dir, f"{dataset}_{protocol}_{algorithm}_seed{seed}")


def _ensure_run(algorithm, dataset, seed, device, results_dir, data_dir):
    d = _setting_dir(results_dir, dataset, algorithm, seed)
    if os.path.exists(os.path.join(d, "predictions.npz")):
        # Reuse only after checking *what* was cached. The directory name encodes
        # dataset, protocol, algorithm and seed but not the preset's contents, the
        # data, or the code version, so a bare "the file exists" test happily feeds
        # stale base-model outputs into a published ensemble number. ``metrics.json``
        # carries the full resolved config next to it; confirm it at least describes
        # the run being asked for.
        meta = os.path.join(d, "metrics.json")
        if os.path.exists(meta):
            with open(meta) as fh:
                m = json.load(fh)
            got = (m.get("dataset"), m.get("algorithm"), (m.get("config") or {}).get("seed"))
            want = (dataset, algorithm, seed)
            if got[2] is not None and got != want:
                raise RuntimeError(
                    f"cached predictions in {d} describe {got}, not {want}; refusing to "
                    f"ensemble them. Delete the directory to regenerate."
                )
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


def _check_alignment(loaded, seeds):
    """Every seed must describe the same subjects, in the same order, with the
    same ground truth.

    The aggregation stacks ``loaded[si][2][j]`` positionally and scores the result
    against ``loaded[0]``'s labels. If one cached seed has a different subject
    order or a different trial order, that silently combines predictions for
    different held-out subjects and scores them against the wrong labels — and the
    output is a plausible accuracy, not an error.
    """
    ref_subjects, ref_true = loaded[0][0], loaded[0][1]
    for si in range(1, len(loaded)):
        subs, yt, _ = loaded[si]
        if not np.array_equal(subs, ref_subjects):
            raise RuntimeError(
                f"seed {seeds[si]} covers subjects {list(subs)} but seed {seeds[0]} "
                f"covers {list(ref_subjects)}; the runs are not aligned.")
        for j in range(len(ref_subjects)):
            if not np.array_equal(yt[j], ref_true[j]):
                raise RuntimeError(
                    f"seed {seeds[si]} has different ground-truth labels for subject "
                    f"{ref_subjects[j]} than seed {seeds[0]}; the trial order differs.")


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
    p.add_argument("--allow_failed_combiners", action="store_true",
                   help="continue when a combiner raises, recording it as failed "
                        "instead of aborting (off by default: a crashed aggregator "
                        "is not a measured low score)")
    a = p.parse_args(argv)

    np.random.seed(0)                                   # voting tie-breaks
    seeds = [int(s) for s in a.seeds.split(",")]
    combiners = [c for c in a.combiners.split(",") if c]
    K = len(seeds)

    dirs = [_ensure_run(a.algorithm, a.dataset, s, a.device, a.results_dir, a.data_dir) for s in seeds]
    loaded = [_load(d) for d in dirs]
    _check_alignment(loaded, seeds)
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
            if c in failed or (c == "SML" and C != 2):
                continue
            # SML-OVR is the multi-class one-vs-rest extension of SML and reduces
            # exactly to binary SML on two classes; report that under the SML-OVR
            # key so all three ensemble runners agree on what the row means,
            # rather than skipping it here and aliasing it in decentralized.py.
            fn = COMBINERS["SML"] if (c == "SML-OVR" and C == 2) else COMBINERS[c]
            try:
                results[c].append(accuracy_score(yt, fn(scores)))
            except Exception as e:
                # A combiner that crashed was not measured, and that is not the
                # same thing as one that scored poorly — but a dropped row and a
                # low row look identical downstream. Fail the run so the
                # difference cannot be lost; ``--allow_failed_combiners`` keeps
                # the old skip-and-continue behaviour for exploratory use.
                failed[c] = f"{type(e).__name__}: {e}"
                results[c] = []
                if not a.allow_failed_combiners:
                    raise RuntimeError(
                        f"combiner {c!r} failed on subject {subjects[j]}: {failed[c]}. "
                        f"Pass --allow_failed_combiners to record it as failed and continue."
                    ) from e
                print(f"[warn] combiner {c!r} failed, skipping it — {failed[c]}")

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
