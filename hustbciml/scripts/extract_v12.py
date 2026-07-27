#!/usr/bin/env python3
# extract_v12.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Aggregate the v1.2.0 re-measurement sweep into a JSON report.

Reads the result tree written by ``rerun_v12.sh`` and applies the benchmark's
established aggregation, unchanged, so the new cells are comparable with the old:

    per seed  -> accuracy = mean over the per-subject accuracies (cross-subject LOSO)
    per cell  -> acc_mean = mean over seeds, acc_std = population std (ddof=0)

Per-subject accuracies are carried through as well. The control rows are only
meaningful at that resolution: they claim to be *untouched* by the release, and a
matching two-decimal mean would not distinguish "identical" from "different runs
that happened to average out". Comparison against the published values happens off
the server, against ``gallery/data/benchmark.yml``.

    python -m hustbciml.scripts.extract_v12 --out /home/sylyoung/v12_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

DATASETS = ["BNCI2014001", "BNCI2014002", "BNCI2015001"]


def read_run(res_dir, ds, algo, seed):
    """One run -> per-subject accuracies plus the seed-level means, or None."""
    path = os.path.join(res_dir, f"{ds}_cross_subject_{algo}_seed{seed}", "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        m = json.load(fh)
    per = m.get("per_subject", [])
    if not per:
        return None
    return {
        "per_subject_acc": [round(x["accuracy"], 6) for x in per],
        "acc": st.mean(x["accuracy"] for x in per),
        "kappa": st.mean(x["kappa"] for x in per),
    }


def aggregate(res_dir, algo, ds, seeds):
    runs = {s: read_run(res_dir, ds, algo, s) for s in seeds}
    have = {s: r for s, r in runs.items() if r}
    if not have:
        return None
    accs = [r["acc"] for r in have.values()]
    return {
        "seeds_done": sorted(have),
        "seeds_missing": sorted(s for s in seeds if s not in have),
        "acc_mean": round(st.mean(accs), 2),
        # Population std, matching every other cell in the leaderboard. With a
        # single seed it is 0.0 by definition, which is why a partial sweep must
        # never be read as a finished cell — `seeds_missing` says which.
        "acc_std": round(st.pstdev(accs) if len(accs) > 1 else 0.0, 2),
        "kappa_mean": round(st.mean(r["kappa"] for r in have.values()), 4),
        "per_seed": {str(s): r for s, r in have.items()},
    }


def collect_tuning(root):
    """Merge per-pair tuning verdicts back into one ``{method: {dataset: verdict}}`` map.

    Serves both tuning tools, because they write the same file: ``tune_networks.py``
    selects a backbone learning rate and ``tune_algorithm.py`` selects a method's own
    hyperparameters, and each records the choice in ``<dir>/tuned_<dataset>.json``.

    Each pair is tuned in its own results dir, because that file is rewritten
    read-modify-write and two workers sharing a dir would drop each other's entries.
    Reassemble here.
    """
    out = {}
    if not root or not os.path.isdir(root):
        return out
    for path in sorted(glob.glob(os.path.join(root, "*", "tuned_*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        for ds, per_bb in d.items():
            for bb, verdict in per_bb.items():
                out.setdefault(bb, {})[ds] = verdict
    return out


def collect_ensemble(results_dir):
    """Read the decentralized ensemble runs: {dataset: {combiner: [per-seed acc]}}."""
    out = {}
    if not results_dir or not os.path.isdir(results_dir):
        return out
    for path in sorted(glob.glob(os.path.join(results_dir, "decentralized_*_hetero_*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        combiners = {c: {"mean": round(st.mean(v), 2),
                         "std": round(st.pstdev(v) if len(v) > 1 else 0.0, 2),
                         "per_seed": [round(x, 4) for x in v]}
                     for c, v in (d.get("combiners") or {}).items() if v}
        ss = d.get("single_source") or []
        out[d["dataset"]] = {
            "seeds_done": d.get("seeds_done"),
            "single_source": {"mean": round(st.mean(ss), 2),
                              "std": round(st.pstdev(ss) if len(ss) > 1 else 0.0, 2)} if ss else None,
            "combiners": combiners,
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="hustbciml.scripts.extract_v12")
    p.add_argument("--results_dir", default="/home/sylyoung/hustbciml_v12_results")
    p.add_argument("--nettune_dir", default="/home/sylyoung/hustbciml_v12_nettune")
    p.add_argument("--algtune_dir", default="/home/sylyoung/hustbciml_v12_algtune")
    p.add_argument("--ensemble_dir", default="/home/sylyoung/hustbciml_v12_ensemble")
    p.add_argument("--out", default="/home/sylyoung/v12_report.json")
    p.add_argument("--seeds", default="1,2,3")
    a = p.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]

    # Discover what actually ran rather than assuming the intended job list, so a
    # sweep that is still in flight reports honestly instead of showing holes as
    # absent methods.
    algos = sorted({
        d.split("_cross_subject_")[1].rsplit("_seed", 1)[0]
        for d in os.listdir(a.results_dir)
        if "_cross_subject_" in d and os.path.isdir(os.path.join(a.results_dir, d))
    })

    report = {}
    for algo in algos:
        cells = {ds: aggregate(a.results_dir, algo, ds, seeds) for ds in DATASETS}
        report[algo] = {ds: c for ds, c in cells.items() if c}

    nettune = collect_tuning(a.nettune_dir)
    algtune = collect_tuning(a.algtune_dir)
    ensemble = collect_ensemble(a.ensemble_dir)
    # Keep the phases in one file under reserved keys, so compare_v12.py has a single
    # input. The leading underscore cannot collide with a preset name.
    #
    # The two tuning channels are kept apart because their keys mean different things:
    # a _nettune key is a bare backbone whose leaderboard row is EA-<backbone>, while an
    # _algtune key is already the preset name. Merging them would need the reader to
    # guess which naming rule applies.
    payload = dict(report)
    payload["_nettune"] = nettune
    payload["_algtune"] = algtune
    payload["_ensemble"] = ensemble
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)

    print(f"{'algorithm':22s} " + " ".join(f"{d.replace('BNCI',''):>18s}" for d in DATASETS))
    for algo in algos:
        row = []
        for ds in DATASETS:
            c = report[algo].get(ds)
            if not c:
                row.append(f"{'-':>18s}")
            else:
                mark = "" if not c["seeds_missing"] else f"*{len(c['seeds_done'])}"
                row.append(f"{c['acc_mean']:>12.2f}±{c['acc_std']:.2f}{mark:>4s}")
        print(f"{algo:22s} " + " ".join(row))

    if nettune:
        print(f"\n-- learning-rate-tuned backbones --")
        for bb in sorted(nettune):
            cells = " ".join(
                f"{ds.replace('BNCI','')}: {v['acc_mean']:.2f}±{v['acc_std']:.2f} (lr {v['best_lr']:g})"
                for ds, v in sorted(nettune[bb].items()))
            print(f"  {bb:16s} {cells}")
    if algtune:
        print(f"\n-- methods re-selected by tune_algorithm --")
        for name in sorted(algtune):
            for ds, v in sorted(algtune[name].items()):
                cfg = ", ".join(f"{k} {val:g}" for k, val in sorted(v["best_config"].items()))
                print(f"  {name:16s} {ds.replace('BNCI',''):>8s}: "
                      f"{v['acc_mean']:.2f}±{v['acc_std']:.2f}  [{v['select']}] {cfg}")
    if ensemble:
        print(f"\n-- decentralized ensemble --")
        for ds in sorted(ensemble):
            e = ensemble[ds]
            print(f"  {ds} (seeds {e['seeds_done']}): " +
                  " ".join(f"{c} {v['mean']:.2f}" for c, v in sorted(e["combiners"].items())))
    print(f"\nwrote {a.out}   (*N = only N seeds done so far)")


if __name__ == "__main__":
    main()
