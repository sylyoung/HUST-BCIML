#!/usr/bin/env python3
# extract_newmethods_3ds.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Aggregate the 6 new methods' 3-dataset x 3-seed results into YAML-ready numbers.

Methodology (identical to the existing benchmark + tune_networks):
  per seed  -> accuracy = mean over per-subject accuracies (LOSO cross-subject)
  per cell  -> acc_mean = mean over 3 seeds, acc_std = population std (ddof=0),
               kappa_mean = mean over 3 seeds.
Backbones (TIE-EEGNet, KDFNet) are read straight from tuned_<DS>.json, which
already stores acc_mean/acc_std/kappa_mean produced by tune_networks.

Run ON the GPU server (paths are absolute there). Prints a human table plus a
YAML `acc:` block per method for pasting into gallery/data/benchmark.yml.
"""
import json
import os
import statistics as st

RES = "/home/sylyoung/hustbciml_results_3ds"
NT = "/home/sylyoung/hustbciml_results_nettune"
DATASETS = ["BNCI2014001", "BNCI2014002", "BNCI2015001"]
SEEDS = [1, 2, 3]

# sweep methods -> (network-trained transfer/privacy). SAFE runs 4-class on 2014001.
SWEEP = ["MEKT", "MDMAML", "ASFA", "SAFE"]
# backbones tuned via tune_networks (display name == tuned-file key)
BACKBONES = ["TIE-EEGNet", "KDFNet"]


def ds_for(algo, ds):
    if algo == "SAFE" and ds == "BNCI2014001":
        return "BNCI2014001-4"
    return ds


def seed_cell(algo, ds, seed):
    """Return (acc_seed, kappa_seed) for one run, or None if missing."""
    d = os.path.join(RES, f"{ds_for(algo, ds)}_cross_subject_{algo}_seed{seed}", "metrics.json")
    if not os.path.exists(d):
        return None
    m = json.load(open(d))
    per = m.get("per_subject", [])
    if not per:
        return None
    acc = st.mean(x["accuracy"] for x in per)
    kap = st.mean(x["kappa"] for x in per)
    return acc, kap


def agg_sweep(algo, ds):
    cells = [seed_cell(algo, ds, s) for s in SEEDS]
    have = [c for c in cells if c is not None]
    if not have:
        return None
    accs = [c[0] for c in have]
    kaps = [c[1] for c in have]
    return {
        "acc_mean": round(st.mean(accs), 2),
        "acc_std": round(st.pstdev(accs) if len(accs) > 1 else 0.0, 2),
        "kappa_mean": round(st.mean(kaps), 3),
        "n_seeds": len(have),
    }


def agg_backbone(bb, ds):
    f = os.path.join(NT, f"tuned_{ds}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    node = d.get(ds, {})
    if bb not in node:
        return None
    e = node[bb]
    return {
        "acc_mean": round(e["acc_mean"], 2),
        "acc_std": round(e["acc_std"], 2),
        "kappa_mean": round(e["kappa_mean"], 3),
        "n_seeds": len(e.get("per_seed_acc", [])),
        "best_lr": e.get("best_lr"),
    }


def fmt(cell):
    if cell is None:
        return "   --incomplete-- "
    star = "" if cell["n_seeds"] == 3 else f" (!{cell['n_seeds']}/3)"
    return f"{cell['acc_mean']:5.2f}±{cell['acc_std']:.2f} k{cell['kappa_mean']:.3f}{star}"


def yaml_block(name, cells):
    dskey = {"BNCI2014001": "BNCI2014001", "BNCI2014002": "BNCI2014002", "BNCI2015001": "BNCI2015001"}
    lines = [f"      # {name}", "      acc:"]
    for ds in DATASETS:
        c = cells[ds]
        if c is None:
            lines.append(f"        # {dskey[ds]}: MISSING")
            continue
        lines.append(f"        {dskey[ds]}: {{mean: {c['acc_mean']}, std: {c['acc_std']}}}")
    return "\n".join(lines)


def main():
    print("=" * 78)
    print("NEW-METHODS 3-DATASET RESULTS  (acc mean±std, kappa)  [ddof=0]")
    print("=" * 78)
    hdr = f"{'method':14s} | " + " | ".join(f"{d:^22s}" for d in DATASETS)
    print(hdr)
    print("-" * len(hdr))

    all_cells = {}
    for algo in SWEEP:
        cells = {ds: agg_sweep(algo, ds) for ds in DATASETS}
        all_cells[algo] = cells
        print(f"{algo:14s} | " + " | ".join(f"{fmt(cells[d]):^22s}" for d in DATASETS))
    for bb in BACKBONES:
        cells = {ds: agg_backbone(bb, ds) for ds in DATASETS}
        all_cells[bb] = cells
        lr = next((cells[d]["best_lr"] for d in DATASETS if cells[d]), "?")
        print(f"{bb:14s} | " + " | ".join(f"{fmt(cells[d]):^22s}" for d in DATASETS))

    print("\n" + "=" * 78)
    print("YAML acc-blocks (paste into benchmark.yml rows):")
    print("=" * 78)
    for name, cells in all_cells.items():
        print(yaml_block(name, cells))
        print()

    # completeness summary
    print("=" * 78)
    missing = []
    for name, cells in all_cells.items():
        for ds in DATASETS:
            c = cells[ds]
            if c is None:
                missing.append(f"{name}/{ds}=MISSING")
            elif c["n_seeds"] < 3:
                missing.append(f"{name}/{ds}={c['n_seeds']}/3")
    print("INCOMPLETE:" if missing else "ALL COMPLETE (6 methods x 3 datasets x 3 seeds)")
    if missing:
        print("  " + "  ".join(missing))


if __name__ == "__main__":
    main()
