#!/usr/bin/env python3
# extract_augbb_3ds.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Aggregate the new augmenters and backbones (3-dataset x 3-seed) into YAML-ready numbers.

Methodology matches the existing benchmark exactly:
  per seed -> accuracy = mean over the per-subject accuracies (cross-subject LOSO)
  per cell -> acc_mean = mean over the 3 seeds, acc_std = population std (ddof=0),
              kappa_mean = mean over the 3 seeds.
Reads two result trees produced by sweep_augmenters.sh and sweep_backbones.sh, each
holding <dataset>_cross_subject_<algo>_seed<seed>/metrics.json. Prints a human table
plus a YAML `acc:` block per method for pasting into gallery/data/benchmark.yml.

Run ON the GPU server (paths are absolute there).
"""
import json
import os
import statistics as st

RES_AUG = "/home/sylyoung/hustbciml_results_aug"
RES_BB = "/home/sylyoung/hustbciml_results_backbones"
DATASETS = ["BNCI2014001", "BNCI2014002", "BNCI2015001"]
SEEDS = [1, 2, 3]

# preset name (as passed to --algorithm) -> display name for the benchmark row.
AUGMENTERS = {
    "Noise-EEGNet": "Noise", "Flip-EEGNet": "Flip", "Scale-EEGNet": "Scale",
    "FShift-EEGNet": "FShift", "FSurr-EEGNet": "FSurr", "FComb-EEGNet": "FComb",
    "HS-EEGNet": "HS",
}
BACKBONES = {
    "EA-ADFCNN": "ADFCNN", "EA-CTNet": "CTNet", "EA-MSCFormer": "MSCFormer",
    "EA-MSVTNet": "MSVTNet", "EA-TMSANet": "TMSA-Net", "EA-EEGWaveNet": "EEGWaveNet",
    "EA-SlimSeiz": "SlimSeiz", "EA-FBMSNet": "FBMSNet", "EA-EEGNeX": "EEGNeX",
    "EA-EEGDeformer": "EEG-Deformer",
}


def seed_cell(res_dir, algo, ds, seed):
    """Return (acc_seed, kappa_seed) for one run, or None if the metrics are missing."""
    d = os.path.join(res_dir, f"{ds}_cross_subject_{algo}_seed{seed}", "metrics.json")
    if not os.path.exists(d):
        return None
    m = json.load(open(d))
    per = m.get("per_subject", [])
    if not per:
        return None
    return st.mean(x["accuracy"] for x in per), st.mean(x["kappa"] for x in per)


def agg(res_dir, algo, ds):
    cells = [seed_cell(res_dir, algo, ds, s) for s in SEEDS]
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


def fmt(cell):
    if cell is None:
        return "   --missing--  "
    star = "" if cell["n_seeds"] == 3 else f" (!{cell['n_seeds']}/3)"
    return f"{cell['acc_mean']:5.2f}±{cell['acc_std']:.2f} k{cell['kappa_mean']:.3f}{star}"


def yaml_block(name, cells):
    lines = [f"      # {name}", "      acc:"]
    for ds in DATASETS:
        c = cells[ds]
        if c is None:
            lines.append(f"        # {ds}: MISSING")
        else:
            lines.append(f"        {ds}: {{mean: {c['acc_mean']}, std: {c['acc_std']}}}")
    return "\n".join(lines)


def section(title, res_dir, name_map):
    print("=" * 78)
    print(f"{title}  (acc mean±std, kappa)  [ddof=0]")
    print("=" * 78)
    hdr = f"{'method':14s} | " + " | ".join(f"{d:^22s}" for d in DATASETS)
    print(hdr)
    print("-" * len(hdr))
    all_cells = {}
    for algo, disp in name_map.items():
        cells = {ds: agg(res_dir, algo, ds) for ds in DATASETS}
        all_cells[disp] = cells
        print(f"{disp:14s} | " + " | ".join(f"{fmt(cells[d]):^22s}" for d in DATASETS))
    print("\nYAML acc-blocks:")
    for disp, cells in all_cells.items():
        print(yaml_block(disp, cells))
        print()
    # completeness
    missing = []
    for disp, cells in all_cells.items():
        for ds in DATASETS:
            c = cells[ds]
            if c is None:
                missing.append(f"{disp}/{ds}=MISSING")
            elif c["n_seeds"] < 3:
                missing.append(f"{disp}/{ds}={c['n_seeds']}/3")
    print("INCOMPLETE:" if missing else "ALL COMPLETE")
    if missing:
        print("  " + "  ".join(missing))
    print()


def main():
    section("AUGMENTERS 3-DATASET RESULTS", RES_AUG, AUGMENTERS)
    section("BACKBONES 3-DATASET RESULTS", RES_BB, BACKBONES)


if __name__ == "__main__":
    main()
