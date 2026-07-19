# compare.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Controlled comparisons for the hustbciml benchmark.

A single flat ranking across all algorithms is NOT a valid comparison: rows can
differ on different stage axes (e.g. a transfer strategy vs a backbone swap
change different things), so ranking them together compares apples to oranges.

This renders one table per stage axis instead. Each table varies exactly ONE
axis and holds the others at the canonical configuration, and always includes
that axis's baseline (no-transfer / no-alignment / no-augmentation / the default
network), with a delta-vs-baseline column. So every row in a table differs from
the baseline in exactly one way -- a controlled comparison.

Canonical configuration: EA aligner | no augmentation | EEGNet backbone |
Linear head | ERM strategy  (== the EA-EEGNet reference algorithm).

The two classical, network-free pipelines (CSP+LDA, Riemannian MDM) have no
backbone to hold fixed, so they form their own category, shown against the deep
EA-EEGNet reference.

Usage: python -m hustbciml.scripts.compare [results_dir] [--dataset NAME]
"""
from __future__ import annotations

import argparse

from hustbciml.scripts.leaderboard import load_runs, aggregate

CANONICAL = {"aligner": "EA", "augmenter": "Identity", "backbone": "EEGNet",
             "head": "Linear", "strategy": "ERM"}

FIT_MODE_STRATEGIES = {"CSP_LDA", "RiemannMDM"}  # network-free classical pipelines
SOURCE_FREE_STRATEGIES = {"LSFT", "MSDT"}        # lab classical source-free transfer

# friendly display names per axis value
DISPLAY = {
    "aligner":   {"EA": "EA", "Identity": "none (no alignment)",
                  "RA": "RA", "LA": "LA", "MEKT": "MEKT"},
    "augmenter": {"Identity": "none (no augmentation)",
                  "ChannelReflection": "Channel Reflection", "CSDA": "CSDA"},
    "backbone":  {"EEGNet": "EEGNet", "ShallowConvNet": "ShallowConvNet",
                  "DeepConvNet": "DeepConvNet", "EEGConformer": "EEGConformer",
                  "DBConformer": "DBConformer", "CSPNet": "CSP-Net"},
    "strategy":  {"ERM": "ERM (no transfer)", "DANN": "DANN", "CDAN": "CDAN",
                  "MCC": "MCC", "DAN": "DAN", "JAN": "JAN", "MDD": "MDD",
                  "TTIME": "T-TIME", "Tent": "Tent", "PL": "PL",
                  "BNAdapt": "BN-adapt", "ABAT": "ABAT", "BFT": "BFT",
                  "CSP_LDA": "CSP-LDA", "RiemannMDM": "Riemann-MDM",
                  "DJPMMD": "DJP-MMD", "LSFT": "LSFT", "MSDT": "MSDT"},
}

COMPARISONS = [
    {"title": "Network (backbone)",
     "axis": "backbone",
     "fixed": {"aligner": "EA", "augmenter": "Identity", "strategy": "ERM"},
     "baseline": "EEGNet",
     "blurb": "EA + ERM, no aug; vary the deep architecture."},
    {"title": "Alignment (transfer by distribution alignment)",
     "axis": "aligner",
     "fixed": {"augmenter": "Identity", "backbone": "EEGNet", "strategy": "ERM"},
     "baseline": "Identity",
     "blurb": "EEGNet + ERM, no aug; vary the aligner. Baseline = no alignment."},
    {"title": "Transfer / adaptation strategy",
     "axis": "strategy",
     "fixed": {"aligner": "EA", "augmenter": "Identity", "backbone": "EEGNet"},
     "baseline": "ERM",
     "exclude": FIT_MODE_STRATEGIES,
     "blurb": "EA + EEGNet, no aug; vary the training/adaptation procedure. "
              "Baseline = ERM (no transfer). All on the identical EEGNet network."},
    {"title": "Augmentation (no-alignment regime)",
     "axis": "augmenter",
     "fixed": {"aligner": "Identity", "backbone": "EEGNet", "strategy": "ERM"},
     "baseline": "Identity",
     "blurb": "EEGNet + ERM; vary the augmenter. Baseline = no augmentation. "
              "Held at no-alignment (aligner: Identity): Channel Reflection is an "
              "electrode-space transform and must precede any spatial whitening."},
    {"title": "Augmentation (EA regime)",
     "axis": "augmenter",
     "fixed": {"aligner": "EA", "backbone": "EEGNet", "strategy": "ERM"},
     "baseline": "Identity",
     "blurb": "EA + EEGNet + ERM; vary the augmenter. Baseline = EA-EEGNet (no "
              "augmentation). CSDA (db4-wavelet cross-subject detail-swap) operates "
              "on EA-aligned trials, so it is measured in the EA regime -- unlike "
              "Channel Reflection, an electrode-space transform held at no-alignment."},
]


def _matches(run, fixed):
    return all(run["stages"].get(k) == v for k, v in fixed.items())


def _render_axis(runs, spec):
    axis = spec["axis"]
    exclude = spec.get("exclude", set())
    sel = [r for r in runs
           if _matches(r, spec["fixed"]) and r["stages"].get(axis) not in exclude]
    rows = aggregate(sel, key=lambda r: r["stages"].get(axis))
    disp = DISPLAY.get(axis, {})
    base = next((x for x in rows if x["stages"].get(axis) == spec["baseline"]), None)
    base_acc = base["acc"] if base else None
    rows.sort(key=lambda r: -(r["acc"] or 0))

    L = [f"### {spec['title']}",
         f"_{spec['blurb']}_", "",
         f"| {axis.capitalize()} | Acc | Kappa | Δacc vs base | Seeds |",
         "|---|--:|--:|--:|--:|"]
    if not rows:
        L.append("| _(no runs yet)_ | | | | |")
        return "\n".join(L)
    for r in rows:
        val = r["stages"].get(axis)
        name = disp.get(val, val)
        is_base = (val == spec["baseline"])
        if is_base:
            delta = "(baseline)"
        elif base_acc is not None:
            delta = f"{r['acc'] - base_acc:+.2f}"
        else:
            delta = "-"
        L.append(f"| {name} | {r['acc']:.2f} | {r['kappa']:.3f} | {delta} | "
                 f"{r['n_seeds']} |")
    return "\n".join(L)


def _render_classical(runs):
    disp = DISPLAY["strategy"]
    sel = [r for r in runs if r["stages"].get("strategy") in FIT_MODE_STRATEGIES]
    rows = aggregate(sel, key=lambda r: r["stages"].get("strategy"))
    rows.sort(key=lambda r: -(r["acc"] or 0))
    ref_rows = aggregate([r for r in runs if _matches(r, CANONICAL)],
                         key=lambda r: "ref")

    L = ["### Classical (network-free) baselines",
         "_EA-aligned trials -> a classical pipeline (no backbone). Shown against "
         "the deep EA-EEGNet reference._", "",
         "| Method | Acc | Kappa | Seeds |", "|---|--:|--:|--:|"]
    for r in rows:
        name = disp.get(r["stages"].get("strategy"), r["stages"].get("strategy"))
        L.append(f"| {name} | {r['acc']:.2f} | {r['kappa']:.3f} | {r['n_seeds']} |")
    for r in ref_rows:
        L.append(f"| _EA-EEGNet (deep reference)_ | {r['acc']:.2f} | "
                 f"{r['kappa']:.3f} | {r['n_seeds']} |")
    return "\n".join(L)


def _render_sourcefree(runs):
    """Lab classical source-free transfer methods (LSFT, MSDT). They use their own
    tangent-space alignment (aligner Identity) and their own strategy names, so they
    match none of the single-axis tables; shown here against the EA-EEGNet reference."""
    disp = DISPLAY["strategy"]
    sel = [r for r in runs if r["stages"].get("strategy") in SOURCE_FREE_STRATEGIES]
    rows = aggregate(sel, key=lambda r: r["stages"].get("strategy"))
    rows.sort(key=lambda r: -(r["acc"] or 0))
    ref_rows = aggregate([r for r in runs if _matches(r, CANONICAL)], key=lambda r: "ref")
    ref_acc = ref_rows[0]["acc"] if ref_rows else None

    L = ["### Source-free transfer (classical, lab)",
         "_Lab source-free / privacy-preserving transfer on Riemannian tangent-space "
         "features (no raw source data at transfer time; transductive). Distinct from the "
         "no-transfer classical baselines. Shown against the deep EA-EEGNet reference._", "",
         "| Method | Acc | Kappa | Δacc vs ref | Seeds |", "|---|--:|--:|--:|--:|"]
    if not rows:
        L.append("| _(no runs yet)_ | | | | |")
        return "\n".join(L)
    for r in rows:
        name = disp.get(r["stages"].get("strategy"), r["stages"].get("strategy"))
        delta = f"{r['acc'] - ref_acc:+.2f}" if ref_acc is not None else "-"
        L.append(f"| {name} | {r['acc']:.2f} | {r['kappa']:.3f} | {delta} | {r['n_seeds']} |")
    for r in ref_rows:
        L.append(f"| _EA-EEGNet (deep reference)_ | {r['acc']:.2f} | {r['kappa']:.3f} | "
                 f"(ref) | {r['n_seeds']} |")
    return "\n".join(L)


def render(runs) -> str:
    if not runs:
        return "(no results found)"
    blocks = [_render_axis(runs, spec) for spec in COMPARISONS]
    blocks.append(_render_classical(runs))
    blocks.append(_render_sourcefree(runs))
    header = ("# Controlled comparisons -- BNCI2014001, cross-subject LOSO\n"
              "Each table varies ONE stage axis; the rest are held at the canonical "
              "config (EA . no-aug . EEGNet . Linear . ERM). "
              "Δacc = accuracy minus that table's baseline.\n")
    return header + "\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="./results")
    ap.add_argument("--dataset", default=None)
    ns = ap.parse_args()
    print(render(load_runs(ns.results_dir, ns.dataset)))


if __name__ == "__main__":
    main()
