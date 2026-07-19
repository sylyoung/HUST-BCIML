# leaderboard.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Aggregate results/*/metrics.json into a ranked results index.

NOTE: this is a raw index of every run, ranked by accuracy — it is NOT a valid
head-to-head comparison, because rows can differ on different stage axes (a
transfer strategy vs a backbone swap change different things). For controlled,
one-axis-at-a-time comparisons with the right baseline in each group, use
``python -m hustbciml.scripts.compare``.

One row per algorithm. When an algorithm has several seed runs (setting dirs
that differ only by seed), they are collapsed into a single row: the primary
metric is averaged across seeds and the ``±`` becomes the std ACROSS SEEDS
(reproducibility). With a single seed, the ``±`` is that run's cross-subject
std (per-subject spread). The Seeds column disambiguates the two.

Usage: python -m hustbciml.scripts.leaderboard [results_dir] [--dataset NAME]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict


def _clean(xs):
    return [x for x in xs if x is not None and x == x]  # drop None + NaN


def _mean(xs):
    xs = _clean(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    """Population std; 0.0 for a single value."""
    xs = _clean(xs)
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def load_runs(results_dir: str, dataset: str = None):
    """One dict per metrics.json (one seed run), keeping the full stages map."""
    runs = []
    for path in glob.glob(os.path.join(results_dir, "*", "metrics.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if dataset and d.get("dataset") != dataset:
            continue
        s = d.get("summary", {})
        stages = d.get("stages", {})
        runs.append({
            "dataset": d.get("dataset"),
            "protocol": d.get("protocol"),
            "algorithm": d.get("algorithm") or "+".join(stages.values()),
            "stages": stages,
            "primary_mean": s.get("primary", {}).get("mean", float("nan")),
            "primary_std": s.get("primary", {}).get("std", float("nan")),
            "acc": s.get("accuracy", {}).get("mean", float("nan")),
            "kappa": s.get("kappa", {}).get("mean", float("nan")),
            "n_subjects": len(d.get("per_subject", [])),
        })
    return runs


def aggregate(runs, key):
    """Collapse seed runs sharing key(run) into one row (mean over seeds; ± is
    std across seeds when >1, else the single run's cross-subject std)."""
    groups = defaultdict(list)
    for r in runs:
        groups[key(r)].append(r)
    rows = []
    for _, rs in groups.items():
        per_seed = [r["primary_mean"] for r in rs]
        n_seeds = len(rs)
        std = _std(per_seed) if n_seeds >= 2 else rs[0]["primary_std"]
        rows.append({
            "dataset": rs[0]["dataset"],
            "protocol": rs[0]["protocol"],
            "algorithm": rs[0]["algorithm"],
            "stages": rs[0]["stages"],
            "primary_mean": _mean(per_seed),
            "primary_std": std,
            "acc": _mean([r["acc"] for r in rs]),
            "kappa": _mean([r["kappa"] for r in rs]),
            "n_seeds": n_seeds,
            "n_subjects": max((r["n_subjects"] for r in rs), default=0),
        })
    return rows


def collect(results_dir: str, dataset: str = None):
    runs = load_runs(results_dir, dataset)
    rows = aggregate(runs, key=lambda r: (r["dataset"], r["protocol"], r["algorithm"]))
    rows.sort(key=lambda r: (r["dataset"] or "", -(r["primary_mean"] or 0)))
    return rows


def render(rows) -> str:
    if not rows:
        return "(no results found)"
    L = ["| Dataset | Protocol | Algorithm | Primary | Acc | Kappa | Seeds | Subj |",
         "|---|---|---|--:|--:|--:|--:|--:|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['protocol']} | {r['algorithm']} | "
                 f"{r['primary_mean']:.2f}±{r['primary_std']:.2f} | {r['acc']:.2f} | "
                 f"{r['kappa']:.3f} | {r['n_seeds']} | {r['n_subjects']} |")
    L.append("")
    L.append("_Raw index, not a controlled comparison — use `compare` for that. "
             "± = std across seeds when Seeds>1, else the single run's cross-subject std._")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="./results")
    ap.add_argument("--dataset", default=None)
    ns = ap.parse_args()
    print(render(collect(ns.results_dir, ns.dataset)))


if __name__ == "__main__":
    main()
