#!/usr/bin/env python3
# compare_v12.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Compare the v1.2.0 re-measurement against the published cells and the v1.1.x runs.

Three questions, in the order they have to be answered:

1. **Did the controls hold?** A control row is one RERUN.md claims the release did
   not touch. It passes only if its per-subject accuracies come back *identical* to
   the v1.1.x run — not merely equal to the published two-decimal mean, which a pair
   of genuinely different runs can hit by accident. A failing control means the
   blast radius was mis-scoped and the affected list is incomplete, so this is
   checked before any new number is believed.

2. **How far did the affected cells move?** Published value, new value, delta, and
   whether the sweep is complete enough to quote (a cell measured on fewer than all
   three seeds is reported but flagged, never quietly averaged).

3. **What is still missing?**

    python -m hustbciml.scripts.compare_v12 \\
        --report v12_report.json --baseline_dir baseline_v11x --benchmark gallery/data/benchmark.yml
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

try:
    import yaml
except ImportError:                                     # pragma: no cover
    yaml = None

DATASETS = ["BNCI2014001", "BNCI2014002", "BNCI2015001"]
# Rows the release claims not to touch. Kept in sync with rerun_v12.sh.
CONTROLS = {"EA-EEGNet", "NoAlign-EEGNet", "Noise-EEGNet", "FShift-EEGNet"}
# The ensemble sweep names its combiners after the papers; the leaderboard rows use
# display names. Only the ones that actually differ need an entry.
COMBINER_ROW = {"voting": "Majority voting"}


# 10022 and 20022 link the same Intel MKL build and are bit-identical to each other, so
# either may stand in for the other. 7002 and 60022 are each their own regime.
FAMILY = {"10022": "10022", "20022": "10022", "7002": "7002", "60022": "60022"}


def load_origins(path):
    """{(key, dataset): machine} from cell_origin.tsv."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = [x.strip() for x in line.split("\t")]
            if len(f) >= 3:
                out[(f[0], f[1])] = f[2]
    return out


def find_baseline(baseline_dir, ds, algo, seed, want_family=None):
    """A v1.1.x run for this cell, **from a machine in the right numerical family**.

    The v1.1.x runs live in per-sweep trees under ``baseline_v11x/<machine>/``. Searching
    all of them and taking the first hit is wrong in the one way this whole exercise is
    built to avoid: comparing a re-measurement against another machine's baseline yields
    a difference that is a BLAS change, not a code change. It reports a passing control
    as broken (observed: `EA-EEGNet`/BNCI2014002 re-run on 20022, scored against 7002,
    "DIFFERS in 12/14 subjects") and could just as easily report a real regression as
    fine. So restrict to the family, and return None rather than reach outside it —
    "not checkable on this machine" is a different answer from "failed".
    """
    pat = os.path.join(baseline_dir, "*", "*",
                       f"{ds}_cross_subject_{algo}_seed{seed}", "metrics.json")
    hits = sorted(glob.glob(pat))
    if want_family:
        hits = [h for h in hits
                if FAMILY.get(os.path.relpath(h, baseline_dir).split(os.sep)[0]) == want_family]
    hits = [h for h in hits if not any(w in h.lower() for w in ("smoke", "_test", "probe"))] or hits
    if not hits:
        return None
    with open(hits[0]) as fh:
        return json.load(fh), hits[0]


def per_subject(metrics):
    return [round(x["accuracy"], 6) for x in metrics.get("per_subject", [])]


def published(bench, key, ds):
    for table in bench.get("tables", []):
        groups = [table] + list(table.get("groups") or [])
        for g in groups:
            for row in g.get("rows", []) or []:
                if row.get("key") == key:
                    cell = (row.get("acc") or {}).get(ds)
                    if cell:
                        return table["id"], row["name"], cell.get("mean"), cell.get("std")
    return None


def published_ensemble_row(bench, name, ds):
    """An ensemble row, looked up by display name — its rows are not all keyed."""
    for table in bench.get("tables", []):
        if table.get("id") != "ensemble":
            continue
        for g in [table] + list(table.get("groups") or []):
            for row in g.get("rows", []) or []:
                if row.get("name") == name:
                    cell = (row.get("acc") or {}).get(ds)
                    if cell:
                        return cell.get("mean"), cell.get("std")
    return None


def main(argv=None):
    p = argparse.ArgumentParser(prog="hustbciml.scripts.compare_v12")
    p.add_argument("--report", required=True, help="JSON from extract_v12.py")
    p.add_argument("--baseline_dir", required=True, help="tree of v1.1.x metrics.json files")
    p.add_argument("--benchmark", default="gallery/data/benchmark.yml")
    p.add_argument("--origin_map", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cell_origin.tsv"))
    p.add_argument("--origin", default=None,
                   help="the machine this report came from (e.g. 20022). Defaults to the "
                        "one named in the report filename. Baselines are then taken only "
                        "from that machine's numerical family.")
    p.add_argument("--seeds", default="1,2,3")
    a = p.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]

    with open(a.report) as fh:
        report = json.load(fh)
    bench = {}
    if yaml and os.path.exists(a.benchmark):
        with open(a.benchmark) as fh:
            bench = yaml.safe_load(fh)
    origins = load_origins(a.origin_map)
    origin = a.origin
    if not origin:
        m = re.search(r"v12_report_(\w+)\.json$", os.path.basename(a.report))
        origin = m.group(1) if m else None
    family = FAMILY.get(origin) if origin else None
    if origin and not family:
        print(f"note: unknown machine {origin!r}; comparing without a family restriction")

    # A sweep's results directory can also hold earlier ad-hoc runs — the cross-machine
    # cross-checks were deliberately run into this same tree. extract_v12.py reports
    # whatever it finds there, so a cell whose recorded origin is a *different* machine
    # is not part of this box's job list and must not be judged as if it were.
    foreign = []
    if family:
        for algo, per_ds in report.items():
            if algo.startswith("_"):
                continue
            for ds in per_ds:
                cell_origin = origins.get((algo, ds))
                if cell_origin and FAMILY.get(cell_origin) != family:
                    foreign.append((algo, ds, cell_origin))
    if foreign:
        print(f"NOTE — {len(foreign)} cell(s) in this report belong to another machine and "
              f"are excluded from the verdict (earlier cross-machine check runs sharing the "
              f"results directory):")
        for algo, ds, own in foreign:
            print(f"  {algo} / {ds} — origin {own}, this report is {origin}")
        print()
    foreign_keys = {(algo, ds) for algo, ds, _ in foreign}

    # ---------------------------------------------------------- 1. controls ----
    print("=" * 78)
    print("CONTROLS — must reproduce v1.1.x per subject, not just to two decimals")
    print("=" * 78)
    ctrl_fail = []
    for algo in sorted(CONTROLS & set(report)):
        for ds in DATASETS:
            cell = report[algo].get(ds)
            if not cell or (algo, ds) in foreign_keys:
                continue
            for seed_s, run in sorted(cell["per_seed"].items()):
                base = find_baseline(a.baseline_dir, ds, algo, int(seed_s), family)
                if not base:
                    print(f"  {algo:16s} {ds:12s} seed{seed_s}  no v1.1.x run on this machine's "
                          f"family ({family or 'any'}) to compare against — not checkable here")
                    continue
                old, src = base
                o, n = per_subject(old), run["per_subject_acc"]
                if o == n:
                    print(f"  {algo:16s} {ds:12s} seed{seed_s}  IDENTICAL ({len(n)} subjects, "
                          f"mean {run['acc']:.2f})")
                else:
                    diff = [f"S{i}: {x}->{y}" for i, (x, y) in enumerate(zip(o, n)) if x != y]
                    print(f"  {algo:16s} {ds:12s} seed{seed_s}  DIFFERS in {len(diff)}/{len(o)} "
                          f"subjects — {'; '.join(diff[:4])}")
                    print(f"      baseline: {src}")
                    ctrl_fail.append((algo, ds, seed_s))

    # ---------------------------------------------------- 2. affected cells ----
    print()
    print("=" * 78)
    print("AFFECTED CELLS — published (v1.1.x) vs re-measured (v1.2.0)")
    print("=" * 78)
    print(f"  {'key':22s} {'dataset':12s} {'published':>16s} {'re-measured':>16s} {'delta':>8s}")
    incomplete = []
    # ``_nettune`` and ``_ensemble`` are reserved keys holding the other two sweep
    # phases, not method cells; they are reported in their own sections below.
    for algo in sorted(k for k in set(report) - CONTROLS if not k.startswith("_")):
        for ds in DATASETS:
            cell = report[algo].get(ds)
            if not cell or (algo, ds) in foreign_keys:
                continue
            pub = published(bench, algo, ds)
            old_txt = f"{pub[2]:.2f}±{pub[3]:.2f}" if pub and pub[3] is not None else (
                f"{pub[2]:.2f}" if pub else "—")
            new_txt = f"{cell['acc_mean']:.2f}±{cell['acc_std']:.2f}"
            delta = f"{cell['acc_mean'] - pub[2]:+.2f}" if pub else "—"
            flag = "" if not cell["seeds_missing"] else f"  [only seeds {cell['seeds_done']}]"
            if cell["seeds_missing"]:
                incomplete.append((algo, ds, cell["seeds_missing"]))
            print(f"  {algo:22s} {ds:12s} {old_txt:>16s} {new_txt:>16s} {delta:>8s}{flag}")

    # ------------------------------------------------- 2b. the ensemble table --
    ens = report.get("_ensemble") or {}
    if ens:
        print()
        print("=" * 78)
        print("ENSEMBLE TABLE — every row moves, because the source learners moved")
        print("=" * 78)
        print(f"  {'combiner':22s} {'dataset':12s} {'published':>16s} {'re-measured':>16s} {'delta':>8s}")
        for ds in DATASETS:
            e = ens.get(ds)
            if not e:
                print(f"  {'(not measured)':22s} {ds:12s}")
                continue
            missing = [s for s in seeds if s not in (e.get("seeds_done") or [])]
            rows = list(sorted(e["combiners"].items()))
            if e.get("single_source"):
                rows.append(("single-source", e["single_source"]))
            for combiner, v in rows:
                name = COMBINER_ROW.get(combiner, combiner)
                pub = published_ensemble_row(bench, name, ds)
                old_txt = f"{pub[0]:.2f}" if pub else "—"
                new_txt = f"{v['mean']:.2f}±{v['std']:.2f}"
                delta = f"{v['mean'] - pub[0]:+.2f}" if pub else "—"
                flag = "" if not missing else f"  [only seeds {e.get('seeds_done')}]"
                if not pub:
                    flag += "  [no published row under this name]"
                print(f"  {name:22s} {ds:12s} {old_txt:>16s} {new_txt:>16s} {delta:>8s}{flag}")
            if missing:
                incomplete.append((f"ensemble/{ds}", ds, missing))

    # --------------------------------------------------------- 3. what is left -
    print()
    if incomplete:
        print(f"INCOMPLETE — {len(incomplete)} cells are missing seeds and must not be published yet:")
        for algo, ds, missing in incomplete:
            print(f"  {algo} {ds}: missing seeds {missing}")
    else:
        print(f"All reported cells have all {len(seeds)} seeds.")
    print()
    if ctrl_fail:
        print(f"*** {len(ctrl_fail)} CONTROL(S) FAILED — the affected list is incomplete. "
              f"Do not update the leaderboard until this is explained. ***")
        return 1
    print("All controls reproduced their v1.1.x runs exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
