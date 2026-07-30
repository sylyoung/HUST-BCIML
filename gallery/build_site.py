#!/usr/bin/env python3
"""Generate the web-app data files from the source-of-truth YAML.

Reads : gallery/data/publications.yml   (papers, hand-seeded)
        gallery/data/lab.yml            (lab bio, anchor project, flagship repos)
        gallery/data/benchmark.yml      (controlled-comparison leaderboard)

Writes: docs/data/lab.js           window.LAB, window.SITE
        docs/data/publications.js  window.PUBLICATIONS
        docs/data/benchmark.js     window.BENCHMARK

Each output is a plain `window.X = <json>;` assignment. The data is inlined this
way (rather than fetched) so docs/index.html opens directly as a local file
(file://) with no server, and works unchanged when served by GitHub Pages.

Run: python3 gallery/build_site.py      (from the repo root)
"""
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))   # .../gallery
ROOT = os.path.dirname(HERE)                         # repo root
DATA = os.path.join(HERE, "data")
OUT = os.path.join(ROOT, "docs", "data")

# publication fields the web app actually displays (drop internal bookkeeping)
KEEP = ("id", "title", "authors", "year", "venue", "doi",
        "topic", "paradigm", "code_url", "tldr", "in_press")


def load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_js(name, varname, obj, indent=None):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    sep = None if indent else (",", ":")
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.%s = " % varname)
        json.dump(obj, f, ensure_ascii=False, indent=indent, separators=sep)
        f.write(";\n")
    return path


def _mean_of(cell):
    """The mean of an acc cell {mean, std}, or None for an absent (n/a) cell."""
    return cell.get("mean") if isinstance(cell, dict) else None


def _row_out(r, base_map, is_baseline, datasets, where=""):
    """One display row under the three-dataset schema: for every dataset the row
    carries an `acc` cell ({mean, std}, or absent = n/a), the Δ vs that dataset's
    applicable baseline, a reader-facing description, its citation, and the
    code-link path. A row can override the baseline it is compared against with its
    own per-dataset `base` map (used in the augmentation table, whose two augmenters
    sit in different alignment regimes)."""
    is_ref = bool(r.get("reference"))
    acc_in = r.get("acc") or {}
    # A misspelled dataset key would otherwise vanish into an "n/a" cell, losing a
    # measured result with no build failure — the same class of silent loss the
    # baseline check below guards against.
    unknown = sorted(set(acc_in) - set(datasets))
    if unknown:
        raise ValueError(
            f"{where}row {r['name']!r} has acc entries for unknown dataset(s) {unknown}; "
            f"meta.datasets declares {datasets}. A typo here disappears as an n/a cell.")
    override = r.get("base") if isinstance(r.get("base"), dict) else {}
    acc_out, delta_out = {}, {}
    for d in datasets:
        cell = acc_in.get(d)
        if not isinstance(cell, dict) or cell.get("mean") is None:
            acc_out[d] = None
            delta_out[d] = None
            continue
        acc_out[d] = {"mean": cell.get("mean"), "std": cell.get("std")}
        b = override.get(d)
        if b is None and base_map is not None:
            b = base_map.get(d)
        delta_out[d] = (None if (is_baseline or is_ref or b is None)
                        else round(cell["mean"] - b, 2))
    return {"name": r["name"], "acc": acc_out, "delta": delta_out,
            "isBaseline": is_baseline, "isReference": is_ref,
            "key": r.get("key"), "lab": bool(r.get("lab")),
            "code": r.get("code"), "desc": r.get("desc"),
            "ref": r.get("ref"), "doi": r.get("doi"),
            # Why a dataset cell is empty: "n/a" alone cannot distinguish
            # "inapplicable here" from "not run yet" from "run but not reported".
            "naReason": r.get("na_reason"),
            # Extra pipeline stages this row changes beyond the one its table
            # varies, so the site can qualify the "differs in exactly one way"
            # reading instead of the reader having to open the preset.
            "alsoVaries": r.get("also_varies"),
            "pinAfter": r.get("pin_after")}


def _sort_lab_first(rows, datasets):
    """Order rows lab-proposed first, then the other methods, then the baseline
    row(s) last — the presentation rule "lab work on top, baselines lower". Within
    the lab and non-lab groups, order by descending accuracy averaged over the
    datasets where the method applies (an n/a cell is skipped)."""
    def acc_key(r):
        vals = [r["acc"][d]["mean"] for d in datasets
                if isinstance(r["acc"].get(d), dict) and r["acc"][d].get("mean") is not None]
        return sum(vals) / len(vals) if vals else float("-inf")
    lab = [r for r in rows if r.get("lab") and not r.get("isBaseline")]
    other = [r for r in rows if not r.get("lab") and not r.get("isBaseline")]
    base = [r for r in rows if r.get("isBaseline")]
    lab.sort(key=acc_key, reverse=True)
    other.sort(key=acc_key, reverse=True)
    return lab + other + base


def _apply_pins(rows):
    """Honor a row's optional `pin_after: <name>`: lift that row out of its sorted
    position and reinsert it immediately after the named anchor row. Keeps a
    reduces-to companion adjacent regardless of the lab-first ordering — used so the
    binary SML sits directly under the lab's SML-OVR, which it coincides with on the
    two-class tasks. If the anchor is absent the row keeps its sorted place."""
    pinned = [r for r in rows if r.get("pinAfter")]
    if not pinned:
        return rows
    rest = [r for r in rows if not r.get("pinAfter")]
    for p in pinned:
        idx = next((i for i, r in enumerate(rest) if r["name"] == p["pinAfter"]), None)
        rest.insert(len(rest) if idx is None else idx + 1, p)
    return rest


def _group_out(g, datasets, where=""):
    """Normalise one sub-category group: resolve its per-dataset baseline accuracy
    (from a named baseline row, or a context `reference` whose `acc` is itself a
    per-dataset map), then render each row with per-dataset deltas."""
    base_name = g.get("baseline")
    ref = g.get("reference")
    base_map = None
    if base_name is not None:
        for r in g["rows"]:
            if r["name"] == base_name:
                base_map = {d: _mean_of((r.get("acc") or {}).get(d)) for d in datasets}
        if base_map is None:
            # Without this the whole group renders with an empty Δ column and looks
            # like a valid leaderboard, so a one-character typo in `baseline:`
            # silently erases every delta in the table.
            raise ValueError(
                f"{where}baseline {base_name!r} names no row in this group; rows are "
                f"{[r['name'] for r in g['rows']]}")
    elif ref is not None:
        racc = ref.get("acc") or {}
        base_map = {d: _mean_of(racc.get(d)) for d in datasets}
    rows = [_row_out(r, base_map, base_name is not None and r["name"] == base_name,
                     datasets, where)
            for r in g["rows"]]
    rows = _apply_pins(_sort_lab_first(rows, datasets))
    return {"subcat": g.get("subcat"), "blurb": g.get("blurb", ""),
            "baseline": base_name, "reference": ref, "rows": rows}


def build_benchmark(bench):
    """Every table is emitted as a list of sub-category `groups` (a flat table is
    one unnamed group), so the web app renders both shapes uniformly. The
    per-dataset delta-vs-baseline is computed here so the app just displays. The
    dataset column order comes from meta.datasets; the ensemble table also carries
    its per-dataset `context`."""
    datasets = [d["name"] for d in bench.get("meta", {}).get("datasets", [])]
    tables = []
    for t in bench["tables"]:
        if "groups" in t:
            groups = [_group_out(g, datasets, f"table {t['id']!r}, group {g.get('subcat')!r}: ")
                      for g in t["groups"]]
        else:
            g = {"subcat": None, "blurb": "", "baseline": t.get("baseline"),
                 "reference": t.get("reference"), "rows": t["rows"]}
            groups = [_group_out(g, datasets, f"table {t['id']!r}: ")]
        tables.append({"id": t["id"], "title": t["title"], "blurb": t.get("blurb", ""),
                       "groups": groups})
    return {"meta": bench.get("meta", {}), "library": bench.get("library", {}),
            "datasets": datasets, "tables": tables}


def method_keys(benchmark, include_ensemble=False):
    """Distinct algorithms in the leaderboard, by provenance key.

    The ensemble table is separated rather than merged because its rows are
    post-hoc combiners applied to several trained models, not pipeline
    compositions — they are counted, but under their own name.
    """
    keys = set()
    for t in benchmark["tables"]:
        if t["id"] == "ensemble" and not include_ensemble:
            continue
        if t["id"] != "ensemble" and include_ensemble:
            continue
        for g in t["groups"]:
            for r in g["rows"]:
                if not r.get("isReference") and r.get("key"):
                    keys.add(r["key"])
    return keys


def approach_names(benchmark):
    """Every distinct named approach the site renders as a chip on the Overview.

    The Overview shows a headline count and, directly beneath it, the list of
    names. Those were computed from two different populations — the count in
    Python over non-ensemble keyed rows, the chips in JavaScript over every named
    row — so the page disagreed with itself by 18. Both now come from here.

    Every leaderboard table counts, the ensemble table included: its combiners are
    approaches the benchmark measures, two of them the lab's own, so leaving them
    out hid SML-OVR and StackingNet from the one place the site lists the lab's
    methods. What is excluded is a row the data marks ``reference: true`` — the
    ensemble table's single-source and Centralized Training rows, which its own
    sub-category calls non-ensemble references and which bound the combiners
    rather than competing with them. ``benchApproaches()`` in app.js applies the
    same two rules and the two must stay in step, or the heading and the chips it
    introduces disagree again.
    """
    names = set()
    for t in benchmark["tables"]:
        for g in t["groups"]:
            for r in g["rows"]:
                if not r.get("isReference") and r.get("name") and r["name"] != "none":
                    names.add(r["name"])
    return names


def lab_keys(benchmark, keys):
    """The subset of ``keys`` whose rows are lab-proposed."""
    out = set()
    for t in benchmark["tables"]:
        for g in t["groups"]:
            for r in g["rows"]:
                if r.get("lab") and r.get("key") in keys:
                    out.add(r["key"])
    return out


REQUIRED_PUB_FIELDS = ("id", "title", "year")


def validate_publications(pubs):
    """Every publication needs the fields BUILD.md documents as required.

    ``id`` is the stable identifier the gallery deep-links, deduplicates and
    cross-references by; a null one is not a cosmetic gap, it means that record
    cannot be referred to at all. The generator used ``p.get(k)``, so nulls flowed
    straight through into the shipped JavaScript.
    """
    bad = []
    for i, p in enumerate(pubs):
        for field in REQUIRED_PUB_FIELDS:
            if not p.get(field):
                bad.append(f"  entry {i} ({(p.get('title') or '?')[:60]}): missing {field!r}")
    if bad:
        raise ValueError(
            f"{len(bad)} publication field(s) missing (BUILD.md lists them as "
            f"required):\n" + "\n".join(bad))


# There is deliberately no build stamp in the emitted data. `window.SITE` used to
# carry a `generated` date, which nothing on the page ever displayed and nothing ever
# read, and which CI regenerates and compares against the committed file. A stamp that
# moves with the clock cannot survive that comparison: the build would have gone red on
# the date alone every day after a release, reporting the generated files as stale when
# nothing about them had changed. Honouring SOURCE_DATE_EPOCH only narrows the window —
# whatever the developer stamps before committing, CI recomputes after, so any stamp
# derived from the commit disagrees by construction. If a "data as of" line is ever
# wanted on the page, take it from the commit at display time rather than baking it in.


def main():
    pubs = load(os.path.join(DATA, "publications.yml"))
    lab = load(os.path.join(DATA, "lab.yml"))
    bench = load(os.path.join(DATA, "benchmark.yml"))

    validate_publications(pubs)
    papers = [{k: p.get(k) for k in KEEP} for p in pubs]
    # newest first; within a year, papers with code first; then title
    papers.sort(key=lambda p: (-(p.get("year") or 0),
                               0 if p.get("code_url") else 1,
                               (p.get("title") or "").lower()))
    n_code = sum(1 for p in papers if p.get("code_url"))

    benchmark = build_benchmark(bench)
    pipeline_keys = method_keys(benchmark)
    ensemble_keys = method_keys(benchmark, include_ensemble=True)
    site = {"n_papers": len(papers), "n_code": n_code,
            # Pipeline compositions (alignment, augmentation, backbone, transfer).
            "n_methods": len(pipeline_keys),
            # Of those, how many are the lab's own — computed over the same
            # population as n_methods, so "N of M are the lab's" is a true reading.
            "n_lab_methods": len(lab_keys(benchmark, pipeline_keys)),
            # Ensemble combiners, counted separately because they are post-hoc
            # aggregators over trained models rather than pipeline stages.
            "n_ensemble_methods": len(ensemble_keys),
            # Every named approach the Overview renders as a chip, so the headline
            # number and the list beneath it describe the same set.
            "n_approaches": len(approach_names(benchmark))}

    write_js("lab.js", "LAB", lab, indent=2)
    with open(os.path.join(OUT, "lab.js"), "a", encoding="utf-8") as f:
        f.write("window.SITE = ")
        json.dump(site, f, ensure_ascii=False)
        f.write(";\n")
    write_js("publications.js", "PUBLICATIONS", papers)      # compact
    write_js("benchmark.js", "BENCHMARK", benchmark, indent=2)

    n_rows = sum(len(g["rows"]) for t in benchmark["tables"] for g in t["groups"])
    print("papers=%d with_code=%d tables=%d rows=%d methods=%d (lab %d) "
          "ensembles=%d approach_chips=%d"
          % (len(papers), n_code, len(benchmark["tables"]), n_rows, site["n_methods"],
             site["n_lab_methods"], site["n_ensemble_methods"], site["n_approaches"]))
    print("wrote docs/data/{lab.js,publications.js,benchmark.js}")


if __name__ == "__main__":
    main()
