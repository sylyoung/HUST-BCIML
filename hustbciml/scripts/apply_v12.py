#!/usr/bin/env python3
# apply_v12.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Write a finished v1.2.0 re-measurement into the two hand-maintained sources.

``gallery/data/benchmark.yml`` (the leaderboard) and
``hustbciml/tests/repro/repro_targets.yaml`` (the reproduction registry) hold the same
measurements twice, by hand. The v1.2.0 sweep moves on the order of a hundred cells
across four machines, and transcribing that by hand is exactly the process that put ten
stale values in the cards and left three published numbers no run reproduces. So the
report drives the edit.

Two properties matter more than convenience:

* **Formatting survives.** Both files are heavily commented and the comments carry the
  reasoning. A YAML load-and-dump would delete all of it, so this edits the specific
  ``mean:`` / ``std:`` / ``reproduced:`` lines in place and leaves every other byte
  alone. ``--dry-run`` (the default) prints the edit without writing.
* **A partial cell is never published.** A cell missing any of its three seeds has a
  std that is meaningless (0.00 for a single seed, by definition) and is refused, not
  averaged. Same for a combiner whose ensemble run did not cover all three seeds.
* **A cell is published only from the machine it was published on.** ``extract_v12.py``
  aggregates whatever is in a results directory, and a directory accumulates strays —
  20022's holds four cells left by earlier cross-machine checks, whose published values
  came from 60022 and 7002. Trusting the report's filename as the origin of every cell
  in it would replace one of those numbers with a measurement from a different BLAS
  build, which is the single error this release's whole provenance apparatus exists to
  prevent. So each cell is checked against ``cell_origin.tsv`` and dropped if it did not
  come from its own machine's family.

    python -m hustbciml.scripts.apply_v12 --report v12_report_7002.json \\
        --report v12_report_60022.json --write
"""
from __future__ import annotations

import argparse
import json
import os
import re

import yaml

DATASETS = ("BNCI2014001", "BNCI2014002", "BNCI2015001")
# The tuned backbones are reported under the bare backbone name; the leaderboard keys
# them by the preset that produces the row.
NETTUNE_KEY = "EA-{}".format
# The ensemble sweep names combiners after their papers; two leaderboard rows differ.
COMBINER_ROW = {"voting": "Majority voting", "single-source": "single-source"}
# 10022 and 20022 link the same Intel MKL build and produce bit-identical numbers, so a
# cell published from one may be re-measured on the other. 7002 and 60022 are each their
# own regime. Kept in step with the same table in compare_v12.py.
FAMILY = {"10022": "10022", "20022": "10022", "7002": "7002", "60022": "60022"}


def load_origins(path):
    """{(key, dataset): (machine, tuned)} from cell_origin.tsv.

    ``tuned`` is True when the published value's evidence is a ``tuned_<ds>.json``
    verdict rather than an ordinary run tree. That distinction decides how the cell may
    legitimately be re-measured, so it has to travel with the machine — see
    ``measured_as_published`` below.
    """
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = [x.strip() for x in line.split("\t")]
            if len(f) >= 3:
                evidence = f[4] if len(f) > 4 else ""
                out[(f[0], f[1])] = (f[2], "tuned_" in evidence)
    return out


def load_reports(paths, origins=None):
    """Merge per-machine reports into {(key, dataset): (mean, std, origin)}.

    Each report should cover only the cells belonging to its own machine, so the union
    is the whole sweep — but "should" is doing real work in that sentence, because the
    report is just an aggregate of a results directory and directories accumulate runs
    from earlier cross-machine checks. `origins` (from `cell_origin.tsv`) is therefore
    consulted per cell, and a cell whose published value came from another numerical
    family is skipped here rather than published from the wrong machine.

    Overlap between two reports after that filter would mean a cell genuinely was
    measured twice, so it is reported rather than silently resolved.

    The ensemble table is exempt: its rows have no provenance entries because the whole
    table comes from one sweep on one machine, so there is nothing to disagree with.
    """
    origins = origins or {}
    cells, ens_rows, clashes, foreign, unmapped, mistuned = {}, {}, [], [], [], []

    def in_family(key, ds, report_origin):
        """False only when the map positively places this cell on another family."""
        entry = origins.get((key, ds))
        if entry is None:
            unmapped.append((key, ds, report_origin))
            return True
        cell_origin = entry[0]
        if FAMILY.get(cell_origin, cell_origin) == FAMILY.get(report_origin, report_origin):
            return True
        foreign.append((key, ds, cell_origin, report_origin))
        return False

    def measured_as_published(key, ds, from_tuning):
        """False when a cell published from a grid search arrives as a plain preset run.

        A ``tuned_<ds>.json`` records the configuration a search selected, not the
        preset's defaults. Re-running ``--algorithm <preset>`` therefore measures a
        different configuration, and the delta it produces mixes a code change with a
        configuration change — a difference that looks exactly like a behaviour change
        and is not one. This cost a full re-measurement pass to discover: `MDMAML` on
        BNCI2015001 came back 72.19 against a published 73.06, was diagnosed as a
        cross-machine artefact, re-run on the right machine, and returned 72.19 again.
        Two BLAS families agreeing to two decimals was the tell.
        """
        entry = origins.get((key, ds))
        if entry is None or not entry[1] or from_tuning:
            return True
        mistuned.append((key, ds))
        return False

    def put(store, key, value, origin):
        prev = store.get(key)
        if prev and abs(prev[0] - value[0]) > 0.005:
            clashes.append((key, prev, value + (origin,)))
            return
        store[key] = value + (origin,)

    for path in paths:
        m = re.search(r"v12_report_(\w+)\.json$", os.path.basename(path))
        origin = m.group(1) if m else os.path.basename(path)
        with open(path) as fh:
            rep = json.load(fh)

        for algo, per_ds in rep.items():
            if algo.startswith("_"):
                continue
            for ds, cell in per_ds.items():
                if cell.get("seeds_missing") or not in_family(algo, ds, origin):
                    continue
                if not measured_as_published(algo, ds, from_tuning=False):
                    continue
                put(cells, (algo, ds), (cell["acc_mean"], cell["acc_std"]), origin)

        # Both tuning channels re-run a selection, so a cell arriving through either one
        # was measured the way it is published.
        for channel in ("_nettune", "_algtune"):
            for name, per_ds in (rep.get(channel) or {}).items():
                for ds, v in per_ds.items():
                    key = NETTUNE_KEY(name) if channel == "_nettune" else name
                    if v.get("acc_mean") is None or not in_family(key, ds, origin):
                        continue
                    if not measured_as_published(key, ds, from_tuning=True):
                        continue
                    put(cells, (key, ds), (v["acc_mean"], v.get("acc_std")), origin)

        for ds, e in (rep.get("_ensemble") or {}).items():
            done = e.get("seeds_done") or []
            if len(done) < 3:
                continue
            for combiner, v in (e.get("combiners") or {}).items():
                put(ens_rows, (COMBINER_ROW.get(combiner, combiner), ds),
                    (v["mean"], v["std"]), origin)
            if e.get("single_source"):
                put(ens_rows, ("single-source", ds),
                    (e["single_source"]["mean"], e["single_source"]["std"]), origin)

    # A cell only stays on the mistuned list if nothing else supplied it. Rejecting its
    # plain preset run is the normal, expected event once the cell has been re-tuned:
    # both arrive, one through each channel. Reporting the rejection anyway would
    # announce finished work as outstanding, and bury the cell that really is missing
    # among cells that are not — which is how a warning stops being read.
    mistuned = [c for c in dict.fromkeys(mistuned) if c not in cells]

    return cells, ens_rows, clashes, foreign, unmapped, mistuned


def ambiguous_keys(path, cells):
    """Keys that identify more than one row, where the rows publish different values.

    ``key`` is this file's identity for a cell — it is what the report, the provenance
    map and the writer below all match on — but nothing enforces that one key means one
    measurement. ``EA-EEGNet`` is carried both by the reference row every table measures
    against (72.07 on BNCI2014001) and by the Networks table's EEGNet row, which is the
    same architecture at a *grid-searched* learning rate (72.53). Two real measurements,
    one name.

    Writing one report value into both is silent corruption: the rows differ by less
    than half a point, so the wrong one still looks entirely plausible. Nothing has been
    written through this path yet only because EA-EEGNet is a control that reproduced —
    luck, not a guard. This is the guard.

    Returns the subset that ``cells`` would actually write, so a latent collision in a
    row nobody is touching stays a note rather than an obstacle.
    """
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    seen = {}
    for table in doc.get("tables") or []:
        for group in (table.get("groups") or [table]):
            rows = list(group.get("rows") or [])
            if group.get("reference"):
                rows.append(group["reference"])
            for row in rows:
                key = row.get("key")
                if not key:
                    continue
                for ds, acc in (row.get("acc") or {}).items():
                    seen.setdefault((key, ds), []).append(
                        (table.get("id"), row.get("name"),
                         acc.get("mean"), acc.get("std")))
    out = []
    for (key, ds), rows in sorted(seen.items()):
        if len(rows) < 2 or (key, ds) not in cells:
            continue
        if len({(m, s) for _, _, m, s in rows}) > 1:
            out.append((key, ds, rows))
    return out


def edit_benchmark(path, cells, ens_rows):
    """Rewrite matched mean/std lines in benchmark.yml, preserving everything else.

    Method rows are matched on their ``key``; the ensemble table's rows are not all
    keyed, so those are matched on ``name`` — hence two lookup maps rather than one.
    """
    lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
    out, changed = [], []
    cur_key = cur_name = cur_ds = None
    pending = None                                  # the (mean, std) for the open cell

    def lookup():
        if cur_ds is None:
            return None
        if cur_key and (cur_key, cur_ds) in cells:
            return cells[(cur_key, cur_ds)], cur_key
        if cur_name and (cur_name, cur_ds) in ens_rows:
            return ens_rows[(cur_name, cur_ds)], cur_name
        return None

    for line in lines:
        m = re.match(r"^(\s*)-\s+name:\s*(.+?)\s*$", line)
        if m:
            cur_name, cur_key, cur_ds, pending = m.group(2), None, None, None
            out.append(line)
            continue
        m = re.match(r"^(\s*)key:\s*(.+?)\s*$", line)
        if m:
            cur_key, cur_ds, pending = m.group(2), None, None
            out.append(line)
            continue
        m = re.match(r"^(\s*)(BNCI\d+):\s*$", line)
        if m:
            cur_ds = m.group(2)
            pending = lookup()
            out.append(line)
            continue
        # A new table or a non-acc key at low indent ends the current row's acc block.
        if re.match(r"^-\s+id:", line):
            cur_name = cur_key = cur_ds = None
            pending = None
            out.append(line)
            continue

        if pending:
            (mean, std, _origin), label = pending
            m = re.match(r"^(\s*)mean:\s*(\S+)(.*)$", line)
            if m:
                old = m.group(2)
                new = f"{mean:g}"
                if old != new:
                    changed.append((label, cur_ds, "mean", old, new))
                out.append(f"{m.group(1)}mean: {new}{m.group(3)}\n")
                continue
            m = re.match(r"^(\s*)std:\s*(\S+)(.*)$", line)
            if m and std is not None:
                old = m.group(2)
                new = f"{std:g}"
                if old != new:
                    changed.append((label, cur_ds, "std", old, new))
                out.append(f"{m.group(1)}std: {new}{m.group(3)}\n")
                continue
        out.append(line)

    # A row that gained a std it never had (the ensemble table was single-seed, so its
    # rows carry a mean alone) needs the line inserted, not rewritten.
    out, added = _insert_missing_std(out, ens_rows)
    changed += added
    if added:
        out, hdr = _restate_meanonly_comment(out)
        changed += hdr
    return "".join(out), changed


# The file's own header explains the convention that combiner rows carry a mean and no
# std. Inserting those stds makes that sentence false, and a false comment in the source
# of truth is worse than no comment: the next reader trusts it over the data. Restated
# here rather than left as a manual follow-up, because a manual follow-up is exactly what
# was forgotten the last time a convention changed.
_MEANONLY = "Combiner rows are mean-only (no std), per the leaderboard convention."
_MEANONLY_NEW = ("Combiner rows carry mean ± std over three seeds like every other "
                 "table; before v1.2.0 they were single-seed and mean-only.")


def _restate_meanonly_comment(lines):
    out, changed = [], []
    for line in lines:
        if _MEANONLY in line and line.lstrip().startswith("#"):
            out.append(line.replace(_MEANONLY, _MEANONLY_NEW))
            changed.append(("(header comment)", "-", "comment",
                            "combiner rows are mean-only", "mean ± std over three seeds"))
            continue
        out.append(line)
    return out, changed


def _insert_missing_std(lines, ens_rows):
    """Add ``std:`` under an ensemble ``mean:`` that has none."""
    out, added = [], []
    cur_name = cur_ds = None
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)-\s+name:\s*(.+?)\s*$", line)
        if m:
            cur_name, cur_ds = m.group(2), None
        m = re.match(r"^(\s*)(BNCI\d+):\s*$", line)
        if m:
            cur_ds = m.group(2)
        out.append(line)
        m = re.match(r"^(\s*)mean:\s*(\S+)\s*$", line)
        if not (m and cur_name and cur_ds):
            continue
        hit = ens_rows.get((cur_name, cur_ds))
        if not hit or hit[1] is None:
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if re.match(r"^\s*std:", nxt):
            continue
        out.append(f"{m.group(1)}std: {hit[1]:g}\n")
        added.append((cur_name, cur_ds, "std", "(absent)", f"{hit[1]:g}"))
    return out, added


def edit_repro_targets(path, cells):
    """Rewrite ``reproduced`` / ``reproduced_std`` for the BNCI2014001 rows.

    The registry records one dataset per method. Its trailing comments date the run
    that produced the number, so a row whose value moves gets its comment restated
    rather than left pointing at the superseded sweep.
    """
    lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
    out, changed = [], []
    cur_key, cur_ds = None, None
    for line in lines:
        m = re.match(r"^([A-Za-z0-9][\w.+-]*):\s*$", line)
        if m:
            cur_key, cur_ds = m.group(1), None
            out.append(line)
            continue
        m = re.match(r"^\s*dataset:\s*(\S+)", line)
        if m:
            cur_ds = m.group(1)
            out.append(line)
            continue
        hit = cells.get((cur_key, cur_ds)) if cur_key and cur_ds else None
        if hit:
            mean, std, origin = hit
            m = re.match(r"^(\s*)reproduced:\s*(\S+)(\s*#.*)?$", line)
            if m and f"{mean:g}" != m.group(2):
                changed.append((cur_key, cur_ds, "reproduced", m.group(2), f"{mean:g}"))
                out.append(_commented(m.group(1), "reproduced", f"{mean:g}",
                                      f"3-seed mean, v1.2.0 re-measurement on {origin}"))
                continue
            m = re.match(r"^(\s*)reproduced_std:\s*(\S+)(\s*#.*)?$", line)
            if m and std is not None and f"{std:g}" != m.group(2):
                changed.append((cur_key, cur_ds, "reproduced_std", m.group(2), f"{std:g}"))
                out.append(_commented(m.group(1), "reproduced_std", f"{std:g}",
                                      "std across seeds 1,2,3"))
                continue
        out.append(line)
    return "".join(out), changed


# The registry's trailing comments line up in a column; pad to it so a re-measured row
# does not visibly break the block it sits in.
_COMMENT_COL = 29


def _commented(indent, field, value, comment):
    head = f"{indent}{field}: {value}"
    return f"{head}{' ' * max(1, _COMMENT_COL - len(head))}# {comment}\n"


def _rows_present(path):
    """Every (key-or-name, dataset) the leaderboard actually publishes."""
    import yaml
    with open(path, encoding="utf-8") as fh:
        bench = yaml.safe_load(fh)
    out = set()
    for table in bench.get("tables", []):
        for g in [table] + list(table.get("groups") or []):
            for row in (g.get("rows") or []):
                for ds, cell in (row.get("acc") or {}).items():
                    if cell and cell.get("mean") is not None:
                        for label in (row.get("key"), row.get("name")):
                            if label:
                                out.add((label, ds))
    return out


def audit_prose(path, changed):
    """Lines of a hand-written document that quote a value this run just replaced.

    `test_results_md.py` holds RESULTS.md's *tables* to the leaderboard, but the prose
    around them quotes the same numbers inline — "MSCFormer (76.29) and MSVTNet (75.95)
    top the table", "+2.73", "climbs from 69.65 there to 74.79" — and no check can reach
    those: they are claims, not cells. Rewriting them is a judgment call, but *finding*
    them is not, so it should not be done by eye across 650 lines.

    Matching on the replaced value rather than trying to resolve each number semantically
    is what makes this precise. A prose "76.29" when 76.29 is a value this run overwrote
    is almost certainly referring to that cell; asking instead whether some number could
    be explained by some published cell or difference of cells matches nearly anything in
    a 20-point range.

    Returns [(line_no, old_value, line, is_table_row)]. Table rows are reported apart
    from prose because they are corrected wholesale from benchmark.yml.
    """
    olds = {}
    for key, ds, field, old, new in changed:
        if re.fullmatch(r"\d+\.\d+", old or ""):
            olds.setdefault(f"{float(old):.2f}", []).append((key, ds, field))
    hits = []
    for line_no, line in enumerate(open(path, encoding="utf-8"), 1):
        for value in sorted(olds):
            # \b would not fire before a "." so bound the number explicitly
            if re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", line):
                hits.append((line_no, value, line.rstrip(), line.lstrip().startswith("|")))
    return hits


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    p = argparse.ArgumentParser(prog="hustbciml.scripts.apply_v12")
    p.add_argument("--report", action="append", required=True,
                   help="JSON from extract_v12.py; repeat for each machine")
    p.add_argument("--benchmark", default=os.path.join(repo, "gallery", "data", "benchmark.yml"))
    p.add_argument("--repro", default=os.path.join(
        repo, "hustbciml", "tests", "repro", "repro_targets.yaml"))
    p.add_argument("--write", action="store_true",
                   help="apply the edit (default is a dry run)")
    p.add_argument("--audit_prose", default=os.path.join(repo, "hustbciml", "RESULTS.md"),
                   help="hand-written document to scan for now-superseded values "
                        "(empty string to skip)")
    p.add_argument("--origin_map", default=os.path.join(here, "cell_origin.tsv"))
    a = p.parse_args(argv)

    origins = load_origins(a.origin_map)
    if not origins:
        print(f"REFUSING: no provenance map at {a.origin_map}. Without it a cell measured "
              f"on the wrong machine cannot be told from one measured on the right one.")
        return 1
    cells, ens_rows, clashes, foreign, unmapped, mistuned = load_reports(a.report, origins)
    if clashes:
        print("REFUSING: the same cell was measured on more than one machine, so its "
              "origin is ambiguous and the delta would mix a BLAS change into it:")
        for key, prev, new in clashes:
            print(f"  {key[0]} / {key[1]}: {prev[0]} ({prev[2]}) vs {new[0]} ({new[2]})")
        return 1
    if foreign:
        print(f"excluded {len(foreign)} cell(s) measured outside their own machine's "
              f"family (stray runs in a shared results directory):")
        for key, ds, cell_origin, report_origin in foreign:
            print(f"  {key:22s} {ds:12s} published from {cell_origin}, "
                  f"report is {report_origin}")
    if mistuned:
        print(f"\n*** excluded {len(mistuned)} cell(s) published from a grid search but "
              f"re-measured as a plain preset run: ***")
        for key, ds in mistuned:
            print(f"  {key:22s} {ds:12s}")
        print("   Their published value is whatever the tuner selected, so a run of the "
              "preset's\n   defaults measures a different configuration and its delta is "
              "not a code change.\n   Re-run these with tune_algorithm.py (or "
              "tune_networks.py for a backbone learning\n   rate). Until a tuning report "
              "supplies them they keep their v1.1.x value, which is\n   stale if the "
              "release touched them — so this is a thing to finish, not to note.")
    if unmapped:
        # Not fatal — a genuinely new row has no v1.1.x provenance to record — but it is
        # the one case where the family check cannot protect anything, so say so.
        print(f"note: {len(unmapped)} cell(s) have no entry in the provenance map and are "
              f"taken on the report's word:")
        for key, ds, report_origin in unmapped:
            print(f"  {key:22s} {ds:12s} from {report_origin}")
    ambiguous = ambiguous_keys(a.benchmark, cells)
    if ambiguous:
        print(f"\n*** refusing {len(ambiguous)} cell(s) whose key identifies more than one "
              f"row: ***")
        for key, ds, rows in ambiguous:
            print(f"  {key} / {ds}")
            for table_id, name, mean, std in rows:
                print(f"      {table_id:12s} {name:22s} {mean} ± {std}")
        print("   One value would be written into every row sharing the key, and the rows "
              "do not\n   publish the same number — so at least one would become wrong "
              "while still looking\n   plausible. Give the rows distinct keys, or measure "
              "and write them separately.")
        for key, ds, _ in ambiguous:
            cells.pop((key, ds), None)

    print(f"{len(cells)} method cell(s) and {len(ens_rows)} ensemble cell(s) ready "
          f"from {len(a.report)} report(s)")

    bench_text, bench_changed = edit_benchmark(a.benchmark, cells, ens_rows)
    repro_text, repro_changed = edit_repro_targets(a.repro, cells)

    for label, path, text, changed in (
            ("benchmark.yml", a.benchmark, bench_text, bench_changed),
            ("repro_targets.yaml", a.repro, repro_text, repro_changed)):
        print(f"\n== {label}: {len(changed)} value(s) change ==")
        for key, ds, field, old, new in changed:
            print(f"  {key:22s} {ds or '':12s} {field:15s} {old:>9s} -> {new}")
        if a.write:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)

    # Not every measured cell produces an edit, and the two reasons are not equivalent:
    # a cell the leaderboard already agrees with is fine, a cell that matched no row at
    # all is a measurement being silently dropped. Separate them, so a clean-looking
    # apply cannot hide the second.
    touched = {(k, d) for k, d, _, _, _ in bench_changed}
    present = _rows_present(a.benchmark)
    agreed, missing = [], []
    for key in sorted(set(list(cells) + list(ens_rows))):
        if key in touched or not key[1]:
            continue
        (agreed if key in present else missing).append(key)
    if agreed:
        print(f"\n{len(agreed)} measured cell(s) already matched the published value:")
        for key, ds in agreed:
            print(f"  {key} / {ds}")
    if missing:
        print(f"\n*** {len(missing)} measured cell(s) matched NO leaderboard row — these "
              f"measurements go nowhere: ***")
        for key, ds in missing:
            print(f"  {key} / {ds}")
    if a.audit_prose and os.path.exists(a.audit_prose):
        hits = audit_prose(a.audit_prose, bench_changed)
        prose = [h for h in hits if not h[3]]
        rows = [h for h in hits if h[3]]
        name = os.path.relpath(a.audit_prose, repo)
        print(f"\n== {name}: {len(prose)} prose line(s) quote a value this run "
              f"replaced ({len(rows)} table row(s) too, corrected from benchmark.yml) ==")
        if prose:
            print("   Each needs reading, not replacing: the number may be a claim about "
                  "a ranking or a gap that the new value changes.")
        for line_no, value, line, _ in prose:
            excerpt = line.strip()
            print(f"  {name}:{line_no}  [{value}]  "
                  f"{excerpt[:120]}{'…' if len(excerpt) > 120 else ''}")
    print("\n(dry run — pass --write to apply)" if not a.write else "\nwritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
