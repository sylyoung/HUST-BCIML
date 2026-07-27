# sync_results_md.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Correct RESULTS.md's tables from ``gallery/data/benchmark.yml``.

``benchmark.yml`` is the source of truth. The web app is generated from it and
``build_cards.py`` regenerates the per-method cards from the reproduction registry,
but RESULTS.md's tables were still hand-maintained — roughly 240 numbers updated by
whoever remembered. ``tests/repro/test_results_md.py`` closed the *detection* half of
that gap: the file can no longer disagree with the leaderboard without a test failing.
This script closes the *correction* half. A re-measurement that moves 120 cells is not
something to retype.

It shares the guard's code rather than reimplementing it — same cell pattern, same
display-name mapping, same candidate lookup. A fixer with its own idea of which
benchmark row a table row refers to would be free to "correct" a cell to a number from
a different row, and the guard, agreeing with itself, would pass it.

What it deliberately does **not** do:

* **Cells carrying a delta** — ``76.16 (+1.85)`` in the ensemble table. The mean can be
  rewritten mechanically; the annotation next to it cannot, because it is measured
  against a reference row that is moving in the same edit. These are reported and left
  alone.
* **Row order.** Several tables are sorted by their first dataset column. New numbers
  reorder them, and a re-sort has to carry the whole row — including the prose that
  reads a ranking off it. The script reports which tables fell out of order.
* **Bold.** Bolding here marks a result worth the reader's eye, not a computed column
  maximum (the Networks table bolds one cell in one column and leaves two other column
  leaders plain). There is no rule to apply, so the markup is preserved as written and
  any bolded cell that is no longer its column's best is reported.

Usage::

    python -m hustbciml.scripts.sync_results_md            # dry run
    python -m hustbciml.scripts.sync_results_md --write

Then run ``pytest hustbciml/tests/repro/test_results_md.py`` — that, not this script,
is what certifies the file.
"""
import argparse
import os
import re

# The guard owns the format and the name mapping; importing it is what keeps the two
# from drifting apart.
from hustbciml.tests.repro.test_results_md import (
    CELL,
    DATASETS,
    RESULTS_MD,
    candidates,
    clean_name,
    load_published,
)

# A cell whose text ends in a parenthetical: "76.16 (+1.85)". CELL already tolerates
# these; this tells them apart from the plain "mean ± std" ones.
ANNOTATED = re.compile(r"\)\s*\**$")


def split_row(raw):
    """A table line's raw ``|``-separated segments, or None if it is not one.

    Segments are returned unstripped so a rewrite can put a cell back with the spacing
    the author used, instead of reflowing every table it touches.
    """
    core = raw.strip()
    if not (core.startswith("|") and core.endswith("|")):
        return None
    return core[1:-1].split("|")


def join_row(raw, segs):
    """Rebuild a table line, preserving the original leading indentation."""
    lead = raw[:len(raw) - len(raw.lstrip())]
    return f"{lead}|{'|'.join(segs)}|"


def render(mean, std, bold, with_std):
    """Format a cell the way the one being replaced was written."""
    text = f"{mean:.2f}" + (f" ± {float(std):.2f}" if with_std and std is not None else "")
    return f"**{text}**" if bold else text


def sync(path, published):
    """Rewrite what can be rewritten; return the new text and everything to report."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines(keepends=True)

    changed, skipped, unmatched = [], [], {}
    # (table_start_line, {dataset: [[row_name, old_mean, new_mean, was_bold, line_no], …]}).
    # Both means are kept because the useful question is not "is this column sorted" —
    # most columns of a table sorted by its first dataset never were — but "did this
    # edit break an order, or a bold, that held before it".
    tables, cols, current = [], None, None

    for idx, raw in enumerate(lines):
        line_no = idx + 1
        segs = split_row(raw)
        if segs is None:
            cols, current = None, None
            continue
        parts = [s.strip() for s in segs]
        if len([p for p in parts if p in DATASETS]) >= 2:        # a header row
            cols = parts
            current = {}
            tables.append((line_no, current))
            continue
        if cols is None or set(parts) <= {""} or all(set(p) <= set("-: ") for p in parts):
            continue

        name = clean_name(parts[0])
        for i, cell in enumerate(parts):
            if i >= len(cols) or cols[i] not in DATASETS:
                continue
            m = CELL.match(cell)
            if not m:
                continue
            ds = cols[i]
            mean = float(m.group(1))
            std = float(m.group(2)) if m.group(2) else None
            bold = cell.startswith("**")
            current.setdefault(ds, []).append([name, mean, mean, bold, line_no])

            cands = candidates(published, name, ds)
            if not cands:
                unmatched.setdefault(name, line_no)
                continue
            if any(abs(pm - mean) <= 0.005
                   and (ps is None or std is None or abs(float(ps) - std) <= 0.005)
                   for pm, ps in cands):
                continue                                          # already correct

            if ANNOTATED.search(cell):
                skipped.append((line_no, name, ds, cell,
                                "carries a delta that this script cannot recompute"))
                continue
            if len(cands) > 1:
                # An ambiguous display name ("none") is checked permissively by the
                # guard — it passes against any row of that name. Permissive is fine for
                # a check and useless for a rewrite, so decline rather than guess.
                skipped.append((line_no, name, ds, cell,
                                f"matches {len(cands)} published rows; ALIAS must pin it "
                                f"to one table before it can be corrected"))
                continue
            new_mean, new_std = cands[0]
            if std is not None and new_std is None:
                skipped.append((line_no, name, ds, cell,
                                "states a std that benchmark.yml does not publish"))
                continue

            text = render(new_mean, new_std, bold, std is not None)
            old = segs[i]
            segs[i] = re.sub(r"^(\s*).*?(\s*)$", lambda mm: mm.group(1) + text + mm.group(2),
                             old, flags=re.S)
            lines[idx] = join_row(raw.rstrip("\n"), segs) + ("\n" if raw.endswith("\n") else "")
            raw = lines[idx]
            changed.append((line_no, name, ds, cell.strip(), text))
            current[ds][-1][2] = new_mean

    return "".join(lines), changed, skipped, unmatched, tables


def order_and_bold_reports(tables):
    """What this edit broke: orders that held before it, and bolds that led before it.

    Reporting every column that merely happens not to be descending would bury the
    signal — a table sorted by its first dataset is unsorted in its other two by
    construction, and saying so on every run trains the reader to skip the section.
    """
    disordered, stale_bold = [], []
    for start, by_ds in tables:
        for ds, rows in by_ds.items():
            if len(rows) < 3:
                continue
            before = [r[1] for r in rows]
            after = [r[2] for r in rows]
            was_sorted = before == sorted(before, reverse=True)
            if was_sorted and after != sorted(after, reverse=True):
                inversions = sum(1 for a, b in zip(after, after[1:]) if a < b)
                disordered.append((start, ds, inversions, len(rows)))
            best_before, best_after = max(before), max(after)
            for name, old, new, bold, line_no in rows:
                if bold and old >= best_before - 0.005 and new < best_after - 0.005:
                    stale_bold.append((line_no, name, ds, new, best_after))
    return disordered, stale_bold


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=RESULTS_MD)
    p.add_argument("--write", action="store_true",
                   help="apply the corrections (default: report only)")
    a = p.parse_args(argv)

    published = load_published()
    text, changed, skipped, unmatched, tables = sync(a.results, published)
    name = os.path.basename(a.results)

    print(f"== {name}: {len(changed)} cell(s) corrected from benchmark.yml ==")
    for line_no, row, ds, old, new in changed:
        print(f"  {name}:{line_no:<5d} {row:34s} {ds:12s} {old:>16s} -> {new}")

    if skipped:
        print(f"\n*** {len(skipped)} cell(s) disagree with benchmark.yml and were NOT "
              f"corrected — each needs a person: ***")
        for line_no, row, ds, cell, why in skipped:
            print(f"  {name}:{line_no:<5d} {row:34s} {ds:12s} {cell:>16s}  — {why}")

    disordered, stale_bold = order_and_bold_reports(tables)
    if disordered:
        print(f"\n{len(disordered)} table column(s) are no longer in descending order. "
              f"A table sorted by its first dataset needs re-sorting as whole rows, and "
              f"any prose reading a ranking off it needs re-reading:")
        for start, ds, inv, n in disordered:
            print(f"  table at {name}:{start:<5d} {ds:12s} {inv} inversion(s) over {n} rows")
    if stale_bold:
        print(f"\n{len(stale_bold)} bolded cell(s) are no longer the best in their column:")
        for line_no, row, ds, mean, best in stale_bold:
            print(f"  {name}:{line_no:<5d} {row:34s} {ds:12s} {mean:.2f} bold, "
                  f"column best is now {best:.2f}")
    if unmatched:
        print(f"\n{len(unmatched)} row name(s) match no benchmark.yml row and were left "
              f"untouched (the guard reports these too):")
        for row, line_no in sorted(unmatched.items(), key=lambda kv: kv[1]):
            print(f"  {name}:{line_no:<5d} {row!r}")

    if a.write:
        with open(a.results, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"\nwritten — now run "
              f"`pytest hustbciml/tests/repro/test_results_md.py` to certify it")
    else:
        print("\n(dry run — pass --write to apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
