# test_results_md.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""RESULTS.md must agree with the leaderboard it summarises.

`gallery/data/benchmark.yml` is the source of truth: the web app is generated from
it, and `build_cards.py` regenerates the per-method cards from the reproduction
registry, with a test holding those two together. `RESULTS.md` was the remaining
artifact carrying the same numbers with nothing checking it — hand-maintained, 80-odd
accuracies, updated by whoever remembered. That is exactly the arrangement that let
ten card values go stale before anyone noticed, and a stale RESULTS.md is worse than
a stale card because it reads like the authoritative record.

The check is deliberately narrow. It parses only the tables whose header names the
three datasets, matches each row against `benchmark.yml` by display name, and
compares the numbers.

A row RESULTS.md states but the leaderboard does not publish must be **declared**
(`NOT_ON_LEADERBOARD`) rather than silently skipped. Tolerating unmatched names is
how a guard turns into decoration: 42 of 244 cells here went unmatched on the first
version of this test, and they were the worst possible 42 — every baseline and
reference row, the values every Δ in the file is measured against — because RESULTS.md
writes `_majority voting (baseline)_` where `benchmark.yml` says `Majority voting`.
"""
import os
import re

import pytest
import yaml

from . import package_root, repo_root

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = package_root()                                   # .../src/hustbciml
ROOT = repo_root()
RESULTS_MD = os.path.join(PKG, "RESULTS.md")
BENCHMARK = os.path.join(ROOT, "gallery", "data", "benchmark.yml")

DATASETS = ("BNCI2014001", "BNCI2014002", "BNCI2015001")
# "75.15 ± 1.06", tolerating the bolding RESULTS.md puts on a column winner.
#
# The std and a trailing parenthetical are both optional, because the tables are not
# written the same way: the pipeline-stage tables give "mean ± std", while the ensemble
# table gives "mean (Δ vs voting)" and — before this release — no std at all. Requiring
# "mean ± std" silently skipped every cell of the ensemble table, which is the largest
# single block of hand-written numbers in the file. A cell with no std is compared on
# its mean alone rather than ignored.
CELL = re.compile(r"^\*{0,2}(\d+\.\d+)"          # the mean
                  r"(?:\s*±\s*(\d+\.\d+))?"      # optional ± std
                  r"\*{0,2}"                     # closing bold
                  r"(?:\s*\([^)]*\))?$")         # optional "(+1.85)" delta annotation
# A minimum that must actually be compared. Set well below the current count so
# ordinary edits do not trip it, but above zero so a parser that silently stops
# matching cannot pass.
MIN_CELLS_CHECKED = 200

# RESULTS.md names a row by what it *is* in its table — "none (no alignment)",
# "_EA-EEGNet (deep reference)_" — while benchmark.yml carries the bare name. That
# decoration is not noise to be stripped blindly: "none" is a different row with a
# different value in the alignment table than in the augmentation table, so stripping
# the parenthetical would make the two interchangeable and let either value pass for
# either row. Each decorated form is therefore mapped explicitly, and where the bare
# name is ambiguous the alias also names the table it belongs to, which makes these
# rows checked *more* precisely than the ones matched by name alone.
ALIAS = {
    "EEGNet (baseline)":                          ("EEGNet", "network"),
    "MVCNet (IFNet + multi-view contrastive)":    ("MVCNet", "network"),
    "ERM (no transfer)":                          ("ERM", "transfer"),
    "none (no alignment)":                        ("none", "alignment"),
    "none (EA-EEGNet)":                           ("none", "augmentation"),
    "EA-EEGNet (deep reference)":                 ("EA (Euclidean)", "alignment"),
    "EA-EEGNet (reference)":                      ("EA (Euclidean)", "alignment"),
    "majority voting (baseline)":                 ("Majority voting", "ensemble"),
    "single-source (3-learner mean)":             ("single-source", "ensemble"),
    "Centralized Training (reference)":           ("Centralized Training", "ensemble"),
    "Centralized Training (EA-EEGNet, reference)": ("Centralized Training", "transfer"),
}

# Rows RESULTS.md reports that the leaderboard deliberately does not carry. Empty, and
# worth keeping that way: the two entries it held — the network-free CSP-LDA and
# Riemann-MDM — turned out not to be a deliberate omission at all. Declaring them here
# is what made it visible that six published numbers had no source of truth, which is
# why they are now a leaderboard table of their own.
NOT_ON_LEADERBOARD = set()


def clean_name(cell):
    """The display name in RESULTS.md, stripped of its markdown decoration."""
    s = cell.strip()
    s = re.sub(r"\*\(new\)\*|\*\*\(lab\)\*\*|\((?:lab|new)\)", "", s)
    s = s.replace("**", "").replace("`", "").strip()
    s = re.sub(r"^_(.*)_$", r"\1", s).strip()      # italics on the reference rows
    return re.sub(r"\s+", " ", s)


def load_published():
    """{display name: {dataset: [(mean, std, table_id), ...]}} over every table.

    A display name is not unique across tables: "none" is the unaligned row in the
    alignment table and the un-augmented row (the EA-EEGNet baseline) in the
    augmentation table, and the two carry different numbers. Collecting candidates
    per name rather than overwriting keeps those honest — a cell passes if it matches
    any row published under that name, so an ambiguous name is checked permissively
    while every unique one is still checked exactly. The table id rides along so an
    `ALIAS` entry can pin an ambiguous name to the one table it means.
    """
    with open(BENCHMARK, encoding="utf-8") as fh:
        bm = yaml.safe_load(fh)
    out = {}
    for table in bm.get("tables", []):
        for group in (table.get("groups") or [table]):
            rows = list(group.get("rows") or [])
            # A group's `reference` is a row too — it renders as the baseline line at
            # the foot of the table, and RESULTS.md states its numbers like any other.
            if group.get("reference"):
                rows.append(group["reference"])
            for row in rows:
                name = row.get("name")
                acc = row.get("acc") or {}
                if not name:
                    continue
                for ds in DATASETS:
                    c = acc.get(ds)
                    if isinstance(c, dict) and c.get("mean") is not None:
                        (out.setdefault(clean_name(name), {})
                            .setdefault(ds, [])
                            .append((float(c["mean"]), c.get("std"), table.get("id"))))
    return out


def candidates(published, name, ds):
    """Published (mean, std) pairs a RESULTS.md row may legitimately match."""
    alias = ALIAS.get(name)
    canonical, want_table = alias if alias else (name, None)
    hits = published.get(canonical, {}).get(ds) or []
    if want_table:
        hits = [h for h in hits if h[2] == want_table]
    return [(m, s) for m, s, _ in hits]


def parse_results_md():
    """Every (name, dataset, mean, std, line_no) RESULTS.md states in a dataset table."""
    with open(RESULTS_MD, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    found, cols = [], None
    for line_no, line in enumerate(lines, 1):
        if not line.startswith("|"):
            cols = None
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        header_hits = [p for p in parts if p in DATASETS]
        if len(header_hits) >= 2:                      # this is a header row
            cols = parts
            continue
        if cols is None or set(parts) <= {""} or all(set(p) <= set("-: ") for p in parts):
            continue
        for i, cell in enumerate(parts):
            if i >= len(cols) or cols[i] not in DATASETS:
                continue
            m = CELL.match(cell)
            if m:
                found.append((clean_name(parts[0]), cols[i], float(m.group(1)),
                              float(m.group(2)) if m.group(2) else None, line_no))
    return found


def test_results_md_matches_the_published_leaderboard():
    published = load_published()
    stated = parse_results_md()
    assert stated, "parsed no numbers out of RESULTS.md — the parser or the format broke"

    mismatches, unmatched, checked = [], {}, 0
    for name, ds, mean, std, line_no in stated:
        cands = candidates(published, name, ds)
        if not cands:
            unmatched.setdefault(name, line_no)
            continue
        checked += 1
        # A std is compared only when both sides state one: RESULTS.md omits it on the
        # ensemble table's delta-annotated cells, and demanding it there would fail on
        # formatting rather than on a number being wrong.
        if any(abs(pm - mean) <= 0.005
               and (ps is None or std is None or abs(float(ps) - std) <= 0.005)
               for pm, ps in cands):
            continue
        shown = ", ".join(f"{pm:.2f}±{float(ps):.2f}" if ps is not None else f"{pm:.2f}"
                          for pm, ps in cands)
        got = f"{mean:.2f}±{std:.2f}" if std is not None else f"{mean:.2f}"
        mismatches.append(f"  RESULTS.md:{line_no} {name} / {ds}: "
                          f"states {got}, benchmark.yml has {shown}")

    assert not mismatches, (
        f"RESULTS.md disagrees with gallery/data/benchmark.yml in {len(mismatches)} "
        f"cell(s). benchmark.yml is the source of truth; update RESULTS.md to match.\n"
        + "\n".join(mismatches))
    stray = {n: ln for n, ln in unmatched.items() if n not in NOT_ON_LEADERBOARD}
    assert not stray, (
        f"{len(stray)} row(s) in RESULTS.md match no row in benchmark.yml, so their "
        f"numbers are checked by nothing:\n"
        + "\n".join(f"  RESULTS.md:{ln} {n!r}" for n, ln in sorted(stray.items(),
                                                                  key=lambda kv: kv[1]))
        + "\n\nEither the row is on the leaderboard under a different display name — add "
          "it to ALIAS — or it is deliberately not published there, in which case add it "
          "to NOT_ON_LEADERBOARD and say why.")
    assert checked >= MIN_CELLS_CHECKED, (
        f"only {checked} cells were compared (floor {MIN_CELLS_CHECKED}). Either the "
        f"tables moved or the name matching broke.")


@pytest.mark.parametrize("path", [RESULTS_MD, BENCHMARK])
def test_the_two_artifacts_exist(path):
    assert os.path.exists(path), f"missing {path}"


def test_the_fixer_is_a_no_op_when_the_two_already_agree():
    """``sync_results_md.py`` must change nothing once this file is correct.

    The script rewrites RESULTS.md's table cells from the leaderboard, and the test
    above is what certifies its output — but only for the cells it decided to touch.
    A bug that reformatted an untouched cell, dropped a std, or mangled a row it could
    not match would leave the numbers right and the file wrong, and nothing here would
    notice. On a file that already agrees, the only correct output is the input,
    byte for byte.
    """
    from hustbciml.scripts.sync_results_md import sync

    text, changed, skipped, _unmatched, _tables = sync(RESULTS_MD, load_published())
    assert not changed, (
        f"the fixer wants to change {len(changed)} cell(s) that the guard accepts — "
        f"the two disagree about what this file should say:\n"
        + "\n".join(f"  RESULTS.md:{ln} {n} / {ds}: {old} -> {new}"
                    for ln, n, ds, old, new in changed[:10]))
    assert not skipped, (
        f"{len(skipped)} cell(s) disagree with benchmark.yml and the fixer cannot "
        f"correct them, so they must be corrected by hand:\n"
        + "\n".join(f"  RESULTS.md:{ln} {n} / {ds} {cell} — {why}"
                    for ln, n, ds, cell, why in skipped[:10]))
    with open(RESULTS_MD, encoding="utf-8") as fh:
        assert text == fh.read(), (
            "the fixer rewrote RESULTS.md even though it reported no cell changes — "
            "it is altering formatting somewhere it should not touch")
