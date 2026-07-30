# test_leaderboard_links.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Every "code" link the leaderboard renders must point at a file that exists.

Each row of `gallery/data/benchmark.yml` carries a `code:` path, and the web app turns
it into `https://github.com/…/blob/main/<path>`. Nothing checked those paths. The link
checker reads the same file but only follows its DOIs, and it runs weekly against the
network, which cannot help here anyway: the link is built from a path in the repository,
so a file renamed in the same commit that publishes it is already broken before anything
could fetch it.

The failure is silent and public. The page keeps rendering, the row keeps its number,
and the reader who clicks through to see how a method is implemented gets a 404 — on the
one link that exists to make the benchmark checkable.

Checking that the path exists locally is enough, costs nothing, and needs no network.
"""
import os

import pytest
import yaml

from . import repo_root

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = repo_root()
BENCHMARK = os.path.join(ROOT, "gallery", "data", "benchmark.yml")

# Set well below the current count so ordinary edits do not trip it, but above zero so a
# walk that silently stops finding rows cannot pass by checking nothing.
MIN_PATHS = 50


def code_paths():
    """[(row name, path)] for every in-repo code link the leaderboard renders."""
    with open(BENCHMARK, encoding="utf-8") as fh:
        bm = yaml.safe_load(fh) or {}
    out = []
    for table in bm.get("tables") or []:
        for group in (table.get("groups") or [table]):
            rows = list(group.get("rows") or [])
            # A group's `reference` renders as a row too, with its own code link.
            if group.get("reference"):
                rows.append(group["reference"])
            for row in rows:
                path = row.get("code")
                if path:
                    out.append((row.get("name") or row.get("key") or "?", path))
    return out


def test_every_leaderboard_code_link_resolves():
    paths = code_paths()
    assert len(paths) >= MIN_PATHS, (
        f"only {len(paths)} code path(s) found in benchmark.yml (floor {MIN_PATHS}) — "
        f"either the rows moved or this walk stopped matching them")
    missing = [(name, p) for name, p in paths if not os.path.exists(os.path.join(ROOT, p))]
    assert not missing, (
        f"{len(missing)} leaderboard row(s) link to a file that does not exist, so the "
        f"published page has that many dead 'code' links:\n"
        + "\n".join(f"  {name}: {p}" for name, p in missing))


def test_code_paths_are_repo_relative():
    """A path that escapes the repo or starts at / would build a nonsense blob URL."""
    bad = [(name, p) for name, p in code_paths()
           if p.startswith("/") or p.startswith("..") or "://" in p]
    assert not bad, (
        "code paths are joined onto the repository's blob URL, so they must be "
        "repo-relative:\n" + "\n".join(f"  {name}: {p}" for name, p in bad))
