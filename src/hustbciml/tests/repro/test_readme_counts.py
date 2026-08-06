# test_readme_counts.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""The headline counts in both READMEs must match the leaderboard they describe.

The approach count is stated in a shields.io badge and in each README's opening
description, and every copy is typed by hand. Adding one leaderboard row makes all
of them wrong at once, while a stale badge still looks authoritative.

The built site data is the reference: it is derived from ``benchmark.yml`` by
``gallery/build_site.py``, and CI already fails if it is stale. This test therefore
matches only the live badge and opening description rather than arbitrary numbers
elsewhere in the documents.
"""
import json
import os
import re

import pytest

from . import repo_root

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = repo_root()
LAB_JS = os.path.join(ROOT, "docs", "data", "lab.js")

# (file, what, pattern with the number as group 1)
CLAIMS = [
    ("README.md", "approaches badge", r"badge/approaches-(\d+)-"),
    ("README.md", "approaches prose", r"re-implements \*\*(\d+)\s*\n?EEG-decoding approaches\*\*"),
    ("README.zh-CN.md", "approaches badge", r"badge/approaches-(\d+)-"),
    ("README.zh-CN.md", "approaches prose", r"重新实现了 \*\*(\d+) 种脑电解码方法\*\*"),
]
ENSEMBLE_CLAIMS = [
    ("README.md", "ensemble prose", r"\*\*(\d+) ensemble combiners\*\*"),
    ("README.zh-CN.md", "ensemble prose", r"\*\*(\d+) 种集成聚合方法\*\*"),
]


def site_counts():
    """n_methods / n_ensemble_methods as the build computed them."""
    if not os.path.exists(LAB_JS):
        pytest.skip("no built docs/data/lab.js — run gallery/build_site.py first")
    raw = open(LAB_JS, encoding="utf-8").read()
    marker = "window.SITE = "
    return json.loads(raw[raw.index(marker) + len(marker):].strip().rstrip(";"))


@pytest.mark.parametrize("path,what,pattern",
                         [(p, w, r) for p, w, r in CLAIMS + ENSEMBLE_CLAIMS])
def test_readme_count_matches_the_leaderboard(path, what, pattern):
    site = site_counts()
    expected = site["n_ensemble_methods"] if "ensemble" in what else site["n_methods"]
    text = open(os.path.join(ROOT, path), encoding="utf-8").read()
    m = re.search(pattern, text)
    assert m, (f"{path}: the {what} is no longer where this test looks for it "
               f"(pattern {pattern!r}). Either the sentence was reworded — update the "
               f"pattern — or the claim was dropped.")
    assert int(m.group(1)) == expected, (
        f"{path}: the {what} says {m.group(1)}, but the leaderboard has {expected}. "
        f"gallery/data/benchmark.yml is the source of truth; the READMEs state this "
        f"count by hand in several places, so update all of them.")
