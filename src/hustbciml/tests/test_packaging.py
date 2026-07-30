# test_packaging.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""``pyproject.toml`` restates two things that are written down elsewhere, so both are checked.

The version is one of them. It lives in ``hustbciml/__init__.py`` and ``pyproject.toml``
reads it from there, so those two cannot disagree — but the release tags and CHANGELOG.md
are maintained by hand, and the number sat at 1.2.0 through four tagged releases without
anyone noticing, because nothing reads ``__version__`` in the course of ordinary use.

The dependency floors are the other. requirements.txt stays the documented install path,
and it is the file that records *why* each bound is where it is; ``pyproject.toml`` needs
its own core list so that ``pip install hustbciml`` installs something that runs. Two lists
of version ranges for the same packages will drift, and the way they drift is silent: the
loose one keeps working on the machine that already has the right versions installed.

So this compares the ranges rather than the files. requirements.txt may name packages
pyproject does not (the optional groups are declared as extras there), but where both name
a package, the specifier has to be identical.
"""
import os
import re
import sys

import pytest

from . import repo_root

if sys.version_info >= (3, 11):
    import tomllib
else:  # 3.10 has no tomllib; the parse is trivial enough to skip there.
    tomllib = None

ROOT = repo_root()
PYPROJECT = os.path.join(ROOT, "pyproject.toml")
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")

# `name>=1.2,<3` → ("name", ">=1.2,<3"), ignoring any trailing comment.
_REQ = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*([<>=!~][^#\s]*)?\s*(?:#.*)?$")


def _pyproject():
    if tomllib is None:
        pytest.skip("tomllib needs Python 3.11+")
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def _requirements():
    """{package: specifier} for every requirement line in requirements.txt."""
    out = {}
    with open(REQUIREMENTS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _REQ.match(line)
            assert m, f"unparsable requirement line: {line!r}"
            out[m.group(1).lower()] = (m.group(2) or "").strip()
    return out


def test_version_matches_the_changelog():
    """``__version__`` is the newest release CHANGELOG.md announces."""
    from hustbciml import __version__

    with open(CHANGELOG, encoding="utf-8") as fh:
        headings = re.findall(r"^##\s*\[(\d+\.\d+\.\d+)\]", fh.read(), re.M)
    assert headings, "no `## [x.y.z]` heading found in CHANGELOG.md"
    assert __version__ == headings[0], (
        f"hustbciml.__version__ is {__version__} but the newest CHANGELOG.md entry is "
        f"{headings[0]} — bump the version in src/hustbciml/__init__.py when tagging."
    )


def test_pyproject_reads_the_version_from_the_package():
    """One source of truth: pyproject must not hard-code a second copy of the number."""
    cfg = _pyproject()
    assert "version" in cfg["project"].get("dynamic", []), (
        "pyproject.toml should declare `dynamic = [\"version\"]` and read the number "
        "from hustbciml.__version__, not restate it."
    )
    attr = cfg["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "hustbciml.__version__"


def test_core_dependencies_agree_with_requirements():
    """Where both files name a package, they must specify the same version range."""
    reqs = _requirements()
    declared = {}
    cfg = _pyproject()
    for spec in cfg["project"]["dependencies"]:
        m = _REQ.match(spec)
        assert m, f"unparsable pyproject dependency: {spec!r}"
        declared[m.group(1).lower()] = (m.group(2) or "").strip()

    missing = sorted(set(declared) - set(reqs))
    assert not missing, (
        f"{missing} are required by pyproject.toml but absent from requirements.txt, "
        "which is the install path both READMEs document."
    )
    disagree = {p: (declared[p], reqs[p]) for p in declared if declared[p] != reqs[p]}
    assert not disagree, (
        "pyproject.toml and requirements.txt specify different version ranges for "
        f"{disagree} — the bound and the reason for it belong together."
    )


def test_optional_groups_are_installable_from_requirements_too():
    """Every extra's packages are in requirements.txt, which installs the lot."""
    reqs = _requirements()
    cfg = _pyproject()
    for group, specs in cfg["project"]["optional-dependencies"].items():
        if group == "dev":  # pytest is a developer tool, not a benchmark dependency
            continue
        for spec in specs:
            name = _REQ.match(spec).group(1).lower()
            assert name in reqs, (
                f"extra `{group}` requires {name}, which requirements.txt does not install"
            )
