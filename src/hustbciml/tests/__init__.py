"""The test suite, and the two path helpers it shares.

Several tests read a file that belongs to the repository rather than to the installed
package — the leaderboard YAML under ``gallery/``, the two READMEs, the generated site
data, ``pyproject.toml``. Each used to find the checkout by counting ``..`` segments up
from its own location, which is correct exactly until the layout changes: moving the
package under ``src/`` left all four counts one level short, pointing them at a
directory with no ``gallery/`` in it. ``repo_root()`` looks for a marker file, so the
answer no longer depends on how deeply a test is nested.
"""

from __future__ import annotations

import os

_MARKER = "pyproject.toml"


def repo_root() -> str:
    """The root of the checkout, found by walking up to the marker file."""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(d, _MARKER)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                f"no {_MARKER} above {os.path.abspath(__file__)}: these tests read files "
                "that ship with the repository rather than with the installed package, "
                "so they have to run from a checkout."
            )
        d = parent


def package_root() -> str:
    """The ``hustbciml`` package directory (this file sits one level inside it)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
