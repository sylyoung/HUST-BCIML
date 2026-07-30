"""Reproduction tests: the published numbers, checked against the files they came from.

The path helpers live one level up, in ``hustbciml.tests``, because the packaging test
there needs them too. They are re-exported here so a module in this directory can say
``from . import repo_root`` and not care where they are defined.
"""

from .. import package_root, repo_root

__all__ = ["package_root", "repo_root"]
