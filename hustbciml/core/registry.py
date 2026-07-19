# registry.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Auto-scan plug-in registry: filename == key == class name.

No decorators, no manual registration lists. To add an algorithm you drop
``algorithms/<group>/<Name>.py`` defining ``class <Name>`` and it becomes
available as ``--<group-singular> <Name>``. Imports are lazy so heavy
optional deps (MOABB, pyriemann) only load when their stage is selected.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List, Type

GROUPS = ("aligners", "augmenters", "models", "heads", "strategies")
_PKG = "hustbciml.algorithms"


def available(group: str) -> List[str]:
    """List plug-in names (module stems) in a group, excluding dunder files."""
    pkg = importlib.import_module(f"{_PKG}.{group}")
    names = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.name.startswith("_"):
            names.append(info.name)
    return sorted(names)


def resolve(group: str, name: str) -> Type:
    """Return the class ``name`` from ``algorithms/<group>/<name>.py``."""
    if group not in GROUPS:
        raise KeyError(f"unknown stage group {group!r}; expected one of {GROUPS}")
    module = importlib.import_module(f"{_PKG}.{group}.{name}")
    if not hasattr(module, name):
        raise AttributeError(
            f"module {module.__name__} must define a class named {name!r} "
            f"(filename == class name); found: "
            f"{[a for a in dir(module) if not a.startswith('_')]}"
        )
    return getattr(module, name)


def build(group: str, name: str, **kwargs):
    """Instantiate a plug-in by group and name."""
    return resolve(group, name)(**kwargs)


def catalog() -> Dict[str, List[str]]:
    """All available plug-ins, grouped — for ``run.py --list``."""
    out = {}
    for g in GROUPS:
        try:
            out[g] = available(g)
        except Exception as exc:  # pragma: no cover - diagnostic only
            out[g] = [f"<error: {exc}>"]
    return out
