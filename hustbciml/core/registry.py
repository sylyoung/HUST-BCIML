# registry.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Auto-scan plug-in registry: filename == key == class name.

This is how the benchmark finds its plug-ins. There are no decorators and no
manual registration lists. The single convention that ties everything together
is that inside a group folder, the module file name, the class it defines, and
the name a user types on the command line are all the same string. So the
aligner ``EA`` lives in ``algorithms/aligners/EA.py`` as ``class EA`` and is
selected with ``--aligner EA``.

Discovery works by listing, not importing. ``available`` scans a group's folder
for module files (skipping ``_``-prefixed ones like ``__init__``) to enumerate
the plug-ins, so listing the catalog never executes any plug-in code.
``resolve`` imports a single module only when that specific plug-in is asked
for. This lazy import is deliberate. A plug-in that needs a heavy optional
dependency such as MOABB or pyriemann pulls it in only when that plug-in is
selected, so the rest of the benchmark runs without those packages installed.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List, Type

# The plug-in folders under ``hustbciml.algorithms``. Each name is also the
# ``<group>`` used to build a plug-in's import path and to key the catalog. The
# first five are the pipeline stages that ``build_pipeline`` wires together; the
# last, ``ensembles``, holds the post-hoc black-box combiners applied by the
# ensemble runner scripts (they are discovered the same way but are not stages).
GROUPS = ("aligners", "augmenters", "models", "heads", "strategies", "ensembles")
_PKG = "hustbciml.algorithms"


def available(group: str) -> List[str]:
    """List plug-in names (module stems) in a group, excluding dunder files.

    Imports only the group *package* to reach its folder path, then walks that
    folder for module files. It does not import the plug-in modules themselves,
    so this stays cheap and side-effect free even for plug-ins with heavy
    optional dependencies. The returned names are exactly the strings the CLI
    accepts for that group.
    """
    pkg = importlib.import_module(f"{_PKG}.{group}")
    names = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.name.startswith("_"):   # skip __init__ and private helpers
            names.append(info.name)
    return sorted(names)


def resolve(group: str, name: str) -> Type:
    """Return the class ``name`` from ``algorithms/<group>/<name>.py``.

    This is where the filename-equals-classname convention is enforced. The
    module path is built straight from ``group`` and ``name``, then the class of
    the same ``name`` is pulled out of it. If the module exists but does not
    define that class, the error lists what the module does define, which is the
    usual symptom of a file whose class name drifted from its file name.
    """
    if group not in GROUPS:
        raise KeyError(f"unknown plug-in group {group!r}; expected one of {GROUPS}")
    module = importlib.import_module(f"{_PKG}.{group}.{name}")   # lazy: loads this plug-in only
    if not hasattr(module, name):
        raise AttributeError(
            f"module {module.__name__} must define a class named {name!r} "
            f"(filename == class name); found: "
            f"{[a for a in dir(module) if not a.startswith('_')]}"
        )
    return getattr(module, name)


def build(group: str, name: str, **kwargs):
    """Resolve a plug-in class and instantiate it, forwarding ``kwargs``.

    The kwargs are the construction arguments the pipeline passes for that stage
    kind (for example ``n_chans`` and ``n_times`` for a backbone). A plug-in
    that ignores an argument simply accepts and drops it.
    """
    return resolve(group, name)(**kwargs)


def catalog() -> Dict[str, List[str]]:
    """All available plug-ins, grouped — for ``run.py --list``.

    Best-effort by design: if one group fails to scan (a broken folder, say),
    its entry becomes an error string instead of aborting the whole listing, so
    the user still sees every other group.
    """
    out = {}
    for g in GROUPS:
        try:
            out[g] = available(g)
        except Exception as exc:  # pragma: no cover - diagnostic only
            out[g] = [f"<error: {exc}>"]
    return out
