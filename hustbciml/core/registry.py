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


def _expected_base(group: str):
    """The ABC a plug-in in ``group`` must subclass.

    Imported lazily so ``registry`` stays importable from ``stages`` without a
    cycle, and so listing the catalog never pulls torch in.
    """
    from . import stages
    return {
        "aligners": stages.Aligner, "augmenters": stages.Augmenter,
        "models": stages.Backbone, "heads": stages.Head,
        "strategies": stages.Strategy, "ensembles": stages.Combiner,
    }[group]


def resolve(group: str, name: str) -> Type:
    """Return the class ``name`` from ``algorithms/<group>/<name>.py``.

    This is where the filename-equals-classname convention is enforced. The
    module path is built straight from ``group`` and ``name``, then the class of
    the same ``name`` is pulled out of it. If the module exists but does not
    define that class, the error lists what the module does define, which is the
    usual symptom of a file whose class name drifted from its file name.

    The class must also subclass the ABC for its group. Filename and class name
    matching is not enough on its own: a file in ``strategies/`` that defines a
    class of the right name but the wrong base is accepted by the naming
    convention and only fails much later, at the first call whose contract it
    does not implement — or worse, not at all, if it happens to be call-compatible
    with a different stage's contract.
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
    cls = getattr(module, name)
    base = _expected_base(group)
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise TypeError(
            f"{module.__name__}.{name} must subclass {base.__name__} to be a "
            f"{group[:-1]} plug-in; got {cls!r}"
        )
    return cls


def build(group: str, name: str, **kwargs):
    """Resolve a plug-in class and instantiate it, forwarding ``kwargs``.

    The kwargs are the construction arguments the pipeline passes for that stage
    kind (for example ``n_chans`` and ``n_times`` for a backbone). A plug-in
    that ignores an argument simply accepts and drops it.
    """
    return resolve(group, name)(**kwargs)


def catalog(strict: bool = True) -> Dict[str, List[str]]:
    """All available plug-ins, grouped — for ``run.py --list``.

    ``strict=True`` (the default) propagates a scan failure. A whole stage family
    that cannot be loaded is a broken installation, and rendering it as a
    plausible-looking catalog entry — the previous behaviour, which stored the
    error string *as if it were a plug-in name* — hides that from the one command
    a user runs to check the installation.

    ``strict=False`` keeps the best-effort listing for diagnostics, where seeing
    which groups do load is the point.
    """
    out = {}
    for g in GROUPS:
        try:
            out[g] = available(g)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"plug-in group {g!r} failed to scan: {exc}") from exc
            out[g] = [f"<error: {exc}>"]
    return out
