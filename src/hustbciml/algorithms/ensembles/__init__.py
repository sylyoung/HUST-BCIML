# __init__.py  —  hustbciml.algorithms.ensembles
"""Ensemble combiners: post-hoc black-box aggregators of base-model predictions.

Each combiner lives in its own file (filename == class name), exactly like the
pipeline-stage plug-ins under ``models/``, ``aligners/`` and so on. Unlike a stage,
a combiner is applied *after* training by the ensemble runner scripts
(``scripts/ensemble.py``, ``decentralized.py``, ``combined_ensemble.py``), not by
``build_pipeline``. Because a combiner's public name may contain characters a file
name cannot (``SML-OVR``, ``M-MSR``, ``Dawid-Skene``), the display name is a class
attribute and ``build_combiners`` maps it to an instance.

Keeping this module's top level import-free preserves the registry's contract that
listing a group never imports its plug-ins: ``build_combiners`` pulls in the
combiner modules only when a runner actually asks to build the table.
"""
from __future__ import annotations


def build_combiners():
    """Return the ``name -> combiner-instance`` table the ensemble runners consume.

    Uses the same auto-scan registry as the pipeline stages: every non-underscore
    module in this package defines one ``Combiner`` subclass whose class name
    equals the file name. Each instance's ``name`` attribute (which may contain
    hyphens or spaces, e.g. ``SML-OVR``) becomes the key a runner selects it by via
    ``COMBINERS[name](scores)``. Adding a combiner is just dropping in a new file;
    there is no list here to keep in sync.
    """
    from hustbciml.core import registry

    out = {}
    for stem in registry.available("ensembles"):
        cls = registry.resolve("ensembles", stem)
        out[cls.name] = cls()
    return out
