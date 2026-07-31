# __init__.py  —  hustbciml.algorithms.ensembles
"""Discovery and configured construction of post-hoc ensemble combiners."""
from __future__ import annotations

from collections.abc import Iterable, Mapping


def _classes_by_display_name():
    from hustbciml.core import registry

    classes = {}
    for stem in registry.available("ensembles"):
        cls = registry.resolve("ensembles", stem)
        if cls.name in classes:
            raise RuntimeError(f"duplicate combiner display name {cls.name!r}")
        classes[cls.name] = cls
    return classes


def build_combiners(names: Iterable[str] | None = None,
                    settings: Mapping[str, Mapping] | None = None):
    """Build a validated ``display-name -> instance`` mapping.

    ``settings`` is keyed by display name and forwarded to that class constructor.
    Unknown names/settings and invalid constructor keys fail before measurement.
    """
    classes = _classes_by_display_name()
    requested = list(classes) if names is None else list(names)
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate combiner names requested: {requested}")
    unknown = sorted(set(requested) - set(classes))
    settings = dict(settings or {})
    unknown_settings = sorted(set(settings) - set(classes))
    if unknown or unknown_settings:
        raise KeyError(
            f"unknown combiners: requested={unknown}, configured={unknown_settings}; "
            f"available={sorted(classes)}"
        )

    out = {}
    for name in requested:
        kwargs = dict(settings.get(name) or {})
        try:
            out[name] = classes[name](**kwargs)
        except TypeError as exc:
            raise TypeError(f"invalid settings for combiner {name!r}: {kwargs}") from exc
    return out


def combiner_manifest(combiners: Mapping[str, object]) -> dict:
    """Serializable effective configurations, including backend versions."""
    return {name: combiner.configuration() for name, combiner in combiners.items()}
