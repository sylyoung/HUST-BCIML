"""hustbciml — a unified EEG-decoding benchmark library.

Built in place from sylyoung/DeepTransferEEG. The value proposition is
*algorithm coverage*: many EEG-decoding methods (aligners, backbones,
augmenters, strategies, heads) ported onto one stage architecture and
evaluated on MOABB public data.

Entry point: ``python -m hustbciml.run``.
"""

# The package and website share one release number. ``pyproject.toml`` reads it
# from here, and the packaging test checks it against the changelog heading.
__version__ = "1.6.7"
