"""hustbciml — a unified EEG-decoding benchmark library.

Built in place from sylyoung/DeepTransferEEG. The value proposition is
*algorithm coverage*: many EEG-decoding methods (aligners, backbones,
augmenters, strategies, heads) ported onto one stage architecture and
evaluated on MOABB public data.

Entry point: ``python -m hustbciml.run``.
"""

# The repository's release tags cover the library and the website together, and this
# is the one place the number is written: ``pyproject.toml`` reads it from here rather
# than restating it. Hand-maintained all the same, so it is checked — it had been left
# at 1.2.0 through four releases before ``tests/test_packaging.py`` started comparing
# it against the newest heading in CHANGELOG.md.
__version__ = "1.5.0"
