"""hustbciml — a unified EEG-decoding benchmark library.

Built in place from sylyoung/DeepTransferEEG. The value proposition is
*algorithm coverage*: many EEG-decoding methods (aligners, backbones,
augmenters, strategies, heads) ported onto one stage architecture and
evaluated on MOABB public data.

Entry point: ``python -m hustbciml.run``.
"""

# Kept in step with the repository's release tags (v1.1.0 … v1.2.0), which cover
# the library and the website together. There is no packaging metadata to read it
# from, so it is set here by hand when a release is tagged.
__version__ = "1.2.0"
