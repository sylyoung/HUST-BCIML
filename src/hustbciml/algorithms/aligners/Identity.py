# ===========================================================================
# Identity.py  —  HUST-BCIML EEG-decoding benchmark

# Trivial pass-through baseline; no external reference.
# ===========================================================================
"""No-op aligner (the 'no alignment' baseline)."""
from hustbciml.core.batch import EEGEpochs
from hustbciml.core.stages import Aligner


class Identity(Aligner):
    requires_labels = False
    supports_online = True

    def __init__(self, **_):
        pass

    def fit(self, epochs: EEGEpochs) -> "Identity":
        return self

    def transform(self, epochs: EEGEpochs) -> EEGEpochs:
        return epochs
