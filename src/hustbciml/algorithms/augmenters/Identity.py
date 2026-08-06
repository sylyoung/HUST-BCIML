# ===========================================================================
# Identity.py  —  HUST-BCIML EEG-decoding benchmark

# Trivial pass-through baseline; no external reference.
# ===========================================================================
"""No-op augmenter (default; the 'no augmentation' baseline)."""
from hustbciml.core.batch import EEGBatch
from hustbciml.core.stages import Augmenter


class Identity(Augmenter):
    train_only = True

    def __init__(self, **_):
        pass

    def __call__(self, batch: EEGBatch) -> EEGBatch:
        return batch
