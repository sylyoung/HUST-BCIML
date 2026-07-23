# data_factory.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Hard-coded dataset registry (a ``data_dict`` map) + accessor.

``DATA_DICT`` maps each dataset name the CLI accepts to the loader class that
builds it and to its paradigm. ``get_epochs`` is the single entry point the
experiments call: it looks the name up, constructs that loader from the run
config, and returns the assembled ``EEGEpochs``. Adding a dataset means adding
one row here plus its loader, and nothing else in the pipeline changes.
"""
from __future__ import annotations

from hustbciml.core.batch import EEGEpochs
from hustbciml.core.config import Config
from .datasets import MOABBAdapter, NumpyDataset, ToyDataset

DATA_DICT = {
    "Toy":           dict(loader=ToyDataset,   paradigm="MI"),
    "BNCI2014001":   dict(loader=MOABBAdapter, paradigm="MI"),   # 9 subj, 2-class L/R, session T
    "BNCI2014001-4": dict(loader=MOABBAdapter, paradigm="MI"),   # 9 subj, 4-class (L/R/feet/tongue), session T
    "BNCI2014002":   dict(loader=MOABBAdapter, paradigm="MI"),   # 14 subj, 2-class (right_hand/feet), 15 ch, 512 Hz
    "BNCI2015001":   dict(loader=MOABBAdapter, paradigm="MI"),   # 12 subj, 2-class (right_hand/feet), 13 ch, 512 Hz
}


def get_epochs(cfg: Config) -> EEGEpochs:
    """Build the dataset named by ``cfg.dataset`` and return its ``EEGEpochs``.

    All three loaders take the same ``name``/``data_dir``/``seed`` keyword
    arguments, so the call is uniform. Loaders that do not use a given argument
    (for example the synthetic ``ToyDataset`` ignores ``data_dir``) absorb it
    through their ``**_`` catch-all.
    """
    if cfg.dataset not in DATA_DICT:
        raise KeyError(f"unknown dataset {cfg.dataset!r}; known: {sorted(DATA_DICT)}")
    entry = DATA_DICT[cfg.dataset]
    loader = entry["loader"](name=cfg.dataset, data_dir=cfg.data_dir, seed=cfg.seed)
    return loader.load()
