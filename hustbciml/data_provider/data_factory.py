# data_factory.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Hard-coded dataset registry (a ``data_dict`` map) + accessor."""
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
    if cfg.dataset not in DATA_DICT:
        raise KeyError(f"unknown dataset {cfg.dataset!r}; known: {sorted(DATA_DICT)}")
    entry = DATA_DICT[cfg.dataset]
    loader = entry["loader"](name=cfg.dataset, data_dir=cfg.data_dir, seed=cfg.seed)
    return loader.load()
