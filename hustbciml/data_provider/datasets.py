# datasets.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Dataset adapters -> a single ``EEGEpochs`` spanning all subjects.

Three loaders:
  * ToyDataset   — deterministic synthetic MI (learnable + cross-subject shift),
                   bundled, no download; used by the integration test.
  * NumpyDataset — reads DeepTransferEEG-format ``data/<name>/X.npy`` +
                   ``labels.npy`` and replicates the session/class selection of
                   ``tl/utils/dataloader.data_process``.
  * MOABBAdapter — downloads via MOABB and produces the same epochs (the real
                   BNCI path; runs only when data is fetched).
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
from sklearn import preprocessing

from hustbciml.core.batch import EEGEpochs


# ---------------------------------------------------------------- Toy ---------
class ToyDataset:
    """Synthetic 2-class MI designed so cross-subject transfer is possible and
    EA demonstrably helps:

      * *Shared* class structure — each class has a fixed spatial pattern
        (topography) common to every subject, carrying a band-limited
        oscillation. A model trained on some subjects can transfer to another.
      * *Per-subject covariance shift* — each subject applies its own random
        per-channel gain (a diagonal covariance change). Without alignment this
        wrecks transfer; Euclidean Alignment whitens it away.
    """

    def __init__(self, n_subjects=4, n_per_class=50, n_chans=8, n_times=128,
                 sfreq=128.0, freq=10.0, amp=1.5, noise=0.5, gain_std=0.6,
                 seed=0, **_):
        self.n_subjects = n_subjects
        self.n_per_class = n_per_class
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = sfreq
        self.freq = freq
        self.amp = amp
        self.noise = noise
        self.gain_std = gain_std
        self.seed = seed
        self.paradigm = "MI"

    def load(self) -> EEGEpochs:
        # shared class patterns — fixed across subjects (seed independent of self.seed)
        pat_rng = np.random.RandomState(777)
        patterns = pat_rng.randn(2, self.n_chans)
        patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)

        t = np.arange(self.n_times) / self.sfreq
        X, y, dom = [], [], []
        for s in range(self.n_subjects):
            rng = np.random.RandomState(self.seed * 1000 + s)
            gain = np.exp(self.gain_std * rng.randn(self.n_chans))  # per-channel subject shift
            for c in range(2):
                for _ in range(self.n_per_class):
                    phase = rng.uniform(0, 2 * np.pi)
                    sig = self.amp * np.sin(2 * np.pi * self.freq * t + phase)      # (T,)
                    clean = np.outer(patterns[c], sig) + self.noise * rng.randn(self.n_chans, self.n_times)
                    X.append(gain[:, None] * clean)                                # diagonal covariance shift
                    y.append(c)
                    dom.append(s)
        X = np.stack(X).astype(np.float32)
        return EEGEpochs(
            X=X, y=np.array(y), domain=np.array(dom), sfreq=self.sfreq,
            n_classes=2, ch_names=[f"ch{i}" for i in range(self.n_chans)],
            paradigm="MI", classes=["class0", "class1"],
        )


# --------------------------------------------------------- DeepTransferEEG npy -
# session/class selection replicating tl/utils/dataloader.data_process
_MI_SPEC = {
    "BNCI2014001": dict(n_subjects=9, sfreq=250, ch_num=22, per_subject_total=576,
                        session_slice=(0, 288), two_class=("left_hand", "right_hand")),
    "BNCI2014002": dict(n_subjects=14, sfreq=512, ch_num=15, per_subject_total=160,
                        session_slice=(0, 100), two_class=None),
    "BNCI2015001": dict(n_subjects=12, sfreq=512, ch_num=13, per_subject_total=None,
                        session_slice=None, two_class=None),
}


class NumpyDataset:
    def __init__(self, name: str, data_dir: str = "./data", **_):
        self.name = name
        self.data_dir = data_dir
        self.paradigm = "MI"

    def load(self) -> EEGEpochs:
        base = "BNCI2014001" if self.name == "BNCI2014001-4" else self.name
        X = np.load(os.path.join(self.data_dir, base, "X.npy"))
        y = np.load(os.path.join(self.data_dir, base, "labels.npy"))
        spec = _MI_SPEC[base]
        n_sub = spec["n_subjects"]

        if spec["session_slice"] is not None:
            lo, hi = spec["session_slice"]
            total = spec["per_subject_total"]
            idx = np.concatenate([np.arange(lo, hi) + total * i for i in range(n_sub)])
            X, y = X[idx], y[idx]

        # per-subject domain ids (equal split after session selection)
        per = len(X) // n_sub
        domain = np.repeat(np.arange(n_sub), per)

        if spec["two_class"] is not None and self.name != "BNCI2014001-4":
            keep = np.isin(y, list(spec["two_class"]))
            X, y, domain = X[keep], y[keep], domain[keep]

        classes = sorted(np.unique(y).tolist())
        y_enc = preprocessing.LabelEncoder().fit_transform(y)
        return EEGEpochs(
            X=X, y=y_enc, domain=domain, sfreq=spec["sfreq"], n_classes=len(classes),
            ch_names=[f"ch{i}" for i in range(spec["ch_num"])],
            paradigm="MI", classes=[str(c) for c in classes],
        )


# ---------------------------------------------------------------- MOABB --------
# BNCI2014001 channel montage (needed by Channel Reflection later).
_BNCI2014001_CH = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
                   "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
# BNCI2014002 exposes only generic labels EEG1..EEG15 (no 10-20 montage in moabb).
_BNCI2014002_CH = [f"EEG{i}" for i in range(1, 16)]
# BNCI2015001: 13 sensorimotor electrodes (real 10-20 names from moabb).
_BNCI2015001_CH = ["FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                   "CP3", "CPz", "CP4"]

# Class name is version-dependent (moabb >=1.1 underscores: BNCI2014_001;
# moabb 1.0 exposes both). Session labels also differ across versions
# ('0train'/'1test' vs 'session_T'/'session_E'), so we select the *first-
# occurring* session (the training session, in moabb's output order) rather
# than a hard-coded name — matching DeepTransferEEG's positional selection.
#
# ``run_contains`` is an alternative to ``session_first`` for datasets that put
# their train/test split in the RUN label under a single session (BNCI2014002:
# one session '0' with runs '0train'..'4train' + '5test'..'7test'); keeping the
# runs whose label contains 'train' reproduces DeepTransferEEG's first-100/subject
# selection. Selections are per moabb 1.5 output, since the .npz cache is built
# once with that version and shipped to the (offline) GPU server.
_MOABB_SPEC = {
    "BNCI2014001":   dict(cls=["BNCI2014_001", "BNCI2014001"], n_classes=4, sfreq=250.0,
                          session_first=True, two_class=["left_hand", "right_hand"],
                          ch_names=_BNCI2014001_CH),
    "BNCI2014001-4": dict(cls=["BNCI2014_001", "BNCI2014001"], n_classes=4, sfreq=250.0,
                          session_first=True, two_class=None, ch_names=_BNCI2014001_CH),
    # 14 subj, 2-class (right_hand/feet), 15 ch, 512 Hz; train runs only (100/subj).
    "BNCI2014002":   dict(cls=["BNCI2014_002", "BNCI2014002"], n_classes=2, sfreq=512.0,
                          run_contains="train", two_class=None, ch_names=_BNCI2014002_CH),
    # 12 subj, 2-class (right_hand/feet), 13 ch, 512 Hz; first session '0A' (200/subj).
    "BNCI2015001":   dict(cls=["BNCI2015_001", "BNCI2015001"], n_classes=2, sfreq=512.0,
                          session_first=True, two_class=None, ch_names=_BNCI2015001_CH),
}


class MOABBAdapter:
    """Load a MOABB dataset from the MNE cache and apply DeepTransferEEG's
    session/class selection via the ``meta`` DataFrame. Version-robust across
    moabb 1.0/1.5. MOABB is imported lazily so it's only needed when this runs."""

    def __init__(self, name: str, data_dir: str = "./data", **_):
        if name not in _MOABB_SPEC:
            raise KeyError(f"MOABBAdapter has no spec for {name!r}; known: {sorted(_MOABB_SPEC)}")
        self.name = name
        self.data_dir = data_dir
        self.spec = _MOABB_SPEC[name]
        self.paradigm = "MI"

    def _resolve_class(self, D):
        for name in self.spec["cls"]:
            if hasattr(D, name):
                return getattr(D, name)
        raise ImportError(f"none of {self.spec['cls']} found in moabb.datasets")

    def load(self) -> EEGEpochs:
        # npz cache — skip the ~90s moabb filtering on repeat runs/seeds
        cache = os.path.join(self.data_dir, f"{self.name}_epochs.npz")
        if os.path.exists(cache):
            d = np.load(cache, allow_pickle=True)
            return EEGEpochs(
                X=d["X"], y=d["y"], domain=d["domain"], sfreq=float(d["sfreq"]),
                n_classes=int(d["n_classes"]), ch_names=[str(c) for c in d["ch_names"]],
                paradigm="MI", classes=[str(c) for c in d["classes"]])

        import moabb
        import moabb.datasets as D
        from moabb.paradigms import MotorImagery
        from sklearn import preprocessing
        moabb.set_log_level("ERROR")

        ds = self._resolve_class(D)()
        X, labels, meta = MotorImagery(n_classes=self.spec["n_classes"]).get_data(
            dataset=ds, subjects=ds.subject_list)
        labels = np.asarray([str(v) for v in labels])
        subj = meta["subject"].to_numpy()
        sess = meta["session"].to_numpy().astype(str)

        mask = np.ones(len(X), dtype=bool)
        if self.spec.get("session_first"):
            first_session = list(dict.fromkeys(sess.tolist()))[0]  # training session, in output order
            mask &= (sess == first_session)
        if self.spec.get("run_contains"):                # train/test split lives in the run label
            run = meta["run"].to_numpy().astype(str)
            mask &= np.array([self.spec["run_contains"] in r for r in run])
        if self.spec["two_class"] is not None:
            mask &= np.isin(labels, self.spec["two_class"])
        X, labels, subj = X[mask], labels[mask], subj[mask]

        y = preprocessing.LabelEncoder().fit_transform(labels)          # left_hand=0, right_hand=1
        domain = preprocessing.LabelEncoder().fit_transform(subj)       # subjects -> 0..N-1
        classes = sorted(set(labels.tolist()))
        ep = EEGEpochs(
            X=X, y=y, domain=domain, sfreq=self.spec["sfreq"], n_classes=len(classes),
            ch_names=self.spec["ch_names"], paradigm="MI", classes=classes,
        )
        os.makedirs(self.data_dir, exist_ok=True)
        # Store ch_names/classes as unicode string arrays (dtype="U"), not object
        # arrays: object arrays are pickled, and a pickle written by numpy 2.x
        # (``numpy._core``) cannot be read by numpy 1.x. Plain string arrays use
        # the version-stable .npy format, so the cache loads across numpy versions.
        np.savez(cache, X=ep.X, y=ep.y, domain=ep.domain, sfreq=ep.sfreq,
                 n_classes=ep.n_classes, ch_names=np.asarray(ep.ch_names, dtype="U"),
                 classes=np.asarray(ep.classes, dtype="U"))
        return ep
