# datasets.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Dataset adapters -> a single ``EEGEpochs`` spanning all subjects.

Every loader returns one ``EEGEpochs`` that stacks the trials of all subjects
together, with a per-trial ``domain`` array recording which subject each trial
came from. The cross-subject splitter later slices that single container by
domain, so all subject bookkeeping lives in one place rather than in separate
per-subject files. The three per-trial arrays a loader must fill are ``X`` of
shape (N, C, T), ``y`` the integer class index in [0, n_classes), and
``domain`` the subject id. The rest of the fields (``sfreq``, ``n_classes``,
``ch_names``, ``paradigm``, ``classes``) are dataset-wide metadata read from
the dataset spec.

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
    """Read a pre-exported DeepTransferEEG ``.npy`` dump and reconstruct the
    same trials, subject ids, and class subset that its ``data_process`` builds.

    The on-disk arrays hold every subject's trials concatenated in subject
    order, so subject identity is positional (the first ``per_subject_total``
    rows are subject 0, and so on) rather than stored explicitly. ``load``
    therefore recovers ``domain`` by slicing that fixed stride, and it selects
    the training session by index because the dump keeps sessions concatenated
    the same way.
    """

    def __init__(self, name: str, data_dir: str = "./data", **_):
        self.name = name
        self.data_dir = data_dir
        self.paradigm = "MI"

    def load(self) -> EEGEpochs:
        # "BNCI2014001-4" is the 4-class variant of the same recordings, so it
        # reads the same files and only differs in whether the 2-class subset is
        # applied further down.
        base = "BNCI2014001" if self.name == "BNCI2014001-4" else self.name
        X = np.load(os.path.join(self.data_dir, base, "X.npy"))
        y = np.load(os.path.join(self.data_dir, base, "labels.npy"))
        spec = _MI_SPEC[base]
        n_sub = spec["n_subjects"]

        if spec["session_slice"] is not None:
            # Keep only the training session. Within each subject's block of
            # ``per_subject_total`` rows the first ``[lo, hi)`` rows are the
            # training session, so shifting that window by ``total * i`` selects
            # subject i's training trials, and the concatenation gathers them for
            # every subject in one index array.
            lo, hi = spec["session_slice"]
            total = spec["per_subject_total"]
            idx = np.concatenate([np.arange(lo, hi) + total * i for i in range(n_sub)])
            X, y = X[idx], y[idx]

        # After session selection every subject contributes the same number of
        # rows, so an equal split recovers the subject id of each trial.
        per = len(X) // n_sub
        domain = np.repeat(np.arange(n_sub), per)

        if spec["two_class"] is not None and self.name != "BNCI2014001-4":
            # Restrict to the two motor-imagery classes of interest; ``domain``
            # is filtered by the same mask so trials stay aligned to subjects.
            keep = np.isin(y, list(spec["two_class"]))
            X, y, domain = X[keep], y[keep], domain[keep]

        # Map the surviving string labels to contiguous integers 0..K-1 in
        # sorted order, so the numeric class index is stable across runs.
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
        # Fast path: a previous run cached the fully processed epochs to
        # ``{name}_epochs.npz``, so reload them and skip the ~90s of moabb
        # download and band-pass filtering. This is what runs on the offline GPU
        # server, where the cache is shipped in and moabb itself is never called.
        cache = os.path.join(self.data_dir, f"{self.name}_epochs.npz")
        if os.path.exists(cache):
            # The .npz stores exactly the seven fields ``EEGEpochs`` needs, one
            # per archive key. ``X``/``y``/``domain`` come back as arrays as
            # saved. The scalars ``sfreq`` and ``n_classes`` were stored as 0-d
            # arrays, so they are cast back to float/int. ``ch_names`` and
            # ``classes`` were stored as fixed-width unicode string arrays (see
            # the savez comment below) and are turned back into plain str lists.
            # ``allow_pickle=True`` is kept for backward compatibility with any
            # older object-array cache.
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

        # ``get_data`` returns the epoched, filtered trials ``X`` (N, C, T), the
        # string ``labels``, and a ``meta`` DataFrame with one row per trial
        # whose ``subject``/``session``/``run`` columns say where each trial came
        # from. Selection below is expressed as a boolean mask over those rows.
        ds = self._resolve_class(D)()
        X, labels, meta = MotorImagery(n_classes=self.spec["n_classes"]).get_data(
            dataset=ds, subjects=ds.subject_list)
        labels = np.asarray([str(v) for v in labels])
        subj = meta["subject"].to_numpy()
        sess = meta["session"].to_numpy().astype(str)

        mask = np.ones(len(X), dtype=bool)
        if self.spec.get("session_first"):
            # Session labels are named differently across moabb versions, so the
            # training session is picked by position, not by name: dict.fromkeys
            # preserves first-seen order, and its first key is the session moabb
            # emitted first, which is the training session in its output order.
            first_session = list(dict.fromkeys(sess.tolist()))[0]  # training session, in output order
            mask &= (sess == first_session)
        if self.spec.get("run_contains"):                # train/test split lives in the run label
            # Some datasets keep only one session and put the train/test split in
            # the run name instead, so keep the runs whose label contains the
            # marker substring (e.g. "train").
            run = meta["run"].to_numpy().astype(str)
            mask &= np.array([self.spec["run_contains"] in r for r in run])
        if self.spec["two_class"] is not None:
            # Optionally drop down to the two classes of interest.
            mask &= np.isin(labels, self.spec["two_class"])
        X, labels, subj = X[mask], labels[mask], subj[mask]

        # Encode the surviving string labels and subject ids to contiguous
        # integers. Sorted label order fixes the class-to-index mapping, and the
        # subject encoding renumbers moabb's subject ids to a dense 0..N-1 domain
        # axis that the splitter iterates over.
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
