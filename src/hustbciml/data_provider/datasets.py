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

import json
import os
from typing import List

import numpy as np
from sklearn import preprocessing

from hustbciml.core.batch import EEGEpochs
from hustbciml.utils.io import atomic_json_dump, atomic_savez, file_sha256
from hustbciml.utils.provenance import arrays_digest, dependency_versions


# ---------------------------------------------------------------- Toy ---------
# Symmetric sensorimotor montage for the synthetic dataset: every lateral
# electrode has its mirror present (C5/C6, C3/C4, C1/C2, FC1/FC2) and Cz/CPz sit
# on the midline, so ``reflection_permutation`` accepts it. Sliced to the
# requested channel count, which is why the first eight already form a closed
# mirror set.
_TOY_CH = ["C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CPz", "FC1", "FC2"]


def _epochs_digest(ep: EEGEpochs) -> str:
    """Stable identity of the arrays and dataset-wide metadata behind a run."""
    return arrays_digest(
        {"X": ep.X, "y": ep.y, "domain": ep.domain},
        metadata={
            "sfreq": float(ep.sfreq),
            "n_classes": int(ep.n_classes),
            "ch_names": list(ep.ch_names),
            "classes": list(ep.classes),
            "paradigm": ep.paradigm,
        },
    )


def _attach_provenance(ep: EEGEpochs, provenance: dict) -> EEGEpochs:
    provenance = dict(provenance)
    provenance["content_sha256"] = _epochs_digest(ep)
    ep.provenance = provenance
    return ep


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
        # A real, left/right-symmetric sensorimotor montage rather than ch0..ch7,
        # so montage-aware stages (Channel Reflection, MVCNet's reflected view)
        # are exercisable on the bundled data. Note the two class topographies are
        # independent random vectors, not mirror images, so a reflection test on
        # Toy checks the plumbing, not the augmentation's validity.
        ep = EEGEpochs(
            X=X, y=np.array(y), domain=np.array(dom), sfreq=self.sfreq,
            n_classes=2, ch_names=_TOY_CH[:self.n_chans],
            paradigm="MI", classes=["left_hand", "right_hand"],
        )
        return _attach_provenance(ep, {
            "schema_version": 1,
            "is_measurement": True,
            "loader": "ToyDataset",
            "dataset": "Toy",
            "generator": {
                "n_subjects": self.n_subjects, "n_per_class": self.n_per_class,
                "n_chans": self.n_chans, "n_times": self.n_times,
                "sfreq": self.sfreq, "freq": self.freq, "amp": self.amp,
                "noise": self.noise, "gain_std": self.gain_std, "seed": self.seed,
            },
        })


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

    Not wired into ``DATA_DICT``: every benchmark dataset is served by
    ``MOABBAdapter``. Note that this loader emits placeholder channel names
    (``ch0 … chN``) because the ``.npy`` dump carries no montage, so
    montage-aware stages — Channel Reflection, MVCNet's reflected view — will
    correctly refuse to run on it rather than inventing a mirror.
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
        ep = EEGEpochs(
            X=X, y=y_enc, domain=domain, sfreq=spec["sfreq"], n_classes=len(classes),
            ch_names=[f"ch{i}" for i in range(spec["ch_num"])],
            paradigm="MI", classes=[str(c) for c in classes],
        )
        return _attach_provenance(ep, {
            "schema_version": 1,
            "is_measurement": False,
            "loader": "NumpyDataset",
            "dataset": self.name,
            "status": "preprocessing_unknown",
            "reason": "pre-exported arrays do not record their preprocessing environment",
            "preprocessing": "pre-exported arrays; preprocessing not performed by HUST-BCIML",
            "selection": {
                "session_slice": spec["session_slice"],
                "two_class": spec["two_class"] if self.name != "BNCI2014001-4" else None,
            },
        })


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
# These values used to be inherited from MOABB defaults. They are part of the
# benchmark definition, so they are explicit in both the call and cache identity.
_MI_PARADIGM = dict(fmin=8.0, fmax=32.0, tmin=0.0, tmax=None)
_CACHE_PROVENANCE_SCHEMA = 2
_CACHE_MANIFEST_SCHEMA = 2

_MOABB_SPEC = {
    "BNCI2014001":   dict(cls=["BNCI2014_001", "BNCI2014001"], n_classes=4, sfreq=250.0,
                          session_first=True, two_class=["left_hand", "right_hand"],
                          classes=["left_hand", "right_hand"], n_subjects=9,
                          per_subject=144, n_times=1001,
                          ch_names=_BNCI2014001_CH),
    "BNCI2014001-4": dict(cls=["BNCI2014_001", "BNCI2014001"], n_classes=4, sfreq=250.0,
                          session_first=True, two_class=None,
                          classes=["feet", "left_hand", "right_hand", "tongue"], n_subjects=9,
                          per_subject=288, n_times=1001,
                          ch_names=_BNCI2014001_CH),
    # 14 subj, 2-class (right_hand/feet), 15 ch, 512 Hz; train runs only (100/subj).
    "BNCI2014002":   dict(cls=["BNCI2014_002", "BNCI2014002"], n_classes=2, sfreq=512.0,
                          run_contains="train", two_class=None,
                          classes=["feet", "right_hand"], n_subjects=14,
                          per_subject=100, n_times=2561,
                          ch_names=_BNCI2014002_CH),
    # 12 subj, 2-class (right_hand/feet), 13 ch, 512 Hz; first session '0A' (200/subj).
    "BNCI2015001":   dict(cls=["BNCI2015_001", "BNCI2015001"], n_classes=2, sfreq=512.0,
                          session_first=True, two_class=None,
                          classes=["feet", "right_hand"], n_subjects=12,
                          per_subject=200, n_times=2561,
                          ch_names=_BNCI2015001_CH),
}


class MOABBAdapter:
    """Load a MOABB dataset from the MNE cache and apply DeepTransferEEG's
    session/class selection via the ``meta`` DataFrame. Version-robust across
    moabb 1.0/1.5. MOABB is imported lazily so it's only needed when this runs."""

    def __init__(self, name: str, data_dir: str = "./data",
                 allow_legacy_cache: bool = False, **_):
        if name not in _MOABB_SPEC:
            raise KeyError(f"MOABBAdapter has no spec for {name!r}; known: {sorted(_MOABB_SPEC)}")
        self.name = name
        self.data_dir = data_dir
        self.allow_legacy_cache = bool(allow_legacy_cache)
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
            # ``allow_pickle=False`` prevents a copied cache from executing an
            # object-array pickle. Provenance is a scalar Unicode JSON string.
            try:
                with np.load(cache, allow_pickle=False) as d:
                    ep = EEGEpochs(
                        X=d["X"], y=d["y"], domain=d["domain"], sfreq=float(d["sfreq"]),
                        n_classes=int(d["n_classes"]), ch_names=[str(c) for c in d["ch_names"]],
                        paradigm="MI", classes=[str(c) for c in d["classes"]],
                    )
                    provenance_raw = d["provenance_json"].item() \
                        if "provenance_json" in d.files else None
            except ValueError as exc:
                if "allow_pickle" not in str(exc):
                    raise
                # Preserve the exact legacy arrays. Converting their two string
                # fields is safe, but it does not invent preprocessing provenance.
                raise ValueError(
                    f"{cache} stores ch_names/classes as pickled object arrays, which "
                    "measurement code will not execute. Convert only those two fields "
                    "to Unicode arrays, then inspect/mark the cache with "
                    "`python -m hustbciml.scripts.cache_provenance`."
                ) from exc

            if provenance_raw is None:
                if not self.allow_legacy_cache:
                    raise ValueError(
                        f"{cache} has no preprocessing provenance. Its arrays are preserved, "
                        "but using them as a reportable measurement would make the MOABB/MNE "
                        "versions and filter parameters unknowable. Regenerate the cache, or "
                        "pass --allow_legacy_cache for an exploratory run that will be marked "
                        "is_measurement=false."
                    )
                ep = _attach_provenance(ep, {
                    "schema_version": 1,
                    "is_measurement": False,
                    "loader": "MOABBAdapter",
                    "dataset": self.name,
                    "status": "legacy_unknown",
                    "reason": "cache contains no preprocessing provenance",
                })
            else:
                try:
                    provenance = json.loads(str(provenance_raw))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{cache} contains invalid provenance_json") from exc
                expected_digest = provenance.get("content_sha256")
                actual_digest = _epochs_digest(ep)
                if not expected_digest or expected_digest != actual_digest:
                    raise ValueError(
                        f"{cache} content digest does not match its provenance: "
                        f"stored={expected_digest!r}, actual={actual_digest!r}."
                    )
                ep.provenance = provenance
            self._check_cache(ep, cache)
            if ep.provenance.get("is_measurement") is True:
                self._check_file_manifest(cache, ep.provenance)
            return ep

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
        paradigm = MotorImagery(n_classes=self.spec["n_classes"], **_MI_PARADIGM)
        X, labels, meta = paradigm.get_data(dataset=ds, subjects=ds.subject_list)
        labels = np.asarray([str(v) for v in labels])
        subj = meta["subject"].to_numpy()
        sess = meta["session"].to_numpy().astype(str)
        run = meta["run"].to_numpy().astype(str)
        session_order = list(dict.fromkeys(sess.tolist()))
        run_order = list(dict.fromkeys(run.tolist()))

        mask = np.ones(len(X), dtype=bool)
        if self.spec.get("session_first"):
            # Session labels are named differently across moabb versions, so the
            # training session is picked by position, not by name: dict.fromkeys
            # preserves first-seen order, and its first key is the session moabb
            # emitted first, which is the training session in its output order.
            first_session = session_order[0]  # training session, in output order
            mask &= (sess == first_session)
        if self.spec.get("run_contains"):                # train/test split lives in the run label
            # Some datasets keep only one session and put the train/test split in
            # the run name instead, so keep the runs whose label contains the
            # marker substring (e.g. "train").
            mask &= np.array([self.spec["run_contains"] in r for r in run])
        if self.spec["two_class"] is not None:
            # Optionally drop down to the two classes of interest.
            mask &= np.isin(labels, self.spec["two_class"])
        selected_sessions = list(dict.fromkeys(sess[mask].tolist()))
        selected_runs = list(dict.fromkeys(run[mask].tolist()))
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
        versions = dependency_versions()
        ep = _attach_provenance(ep, {
            "schema_version": _CACHE_PROVENANCE_SCHEMA,
            "is_measurement": True,
            "loader": "MOABBAdapter",
            "dataset": self.name,
            "dataset_class": f"{type(ds).__module__}.{type(ds).__name__}",
            "preprocessing": {
                "paradigm": "MotorImagery",
                "n_classes_requested": self.spec["n_classes"],
                **_MI_PARADIGM,
            },
            "selection": {
                "session_first": bool(self.spec.get("session_first")),
                "run_contains": self.spec.get("run_contains"),
                "two_class": self.spec.get("two_class"),
            },
            "selection_resolved": {
                "session_order": session_order,
                "selected_sessions": selected_sessions,
                "run_order": run_order,
                "selected_runs": selected_runs,
                "subject_trial_counts": np.bincount(
                    domain, minlength=self.spec["n_subjects"]
                ).astype(int).tolist(),
                "class_trial_counts": np.bincount(
                    y, minlength=len(classes)
                ).astype(int).tolist(),
            },
            "software": {name: versions[name] for name in ("numpy", "moabb", "mne")},
        })
        self._check_cache(ep, "<freshly built>")
        atomic_savez(
            cache,
            X=ep.X, y=ep.y, domain=ep.domain, sfreq=ep.sfreq,
            n_classes=ep.n_classes,
            ch_names=np.asarray(ep.ch_names, dtype="U"),
            classes=np.asarray(ep.classes, dtype="U"),
            provenance_json=np.asarray(
                json.dumps(ep.provenance, sort_keys=True, ensure_ascii=False), dtype="U"
            ),
        )
        atomic_json_dump(
            {
                "schema_version": _CACHE_MANIFEST_SCHEMA,
                "dataset": self.name,
                "cache_file": os.path.basename(cache),
                "file_sha256": file_sha256(cache),
                "content_sha256": ep.provenance["content_sha256"],
            },
            f"{cache}.manifest.json",
        )
        return ep

    def _check_file_manifest(self, cache: str, provenance: dict) -> None:
        manifest_path = f"{cache}.manifest.json"
        if not os.path.isfile(manifest_path):
            raise ValueError(
                f"{cache} has array provenance but no whole-file manifest; preserve it "
                "and regenerate the cache before reportable measurement"
            )
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception as exc:
            raise ValueError(f"{manifest_path} is not readable strict JSON") from exc
        expected = {
            "schema_version": _CACHE_MANIFEST_SCHEMA,
            "dataset": self.name,
            "cache_file": os.path.basename(cache),
            "file_sha256": file_sha256(cache),
            "content_sha256": provenance.get("content_sha256"),
        }
        if manifest != expected:
            raise ValueError(
                f"{manifest_path} does not match the cache file or content identity"
            )

    def _check_cache(self, ep: EEGEpochs, where: str) -> None:
        """Confirm the loaded epochs are the dataset this spec describes.

        The cache path is only ``<name>_epochs.npz`` — it says nothing about the
        MOABB version, paradigm parameters, channel set, sampling rate or class
        list that produced it. A cache built under different preprocessing loads
        silently through the fast path and becomes the data behind a published
        number. These are the invariants the spec pins, so they are the ones worth
        asserting; a mismatch means the cache does not belong to this dataset
        definition and must be regenerated.
        """
        spec = self.spec
        problems = []
        if float(ep.sfreq) != float(spec["sfreq"]):
            problems.append(f"sfreq {ep.sfreq} != {spec['sfreq']}")
        if list(ep.ch_names) != list(spec["ch_names"]):
            problems.append(f"{len(ep.ch_names)} channels, expected {len(spec['ch_names'])}"
                            f" ({spec['ch_names'][:3]}…)")
        expected_classes = 2 if spec.get("two_class") else spec["n_classes"]
        if int(ep.n_classes) != expected_classes:
            problems.append(f"{ep.n_classes} classes, expected {expected_classes}")
        if list(ep.classes) != list(spec["classes"]):
            problems.append(f"classes {ep.classes!r} != {spec['classes']!r}")
        if len(ep.X) == 0 or ep.X.ndim != 3:
            problems.append(f"X has shape {ep.X.shape}")
        else:
            expected_shape = (len(ep.X), len(spec["ch_names"]), spec["n_times"])
            if tuple(ep.X.shape) != expected_shape:
                problems.append(f"X shape {ep.X.shape} != {expected_shape}")
        domains = np.unique(ep.domain)
        if len(domains) != spec["n_subjects"] or not np.array_equal(
            domains, np.arange(spec["n_subjects"])
        ):
            problems.append(
                f"domains {domains.tolist()} != 0..{spec['n_subjects'] - 1}"
            )
        if len(ep.domain):
            counts = np.bincount(ep.domain.astype(int), minlength=spec["n_subjects"])
            if not np.all(counts == spec["per_subject"]):
                problems.append(
                    f"per-subject trial counts {counts.tolist()} != {spec['per_subject']}"
                )
        provenance = ep.provenance or {}
        if not provenance:
            problems.append("provenance is missing")
        elif provenance.get("is_measurement", True):
            if provenance.get("schema_version") != _CACHE_PROVENANCE_SCHEMA:
                problems.append(
                    f"unsupported provenance schema {provenance.get('schema_version')!r}; "
                    f"expected {_CACHE_PROVENANCE_SCHEMA}"
                )
            if provenance.get("loader") != "MOABBAdapter":
                problems.append(f"provenance loader {provenance.get('loader')!r} != 'MOABBAdapter'")
            if provenance.get("dataset") != self.name:
                problems.append(f"provenance dataset {provenance.get('dataset')!r} != {self.name!r}")
            if not provenance.get("dataset_class"):
                problems.append("dataset_class is missing")
            preprocessing = provenance.get("preprocessing") or {}
            expected_preprocessing = {
                "paradigm": "MotorImagery",
                "n_classes_requested": spec["n_classes"],
                **_MI_PARADIGM,
            }
            if preprocessing != expected_preprocessing:
                problems.append(
                    f"preprocessing {preprocessing!r} != {expected_preprocessing!r}"
                )
            selection = provenance.get("selection") or {}
            expected_selection = {
                "session_first": bool(spec.get("session_first")),
                "run_contains": spec.get("run_contains"),
                "two_class": spec.get("two_class"),
            }
            if selection != expected_selection:
                problems.append(f"selection {selection!r} != {expected_selection!r}")
            resolved = provenance.get("selection_resolved") or {}
            session_order = resolved.get("session_order") or []
            selected_sessions = resolved.get("selected_sessions") or []
            run_order = resolved.get("run_order") or []
            selected_runs = resolved.get("selected_runs") or []
            if spec.get("session_first"):
                if not session_order or selected_sessions != [session_order[0]]:
                    problems.append(
                        "resolved session selection does not prove the first emitted session"
                    )
            elif not selected_sessions:
                problems.append("resolved selected session labels are missing")
            if not set(selected_sessions).issubset(set(session_order)):
                problems.append("resolved selected sessions are absent from session order")
            run_marker = spec.get("run_contains")
            if run_marker:
                if not selected_runs or any(run_marker not in run for run in selected_runs):
                    problems.append(
                        f"resolved runs do not all contain {run_marker!r}"
                    )
            if not set(selected_runs).issubset(set(run_order)):
                problems.append("resolved selected runs are absent from run order")
            actual_subject_counts = np.bincount(
                ep.domain.astype(int), minlength=spec["n_subjects"]
            ).astype(int).tolist()
            if resolved.get("subject_trial_counts") != actual_subject_counts:
                problems.append("resolved subject trial counts do not match cache arrays")
            actual_class_counts = np.bincount(
                ep.y.astype(int), minlength=expected_classes
            ).astype(int).tolist()
            if resolved.get("class_trial_counts") != actual_class_counts:
                problems.append("resolved class trial counts do not match cache arrays")
            software = provenance.get("software") or {}
            missing_software = sorted({"numpy", "moabb", "mne"} - set(software))
            if missing_software:
                problems.append(f"software versions are missing {missing_software}")
            if not provenance.get("content_sha256"):
                problems.append("content_sha256 is missing")
        if problems:
            raise ValueError(
                f"cached epochs for {self.name} at {where} do not match its spec: "
                + "; ".join(problems) + ". Preserve the cache for forensic comparison and "
                  "regenerate a separately named cache with explicit provenance.")
