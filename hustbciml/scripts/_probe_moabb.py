# _probe_moabb.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""One-off probe: extract session labels, class labels, per-subject/session
trial counts, and EEG channel names for MOABB MI datasets, so that correct
``_MOABB_SPEC`` entries can be written for ``data_provider/datasets.py``.

Run on the GPU server (has moabb + the MNE download cache):
    python -m hustbciml.scripts._probe_moabb
It downloads the datasets on first run (needed for the experiment anyway).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import pandas as pd

import moabb
import moabb.datasets as D
from moabb.paradigms import MotorImagery

moabb.set_log_level("ERROR")

TARGETS = [
    ("BNCI2014002", ["BNCI2014_002", "BNCI2014002"], 2),
    ("BNCI2015001", ["BNCI2015_001", "BNCI2015001"], 2),
]


def resolve(cands):
    for name in cands:
        if hasattr(D, name):
            return getattr(D, name), name
    raise ImportError(f"none of {cands} in moabb.datasets")


for key, cands, n_classes in TARGETS:
    print("=" * 70)
    print(f"### {key}")
    cls, resolved_name = resolve(cands)
    print(f"moabb class: {resolved_name}")
    ds = cls()
    print(f"subject_list: {ds.subject_list}  (n={len(ds.subject_list)})")

    para = MotorImagery(n_classes=n_classes)
    X, labels, meta = para.get_data(dataset=ds, subjects=ds.subject_list)
    labels = np.asarray([str(v) for v in labels])
    print(f"X.shape: {X.shape}  dtype={X.dtype}")
    print(f"unique labels: {sorted(set(labels.tolist()))}")

    sess = meta["session"].to_numpy().astype(str)
    subj = meta["subject"].to_numpy()
    run = meta["run"].to_numpy().astype(str) if "run" in meta.columns else None
    sess_order = list(dict.fromkeys(sess.tolist()))
    print(f"session labels (in output order): {sess_order}")
    if run is not None:
        print(f"run labels (in output order): {list(dict.fromkeys(run.tolist()))}")

    # per-subject x per-session counts
    df = pd.DataFrame({"subject": subj, "session": sess})
    tab = df.groupby(["subject", "session"]).size().unstack(fill_value=0)
    print("per-subject x per-session trial counts:")
    print(tab.to_string())

    # what session_first=True would select
    first_session = sess_order[0]
    m = sess == first_session
    print(f"session_first -> keep session '{first_session}': "
          f"{m.sum()} trials, {len(np.unique(subj[m]))} subjects, "
          f"per-subject counts = {np.bincount(subj[m].astype(int))[1:].tolist()}")
    # label balance within that selection
    kept_labels = labels[m]
    print(f"  label balance in kept: "
          f"{dict((c, int((kept_labels==c).sum())) for c in sorted(set(kept_labels.tolist())))}")

    # channel names from a single-subject epochs load
    ep = MotorImagery(n_classes=n_classes).get_data(
        dataset=ds, subjects=[ds.subject_list[0]], return_epochs=True)[0]
    print(f"n_channels: {len(ep.ch_names)}")
    print(f"ch_names: {ep.ch_names}")
    print(f"epoch n_times: {ep.get_data().shape[-1]}  sfreq: {ep.info['sfreq']}")

print("=" * 70)
print("PROBE DONE")
