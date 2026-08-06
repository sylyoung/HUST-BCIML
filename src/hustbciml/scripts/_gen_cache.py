# _gen_cache.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Build and verify provenance-complete epoch caches for all three datasets.

Build on a machine with dataset access, then transfer each NPZ together with its
``.manifest.json`` sibling. The adapter validates exact shape, subject/trial
counts, preprocessing identity, array digest, and whole-file digest on load.

    python -m hustbciml.scripts._gen_cache --data_dir ./data-cache-build
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import numpy as np

from hustbciml.data_provider.datasets import ToyDataset, MOABBAdapter

EXPECT = {
    "BNCI2014001": dict(
        n_subjects=9, per_subject=144, n_chans=22, n_times=1001,
        n_classes=2, sfreq=250.0, classes=["left_hand", "right_hand"],
    ),
    "BNCI2014002": dict(
        n_subjects=14, per_subject=100, n_chans=15, n_times=2561,
        n_classes=2, sfreq=512.0, classes=["feet", "right_hand"],
    ),
    "BNCI2015001": dict(
        n_subjects=12, per_subject=200, n_chans=13, n_times=2561,
        n_classes=2, sfreq=512.0, classes=["feet", "right_hand"],
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data-cache-build")
    p.add_argument(
        "--datasets",
        default=",".join(EXPECT),
        help="comma-separated subset; existing complete cache+manifest pairs are validated and reused",
    )
    a = p.parse_args()
    requested = [value for value in a.datasets.split(",") if value]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("--datasets must be non-empty and contain no duplicates")
    unknown = sorted(set(requested) - set(EXPECT))
    if unknown:
        raise KeyError(f"unknown datasets {unknown}; available: {sorted(EXPECT)}")
    os.makedirs(a.data_dir, exist_ok=True)

    # sanity: imports + Toy still load after the edit
    toy = ToyDataset().load()
    print(f"[toy] X {toy.X.shape} classes={toy.classes} OK")

    for name in requested:
        exp = EXPECT[name]
        # Existing artifacts are never overwritten. A restart loads and validates
        # each complete cache+manifest pair, then continues with the next dataset.
        cache = os.path.join(a.data_dir, f"{name}_epochs.npz")
        existed = os.path.exists(cache)
        ep = MOABBAdapter(name=name, data_dir=a.data_dir).load()
        subs = np.unique(ep.domain)
        counts = np.bincount(ep.domain.astype(int))
        print(f"\n[{name}] X {ep.X.shape}  y {ep.y.shape}  classes={ep.classes} "
              f"n_classes={ep.n_classes} sfreq={ep.sfreq}")
        print(f"  domains={len(subs)}  per-subject counts={counts.tolist()}")
        print(f"  label balance={np.bincount(ep.y.astype(int)).tolist()}  ch={len(ep.ch_names)}")
        # assertions against expected protocol
        assert len(subs) == exp["n_subjects"], f"subjects {len(subs)} != {exp['n_subjects']}"
        assert ep.X.shape[1] == exp["n_chans"], f"chans {ep.X.shape[1]} != {exp['n_chans']}"
        assert ep.X.shape[2] == exp["n_times"], f"times {ep.X.shape[2]} != {exp['n_times']}"
        assert float(ep.sfreq) == exp["sfreq"], f"sfreq {ep.sfreq} != {exp['sfreq']}"
        assert ep.n_classes == exp["n_classes"], f"classes {ep.n_classes} != {exp['n_classes']}"
        assert list(ep.classes) == exp["classes"], f"class map {ep.classes} != {exp['classes']}"
        assert all(c == exp["per_subject"] for c in counts), \
            f"per-subject counts {counts.tolist()} != {exp['per_subject']}"
        assert ep.provenance.get("is_measurement") is True
        assert ep.provenance.get("schema_version") == 2
        assert ep.provenance.get("content_sha256"), "cache provenance has no content digest"
        resolved = ep.provenance.get("selection_resolved") or {}
        assert resolved.get("selected_sessions"), "selected session labels were not recorded"
        assert resolved.get("selected_runs"), "selected run labels were not recorded"
        assert resolved.get("subject_trial_counts") == counts.astype(int).tolist()
        assert resolved.get("class_trial_counts") == np.bincount(
            ep.y.astype(int), minlength=2
        ).astype(int).tolist()
        assert ep.provenance.get("preprocessing") == {
            "paradigm": "MotorImagery", "n_classes_requested": (
                4 if name == "BNCI2014001" else exp["n_classes"]
            ),
            "fmin": 8.0, "fmax": 32.0, "tmin": 0.0, "tmax": None,
        }
        manifest = f"{cache}.manifest.json"
        assert os.path.exists(cache), "cache not written"
        assert os.path.exists(manifest), "whole-file cache manifest not written"
        print(
            f"  cache -> {cache} ({os.path.getsize(cache)/1e6:.1f} MB) "
            f"+ {manifest}  ✓ {'reused and revalidated' if existed else 'generated and validated'}"
        )

    print("\nGEN+VERIFY DONE")


if __name__ == "__main__":
    main()
