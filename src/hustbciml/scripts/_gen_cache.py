# _gen_cache.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Build (and verify) the .npz epoch caches for the two new MOABB datasets on a
machine WITH internet (the GPU servers are offline). The resulting
``<name>_epochs.npz`` files are rsynced to the server's data dir, where
MOABBAdapter reads them directly and never touches moabb.

    python -m hustbciml.scripts._gen_cache --data_dir /tmp/hustbciml_newdata
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
    "BNCI2014002": dict(n_subjects=14, per_subject=100, n_chans=15, n_classes=2),
    "BNCI2015001": dict(n_subjects=12, per_subject=200, n_chans=13, n_classes=2),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/tmp/hustbciml_newdata")
    a = p.parse_args()
    os.makedirs(a.data_dir, exist_ok=True)

    # sanity: imports + Toy still load after the edit
    toy = ToyDataset().load()
    print(f"[toy] X {toy.X.shape} classes={toy.classes} OK")

    for name, exp in EXPECT.items():
        # force a rebuild so we test the real MOABB path (not a stale cache)
        cache = os.path.join(a.data_dir, f"{name}_epochs.npz")
        if os.path.exists(cache):
            os.remove(cache)
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
        assert ep.n_classes == exp["n_classes"], f"classes {ep.n_classes} != {exp['n_classes']}"
        assert all(c == exp["per_subject"] for c in counts), \
            f"per-subject counts {counts.tolist()} != {exp['per_subject']}"
        assert os.path.exists(cache), "cache not written"
        print(f"  cache -> {cache} ({os.path.getsize(cache)/1e6:.1f} MB)  ✓ all checks pass")

    print("\nGEN+VERIFY DONE")


if __name__ == "__main__":
    main()
