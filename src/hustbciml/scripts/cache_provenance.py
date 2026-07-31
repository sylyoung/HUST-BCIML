# cache_provenance.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Inspect an epoch cache or copy a legacy cache with an explicit non-measurement mark.

This command never asserts unknown filter/library metadata and never overwrites its
input. A marked legacy cache remains unsuitable for reportable measurements; the
mark only makes that fact machine-readable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hustbciml.core.batch import EEGEpochs
from hustbciml.data_provider.datasets import _epochs_digest
from hustbciml.utils.io import atomic_savez

_REQUIRED = ("X", "y", "domain", "sfreq", "n_classes", "ch_names", "classes")


def _load(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(_REQUIRED) - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing fields {missing}")
        fields = {key: archive[key] for key in _REQUIRED}
        raw = archive["provenance_json"].item() if "provenance_json" in archive.files else None
    epochs = EEGEpochs(
        X=fields["X"], y=fields["y"], domain=fields["domain"],
        sfreq=float(fields["sfreq"]), n_classes=int(fields["n_classes"]),
        ch_names=[str(v) for v in fields["ch_names"]], classes=[str(v) for v in fields["classes"]],
        paradigm="MI",
    )
    provenance = json.loads(str(raw)) if raw is not None else None
    return fields, epochs, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hustbciml.scripts.cache_provenance")
    parser.add_argument("cache")
    parser.add_argument("--mark-legacy", action="store_true",
                        help="write a separate cache marked is_measurement=false")
    parser.add_argument("--output", help="required with --mark-legacy; must not already exist")
    args = parser.parse_args(argv)

    source = Path(args.cache).resolve()
    fields, epochs, provenance = _load(source)
    actual_digest = _epochs_digest(epochs)
    print(json.dumps({
        "path": str(source), "shape": list(epochs.X.shape),
        "content_sha256": actual_digest, "provenance": provenance,
    }, indent=2))

    if not args.mark_legacy:
        return
    if not args.output:
        parser.error("--output is required with --mark-legacy")
    destination = Path(args.output).resolve()
    if destination == source or destination.exists():
        raise FileExistsError("--output must be a new path; the source cache is never overwritten")
    if provenance is not None:
        raise ValueError("cache already has provenance; refusing to replace it with a legacy mark")

    legacy = {
        "schema_version": 1,
        "is_measurement": False,
        "loader": "MOABBAdapter",
        "dataset": source.name.removesuffix("_epochs.npz"),
        "status": "legacy_unknown",
        "reason": "original cache contained no preprocessing provenance",
        "content_sha256": actual_digest,
    }
    atomic_savez(
        destination,
        **fields,
        provenance_json=np.asarray(json.dumps(legacy, sort_keys=True), dtype="U"),
    )
    print(f"legacy-marked copy -> {destination}")


if __name__ == "__main__":
    main()
