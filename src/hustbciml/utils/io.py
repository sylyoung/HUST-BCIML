# io.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Atomic writers for measurement artifacts.

A killed process must leave either the previous complete artifact or the new
complete artifact, never a truncated JSON/NPZ file that a resume path mistakes
for a finished run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def atomic_json_dump(payload: Any, path: str | os.PathLike, *, indent: int = 2) -> None:
    """Serialize ``payload`` as strict JSON and atomically replace ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_savez(path: str | os.PathLike, **arrays) -> None:
    """Write a NumPy archive atomically without allowing extension drift."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(payload: Any, path: str | os.PathLike) -> None:
    """Write a PyTorch checkpoint atomically and flush it to stable storage."""
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(
    text: str,
    path: str | os.PathLike,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write plain text atomically with the same durability as JSON artifacts."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_sha256(path: str | os.PathLike) -> str:
    """Return the SHA-256 digest of one artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
