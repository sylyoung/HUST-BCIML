# provenance.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Stable source, environment, and array identities for benchmark artifacts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

_NON_MEASUREMENT_TREES = frozenset({"docs", "tests"})
_DEPENDENCIES = (
    "numpy", "scipy", "scikit-learn", "torch", "pyriemann", "PyYAML",
    "moabb", "mne", "crowd-kit", "pandas",
)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_versions() -> dict[str, str | None]:
    """Versions relevant to numerical behavior, including absent optionals."""
    return {name: _version(name) for name in _DEPENDENCIES}


def source_tree_digest(package_root: str | os.PathLike | None = None) -> str:
    """Hash executable package sources and algorithm presets deterministically.

    Tests and generated documentation are intentionally excluded: editing a card or
    regression note must not make an otherwise identical measurement look like new
    executable code. YAML is included only below ``algorithms/presets``.
    """
    root = Path(package_root) if package_root else Path(__file__).resolve().parents[1]

    def included(path: Path) -> bool:
        relative = path.relative_to(root)
        if not path.is_file() or "__pycache__" in relative.parts:
            return False
        if relative.parts and relative.parts[0] in _NON_MEASUREMENT_TREES:
            return False
        if path.suffix == ".py":
            return True
        return (
            path.suffix in {".yaml", ".yml"}
            and relative.parts[:2] == ("algorithms", "presets")
        )

    files = sorted(path for path in root.rglob("*") if included(path))
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def arrays_digest(arrays: Mapping[str, np.ndarray], metadata: Mapping | None = None) -> str:
    """Hash named arrays by name, dtype, shape, and C-order bytes."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        header = f"{name}\0{array.dtype.str}\0{array.shape}".encode("utf-8")
        digest.update(len(header).to_bytes(4, "big"))
        digest.update(header)
        raw = memoryview(array).cast("B")
        for start in range(0, len(raw), 1 << 20):
            digest.update(raw[start:start + (1 << 20)])
    if metadata:
        import json
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _find_git_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_info(package_root: Path) -> dict:
    root = _find_git_root(package_root)
    git = shutil.which("git")
    if root is None or git is None:
        return {"root": None, "commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            [git, "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [git, "-C", str(root), "status", "--porcelain"], check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"root": str(root), "commit": None, "dirty": None}
    return {"root": str(root), "commit": commit, "dirty": bool(status.strip())}


def _numpy_build() -> dict:
    """Portable NumPy BLAS/LAPACK and CPU-dispatch build identity."""
    try:
        config = np.show_config(mode="dicts")
    except TypeError:  # NumPy 1.x
        get_info = getattr(np.__config__, "get_info", None)
        if get_info is None:
            return {}
        out = {}
        for name in ("blas_opt_info", "lapack_opt_info"):
            info = get_info(name) or {}
            out[name] = {
                "libraries": info.get("libraries"),
                "define_macros": info.get("define_macros"),
                "language": info.get("language"),
            }
        return out
    except Exception:
        return {}

    dependencies = config.get("Build Dependencies") or {}
    keep_dependency = (
        "name", "found", "version", "detection method", "openblas configuration",
    )
    return {
        "machine": (config.get("Machine Information") or {}).get("host"),
        "blas": {key: (dependencies.get("blas") or {}).get(key) for key in keep_dependency},
        "lapack": {
            key: (dependencies.get("lapack") or {}).get(key) for key in keep_dependency
        },
        "simd": config.get("SIMD Extensions"),
    }


def _numerical_libraries() -> list[dict]:
    """Loaded BLAS/OpenMP identities and effective thread counts."""
    try:
        from threadpoolctl import threadpool_info
    except Exception:
        return []
    keep = (
        "user_api", "internal_api", "prefix", "version", "threading_layer",
        "architecture", "num_threads",
    )
    records = [
        {key: record.get(key) for key in keep}
        for record in threadpool_info()
    ]
    return sorted(
        records,
        key=lambda record: tuple(str(record.get(key)) for key in keep),
    )


def runtime_provenance() -> dict:
    """Return the execution environment and exact installed source identity."""
    package_root = Path(__file__).resolve().parents[1]
    try:
        from hustbciml import __version__ as hustbciml_version
    except Exception:
        hustbciml_version = _version("hustbciml")

    torch_runtime = {"cuda_runtime": None, "cudnn": None, "cuda_available": None}
    try:
        import torch
        torch_runtime = {
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception:
        pass

    return {
        "schema_version": 1,
        "hustbciml_version": hustbciml_version,
        "source_sha256": source_tree_digest(package_root),
        "git": _git_info(package_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": {
            "node": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "dependencies": dependency_versions(),
        "numpy_build": _numpy_build(),
        "numerical_libraries": _numerical_libraries(),
        "torch_runtime": torch_runtime,
    }
