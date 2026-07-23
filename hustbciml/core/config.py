# config.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Typed run configuration + argparse + YAML preset resolution.

The ``Config`` dataclass is the single object that describes one run: which
dataset and protocol, which five stage plug-ins to compose, and every knob they
read. It flows from here into ``build_pipeline`` and the Exp, so filling it
correctly is what "configuring a run" means.

There are three sources for a field, and ``resolve_config`` layers them in a
fixed precedence, low to high: the dataclass defaults, then a preset YAML, then
whatever the user typed on the command line. Each layer only overrides the ones
below it. A field the user did not pass stays at its preset or default value.

An ``--algorithm`` names a preset in ``algorithms/presets/<name>.yaml``. A
preset is a shorthand that composes the stages and sets hyperparameters in one
word, so ``--algorithm EA-EEGNet`` fills in the aligner, backbone, head, and
strategy together. Individual ``--aligner`` / ``--backbone`` / ... flags then
override whatever the preset chose, which is how you tweak one stage of a preset
without copying the whole thing.

The data-derived dims (``n_chans``, ``n_times``, ``n_classes``, ``sfreq``) are
not set by the user at all. They start at 0 and the Exp measures them from the
loaded dataset and writes them back onto the Config *before* the pipeline is
built. That late fill is the trick that lets one config run unchanged on any
dataset: the architecture is sized to the data at build time, not hard-coded.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, fields
from typing import List, Optional

import yaml

_PRESET_DIR = os.path.join(os.path.dirname(__file__), "..", "algorithms", "presets")


@dataclass
class Config:
    # --- what to run ---
    dataset: str = "Toy"
    protocol: str = "cross_subject"
    algorithm: Optional[str] = None        # preset name (composes the stages below)
    aligner: str = "Identity"
    augmenter: str = "Identity"
    backbone: str = "EEGNet"
    head: str = "Linear"
    strategy: str = "ERM"

    # --- optimization ---
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 2023
    itr: int = 1                           # independent repeats
    device: str = "auto"                   # auto | cpu | cuda | cuda:N
    early_stop_patience: int = 20
    val_ratio: float = 0.2                 # source held-out for early stopping

    # --- backbone architecture (EEGNet / CSP-Net family; other backbones ignore these) ---
    F1: int = 4                            # EEGNet temporal filters
    D: int = 2                             # EEGNet depth multiplier (spatial filters per temporal)
    F2: int = 8                            # EEGNet pointwise filters
    dropout: float = 0.25                  # EEGNet dropout probability

    # --- protocol knobs ---
    calib_ratio: float = 0.0               # target calibration slice (chronological)

    # --- strategy hyperparameters (T-TIME etc.) ---
    test_batch: int = 8
    steps: int = 1
    stride: int = 1
    temperature: float = 2.0
    # method-specific knobs (loss tradeoffs, internal LRs, capacities). Each
    # strategy reads ``cfg.hp.get(<key>, <default>)`` so behaviour is unchanged
    # unless a key is set. Populated from a preset ``hp:`` block and/or a
    # repeatable ``--hp key=value`` CLI (CLI merges over the preset).
    hp: dict = field(default_factory=dict)

    # --- io ---
    data_dir: str = "./data"
    results_dir: str = "./results"

    # --- data-derived (filled by Exp._get_data, not set by the user) ---
    n_chans: int = 0
    n_times: int = 0
    n_classes: int = 0
    sfreq: float = 0.0
    ch_names: List[str] = field(default_factory=list)   # for montage-aware stages

    def setting(self) -> str:
        """The run's identity string, used to name its results folder.

        It fingerprints the four things that make a run distinct: dataset,
        protocol, algorithm, and seed. The algorithm part is the preset name
        when one was given, otherwise it is reconstructed from the four stage
        names, so a hand-composed run still gets a readable, unique key. Two
        runs with the same setting are the same experiment, which is what lets
        results be found and compared by this string.
        """
        algo = self.algorithm or f"{self.aligner}-{self.backbone}-{self.head}-{self.strategy}"
        return f"{self.dataset}_{self.protocol}_{algo}_seed{self.seed}"


def _coerce(v: str):
    """Best-effort scalar coercion for ``--hp key=value`` values: int, then
    float, then bool, else the raw string."""
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    return v


def _load_preset(name: str) -> dict:
    """Load ``presets/<name>.yaml`` as a plain dict of field overrides.

    Returns an empty dict for an empty file. A missing preset raises with the
    list of names that do exist, since a mistyped ``--algorithm`` is the common
    cause. The keys are validated against ``Config`` fields by the caller, not
    here.
    """
    path = os.path.normpath(os.path.join(_PRESET_DIR, f"{name}.yaml"))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"preset {name!r} not found at {path}. "
            f"Available: {sorted(f[:-5] for f in os.listdir(_PRESET_DIR) if f.endswith('.yaml'))}"
        )
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def build_parser() -> argparse.ArgumentParser:
    # Note the split in defaults below. Flags that a preset may set (the stage
    # names and hyperparameters) default to None so ``resolve_config`` can tell
    # "user passed this" from "user left it to the preset". Flags a preset never
    # touches (dataset, seed, dirs) default to the dataclass value directly.
    p = argparse.ArgumentParser(prog="hustbciml.run", description="Unified EEG-decoding benchmark")
    d = Config()
    p.add_argument("--dataset", default=d.dataset)
    p.add_argument("--protocol", default=d.protocol,
                   choices=["cross_subject", "within_subject", "cross_session", "online"])
    p.add_argument("--algorithm", default=d.algorithm, help="preset name (composes stages)")
    p.add_argument("--aligner", default=None)
    p.add_argument("--augmenter", default=None)
    p.add_argument("--backbone", default=None)
    p.add_argument("--head", default=None)
    p.add_argument("--strategy", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--F1", type=int, default=None, help="EEGNet temporal filters")
    p.add_argument("--D", type=int, default=None, help="EEGNet depth multiplier")
    p.add_argument("--F2", type=int, default=None, help="EEGNet pointwise filters")
    p.add_argument("--dropout", type=float, default=None, help="EEGNet dropout")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--itr", type=int, default=d.itr)
    p.add_argument("--device", default=d.device)
    p.add_argument("--calib_ratio", type=float, default=None)
    p.add_argument("--test_batch", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--hp", action="append", default=None, metavar="KEY=VALUE",
                   help="method-specific hyperparameter override, repeatable "
                        "(e.g. --hp asfa_beta=0.3 --hp abat_eps=0.02); merges over the preset")
    p.add_argument("--data_dir", default=d.data_dir)
    p.add_argument("--results_dir", default=d.results_dir)
    p.add_argument("--list", action="store_true", help="list available plug-ins and exit")
    return p


def resolve_config(argv=None) -> Config:
    """Build the final ``Config`` from CLI args and any named preset.

    Applies the precedence rule in three passes. First a fresh ``Config`` holds
    the defaults. Then, if an ``--algorithm`` was given, its preset overwrites
    fields (validated to be real ``Config`` fields). Last, every CLI flag the
    user actually passed overrides again, where "actually passed" means the arg
    parsed to a non-None value. Method-specific ``hp`` entries get their own
    merge so a CLI ``--hp`` wins per key over the preset's ``hp:`` block.

    Returns the pair ``(cfg, ns)``: the resolved config plus the raw argparse
    namespace, because the caller still needs namespace-only flags such as
    ``--list`` that are not part of the run configuration.
    """
    ns = build_parser().parse_args(argv)
    cfg = Config()

    # 1. preset fills stage/hyperparam defaults
    if ns.algorithm:
        cfg.algorithm = ns.algorithm
        preset = _load_preset(ns.algorithm)
        valid = {f.name for f in fields(Config)}
        for k, v in preset.items():
            if k in valid:
                setattr(cfg, k, v)
            else:
                raise KeyError(f"preset {ns.algorithm!r} sets unknown field {k!r}")

    # 2. explicit CLI overrides (anything the user actually passed, i.e. not None)
    passthrough = [
        "dataset", "protocol", "aligner", "augmenter", "backbone", "head", "strategy",
        "epochs", "batch_size", "lr", "F1", "D", "F2", "dropout", "seed", "itr", "device",
        "calib_ratio", "test_batch", "steps", "stride", "temperature", "weight_decay",
        "val_ratio", "early_stop_patience", "data_dir", "results_dir",
    ]
    for k in passthrough:
        v = getattr(ns, k)
        if v is not None:
            setattr(cfg, k, v)

    # 3. method-specific hp: preset ``hp:`` (already copied above) merged under
    # any ``--hp key=value`` from the CLI, so the CLI wins per key.
    merged = dict(cfg.hp or {})
    for kv in (ns.hp or []):
        if "=" not in kv:
            raise ValueError(f"--hp expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        merged[k.strip()] = _coerce(v.strip())
    cfg.hp = merged
    return cfg, ns
