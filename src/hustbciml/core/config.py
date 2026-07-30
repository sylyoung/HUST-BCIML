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

# Protocols that actually have an Exp behind them. ``run.py``'s ``PROTOCOLS``
# table is the implementation; this tuple is what ``--protocol`` advertises, and
# ``run.py`` asserts the two agree at import time. Keeping the CLI choices in
# step with the implementations is why ``--protocol online`` no longer parses
# only to die later with "not implemented yet".
IMPLEMENTED_PROTOCOLS = ("cross_subject",)

# The names ``--hp key=value`` accepts. Every entry is read somewhere by a
# strategy via ``cfg.hp.get(<key>, <default>)``, so an unlisted key would be
# accepted, stored, and never applied — the run would look tuned while executing
# at the default. Validating here turns that silent no-op into an error naming
# the closest known keys. Add a key here in the same commit that adds its
# ``cfg.hp.get`` reader.
KNOWN_HP_KEYS = frozenset({
    # experiment-level
    "dev_targets",
    # ABAT / PAT (adversarial training)
    "abat_eps", "abat_steps", "abat_warmup",
    "pat_alpha", "pat_eps", "pat_steps", "pat_warmup",
    # ASFA
    "asfa_a", "asfa_beta", "asfa_epochs", "asfa_lr", "asfa_temp",
    # BFT
    "bft_lp_epochs", "bft_no_tta", "bft_temp",
    # DJP-MMD
    "djpmmd_align", "djpmmd_mu",
    # LSFT
    "lsft_dim", "lsft_mu", "lsft_niter",
    # MDMAML
    "mdmaml_inner_lr", "mdmaml_meta_lr",
    # MEKT
    "mekt_alpha", "mekt_beta", "mekt_cov", "mekt_dim", "mekt_iter",
    "mekt_k", "mekt_rho", "mekt_t",
    # MSDT
    "msdt_batch", "msdt_bottleneck", "msdt_incons", "msdt_src_epochs",
    "msdt_src_lr", "msdt_tgt_epochs", "msdt_tgt_lr",
    # MVCNet
    "mvc_lamda1", "mvc_lamda2", "mvc_temp", "mvc_f_shift",
    # MMD-family transfer weights
    "dan_align", "jan_align",
    # MCC / MDD / CDAN
    "mcc_temp", "mdd_margin", "mdd_trade_off", "cdan_max_iter",
})


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
    # How the source validation split is drawn. "trial" holds out random source
    # trials (what the published leaderboard used); "subject" holds out whole
    # source subjects, which makes the early-stopping signal cross-subject like
    # the reported metric. See ``strategies._common.split_train_val``.
    val_split: str = "trial"
    # Re-seed at the top of every LOSO fold instead of once per sweep. Off by
    # default: it makes each fold reproducible in isolation, but it changes the
    # RNG stream and therefore every number, so the published leaderboard was
    # produced with it off.
    fold_seed: bool = False

    # --- backbone architecture (EEGNet / CSP-Net family; other backbones ignore these) ---
    F1: int = 4                            # EEGNet temporal filters
    D: int = 2                             # EEGNet depth multiplier (spatial filters per temporal)
    F2: int = 8                            # EEGNet pointwise filters
    dropout: float = 0.25                  # EEGNet dropout probability

    # --- protocol knobs ---
    # Reserved for the calibrated target protocol (a chronological slice of the
    # target used for calibration). No Exp implements it yet, so ``resolve_config``
    # rejects any non-zero value rather than let a run be labelled "calibrated"
    # while measuring the uncalibrated protocol.
    calib_ratio: float = 0.0

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
    run_tag: str = ""                      # optional suffix on the results folder
    overwrite: bool = False                # allow replacing a result from a different config
    verbose: bool = False                  # print per-epoch training progress

    # --- data-derived (filled by Exp._get_data, not set by the user) ---
    n_chans: int = 0
    n_times: int = 0
    n_classes: int = 0
    sfreq: float = 0.0
    ch_names: List[str] = field(default_factory=list)   # for montage-aware stages
    classes: List[str] = field(default_factory=list)    # class names, for label-aware stages

    # --- bookkeeping (filled by resolve_config, not set by the user) ---
    # Stage flags the CLI used to override a named preset, e.g.
    # ``--algorithm EA-EEGNet --backbone ShallowConvNet`` records
    # ``{"backbone": "ShallowConvNet"}``. Recorded so the run identity can say
    # that it is *not* the plain preset.
    stage_overrides: dict = field(default_factory=dict)

    def setting(self) -> str:
        """The run's identity string, used to name its results folder.

        It fingerprints what makes a run distinct: dataset, protocol, algorithm,
        and seed. The algorithm part is the preset name when one was given,
        otherwise it is reconstructed from all *five* stage names, so a
        hand-composed run still gets a readable, unique key — including the
        augmenter, which the benchmark varies across a whole table and which an
        earlier four-stage form silently collapsed (``--augmenter CSDA`` and
        ``--augmenter Identity`` shared one folder).

        A preset name that was partly overridden on the command line is no
        longer filed under the plain preset identity: the overridden stages are
        appended, so ``--algorithm EA-EEGNet --backbone ShallowConvNet`` cannot
        overwrite the genuine ``EA-EEGNet`` result.

        Hyperparameters (``lr``, ``epochs``, ``hp`` …) are deliberately *not* in
        the folder name — it would become unreadable and every consumer script
        reconstructs this string. They are instead recorded in full inside
        ``metrics.json``, and ``Exp_Basic.save_results`` refuses to overwrite a
        result that was produced by a different config. Use ``--run_tag`` (or a
        distinct ``--results_dir``, as the tuner does) to keep several
        hyperparameter settings side by side.
        """
        algo = self.algorithm or (
            f"{self.aligner}-{self.augmenter}-{self.backbone}-{self.head}-{self.strategy}")
        if self.algorithm and self.stage_overrides:
            algo += "+" + "+".join(f"{k}.{v}" for k, v in sorted(self.stage_overrides.items()))
        tag = f"_{self.run_tag}" if self.run_tag else ""
        return f"{self.dataset}_{self.protocol}_{algo}_seed{self.seed}{tag}"


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
    p.add_argument("--protocol", default=d.protocol, choices=list(IMPLEMENTED_PROTOCOLS))
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
    p.add_argument("--calib_ratio", type=float, default=None,
                   help="reserved; the calibrated target protocol is not implemented, "
                        "so any non-zero value is rejected rather than ignored")
    p.add_argument("--test_batch", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--val_split", default=None, choices=["trial", "subject"],
                   help="source validation split for early stopping: random trials "
                        "(default, as published) or held-out source subjects")
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--hp", action="append", default=None, metavar="KEY=VALUE",
                   help="method-specific hyperparameter override, repeatable "
                        "(e.g. --hp asfa_beta=0.3 --hp abat_eps=0.02); merges over the preset")
    p.add_argument("--data_dir", default=d.data_dir)
    p.add_argument("--results_dir", default=d.results_dir)
    p.add_argument("--run_tag", default=None,
                   help="suffix appended to the results folder, to keep several "
                        "hyperparameter settings of one algorithm side by side")
    p.add_argument("--overwrite", action="store_true",
                   help="allow overwriting an existing result produced by a different config")
    p.add_argument("--verbose", action="store_true",
                   help="print per-epoch training/validation progress")
    p.add_argument("--fold_seed", action="store_true",
                   help="re-seed at each LOSO fold so folds are independently "
                        "reproducible (changes the RNG stream, so numbers differ "
                        "from the published leaderboard)")
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
        "val_ratio", "val_split", "early_stop_patience", "data_dir", "results_dir", "run_tag",
        "overwrite", "verbose", "fold_seed",
    ]
    # Stage flags that redefine what the preset composes. Recording which ones
    # the user overrode is what keeps a modified preset from being filed — and
    # published — under the untouched preset's name.
    stage_flags = ("aligner", "augmenter", "backbone", "head", "strategy")
    for k in passthrough:
        v = getattr(ns, k)
        if v is not None:
            if ns.algorithm and k in stage_flags and v != getattr(cfg, k):
                cfg.stage_overrides[k] = v
            setattr(cfg, k, v)

    # 3. method-specific hp: preset ``hp:`` (already copied above) merged under
    # any ``--hp key=value`` from the CLI, so the CLI wins per key.
    merged = dict(cfg.hp or {})
    for kv in (ns.hp or []):
        if "=" not in kv:
            raise ValueError(f"--hp expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        merged[k.strip()] = _coerce(v.strip())
    # Every hp key must be one a strategy actually reads. An unknown key would
    # otherwise be carried silently to the end of the run and applied nowhere,
    # so a typo produces a default-valued run wearing a tuned run's label.
    unknown = sorted(set(merged) - KNOWN_HP_KEYS)
    if unknown:
        raise KeyError(
            f"unknown --hp key(s) {unknown}; no strategy reads them, so setting them "
            f"would have no effect. Known keys: {sorted(KNOWN_HP_KEYS)}"
        )
    cfg.hp = merged

    # ``calib_ratio`` describes a target-calibration protocol that no Exp
    # implements yet: nothing reads it, so accepting a non-zero value would
    # report an uncalibrated run under a calibrated label. Refuse instead.
    if cfg.calib_ratio:
        raise NotImplementedError(
            "--calib_ratio is reserved for the calibrated target protocol, which is not "
            "implemented; a non-zero value would be silently ignored. Use 0.0."
        )
    return cfg, ns
