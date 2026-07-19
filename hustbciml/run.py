# run.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Entry point: ``python -m hustbciml.run --algorithm EA-EEGNet --dataset Toy``.

Resolves the config (preset + CLI), runs the chosen protocol ``--itr`` times
with incrementing seeds, and reports the mean primary metric across repeats.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from hustbciml.core.config import resolve_config
from hustbciml.core.registry import catalog

# protocol registry — extend as protocols land
from hustbciml.exp.exp_cross_subject import Exp_CrossSubject

PROTOCOLS = {
    "cross_subject": Exp_CrossSubject,
    # "within_subject": ...,  (M1)
    # "cross_session": ...,   (M1)
    # "online": ...,          (M1)
}


def _print_catalog():
    print("Available plug-ins:")
    for group, names in catalog().items():
        print(f"  {group:11s}: {', '.join(names)}")


def main(argv=None):
    cfg, ns = resolve_config(argv)
    if ns.list:
        _print_catalog()
        return

    if cfg.protocol not in PROTOCOLS:
        raise SystemExit(
            f"protocol {cfg.protocol!r} not implemented yet; available: {sorted(PROTOCOLS)}"
        )
    Exp = PROTOCOLS[cfg.protocol]

    primaries = []
    for i in range(cfg.itr):
        cfg_i = dataclasses.replace(cfg, seed=cfg.seed + i)
        print(f"\n########## itr {i + 1}/{cfg.itr}  (seed {cfg_i.seed}) ##########")
        summary = Exp(cfg_i).run()
        primaries.append(summary["primary"]["mean"])

    if cfg.itr > 1:
        arr = np.array(primaries)
        print(f"\n######### {cfg.setting()} #########")
        print(f"primary over {cfg.itr} itr: {arr.mean():.2f} +/- {arr.std():.2f}")


if __name__ == "__main__":
    main()
